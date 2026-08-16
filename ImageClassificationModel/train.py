"""
================================================================================
  Kaggle Image Classification Pipeline
  ─────────────────────────────────────
  Features:
    • timm backbone (ConvNeXt-Tiny / EfficientNetV2-S) with ImageNet pretraining
    • Two-phase freeze → unfreeze fine-tuning
    • Mixed-precision (AMP) training
    • AdamW + label-smoothed CrossEntropy (+ optional class weights)
    • TrivialAugmentWide / RandAugment
    • 80/20 train/val split
    • Test-Time Augmentation (TTA) inference
================================================================================
"""

import os
import glob
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast

import timm
from timm.data import resolve_data_config, create_transform

from torchvision import transforms
from torchvision.transforms import (
    Compose, Resize, CenterCrop, RandomHorizontalFlip,
    TrivialAugmentWide, RandAugment, ToTensor, Normalize,
)

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION — Fill in your dataset-specific values here
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration object.
    ► Update the placeholder values below once you know your dataset details.
    """

    # ── Dataset details ──────────────────────────────────────────────────────
    TRAIN_CSV       = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\train.csv"
    TEST_CSV        = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\test.csv"
    IMAGES_DIR      = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\images\images"
    NUM_CLASSES     = 4                        # Number of target classes: tom, jerry, both, none
    CLASS_IMBALANCE = True                     # Enabled
    IMAGE_SIZE      = 224                      # Target H×W (224 for ConvNeXt-Tiny, 300 for EfficientNetV2-S)

    # ── Model ────────────────────────────────────────────────────────────────
    # Options: "convnext_tiny", "tf_efficientnetv2_s"
    MODEL_NAME      = "convnext_tiny"
    PRETRAINED      = True

    # ── Training hyper-parameters ────────────────────────────────────────────
    PHASE1_EPOCHS   = 3                        # Head-only training epochs
    PHASE2_EPOCHS   = 12                       # Full fine-tuning epochs
    BATCH_SIZE      = 32
    NUM_WORKERS     = 0                        # DataLoader workers (set 0 on Windows to avoid issues)
    LR_HEAD         = 1e-3                     # Learning rate for Phase 1 (head only)
    LR_FINETUNE     = 1e-4                     # Learning rate for Phase 2 (1/10th — gentler)
    WEIGHT_DECAY    = 1e-2                     # AdamW weight decay (good default)
    LABEL_SMOOTHING = 0.1                      # Reduces overconfidence on noisy labels

    # ── Augmentation strategy ────────────────────────────────────────────────
    # Options: "trivial" → TrivialAugmentWide, "rand" → RandAugment
    AUG_STRATEGY    = "trivial"

    # ── Paths ────────────────────────────────────────────────────────────────
    SAVE_DIR        = "checkpoints"
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model_fold_{fold}.pth")
    NUM_FOLDS       = 5

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED            = 42
    VAL_SPLIT       = 0.20                     # 80/20 split

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()

# ══════════════════════════════════════════════════════════════════════════════
# 2. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42):
    """Pin every source of randomness for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # Disable for reproducibility

seed_everything(cfg.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ImageClassificationDataset(Dataset):
    """
    Loads images using a CSV file mapping filenames to labels.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, csv_path: str, images_dir: str, transform=None, is_test: bool = False):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_test = is_test

        self.df = pd.read_csv(csv_path)
        self.classes = ["none", "tom", "jerry", "both"]  # 0: none, 1: tom, 2: jerry, 3: both
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for _, row in self.df.iterrows():
            img_path = self.images_dir / row["filename"]
            if is_test:
                label = -1
            else:
                label = int(row["appearance"])
            self.samples.append((str(img_path), label))

        print(f"[Dataset] Loaded {len(self.samples)} images from '{csv_path}' ({'test' if is_test else 'train'} mode)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_path            # Return path for submission mapping
        return image, label


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMS (Augmentation)
# ══════════════════════════════════════════════════════════════════════════════

# ImageNet channel statistics — used for normalising inputs to match
# what the pretrained backbone expects.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int, strategy: str = "trivial") -> Compose:
    """
    Training augmentation pipeline.

    Why TrivialAugmentWide / RandAugment?
      • They are "parameter-free" (or nearly so) augmentation policies that
        consistently outperform hand-tuned pipelines on a wide range of
        datasets, especially when you don't have time to tune aug params.
      • TrivialAugment is the simplest — each image gets exactly ONE random
        transform at a random magnitude.  Great default.
      • RandAugment applies N transforms, each at magnitude M.
    """
    aug_list = [
        Resize((image_size, image_size)),       # Deterministic resize first
        RandomHorizontalFlip(p=0.5),            # Cheap but effective
    ]

    if strategy == "trivial":
        aug_list.append(TrivialAugmentWide())
    elif strategy == "rand":
        aug_list.append(RandAugment(num_ops=2, magnitude=9))
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

    aug_list += [
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return Compose(aug_list)


def get_val_transforms(image_size: int) -> Compose:
    """
    Validation/test transforms — NO augmentation, only resize + normalise.
    This ensures a clean, deterministic evaluation.
    """
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(image_size: int) -> list[Compose]:
    """
    Test-Time Augmentation transforms.
    We predict on:
      1. The original (clean) image
      2. A horizontally-flipped version
    Then average the softmax probabilities.  This simple TTA typically gains
    ~0.5–1.0 % accuracy for free.
    """
    base = [Resize((image_size, image_size)), ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    flipped = [Resize((image_size, image_size)),
               transforms.RandomHorizontalFlip(p=1.0),   # Deterministic flip
               ToTensor(),
               Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    return [Compose(base), Compose(flipped)]


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained backbone via `timm` and swap the classification head
    to match our number of classes.

    Why timm?
      • Largest collection of SOTA pretrained vision models in one API.
      • `num_classes=` automatically replaces the head — no manual surgery.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    print(f"[Model] Loaded '{model_name}' with {num_classes}-class head  "
          f"(pretrained={pretrained})")
    return model


def freeze_backbone(model: nn.Module):
    """
    Freeze every parameter EXCEPT the final classifier head.

    Why?
      • The backbone already knows rich visual features from ImageNet.
      • Training only the head first lets it "catch up" to the backbone
        without destroying pretrained weights with large random gradients.
    """
    # ── Identify the head parameter names (timm convention) ───────────────
    head_names = set()
    # timm models expose `.get_classifier()` which returns the head module
    classifier = model.get_classifier()
    for name, _ in classifier.named_parameters():
        head_names.add(name)

    # Fully-qualified names include the parent module prefix
    classifier_prefix = ""
    for name, module in model.named_modules():
        if module is classifier:
            classifier_prefix = name
            break

    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if name.startswith(classifier_prefix):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1

    print(f"[Freeze] Backbone frozen: {frozen} params frozen, "
          f"{trainable} head params trainable")


def unfreeze_all(model: nn.Module):
    """Unfreeze the entire network for full fine-tuning (Phase 2)."""
    for param in model.named_parameters():
        param[1].requires_grad = True
    total = sum(1 for _ in model.parameters())
    print(f"[Unfreeze] All {total} parameter groups are now trainable")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASS WEIGHTS (for imbalanced datasets)
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency weights so that rare classes contribute more
    to the loss.  Formula: weight_c = N_total / (num_classes * N_c)

    This is the same formula used by sklearn's `compute_class_weight('balanced')`.
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total  = len(labels)

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        n_c = counts.get(cls_idx, 1)           # Avoid division by zero
        weights[cls_idx] = total / (num_classes * n_c)

    print(f"[Weights] Class weights: {weights.tolist()}")
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    phase_name: str = "",
) -> tuple[float, float]:
    """
    Train for a single epoch with mixed-precision.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc=f"  [{phase_name}] Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

        # ── Mixed-precision forward pass ──────────────────────────────────
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Scaled backward pass ──────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics ───────────────────────────────────────────────────────
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate on the validation set (no gradients, no augmentation).
    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in tqdm(loader, desc="  [Val]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training(cfg: Config):
    """
    Orchestrates the Stratified K-Fold training pipeline.
    """
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # ── 8a. Build dataset ─────────────────────────────────────────────────
    full_dataset = ImageClassificationDataset(cfg.TRAIN_CSV, cfg.IMAGES_DIR)
    
    # Extract labels for stratified K-fold split
    labels = [sample[1] for sample in full_dataset.samples]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    # Helper to wrap subset index lists with appropriate transforms
    class TransformIndicesDataset(Dataset):
        def __init__(self, parent_dataset, indices, transform):
            self.parent_dataset = parent_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img_path, label = self.parent_dataset.samples[self.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    oof_preds = np.zeros((len(full_dataset), cfg.NUM_CLASSES))
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_idx + 1} / {cfg.NUM_FOLDS}")
        print("=" * 70)
        
        train_ds = TransformIndicesDataset(full_dataset, train_idx, get_train_transforms(cfg.IMAGE_SIZE, cfg.AUG_STRATEGY))
        val_ds   = TransformIndicesDataset(full_dataset, val_idx,   get_val_transforms(cfg.IMAGE_SIZE))
        
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        
        # Build fresh model
        model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.PRETRAINED)
        model.to(cfg.DEVICE)
        
        # Loss function
        if cfg.CLASS_IMBALANCE:
            # Recompute weights specific to the training fold
            fold_labels = [labels[i] for i in train_idx]
            counts = Counter(fold_labels)
            total = len(fold_labels)
            weights = torch.zeros(cfg.NUM_CLASSES, dtype=torch.float32)
            for cls_idx in range(cfg.NUM_CLASSES):
                n_c = counts.get(cls_idx, 1)
                weights[cls_idx] = total / (cfg.NUM_CLASSES * n_c)
            weights = weights.to(cfg.DEVICE)
            print(f"[Weights] Fold class weights: {weights.tolist()}")
            criterion = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=cfg.LABEL_SMOOTHING,
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
            
        scaler = GradScaler(enabled=(cfg.DEVICE.type == "cuda"))
        best_val_acc = 0.0
        fold_best_path = cfg.BEST_MODEL_PATH.format(fold=fold_idx)
        
        # Phase 1: Train head only
        print(f"\n  [Fold {fold_idx + 1}] Phase 1: Head-only training")
        freeze_backbone(model)
        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.LR_HEAD,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        
        for epoch in range(1, cfg.PHASE1_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_p1,
                scaler, cfg.DEVICE, epoch, phase_name=f"F{fold_idx+1}-P1"
            )
            val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)
            print(f"  Epoch {epoch}/{cfg.PHASE1_EPOCHS}  |  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_best_path)
                print(f"  [OK] New best model saved (val_acc={val_acc:.4f})")
                
        # Phase 2: Full fine-tuning
        print(f"\n  [Fold {fold_idx + 1}] Phase 2: Full fine-tuning")
        unfreeze_all(model)
        optimizer_p2 = optim.AdamW(
            model.parameters(),
            lr=cfg.LR_FINETUNE,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p2,
            T_max=cfg.PHASE2_EPOCHS,
            eta_min=1e-7,
        )
        
        for epoch in range(1, cfg.PHASE2_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_p2,
                scaler, cfg.DEVICE, epoch, phase_name=f"F{fold_idx+1}-P2"
            )
            val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)
            scheduler_p2.step()
            
            current_lr = scheduler_p2.get_last_lr()[0]
            print(f"  Epoch {epoch}/{cfg.PHASE2_EPOCHS}  |  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
                  f"LR: {current_lr:.2e}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_best_path)
                print(f"  [OK] New best model saved (val_acc={val_acc:.4f})")
                
        print(f"\n  Fold {fold_idx + 1} training complete. Best Val Acc: {best_val_acc:.4f}")
        fold_accuracies.append(best_val_acc)
        
        # Load the best weights of this fold to generate OOF predictions
        model.load_state_dict(torch.load(fold_best_path, map_location=cfg.DEVICE))
        model.eval()
        
        val_loader_no_shuffle = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        
        start_idx = 0
        with torch.no_grad():
            for images, _ in val_loader_no_shuffle:
                images = images.to(cfg.DEVICE)
                with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
                    logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_size = images.size(0)
                fold_val_indices = val_idx[start_idx : start_idx + batch_size]
                oof_preds[fold_val_indices] = probs
                start_idx += batch_size

    print(f"\n{'-' * 70}")
    print(f"  All folds training complete. Mean Fold Acc: {np.mean(fold_accuracies):.4f}")
    oof_accuracy = (oof_preds.argmax(axis=1) == np.array(labels)).mean()
    print(f"  Out-of-Fold (OOF) Accuracy: {oof_accuracy:.4f}")
    print(f"{'-' * 70}\n")
    
    # Save the OOF predictions for validation report
    np.save(os.path.join(cfg.SAVE_DIR, "oof_preds.npy"), oof_preds)
    
    return oof_preds


# ══════════════════════════════════════════════════════════════════════════════
# 9. INFERENCE WITH TEST-TIME AUGMENTATION (TTA)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(cfg: Config) -> pd.DataFrame:
    """
    Load all K-fold models and ensemble predict on every image in the test set in batches.
    """
    print("\n" + "=" * 70)
    print("  ENSEMBLE INFERENCE (with TTA - Batched)")
    print("=" * 70)

    # Build and load all K-fold models
    models = []
    for fold_idx in range(cfg.NUM_FOLDS):
        model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, pretrained=False)
        fold_best_path = cfg.BEST_MODEL_PATH.format(fold=fold_idx)
        model.load_state_dict(torch.load(fold_best_path, map_location=cfg.DEVICE))
        model.to(cfg.DEVICE)
        model.eval()
        models.append(model)
        print(f"[Inference] Loaded Fold {fold_idx + 1} weights from '{fold_best_path}'")

    # Load test dataset with validation-style transforms (normalization and resize)
    test_dataset = ImageClassificationDataset(
        cfg.TEST_CSV,
        cfg.IMAGES_DIR,
        transform=get_val_transforms(cfg.IMAGE_SIZE),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    for images, img_paths in tqdm(test_loader, desc="  [Ensemble Inference]"):
        images = images.to(cfg.DEVICE)
        # TTA: Horizontal flip of the batch tensors on GPU
        flipped_images = torch.flip(images, dims=[3])

        # Initialize accumulated predictions
        batch_probs = torch.zeros((images.size(0), cfg.NUM_CLASSES), device=cfg.DEVICE)

        for model in models:
            with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
                # Predictions on original batch
                logits_normal = model(images)
                probs_normal = torch.softmax(logits_normal, dim=1)

                # Predictions on flipped batch
                logits_flipped = model(flipped_images)
                probs_flipped = torch.softmax(logits_flipped, dim=1)

            batch_probs += (probs_normal + probs_flipped)

        # Average across folds and TTA passes (2 passes)
        batch_probs /= (cfg.NUM_FOLDS * 2)
        batch_probs = batch_probs.cpu().numpy()

        for idx, img_path in enumerate(img_paths):
            pred_probs = batch_probs[idx]
            pred_class = pred_probs.argmax().item()
            confidence = pred_probs[pred_class].item()
            filename = Path(img_path).name
            results.append({
                "filename":        filename,
                "appearance":      pred_class,
                "confidence":      round(confidence, 4),
            })

    df = pd.DataFrame(results)
    
    # Save submission-ready file (matching sample_submission.csv format)
    submission_df = df[["filename", "appearance"]]
    submission_path = os.path.join(cfg.SAVE_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"[Inference] Saved submission format to '{submission_path}'")
    
    # Save full prediction details with confidence scores
    predictions_path = os.path.join(cfg.SAVE_DIR, "predictions.csv")
    df.to_csv(predictions_path, index=False)
    print(f"[Inference] Saved full predictions with confidence to '{predictions_path}'")
    
    print(submission_df.head(10).to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 10. EVALUATION HELPERS (optional — useful during development)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_val_set(cfg: Config):
    """
    Load the OOF predictions and print the consolidated classification report + confusion matrix.
    """
    oof_preds = np.load(os.path.join(cfg.SAVE_DIR, "oof_preds.npy"))
    full_dataset = ImageClassificationDataset(cfg.TRAIN_CSV, cfg.IMAGES_DIR)
    labels = [sample[1] for sample in full_dataset.samples]
    
    oof_labels = oof_preds.argmax(axis=1)
    class_names = full_dataset.classes
    
    print("\n-- Out-of-Fold (OOF) Classification Report --")
    print(classification_report(labels, oof_labels, target_names=class_names))
    print("-- Out-of-Fold (OOF) Confusion Matrix --")
    print(confusion_matrix(labels, oof_labels))


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {cfg.DEVICE}")
    print(f"Model:  {cfg.MODEL_NAME}")
    print(f"Image size: {cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE}")
    print(f"Phases: {cfg.PHASE1_EPOCHS} (head) + {cfg.PHASE2_EPOCHS} (fine-tune)")
    print()

    # ── Step 1: Train ─────────────────────────────────────────────────────
    oof_preds = run_training(cfg)

    # ── Step 2: Evaluate on validation set ────────────────────────────────
    evaluate_val_set(cfg)

    # ── Step 3: Inference on test set with TTA ────────────────────────────
    predict_with_tta(cfg)

    print("\n-- Pipeline complete --")
