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
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")

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
    Orchestrates the full two-phase training pipeline:
      Phase 1 — Head-only   (high LR, frozen backbone)
      Phase 2 — Full model  (low LR, everything unfrozen)
    """
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # ── 8a. Build dataset & 80/20 split ───────────────────────────────────
    full_dataset = ImageClassificationDataset(cfg.TRAIN_CSV, cfg.IMAGES_DIR)
    num_val   = int(len(full_dataset) * cfg.VAL_SPLIT)
    num_train = len(full_dataset) - num_val

    train_subset, val_subset = random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    print(f"[Split] Train: {num_train} | Val: {num_val}")

    # ── 8b. Wrap subsets with appropriate transforms ──────────────────────
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    train_ds = TransformSubset(train_subset, get_train_transforms(cfg.IMAGE_SIZE, cfg.AUG_STRATEGY))
    val_ds   = TransformSubset(val_subset,   get_val_transforms(cfg.IMAGE_SIZE))

    # ── 8c. DataLoaders ───────────────────────────────────────────────────
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

    # ── 8d. Model ─────────────────────────────────────────────────────────
    model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.PRETRAINED)
    model.to(cfg.DEVICE)

    # ── 8e. Loss function ─────────────────────────────────────────────────
    if cfg.CLASS_IMBALANCE:
        weights = compute_class_weights(full_dataset, cfg.NUM_CLASSES).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # ── 8f. Mixed-precision scaler ────────────────────────────────────────
    scaler = GradScaler(enabled=(cfg.DEVICE.type == "cuda"))

    best_val_acc  = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ══════════════════════════════════════════════════════════════════════
    #   PHASE 1 — Train HEAD only (backbone frozen)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 1: Head-only training (backbone frozen)")
    print("=" * 70)

    freeze_backbone(model)

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR_HEAD,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    for epoch in range(1, cfg.PHASE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p1,
            scaler, cfg.DEVICE, epoch, phase_name="P1-Head",
        )
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch}/{cfg.PHASE1_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  [OK] New best model saved (val_acc={val_acc:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    #   PHASE 2 — Fine-tune ENTIRE network (backbone unfrozen)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 2: Full fine-tuning (all layers unfrozen)")
    print("=" * 70)

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
            scaler, cfg.DEVICE, epoch, phase_name="P2-Full",
        )
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)
        scheduler_p2.step()

        current_lr = scheduler_p2.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch}/{cfg.PHASE2_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  [OK] New best model saved (val_acc={val_acc:.4f})")

    print(f"\n{'-' * 70}")
    print(f"  Training complete.  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Best model saved to: {cfg.BEST_MODEL_PATH}")
    print(f"{'-' * 70}\n")

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 9. INFERENCE WITH TEST-TIME AUGMENTATION (TTA)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(cfg: Config) -> pd.DataFrame:
    """
    Load the best saved model and predict on every image in `cfg.TEST_DIR`.
    """
    print("\n" + "=" * 70)
    print("  INFERENCE (with TTA)")
    print("=" * 70)

    model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()
    print(f"[Inference] Loaded weights from '{cfg.BEST_MODEL_PATH}'")

    tta_transforms = get_tta_transforms(cfg.IMAGE_SIZE)

    test_dataset = ImageClassificationDataset(
        cfg.TEST_CSV,
        cfg.IMAGES_DIR,
        transform=None,
        is_test=True,
    )

    results = []

    for raw_image_tensor, img_path in tqdm(test_dataset, desc="  [TTA Inference]"):
        pil_image = Image.open(img_path).convert("RGB")
        avg_probs = torch.zeros(cfg.NUM_CLASSES, device=cfg.DEVICE)

        for tta_tfm in tta_transforms:
            tensor = tta_tfm(pil_image).unsqueeze(0).to(cfg.DEVICE)

            with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
                logits = model(tensor)

            probs = torch.softmax(logits, dim=1).squeeze(0)
            avg_probs += probs

        avg_probs /= len(tta_transforms)

        pred_class  = avg_probs.argmax().item()
        confidence  = avg_probs[pred_class].item()

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
    Run the best model on the validation set and print a full
    classification report + confusion matrix.
    """
    full_dataset = ImageClassificationDataset(cfg.TRAIN_CSV, cfg.IMAGES_DIR)
    num_val   = int(len(full_dataset) * cfg.VAL_SPLIT)
    num_train = len(full_dataset) - num_val

    _, val_subset = random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )

    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset, self.transform = subset, transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    val_ds = TransformSubset(val_subset, get_val_transforms(cfg.IMAGE_SIZE))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    for images, labels in tqdm(val_loader, desc="  [Eval]"):
        images = images.to(cfg.DEVICE)
        with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
            logits = model(images)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    class_names = full_dataset.classes
    print("\n-- Classification Report --")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("-- Confusion Matrix --")
    print(confusion_matrix(all_labels, all_preds))


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
    model, history = run_training(cfg)

    # ── Step 2: Evaluate on validation set ────────────────────────────────
    evaluate_val_set(cfg)

    # ── Step 3: Inference on test set with TTA ────────────────────────────
    predict_with_tta(cfg)

    print("\n-- Pipeline complete --")
