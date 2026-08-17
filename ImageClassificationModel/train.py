# -*- coding: utf-8 -*-
"""
================================================================================
  Kaggle Tom & Jerry Classification Pipeline - 0.95+ Architecture
  ──────────────────────────────────────────────────────────────────
  Features Adapted from 0.95 Leaderboard Submission:
    1. EMA (Exponential Moving Average, decay=0.999): Smooths model weights to
       find flat, highly generalizable minima on unseen test data.
    2. Mixup Regularization (alpha=0.2): Linearly blends image pairs and labels
       during Phase 2 to prevent background memorization.
    3. Gradient Clipping (norm=1.0): Prevents gradient shocks in fine-tuning.
    4. Inverse-Frequency Class Weights + Label Smoothing (0.1): Matches the test
       set's dominant 44% 'none' and 19% 'both' distribution.
    5. Macro F1 Checkpointing & Early Stopping (patience=4): Evaluates on the
       exact competition metric.
    6. Uniform Fine-Tuning LR (1e-4) with Linear Warmup + Cosine Decay.
    7. 5-View High-Resolution TTA:
         • View 1: Original 384×384
         • View 2: Horizontal Flip
         • View 3: Zoomed 1.15× Center Crop (enlarges small Jerry characters)
         • View 4: Zoomed 1.15× + Horizontal Flip
         • View 5: Tight Crop 1.25× Center Crop
================================================================================
"""

import os
import copy
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import timm

from torchvision import transforms
from torchvision.transforms import (
    Compose, Resize, CenterCrop, RandomHorizontalFlip,
    TrivialAugmentWide, ToTensor, Normalize,
)
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Dataset Paths ────────────────────────────────────────────────────────
    TRAIN_CSV       = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\train.csv"
    TEST_CSV        = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\test.csv"
    IMAGES_DIR      = r"C:\Users\Yasiru Sithum\OneDrive\Documents\webprojects\MachineLearning\my_dataset\images\images"
    
    NUM_CLASSES     = 4                            # 0: none, 1: tom, 2: jerry, 3: both
    CLASS_NAMES     = ["none", "tom", "jerry", "both"]
    CLASS_IMBALANCE = True                         # Essential for matching test set prior
    IMAGE_SIZE      = 384                          # 384x384 resolution

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_NAME      = "convnext_tiny"
    PRETRAINED      = True

    # ── Training Hyperparameters ─────────────────────────────────────────────
    N_FOLDS             = 5
    PHASE1_EPOCHS       = 2                        # Head-only warmup
    PHASE2_MAX_EPOCHS   = 15                       # Full fine-tuning ceiling
    EARLY_STOP_PATIENCE = 4                        # Early stopping on validation Macro F1
    WARMUP_EPOCHS       = 2                        # Warmup epochs for Phase 2
    BATCH_SIZE          = 16                       # Optimized for 384px on GPU
    NUM_WORKERS         = 0                        # Safe for Windows environment
    
    LR_HEAD             = 1e-3                     # Phase 1 Head LR
    LR_FINETUNE         = 1e-4                     # Phase 2 uniform LR (avoids underfitting)
    WEIGHT_DECAY        = 1e-2
    LABEL_SMOOTHING     = 0.1

    # ── Regularization & EMA ─────────────────────────────────────────────────
    GRAD_CLIP_NORM      = 1.0                      # Gradient clipping
    MIXUP_ALPHA         = 0.2                      # Mixup parameter
    EMA_DECAY           = 0.999                    # Exponential Moving Average weight decay
    AUG_STRATEGY        = "trivial"

    SAVE_DIR            = "checkpoints"
    SEED                = 42
    DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

# ══════════════════════════════════════════════════════════════════════════════
# 2. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 3. EMA & MIXUP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Returns mixed inputs, pairs of targets, and blending lambda."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMS & 5-VIEW TTA
# ══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transforms(sz: int, strategy: str = "trivial") -> Compose:
    aug = [Resize((sz, sz)), RandomHorizontalFlip(p=0.5)]
    if strategy == "trivial":
        aug.append(TrivialAugmentWide())
    aug += [ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return Compose(aug)

def get_val_transforms(sz: int) -> Compose:
    return Compose([Resize((sz, sz)), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def get_5view_tta_transforms(sz: int) -> list[Compose]:
    """
    5-View Multi-Scale TTA:
      1. Original
      2. Horizontal Flip
      3. Zoom 1.15x CenterCrop (enlarges Jerry & small features)
      4. Zoom 1.15x CenterCrop + Horizontal Flip
      5. Tight Crop 1.25x CenterCrop
    """
    norm = [ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    v1_orig     = Compose([Resize((sz, sz))] + norm)
    v2_flip     = Compose([Resize((sz, sz)), RandomHorizontalFlip(p=1.0)] + norm)
    
    z1 = int(sz * 1.15)
    v3_zoom     = Compose([Resize((z1, z1)), CenterCrop(sz)] + norm)
    v4_zoom_flp = Compose([Resize((z1, z1)), CenterCrop(sz), RandomHorizontalFlip(p=1.0)] + norm)
    
    z2 = int(sz * 1.25)
    v5_tight    = Compose([Resize((z2, z2)), CenterCrop(sz)] + norm)
    
    return [v1_orig, v2_flip, v3_zoom, v4_zoom_flp, v5_tight]


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class ImageClassificationDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, transform=None, is_test: bool = False):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_test = is_test
        self.df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in self.df.iterrows():
            img_path = self.images_dir / row["filename"]
            label = -1 if is_test else int(row["appearance"])
            self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img, Path(img_path).name) if self.is_test else (img, label)


class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.base.samples[self.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL & LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

def freeze_backbone(model: nn.Module):
    classifier = model.get_classifier()
    prefix = ""
    for n, m in model.named_modules():
        if m is classifier:
            prefix = n
            break
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefix)

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True

def compute_class_weights(dataset: Dataset, num_classes: int = 4) -> torch.Tensor:
    labels = [l for _, l in dataset.samples]
    counts = Counter(labels)
    total = len(labels)
    w = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        w[c] = total / (num_classes * counts.get(c, 1))
    print(f"[Weights] Class weights: {w.tolist()}")
    return w


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING & VALIDATION ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    phase_name, ema=None, mixup_alpha=0.0, grad_clip=0.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  [{phase_name}] Ep {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mixup = mixup_alpha > 0.0 and phase_name.startswith("P2")
        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  [Val]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return val_loss, macro_f1, acc, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# 8. 5-FOLD K-FOLD PIPELINE (WITH EMA, MIXUP, EARLY STOPPING)
# ══════════════════════════════════════════════════════════════════════════════

def run_training(cfg: Config):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    full_dataset = ImageClassificationDataset(cfg.TRAIN_CSV, cfg.IMAGES_DIR)

    weights = compute_class_weights(full_dataset, cfg.NUM_CLASSES).to(cfg.DEVICE) if cfg.CLASS_IMBALANCE else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.LABEL_SMOOTHING)
    scaler = GradScaler(enabled=(cfg.DEVICE.type == "cuda"))

    labels = [l for _, l in full_dataset.samples]
    kf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)

    oof_preds = np.zeros(len(labels), dtype=int)
    fold_f1s = []
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 70}\n  FOLD {fold + 1}/{cfg.N_FOLDS} (0.95+ Architecture @ {cfg.IMAGE_SIZE}px)\n{'=' * 70}")

        train_ds = TransformSubset(full_dataset, train_idx, get_train_transforms(cfg.IMAGE_SIZE, cfg.AUG_STRATEGY))
        val_ds   = TransformSubset(full_dataset, val_idx,   get_val_transforms(cfg.IMAGE_SIZE))

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.PRETRAINED).to(cfg.DEVICE)
        ema = EMA(model, cfg.EMA_DECAY)

        best_val_f1 = 0.0
        best_val_acc = 0.0
        fold_path = os.path.join(cfg.SAVE_DIR, f"best_model_fold_{fold}.pth")

        # ── Phase 1: Classification Head Warmup ──────────────────────────────
        print(f"\n  [Fold {fold+1}] Phase 1: Head-Only Training ({cfg.PHASE1_EPOCHS} eps)")
        freeze_backbone(model)
        opt_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY)

        for ep in range(1, cfg.PHASE1_EPOCHS + 1):
            tl, ta = train_one_epoch(model, train_loader, criterion, opt_p1, scaler,
                                     cfg.DEVICE, ep, "P1-Head", ema=ema)
            vl, vf1, vacc, _, _ = validate(model, val_loader, criterion, cfg.DEVICE)
            print(f"  Ep {ep}/{cfg.PHASE1_EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Macro F1: {vf1:.4f} Acc: {vacc:.4f}")
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_val_acc = vacc
                torch.save(ema.state_dict(), fold_path)

        # ── Phase 2: Full Fine-Tuning with Mixup, Grad Clip, EMA & Early Stop ─
        print(f"\n  [Fold {fold+1}] Phase 2: Full Fine-Tuning (Max {cfg.PHASE2_MAX_EPOCHS} eps, Patience={cfg.EARLY_STOP_PATIENCE})")
        unfreeze_all(model)

        opt_p2 = optim.AdamW(model.parameters(), lr=cfg.LR_FINETUNE, weight_decay=cfg.WEIGHT_DECAY)
        warmup = LinearLR(opt_p2, start_factor=0.1, total_iters=cfg.WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(opt_p2, T_max=max(1, cfg.PHASE2_MAX_EPOCHS - cfg.WARMUP_EPOCHS), eta_min=1e-7)
        sched  = SequentialLR(opt_p2, [warmup, cosine], milestones=[cfg.WARMUP_EPOCHS])

        patience_counter = 0

        for ep in range(1, cfg.PHASE2_MAX_EPOCHS + 1):
            lr_now = opt_p2.param_groups[0]["lr"]
            tl, ta = train_one_epoch(model, train_loader, criterion, opt_p2, scaler,
                                     cfg.DEVICE, ep, "P2-Full", ema=ema,
                                     mixup_alpha=cfg.MIXUP_ALPHA, grad_clip=cfg.GRAD_CLIP_NORM)

            # Evaluate with EMA weights
            orig_sd = copy.deepcopy(model.state_dict())
            ema.apply(model)
            vl, vf1, vacc, vpreds, _ = validate(model, val_loader, criterion, cfg.DEVICE)
            model.load_state_dict(orig_sd)
            sched.step()

            print(f"  Ep {ep}/{cfg.PHASE2_MAX_EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Macro F1: {vf1:.4f} Acc: {vacc:.4f} | LR: {lr_now:.2e}")

            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_val_acc = vacc
                oof_preds[val_idx] = vpreds
                torch.save(ema.state_dict(), fold_path)
                patience_counter = 0
                print(f"  --> Saved new best EMA checkpoint (Macro F1: {vf1:.4f}, Acc: {vacc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                    print(f"  [Early Stopping] No improvement for {cfg.EARLY_STOP_PATIENCE} epochs. Stopping Fold {fold+1}.")
                    break

        fold_f1s.append(best_val_f1)
        fold_accs.append(best_val_acc)

    overall_f1 = f1_score(labels, oof_preds, average="macro", zero_division=0)
    overall_acc = (oof_preds == np.array(labels)).mean()

    print(f"\n{'=' * 70}\n  5-FOLD TRAINING COMPLETE\n{'=' * 70}")
    for i, (f1_val, acc_val) in enumerate(zip(fold_f1s, fold_accs)):
        print(f"  Fold {i+1}: Macro F1 = {f1_val:.4f} | Acc = {acc_val:.4f}")
    print(f"\n  Overall Out-of-Fold Macro F1: {overall_f1:.4f}")
    print(f"  Overall Out-of-Fold Accuracy: {overall_acc:.4f}")
    print(f"{'=' * 70}\n")

    print("-- Out-of-Fold Classification Report --")
    print(classification_report(labels, oof_preds, target_names=cfg.CLASS_NAMES))
    print("-- Out-of-Fold Confusion Matrix --")
    print(confusion_matrix(labels, oof_preds))

    return fold_f1s


# ══════════════════════════════════════════════════════════════════════════════
# 9. 5-VIEW BATCHED TTA ENSEMBLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_batched_tta_inference(cfg: Config) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  FAST BATCHED ENSEMBLE INFERENCE (5 Models x 5-View TTA)")
    print("=" * 70)

    # Load 5 fold models with EMA weights
    models = []
    for fold in range(cfg.N_FOLDS):
        fpath = os.path.join(cfg.SAVE_DIR, f"best_model_fold_{fold}.pth")
        if os.path.exists(fpath):
            m = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(fpath, map_location=cfg.DEVICE))
            m.to(cfg.DEVICE).eval()
            models.append(m)
            print(f"  [Inference] Loaded Fold {fold+1} model from '{fpath}'")

    if not models:
        print("[Error] No model checkpoints found for inference.")
        return

    tta_tfms = get_5view_tta_transforms(cfg.IMAGE_SIZE)
    test_df = pd.read_csv(cfg.TEST_CSV)
    num_samples = len(test_df)
    total_probs = np.zeros((num_samples, cfg.NUM_CLASSES), dtype=np.float32)
    filenames = []

    for v_idx, tfm in enumerate(tta_tfms):
        test_ds = ImageClassificationDataset(cfg.TEST_CSV, cfg.IMAGES_DIR, transform=tfm, is_test=True)
        loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

        view_probs = []
        batch_filenames = []
        for images, names in tqdm(loader, desc=f"  TTA View {v_idx+1}/5"):
            images = images.to(cfg.DEVICE, non_blocking=True)
            batch_avg = torch.zeros((images.size(0), cfg.NUM_CLASSES), device=cfg.DEVICE)

            for model in models:
                with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
                    logits = model(images)
                batch_avg += torch.softmax(logits, dim=1)

            batch_avg /= len(models)
            view_probs.append(batch_avg.cpu().numpy())
            if v_idx == 0:
                batch_filenames.extend(names)

        total_probs += np.vstack(view_probs)
        if v_idx == 0:
            filenames = batch_filenames

    final_probs = total_probs / len(tta_tfms)
    np.save(os.path.join(cfg.SAVE_DIR, "final_probabilities.npy"), final_probs)

    pred_classes = final_probs.argmax(axis=1)
    confidences  = final_probs.max(axis=1)

    result_df = pd.DataFrame({
        "filename": filenames,
        "appearance": pred_classes,
        "confidence": np.round(confidences, 4)
    })

    # Save final submission
    sub_path = os.path.join(cfg.SAVE_DIR, "submission.csv")
    result_df[["filename", "appearance"]].to_csv(sub_path, index=False)
    print(f"\n[Done] 0.95+ Submission saved to '{sub_path}'")

    pred_path = os.path.join(cfg.SAVE_DIR, "predictions.csv")
    result_df.to_csv(pred_path, index=False)
    print(f"[Done] Detailed predictions saved to '{pred_path}'")

    print("\n=== Prediction Class Distribution ===")
    counts = result_df["appearance"].value_counts().sort_index()
    for cls_idx, count in counts.items():
        pct = 100.0 * count / len(result_df)
        print(f"  Class {cls_idx} ({cfg.CLASS_NAMES[cls_idx]}): {count} images ({pct:.2f}%)")

    print(f"\nMean Confidence: {result_df['confidence'].mean():.4f}")
    print("\nSample Predictions:")
    print(result_df.head(10).to_string(index=False))
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {cfg.DEVICE} | Model: {cfg.MODEL_NAME} | Resolution: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    run_training(cfg)
    run_batched_tta_inference(cfg)
