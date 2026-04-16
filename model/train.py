import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter
import numpy as np

from model.model import CNN
from config import (
    DATA_DIR, VAL_SPLIT, BATCH_SIZE, LEARNING_RATE,
    EPOCHS, IMG_SIZE, MODEL_PATH, device,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(img_size: int):
    """Separate train/val transforms.  Augmentation is critical for emotion
    recognition: real faces tilt, are partially lit, and vary in skin tone.
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.RandomGrayscale(p=0.1),          # mimic B&W webcam feeds
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # occlusion
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Wrap a Subset with its own transform so train and val can differ
    while sharing the same underlying split indices.

    WHY THIS EXISTS: calling random_split twice on two separately-constructed
    ImageFolder instances (same path, different transforms) produces two
    independent random splits.  Some samples appear in both loaders (data
    leakage), others in neither.  This wrapper fixes that by splitting once
    on indices and attaching the transform afterwards.
    """
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]           # PIL image (no transform on base dataset)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_datasets(data_dir, val_split, img_size):
    # Load without any transform so we control it per split
    base = datasets.ImageFolder(data_dir)

    n = len(base)
    val_count = int(n * val_split)
    train_count = n - val_count

    # Deterministic split — fix generator seed for reproducibility
    gen = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        base, [train_count, val_count], generator=gen
    )

    train_tf, val_tf = get_transforms(img_size)
    train_ds = _TransformSubset(train_subset, train_tf)
    val_ds   = _TransformSubset(val_subset,   val_tf)

    return train_ds, val_ds, base.classes


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) down-weights well-classified samples so
    the gradient is dominated by hard / minority-class examples.

    gamma=2 is the standard default.  Combined with class weights it is
    much more effective than weighted CE alone for heavily skewed emotion
    datasets where disgust/fear/contempt can be 10× underrepresented.
    """
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)                      # (B,)
        pt = torch.exp(-ce_loss)                                # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model():

    # --- Data ---
    train_ds, val_ds, class_names = build_datasets(DATA_DIR, VAL_SPLIT, IMG_SIZE)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=2, pin_memory=True,
    )

    # --- Class weights (computed from train indices only) ---
    train_labels = [train_ds.subset.dataset.targets[i]
                    for i in train_ds.subset.indices]
    count = Counter(train_labels)
    # Inverse-frequency weights normalised so they sum to num_classes
    raw = torch.tensor([1.0 / count[i] for i in range(num_classes)])
    class_weights = (raw / raw.sum() * num_classes).to(device)
    print("Class weights:", {class_names[i]: f"{class_weights[i].item():.2f}"
                              for i in range(num_classes)})

    # --- Model ---
    model = CNN(num_classes=num_classes).to(device)

    # --- Loss: Focal + class weights ---
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    # --- Optimiser ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,          # stronger L2 than before (1e-4 → 1e-3)
        amsgrad=True,               # more stable in small-batch regime
    )

    # --- LR schedule: cosine annealing with warm restarts ---
    # Better than ReduceLROnPlateau for emotion models: avoids getting stuck
    # in sharp minima that don't generalise.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15        # stop if val acc doesn't improve for 15 epochs

    for epoch in range(1, EPOCHS + 1):

        # ---- Train ----
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping: prevents the rare exploding-gradient spike
            # that poisons a batch — common when BatchNorm is first warming up
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(epoch + train_total / len(train_ds))   # per-batch step for CAWR

            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            running_loss  += loss.item() * labels.size(0)

        train_acc  = train_correct / train_total
        train_loss = running_loss  / train_total

        # ---- Validate ----
        model.eval()
        val_correct, val_total = 0, 0
        per_class_correct = [0] * num_classes
        per_class_total   = [0] * num_classes

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds  = logits.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                for c in range(num_classes):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c]   += mask.sum().item()

        val_acc = val_correct / val_total

        per_class_acc = {
            class_names[c]: (per_class_correct[c] / per_class_total[c]
                             if per_class_total[c] > 0 else 0.0)
            for c in range(num_classes)
        }

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | Train {train_acc:.3f} | Val {val_acc:.3f} | "
            f"LR {lr_now:.2e}"
        )
        # Print per-class accuracy every 5 epochs to detect prediction collapse early
        if epoch % 5 == 0:
            print("  Per-class val acc:", {k: f"{v:.2f}" for k, v in per_class_acc.items()})

        # ---- Checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "class_names":  class_names,
                "img_size":     IMG_SIZE,
                "val_acc":      val_acc,
            }, MODEL_PATH)
            print(f"  ✓ Saved checkpoint (val acc {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


