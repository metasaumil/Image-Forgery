"""
training/trainer.py
-------------------
Training pipeline for the CNN forgery detector.

Features:
- Clean training loop with validation
- Early stopping (stops if val loss doesn't improve)
- Checkpoint saving (saves best model)
- Learning rate scheduling
- Detailed logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional


# ─────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training if validation loss doesn't improve for `patience` epochs.
    Saves the best model checkpoint.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 checkpoint_path: str = "models/best_model.pth"):
        self.patience  = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter   = 0
        self.should_stop = False

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found — save model, reset counter
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  ✓ Val loss improved to {val_loss:.4f} — checkpoint saved")
        else:
            self.counter += 1
            print(f"  No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  ⏹ Early stopping triggered.")
        return self.should_stop


# ─────────────────────────────────────────────
#  One Epoch (Train)
# ─────────────────────────────────────────────

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device) -> tuple:
    """
    Runs one full pass through the training data.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels, _ in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  One Epoch (Validation)
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> tuple:
    """
    Runs validation without updating weights.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels, _ in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  Full Training Loop
# ─────────────────────────────────────────────

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          num_epochs:   int = 30,
          lr:           float = 1e-3,
          weight_decay: float = 1e-4,
          patience:     int = 7,
          checkpoint_dir: str = "models",
          model_name: str = "model",
          device: Optional[torch.device] = None) -> dict:
    """
    Full training loop.

    Args:
        model:          PyTorch model
        train_loader:   Training DataLoader
        val_loader:     Validation DataLoader
        num_epochs:     Maximum training epochs
        lr:             Initial learning rate
        weight_decay:   L2 regularization factor
        patience:       Early stopping patience
        checkpoint_dir: Where to save checkpoints
        model_name:     Used for checkpoint naming
        device:         CPU or GPU

    Returns:
        history dict with train/val loss and accuracy per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    print(f"\n[Trainer] Using device: {device}")
    print(f"[Trainer] Training for up to {num_epochs} epochs (patience={patience})\n")

    # Loss function: CrossEntropy works for multi-class (including binary)
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW is Adam + better weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    # LR Scheduler: reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Early stopping
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
    early_stopping  = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    # Track history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\n── Epoch {epoch}/{num_epochs} " + "─" * 30)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        # Step the LR scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}  [{elapsed:.1f}s]")

        # Log history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"\n[Trainer] Training stopped at epoch {epoch}")
            break

    print(f"\n[Trainer] Best val loss: {early_stopping.best_loss:.4f}")
    print(f"[Trainer] Checkpoint: {checkpoint_path}")

    return history


# ─────────────────────────────────────────────
#  Plot Training History
# ─────────────────────────────────────────────

def plot_history(history: dict, save_path: str = "training_curves.png"):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Plot] Saved to {save_path}")


if __name__ == "__main__":
    # Quick integration test with dummy data
    import sys
    sys.path.insert(0, "..")
    from deep_learning.models import get_model
    from preprocessing.dataset import get_dataloaders

    # Create dummy dataset first: python preprocessing/prepare_data.py --mode dummy
    model = get_model("resnet18")
    train_loader, val_loader, _ = get_dataloaders("data/processed", batch_size=8, num_workers=0)

    history = train(
        model, train_loader, val_loader,
        num_epochs=3, patience=2,
        checkpoint_dir="models",
        model_name="resnet18_test"
    )
    plot_history(history)
