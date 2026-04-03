"""
preprocessing/dataset.py
------------------------
PyTorch Dataset class for image forgery detection.
Handles loading, transforming, and batching images.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────

def get_train_transforms(img_size: int = 224):
    """Augmentation pipeline for training. Each image is slightly different each epoch."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    """No augmentation for validation/test — just resize and normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  Dataset Class
# ─────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    Expects a folder structure like:
        root/
            real/   ← authentic images
            fake/   ← forged images

    Labels: 0 = real, 1 = fake
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (path, label)
        self.class_names = ["real", "fake"]

        # Walk through real/ and fake/ folders
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Missing folder: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"[Dataset] Loaded {len(self.samples)} images from '{root_dir}'")
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"  Real: {real_count} | Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image as numpy array (albumentations needs numpy)
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # now a torch.Tensor

        return image, label, img_path  # also return path for debugging


# ─────────────────────────────────────────────
#  DataLoader Factory
# ─────────────────────────────────────────────

def get_dataloaders(data_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4):
    """
    Expects:
        data_dir/
            train/real/, train/fake/
            val/real/,   val/fake/
            test/real/,  test/fake/
    """
    train_dataset = ForgeryDataset(os.path.join(data_dir, "train"), get_train_transforms(img_size))
    val_dataset   = ForgeryDataset(os.path.join(data_dir, "val"),   get_val_transforms(img_size))
    test_dataset  = ForgeryDataset(os.path.join(data_dir, "test"),  get_val_transforms(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  Dataset Sanity Checker
# ─────────────────────────────────────────────

def sanity_check(data_dir: str):
    """
    Quick sanity check: loads a few images and prints their shapes.
    Helps catch corrupt files or wrong folder structure early.
    """
    import matplotlib.pyplot as plt

    dataset = ForgeryDataset(data_dir, get_val_transforms())
    if len(dataset) == 0:
        print("[ERROR] Dataset is empty. Check folder structure.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i in range(min(8, len(dataset))):
        tensor, label, path = dataset[i]
        # Unnormalize for display
        img = tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"{'REAL' if label == 0 else 'FAKE'}")
        axes[i].axis("off")

    plt.suptitle("Dataset Sanity Check — Sample Images", fontsize=14)
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150)
    plt.show()
    print("[Sanity Check] Saved to sanity_check.png")


if __name__ == "__main__":
    # Run this to test your dataset
    sanity_check("data/processed/train")
