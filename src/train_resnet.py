# src/train_resnet.py
#
# Task 3.1–3.3: Fine-tune Inception-ResNet V1 on cropped faces
#
# - Uses data/cropped_faces (cleaned) as dataset.
# - Splits into train/validation.
# - Trains for N epochs and saves best model to models/resnet_best.pth
# - Prints training and validation (testing) accuracy for your report.

from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# Your course's InceptionResnetV1 implementation
from inception_resnet_v1 import InceptionResnetV1

# Use facenet-pytorch's standardization
from facenet_pytorch import fixed_image_standardization

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CROPPED_DIR = BASE_DIR / "data" / "cropped_faces"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# CHANGE THIS if your pretrained file has a different name
PRETRAINED_WEIGHTS = MODELS_DIR / "casia-webface.pt"


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
def get_dataloaders(batch_size=8, val_split=0.2):
    """
    Creates training and validation DataLoaders from cropped_faces.
    Training loader uses drop_last=True to avoid batch_size=1
    (BatchNorm can't update with a single sample in training).
    """

    transform = transforms.Compose([
        transforms.Resize((160, 160)),              # required input size
        transforms.ToTensor(),                      # [0,1]
        transforms.Lambda(lambda x: fixed_image_standardization(x))
    ])

    dataset = datasets.ImageFolder(str(CROPPED_DIR), transform=transform)
    num_classes = len(dataset.classes)

    print("Using dataset in:", CROPPED_DIR)
    print("Classes (people):")
    for i, cls in enumerate(dataset.classes):
        print(f"  [{i}] {cls}")
    print("Total cropped images:", len(dataset))

    # Create randomized train/val split
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    split_idx = int((1.0 - val_split) * len(indices))
    train_inds = indices[:split_idx]
    val_inds = indices[split_idx:]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds),
        drop_last=True          # <<< important fix
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds),
        drop_last=False         # fine for eval mode
    )

    return dataset, train_loader, val_loader, num_classes


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_model(
    num_epochs=20,
    batch_size=8,
    lr=1e-3,
    save_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset, train_loader, val_loader, num_classes = get_dataloaders(
        batch_size=batch_size
    )
    class_names = dataset.classes

    if save_path is None:
        save_path = MODELS_DIR / "resnet_best.pth"

    # Build model
    if PRETRAINED_WEIGHTS.exists():
        print("Loading pretrained weights from:", PRETRAINED_WEIGHTS)
        model = InceptionResnetV1(
            classify=True,
            pretrained=str(PRETRAINED_WEIGHTS),  # path to .pt file
            num_classes=num_classes
        ).to(device)
    else:
        print("WARNING: Pretrained weights file not found at:", PRETRAINED_WEIGHTS)
        print("Training from scratch instead (this may give lower accuracy).")
        model = InceptionResnetV1(
            classify=True,
            pretrained=None,                     # no weights
            num_classes=num_classes
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, [5, 10])

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # ===== Train phase =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in tqdm(train_loader, desc="Train"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # IMPORTANT: zero_grad BEFORE loss.backward()
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ===== Validation phase =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val"):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        if val_total > 0:
            val_loss /= val_total
            val_acc = val_correct / val_total
        else:
            # Edge case: if validation set is empty (unlikely), skip
            val_loss = 0.0
            val_acc = 0.0

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                save_path
            )
            print(f">>> New best model saved to {save_path} (val_acc={val_acc:.4f})")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    train_model()
