# src/dataset_prep.py
#
# Task 1.2: Wrap shared-drive photos into a PyTorch dataset
# and verify that all classmates are seen correctly.

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Project root: ...\proj3
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

def main():
    print("Base dir:", BASE_DIR)
    print("Raw dir:", RAW_DIR)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(str(RAW_DIR), transform=transform)

    print("\nFound classes (people):")
    for i, cls in enumerate(dataset.classes):
        print(f"  [{i}] {cls}")
    print("Total images:", len(dataset))

    # Grab one small batch to confirm tensors look right
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print("Sample batch shape:", images.shape)   # [B, 3, 160, 160]
    print("Sample labels tensor:", labels)

if __name__ == "__main__":
    main()
