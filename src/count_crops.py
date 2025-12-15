# src/count_crops.py
#
# Task 2.3: Count how many cropped face images remain after manual cleaning.
# This gives you True Positives (TP).
# Precision = TP / Total_Detections (where Total_Detections is printed by
# mtcnn_crop_faces.py before you cleaned the dataset).

from pathlib import Path
import os
import glob

BASE_DIR = Path(__file__).resolve().parent.parent
CROPPED_DIR = BASE_DIR / "data" / "cropped_faces"

def count_images(root: Path) -> int:
    total = 0
    for person in os.listdir(root):
        person_dir = root / person
        if not person_dir.is_dir():
            continue

        imgs = [
            p for p in glob.glob(str(person_dir / "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        total += len(imgs)
    return total

def main():
    tp = count_images(CROPPED_DIR)
    print("Cropped dir:", CROPPED_DIR)
    print("True Positives (cleaned crops):", tp)
    print("\nPrecision = True Positives / Total Detections")
    print("Use the 'Total face crops saved (detections)' printed by")
    print("mtcnn_crop_faces.py as Total Detections.")

if __name__ == "__main__":
    main()
