# src/mtcnn_crop_faces.py
#
# Task 2.2: Use MTCNN (P-Net, R-Net, O-Net) to detect faces
# in the shared-drive photos, crop them, and save to data/cropped_faces.
#
# After this script, you will manually inspect and delete bad crops
# (wrong people, non-faces, etc.) for Task 2.3.

from pathlib import Path
import os
import glob

import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Project root and data dirs
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CROPPED_DIR = BASE_DIR / "data" / "cropped_faces"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Raw dir:", RAW_DIR)
    print("Cropped dir:", CROPPED_DIR)

    os.makedirs(CROPPED_DIR, exist_ok=True)

    # keep_all=True → detect all faces in the image
    mtcnn = MTCNN(keep_all=True, device=device)

    total_images = 0          # number of original images seen
    total_detections = 0      # number of face crops saved (for precision later)

    for person in os.listdir(RAW_DIR):
        person_dir = RAW_DIR / person
        if not person_dir.is_dir():
            continue

        print(f"\nProcessing person: {person}")
        out_dir = CROPPED_DIR / person
        os.makedirs(out_dir, exist_ok=True)

        for img_path in glob.glob(str(person_dir / "*")):
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total_images += 1

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Failed to open {img_path}: {e}")
                continue

            boxes, probs = mtcnn.detect(img)
            if boxes is None:
                # no faces detected in this image
                continue

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or prob < 0.90:
                    # ignore low-confidence detections
                    continue

                total_detections += 1

                x1, y1, x2, y2 = [int(v) for v in box]
                face = img.crop((x1, y1, x2, y2))
                face = face.resize((160, 160))

                base_name = Path(img_path).stem
                out_name = f"{base_name}_det{i}_p{prob:.2f}.jpg"
                out_path = out_dir / out_name
                face.save(out_path)

    print("\n=== MTCNN Cropping Summary ===")
    print("Total original images scanned:", total_images)
    print("Total face crops saved (detections):", total_detections)
    print("Crops saved under:", CROPPED_DIR)
    print("IMPORTANT: Write down 'Total face crops saved' as Total Detections for Task 2.3.")
    print("Next: manually delete bad crops in data\\cropped_faces, then run count_crops.py")

if __name__ == "__main__":
    main()
