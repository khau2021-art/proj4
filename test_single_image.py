# src/test_single_image.py
#
# Quick test: run the trained model on a single cropped face
# from data/cropped_faces/<Person>/ and print the predicted name + prob.

from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization
from inception_resnet_v1 import InceptionResnetV1

BASE_DIR = Path(__file__).resolve().parent.parent
CROPPED_DIR = BASE_DIR / "data" / "cropped_faces"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "resnet_best.pth"
PRETRAINED_WEIGHTS = MODELS_DIR / "casia-webface.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Pick one of your own images here:
    test_img_path = next((CROPPED_DIR / "Tommy Khau").glob("*.jpg"))
    print("Testing on:", test_img_path)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    print("Classes:", class_names)

    # Build model
    model = InceptionResnetV1(
        classify=True,
        pretrained=str(PRETRAINED_WEIGHTS),
        num_classes=len(class_names)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: fixed_image_standardization(x))
    ])

    img = Image.open(test_img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        top_prob, pred = probs.max(1)
        top_prob = top_prob.item()
        pred_idx = pred.item()

    print(f"Predicted: {class_names[pred_idx]} with prob {top_prob:.4f}")

if __name__ == "__main__":
    main()
