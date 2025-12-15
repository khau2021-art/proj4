# src/realtime_demo.py
#
# Task 4.1 & 4.4: Real-time demo
# - MTCNN for face detection + landmarks
# - Inception-ResNet V1 (fine-tuned) for identity
# - Symmetry + PnP methods for "looking at me"
# - Overlays "Name is Looking at me" on the frame

from pathlib import Path
from collections import deque

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from facenet_pytorch import MTCNN, fixed_image_standardization
from inception_resnet_v1 import InceptionResnetV1
from gaze_utils import is_looking_at_camera_symmetry, is_looking_at_camera_pnp

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "resnet_best.pth"

# Use same pretrained file as in training
PRETRAINED_WEIGHTS = MODELS_DIR / "casia-webface.pt"  # change if you switch to vggface2.pt

# ---------------------------------------------------------------------
# Hyperparameters / thresholds
# ---------------------------------------------------------------------
ID_CONF_THRESH = 0.50      # identity confidence threshold (softmax)
SMOOTHING = True
SMOOTH_WINDOW = 10         # frames to smooth over when exactly one face is present

LOW_LIGHT_THRESH = 50.0    # average grayscale brightness threshold (0–255)
BLUR_THRESH = 80.0         # variance of Laplacian threshold for blur (higher = sharper)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------------------------------------
    # Load trained face recognition model checkpoint
    # -----------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        print("ERROR: Checkpoint not found at:", CHECKPOINT_PATH)
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    print("Loaded checkpoint from:", CHECKPOINT_PATH)
    print("Classes:", class_names)

    # -----------------------------------------------------------------
    # Build model architecture:
    # 1. Initialize with same pretrained weights as training (casia-webface.pt)
    # 2. Overwrite with our fine-tuned checkpoint (resnet_best.pth)
    # -----------------------------------------------------------------
    if not PRETRAINED_WEIGHTS.exists():
        print("WARNING: Pretrained weights not found at:", PRETRAINED_WEIGHTS)
        print("Continuing without pretrained weights (may still work if shapes match).")
        model = InceptionResnetV1(
            classify=True,
            pretrained=None,
            num_classes=len(class_names)
        ).to(device)
    else:
        print("Initializing model with pretrained weights from:", PRETRAINED_WEIGHTS)
        model = InceptionResnetV1(
            classify=True,
            pretrained=str(PRETRAINED_WEIGHTS),
            num_classes=len(class_names)
        ).to(device)

    # Now load our fine-tuned weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Face transform: same as training
    face_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: fixed_image_standardization(x))
    ])

    # -----------------------------------------------------------------
    # MTCNN for face detection & landmarks
    # -----------------------------------------------------------------
    mtcnn = MTCNN(keep_all=True, device=device)

    # -----------------------------------------------------------------
    # Video capture
    # -----------------------------------------------------------------
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Read one frame to estimate camera intrinsics (cx, cy)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read initial frame from camera.")
        cap.release()
        return

    h, w = frame.shape[:2]
    # APPROX camera intrinsics – replace fx/fy/cx/cy with real calibration values if you have them
    fx = 800.0
    fy = 800.0
    cx = w / 2.0
    cy = h / 2.0

    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # or your real distortion coeffs

    print("Camera matrix (approx):")
    print(camera_matrix)

    # For temporal smoothing when exactly one face is present
    prob_buffer = deque(maxlen=SMOOTH_WINDOW)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Detect faces + landmarks
        boxes, probs, landmarks = mtcnn.detect(pil_image, landmarks=True)

        if boxes is not None and landmarks is not None:
            num_faces = len(boxes)

            for box, prob, lm in zip(boxes, probs, landmarks):
                if prob is None or prob < 0.90:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]

                # Safety check for bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop face
                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue

                # Grayscale for both low-light and blur checks
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                fm = cv2.Laplacian(gray, cv2.CV_64F).var()  # focus measure

                # -------- Low-light check --------
                if brightness < LOW_LIGHT_THRESH:
                    label = f"Unknown (low light {brightness:.0f})"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    # Skip classification + gaze for this face
                    continue

                # -------- Blur (autofocus) check --------
                if fm < BLUR_THRESH:
                    label = f"Unknown (blur {fm:.0f})"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    # Skip classification + gaze for this face
                    continue

                # -------- Normal classification path --------
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_tensor = face_transform(face_pil).unsqueeze(0).to(device)

                # Classify identity + probs
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs_vec = torch.softmax(outputs, dim=1)[0].cpu().numpy()  # (num_classes,)

                # ---- Temporal smoothing (single-face case) ----
                if SMOOTHING and num_faces == 1:
                    prob_buffer.append(probs_vec)
                    probs_avg = np.mean(prob_buffer, axis=0)
                    smoothed_probs = probs_avg
                else:
                    smoothed_probs = probs_vec

                pred_idx = int(np.argmax(smoothed_probs))
                top_prob = float(smoothed_probs[pred_idx])
                person_name = class_names[pred_idx]

                # Landmarks: numpy shape (5,2) in full-image coordinates
                lm = np.array(lm)  # [5, 2]

                # Symmetry-based gaze
                sym_looking, eye_sym, mouth_sym = is_looking_at_camera_symmetry(lm)

                # PnP-based gaze
                pnp_looking, yaw, pitch, roll = is_looking_at_camera_pnp(
                    lm, camera_matrix, dist_coeffs
                )

                # Decide final "looking at me" – here we trust PnP more
                looking = pnp_looking

                # Decide final label using confidence threshold
                if top_prob < ID_CONF_THRESH:
                    label = f"Unknown ({top_prob:.2f})"
                else:
                    label = f"{person_name} ({top_prob:.2f})"
                    if looking:
                        label += " is Looking at me"

                # Draw label above bounding box
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # (Optional) show yaw/pitch for debugging in a corner
                if yaw is not None and pitch is not None:
                    debug_text = f"yaw={yaw:.1f}, pitch={pitch:.1f}"
                    cv2.putText(
                        frame,
                        debug_text,
                        (x1, min(h - 5, y2 + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 255, 255),
                        1
                    )

        cv2.imshow("Who Looks at Me?", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
