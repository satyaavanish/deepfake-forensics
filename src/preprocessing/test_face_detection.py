from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("outputs/test_frames")
OUTPUT_DIR = Path("outputs/test_faces")

FACE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MTCNN Face Detection Test")
print("=" * 70)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input directory not found:\n{INPUT_DIR}"
    )


frame_paths = sorted(
    INPUT_DIR.glob("*.jpg")
)

print(f"\nInput frames: {len(frame_paths)}")

if len(frame_paths) == 0:
    raise RuntimeError(
        "No JPG frames found."
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INITIALIZE MTCNN
# ============================================================

print("\nLoading MTCNN...")

mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    keep_all=True,
    device=device
)

print("MTCNN loaded.")


# ============================================================
# PROCESS FRAMES
# ============================================================

detected_count = 0
no_face_count = 0
multi_face_count = 0

print("\nProcessing frames...\n")


for frame_path in frame_paths:

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(frame_path)
    )

    if image_bgr is None:
        print(
            f"WARNING: Could not read {frame_path.name}"
        )
        continue

    # OpenCV BGR → RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    boxes, probabilities = mtcnn.detect(
        pil_image
    )

    # --------------------------------------------------------
    # No face
    # --------------------------------------------------------

    if boxes is None or len(boxes) == 0:

        print(
            f"{frame_path.name}: NO FACE"
        )

        no_face_count += 1

        continue

    # --------------------------------------------------------
    # Multiple faces
    # --------------------------------------------------------

    if len(boxes) > 1:

        multi_face_count += 1

        print(
            f"{frame_path.name}: "
            f"{len(boxes)} faces detected"
        )

    else:

        print(
            f"{frame_path.name}: "
            f"1 face detected"
        )

    detected_count += 1

    # --------------------------------------------------------
    # Select the largest face
    # --------------------------------------------------------

    largest_area = 0
    largest_box = None

    for box in boxes:

        x1, y1, x2, y2 = box

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        area = width * height

        if area > largest_area:

            largest_area = area
            largest_box = box

    if largest_box is None:
        continue

    # --------------------------------------------------------
    # Convert coordinates to integers
    # --------------------------------------------------------

    x1, y1, x2, y2 = largest_box

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image_rgb.shape[1], int(x2))
    y2 = min(image_rgb.shape[0], int(y2))

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    face = image_rgb[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:

        print(
            f"{frame_path.name}: "
            "INVALID FACE CROP"
        )

        continue

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    face = cv2.resize(
        face,
        (FACE_SIZE, FACE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # RGB → BGR for OpenCV saving
    # --------------------------------------------------------

    face_bgr = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        frame_path.name
    )

    cv2.imwrite(
        str(output_path),
        face_bgr
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FACE DETECTION COMPLETE")
print("=" * 70)

print(f"\nInput frames:       {len(frame_paths)}")
print(f"Faces detected:     {detected_count}")
print(f"No face:            {no_face_count}")
print(f"Multiple faces:     {multi_face_count}")

print(
    f"\nFace crops saved to:"
)
print(
    OUTPUT_DIR.resolve()
)

print("\nExpected face size:")
print(f"{FACE_SIZE} × {FACE_SIZE}")

print("\n✓ Test completed.")