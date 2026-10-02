import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "dataset", "raw", "FakeAVCeleb"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "dataset", "processed", "frames"
)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "outputs", "train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "outputs", "val.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "test.csv")

NUM_FRAMES = 16
FACE_SIZE = 224

# IMPORTANT:
# Start with only 5 videos for testing.
# Later change this to None to process everything.
TEST_LIMIT = None


# ============================================================
# CATEGORY -> ACTUAL DATASET FOLDER
# ============================================================

CATEGORY_TO_FOLDER = {
    "fake_video_fake_audio": "fakevide-fake audi",
    "fake_video_real_audio": "fakevidoe-realaudio",
    "real_video_fake_audio": "reak video -fake auio",
    "real_video_real_audio": "reala udio-real video",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FULL DATASET PREPROCESSING TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# LOAD MTCNN
# ============================================================

print("Loading MTCNN...")

mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    keep_all=True,
    device=DEVICE
)

print("MTCNN loaded.")
print()


# ============================================================
# LOAD SPLIT CSVs
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("Dataset loaded:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")
print(f"Total: {len(df)}")
print()


# ============================================================
# LIMIT FOR TESTING
# ============================================================

if TEST_LIMIT is not None:
    df = df.head(TEST_LIMIT).copy()

print(f"Videos to process in this test: {len(df)}")
print()


# ============================================================
# STATISTICS
# ============================================================

total_videos = 0
successful_videos = 0
failed_videos = 0

total_frames = 0
successful_frames = 0
failed_frames = 0


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing videos"
):

    total_videos += 1

    category = row["category"]
    identity = str(row["identity"])
    file_path = row["file"]
    split = row["split"]

    # --------------------------------------------------------
    # Convert relative path to actual video path
    # --------------------------------------------------------

     # --------------------------------------------------------
    # Convert CSV path to actual video path
    # --------------------------------------------------------

    # The CSV already contains paths beginning with:
    # dataset\raw\FakeAVCeleb\...

    if os.path.isabs(file_path):
        video_path = file_path
    else:
        video_path = os.path.join(
            PROJECT_ROOT,
            file_path
        )

    video_path = os.path.normpath(video_path)

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not os.path.exists(video_path):

        print(
            f"\nWARNING: Video not found:\n"
            f"{video_path}"
        )

        failed_videos += 1
        continue

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"\nWARNING: Could not open:\n"
            f"{video_path}"
        )

        failed_videos += 1
        continue

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_video_frames <= 0:

        cap.release()

        print(
            f"\nWARNING: No frames:\n"
            f"{video_path}"
        )

        failed_videos += 1
        continue

    # --------------------------------------------------------
    # Uniform frame sampling
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_video_frames - 1,
        NUM_FRAMES,
        dtype=int
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    video_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    output_dir = os.path.join(
        OUTPUT_ROOT,
        split,
        category,
        identity,
        video_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    video_success = False

    # --------------------------------------------------------
    # Process sampled frames
    # --------------------------------------------------------

    for frame_number, frame_index in enumerate(
        frame_indices,
        start=1
    ):

        total_frames += 1

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            failed_frames += 1
            continue

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb_frame
        )

        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        boxes, probabilities = mtcnn.detect(
            pil_image
        )

        if boxes is None or len(boxes) == 0:

            failed_frames += 1
            continue

        # ----------------------------------------------------
        # Select largest face
        # ----------------------------------------------------

        largest_index = 0
        largest_area = 0

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box

            area = max(0, x2 - x1) * max(
                0,
                y2 - y1
            )

            if area > largest_area:

                largest_area = area
                largest_index = i

        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        x1, y1, x2, y2 = boxes[largest_index]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(pil_image.width, int(x2))
        y2 = min(pil_image.height, int(y2))

        if x2 <= x1 or y2 <= y1:

            failed_frames += 1
            continue

        face = pil_image.crop(
            (x1, y1, x2, y2)
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        face = face.resize(
            (FACE_SIZE, FACE_SIZE),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = os.path.join(
            output_dir,
            f"frame_{frame_number:03d}.jpg"
        )

        face.save(
            output_path,
            quality=95
        )

        successful_frames += 1
        video_success = True

    cap.release()

    if video_success:
        successful_videos += 1
    else:
        failed_videos += 1


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING TEST COMPLETE")
print("=" * 70)

print(f"Videos processed:       {total_videos}")
print(f"Successful videos:      {successful_videos}")
print(f"Failed videos:          {failed_videos}")

print()

print(f"Total frames attempted: {total_frames}")
print(f"Successful face crops:  {successful_frames}")
print(f"Failed frames:          {failed_frames}")

print()

print("Output directory:")
print(OUTPUT_ROOT)

print()

if successful_frames > 0:
    print("✓ Face preprocessing is working.")
else:
    print("✗ No face crops were generated.")

print("=" * 70)