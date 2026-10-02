from pathlib import Path
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = Path(
    "dataset/raw/FakeAVCeleb/reala udio-real video/"
    "id00080/00281.mp4"
)

OUTPUT_DIR = Path("outputs/test_frames")

NUM_FRAMES = 16


# ============================================================
# CHECK VIDEO
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print("=" * 70)
print("Frame Sampling Test")
print("=" * 70)

print(f"\nVideo:")
print(VIDEO_PATH)

print(f"\nTotal frames: {total_frames}")
print(f"FPS:          {fps:.2f}")
print(f"Duration:     {duration:.2f} seconds")

print(f"\nRequested frames: {NUM_FRAMES}")


# ============================================================
# GENERATE UNIFORM FRAME INDICES
# ============================================================

frame_indices = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES,
    dtype=int
)

print("\nFrame indices:")
print(frame_indices)


# ============================================================
# EXTRACT FRAMES
# ============================================================

saved_count = 0

for i, frame_index in enumerate(frame_indices):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index)
    )

    success, frame = cap.read()

    if not success:
        print(
            f"WARNING: Could not read frame "
            f"{frame_index}"
        )
        continue

    output_path = (
        OUTPUT_DIR /
        f"frame_{i + 1:03d}.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        frame
    )

    if success:
        saved_count += 1
    else:
        print(
            f"WARNING: Could not save "
            f"{output_path}"
        )


# ============================================================
# CLEAN UP
# ============================================================

cap.release()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FRAME SAMPLING COMPLETE")
print("=" * 70)

print(f"Requested: {NUM_FRAMES}")
print(f"Saved:     {saved_count}")

print(f"\nOutput directory:")
print(OUTPUT_DIR.resolve())


if saved_count == NUM_FRAMES:
    print("\n✓ All 16 frames saved successfully.")
else:
    print(
        f"\n⚠ Only {saved_count}/{NUM_FRAMES} "
        "frames were saved."
    )