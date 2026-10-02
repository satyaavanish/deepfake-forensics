import os
import subprocess
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "outputs", "train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "outputs", "val.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "test.csv")

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio"
)

# Start with 5 videos for testing
TEST_LIMIT = None


# ============================================================
# LOAD DATASET
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

if TEST_LIMIT is not None:
    df = df.head(TEST_LIMIT).copy()


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# STATISTICS
# ============================================================

successful = 0
failed = 0
skipped = 0


print("=" * 70)
print("AUDIO EXTRACTION")
print("=" * 70)

print(f"Videos to process: {len(df)}")
print()


# ============================================================
# PROCESS VIDEOS
# ============================================================

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Extracting audio"
):

    file_path = row["file"]
    split = row["split"]

    # --------------------------------------------------------
    # Build video path
    # --------------------------------------------------------

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

        failed += 1
        continue

    # --------------------------------------------------------
    # Create output path
    # --------------------------------------------------------

    video_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    output_dir = os.path.join(
        OUTPUT_ROOT,
        split
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        video_name + ".wav"
    )

    # --------------------------------------------------------
    # Skip if already extracted
    # --------------------------------------------------------

    if os.path.exists(output_path):

        skipped += 1
        continue

    # --------------------------------------------------------
    # FFmpeg command
    # --------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            print(
                f"\nWARNING: FFmpeg failed:\n"
                f"{video_path}\n"
                f"{result.stderr[-500:]}"
            )

            failed += 1
            continue

        if os.path.exists(output_path):

            successful += 1

        else:

            failed += 1

    except Exception as e:

        print(
            f"\nERROR:\n"
            f"{video_path}\n"
            f"{e}"
        )

        failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("AUDIO EXTRACTION COMPLETE")
print("=" * 70)

print(f"Videos processed: {len(df)}")
print(f"Successful:       {successful}")
print(f"Skipped:           {skipped}")
print(f"Failed:            {failed}")

print()
print("Output:")
print(OUTPUT_ROOT)

print("=" * 70)