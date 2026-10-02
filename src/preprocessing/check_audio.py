import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

AUDIO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio"
)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "outputs", "train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "outputs", "val.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "test.csv")


train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"

df = pd.concat(
    [train, val, test],
    ignore_index=True
)

missing = []

for _, row in df.iterrows():

    file_path = row["file"]
    split = row["split"]

    video_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    audio_path = os.path.join(
        AUDIO_ROOT,
        split,
        video_name + ".wav"
    )

    if not os.path.exists(audio_path):
        missing.append({
            "split": split,
            "video": file_path,
            "expected_audio": audio_path
        })


print("=" * 70)
print("AUDIO VERIFICATION")
print("=" * 70)

print(f"Total videos:    {len(df)}")
print(f"WAV files found: {len(df) - len(missing)}")
print(f"Missing WAVs:    {len(missing)}")

print()

if missing:

    print("MISSING AUDIO FILES:")
    print()

    for item in missing:
        print(f"[{item['split']}]")
        print(item["video"])
        print()

else:

    print("✓ Every video has an audio file.")

print("=" * 70)