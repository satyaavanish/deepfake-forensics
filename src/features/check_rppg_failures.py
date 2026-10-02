import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

RPPG_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "rppg_features.csv"
)


metadata = pd.read_csv(METADATA_FILE)
rppg = pd.read_csv(RPPG_FILE)

metadata_files = set(
    metadata["file"].astype(str)
)

processed_files = set(
    rppg["file"].astype(str)
)

failed_files = sorted(
    metadata_files - processed_files
)

print("=" * 70)
print("rPPG FAILED VIDEO CHECK")
print("=" * 70)

print(
    f"\nTotal metadata videos: {len(metadata_files)}"
)

print(
    f"Successfully processed: {len(processed_files)}"
)

print(
    f"Failed/skipped: {len(failed_files)}"
)

print("\nFailed videos:")

for i, file_path in enumerate(
    failed_files,
    start=1
):

    print(
        f"{i:02d}. {file_path}"
    )

print("\n" + "=" * 70)