import os
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SPLITS = ["train", "val", "test"]

SPECTROGRAM_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "spectrograms"
)


def check_split(split):

    csv_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        f"{split}.csv"
    )

    spectrogram_dir = os.path.join(
        SPECTROGRAM_ROOT,
        split
    )

    df = pd.read_csv(csv_path)

    missing = []
    invalid = []

    for _, row in df.iterrows():

        file_path = str(row["file"])

        video_stem = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        identity = str(row["identity"])

        video_name = f"{identity}_{video_stem}"

        npy_path = os.path.join(
            spectrogram_dir,
            video_name + ".npy"
        )

        if not os.path.exists(npy_path):

            missing.append(video_name)
            continue

        try:

            data = np.load(npy_path)

            # Check dimensions
            if data.ndim != 2:

                invalid.append(
                    (video_name, f"ndim={data.ndim}")
                )
                continue

            # Check mel bins
            if data.shape[0] != 128:

                invalid.append(
                    (
                        video_name,
                        f"shape={data.shape}"
                    )
                )
                continue

            # Check NaN / Inf
            if not np.isfinite(data).all():

                invalid.append(
                    (
                        video_name,
                        "contains NaN/Inf"
                    )
                )

        except Exception as e:

            invalid.append(
                (video_name, str(e))
            )

    actual_files = len([
        f for f in os.listdir(spectrogram_dir)
        if f.lower().endswith(".npy")
    ])

    print()
    print("=" * 60)
    print(split.upper())
    print("=" * 60)

    print(f"Expected:        {len(df)}")
    print(f"Actual .npy:     {actual_files}")
    print(f"Missing:         {len(missing)}")
    print(f"Invalid:         {len(invalid)}")

    if missing:

        print("\nMissing files:")
        for item in missing[:10]:
            print(" ", item)

    if invalid:

        print("\nInvalid files:")
        for item in invalid[:10]:
            print(" ", item)

    return len(df), actual_files, len(missing), len(invalid)


def main():

    print("=" * 60)
    print("SPECTROGRAM DATASET VERIFICATION")
    print("=" * 60)

    total_expected = 0
    total_actual = 0
    total_missing = 0
    total_invalid = 0

    for split in SPLITS:

        expected, actual, missing, invalid = check_split(split)

        total_expected += expected
        total_actual += actual
        total_missing += missing
        total_invalid += invalid

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Expected: {total_expected}")
    print(f"Actual:   {total_actual}")
    print(f"Missing:  {total_missing}")
    print(f"Invalid:  {total_invalid}")

    if (
        total_expected == total_actual
        and total_missing == 0
        and total_invalid == 0
    ):

        print()
        print("✓ ALL SPECTROGRAMS VERIFIED SUCCESSFULLY")

    else:

        print()
        print("⚠ CHECK THE RESULTS ABOVE")


if __name__ == "__main__":
    main()