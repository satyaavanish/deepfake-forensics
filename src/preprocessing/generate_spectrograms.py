import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SPLITS = ["train", "val", "test"]

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "spectrograms"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    print()
    print("=" * 70)
    print(f"PROCESSING {split.upper()}")
    print("=" * 70)

    csv_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        f"{split}.csv"
    )

    audio_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "audio",
        split
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        split
    )

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    print(f"Videos in {split}: {len(df)}")

    successful = 0
    failed = 0
    skipped = 0

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split}"
    ):

        # Original video path
        file_path = str(row["file"])

        video_stem = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        identity = str(row["identity"])

        video_name = f"{identity}_{video_stem}"

        audio_path = os.path.join(
    audio_dir,
    video_stem + ".wav"
)

        output_path = os.path.join(
            output_dir,
            video_name + ".npy"
        )

        # ----------------------------------------------------
        # Already processed
        # ----------------------------------------------------

        if os.path.exists(output_path):

            skipped += 1
            continue

        # ----------------------------------------------------
        # Check audio
        # ----------------------------------------------------

        if not os.path.exists(audio_path):

            print(
                f"\nWARNING: Audio missing: {audio_path}"
            )

            failed += 1
            continue

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            audio, sr = librosa.load(
                audio_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            # ------------------------------------------------
            # Mel spectrogram
            # ------------------------------------------------

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                power=2.0
            )

            # ------------------------------------------------
            # Convert to Log-Mel
            # ------------------------------------------------

            log_mel = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            np.save(
                output_path,
                log_mel.astype(np.float32)
            )

            successful += 1

        except Exception as e:

            print(
                f"\nERROR: {video_name}"
            )

            print(e)

            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(f"{split.upper()} COMPLETE")
    print(f"Total:      {len(df)}")
    print(f"Successful: {successful}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL DATASET LOG-MEL SPECTROGRAM GENERATION")
    print("=" * 70)

    print()
    print(f"Sample rate : {SAMPLE_RATE}")
    print(f"N_FFT       : {N_FFT}")
    print(f"Hop length  : {HOP_LENGTH}")
    print(f"Mel bins    : {N_MELS}")

    for split in SPLITS:

        process_split(split)

    print()
    print("=" * 70)
    print("ALL SPECTROGRAMS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()