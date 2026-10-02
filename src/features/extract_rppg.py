import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.signal import butter, filtfilt, periodogram
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "rppg_features.csv"
)

RAW_DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "FakeAVCeleb"
)

# rPPG parameters
TARGET_FPS = 25.0

# Use approximately 6 seconds
WINDOW_SECONDS = 6

# Physiological frequency range
LOW_HZ = 0.7
HIGH_HZ = 4.0

# Minimum number of frames
MIN_FRAMES = 60


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "fake_video_fake_audio": 1,
    "fake_video_real_audio": 1,
    "real_video_fake_audio": 0,
    "real_video_real_audio": 0
}


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fs, low=0.7, high=4.0, order=3):

    nyquist = 0.5 * fs

    low_normalized = low / nyquist
    high_normalized = high / nyquist

    b, a = butter(
        order,
        [low_normalized, high_normalized],
        btype="band"
    )

    # filtfilt requires enough samples
    padlen = 3 * max(len(a), len(b))

    if len(signal) <= padlen:
        return None

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ============================================================
# FIND SKIN REGION
# ============================================================

def get_skin_rgb(frame):
    """
    Extract average RGB values from a central facial skin region.

    The processed face frames are already cropped around the face,
    so we avoid using the outermost pixels.
    """

    h, w = frame.shape[:2]

    # Central region of the face.
    # Avoid eyes/mouth as much as possible.
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    y1 = int(h * 0.15)
    y2 = int(h * 0.65)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    # HSV skin filtering
    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [0, 20, 40],
        dtype=np.uint8
    )

    upper = np.array(
        [30, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    pixels = rgb[mask > 0]

    # If skin mask is too small,
    # use the whole central ROI.
    if len(pixels) < 20:

        pixels = rgb.reshape(
            -1,
            3
        )

    if len(pixels) == 0:
        return None

    return np.mean(
        pixels,
        axis=0
    )


# ============================================================
# CHROM rPPG
# ============================================================

def calculate_chrom(rgb_signal):

    # Normalize each channel
    rgb = rgb_signal.astype(
        np.float64
    )

    mean_rgb = np.mean(
        rgb,
        axis=0
    )

    normalized = rgb / (
        mean_rgb + 1e-8
    )

    R = normalized[:, 0]
    G = normalized[:, 1]
    B = normalized[:, 2]

    # CHROM projection
    X = 3.0 * R - 2.0 * G

    Y = 1.5 * R + G - 1.5 * B

    alpha = (
        np.std(X) /
        (np.std(Y) + 1e-8)
    )

    pulse = X - alpha * Y

    # Remove DC component
    pulse = pulse - np.mean(pulse)

    return pulse


# ============================================================
# HEART RATE
# ============================================================

def estimate_heart_rate(
    pulse_signal,
    fs
):

    frequencies, power = periodogram(
        pulse_signal,
        fs=fs
    )

    valid = (
        (frequencies >= LOW_HZ) &
        (frequencies <= HIGH_HZ)
    )

    if not np.any(valid):
        return 0.0

    valid_freq = frequencies[valid]
    valid_power = power[valid]

    peak_index = np.argmax(
        valid_power
    )

    peak_frequency = valid_freq[
        peak_index
    ]

    heart_rate = (
        peak_frequency * 60.0
    )

    return float(
        heart_rate
    )


# ============================================================
# PULSE CONSISTENCY
# ============================================================

def calculate_pulse_consistency(
    pulse_signal
):

    if len(pulse_signal) < 10:
        return 0.0

    std = np.std(
        pulse_signal
    )

    if std < 1e-8:
        return 0.0

    normalized = (
        pulse_signal -
        np.mean(pulse_signal)
    ) / std

    autocorrelation = np.correlate(
        normalized,
        normalized,
        mode="full"
    )

    autocorrelation = autocorrelation[
        len(autocorrelation) // 2:
    ]

    if autocorrelation[0] == 0:
        return 0.0

    autocorrelation = (
        autocorrelation /
        autocorrelation[0]
    )

    # Only physiological lags
    min_lag = max(
        1,
        int(25.0 / HIGH_HZ)
    )

    max_lag = min(
        len(autocorrelation) - 1,
        int(25.0 / LOW_HZ)
    )

    if max_lag <= min_lag:
        return 0.0

    consistency = np.max(
        autocorrelation[
            min_lag:max_lag + 1
        ]
    )

    return float(
        consistency
    )


# ============================================================
# TEMPORAL SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    pulse_signal
):

    if len(pulse_signal) < 10:
        return 0.0

    signal_std = np.std(
        pulse_signal
    )

    mean_absolute = np.mean(
        np.abs(pulse_signal)
    )

    if mean_absolute < 1e-8:
        return 0.0

    quality = (
        signal_std /
        mean_absolute
    )

    return float(
        quality
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return None

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames /
        original_fps
        if original_fps > 0
        else 0
    )

    if original_fps <= 0:
        original_fps = TARGET_FPS

    # --------------------------------------------------------
    # Read consecutive frames
    # --------------------------------------------------------

    rgb_values = []

    max_frames = int(
        WINDOW_SECONDS *
        original_fps
    )

    max_frames = min(
        max_frames,
        total_frames
    )

    for _ in range(max_frames):

        ret, frame = cap.read()

        if not ret:
            break

        rgb = get_skin_rgb(
            frame
        )

        if rgb is not None:
            rgb_values.append(
                rgb
            )

    cap.release()

    rgb_signal = np.array(
        rgb_values,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Check enough frames
    # --------------------------------------------------------

    if len(rgb_signal) < MIN_FRAMES:

        return None

    # --------------------------------------------------------
    # Calculate rPPG
    # --------------------------------------------------------

    pulse_signal = calculate_chrom(
        rgb_signal
    )

    # --------------------------------------------------------
    # Bandpass filtering
    # --------------------------------------------------------

    filtered_pulse = bandpass_filter(
        pulse_signal,
        original_fps,
        LOW_HZ,
        HIGH_HZ
    )

    if filtered_pulse is None:

        return None

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    heart_rate = estimate_heart_rate(
        filtered_pulse,
        original_fps
    )

    pulse_consistency = (
        calculate_pulse_consistency(
            filtered_pulse
        )
    )

    signal_quality = (
        calculate_signal_quality(
            filtered_pulse
        )
    )

    return {
        "heart_rate_estimate": heart_rate,
        "pulse_consistency": pulse_consistency,
        "temporal_signal_quality": signal_quality,
        "rppg_frames": len(rgb_signal),
        "rppg_duration_seconds": (
            len(rgb_signal) /
            original_fps
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL rPPG FEATURE EXTRACTION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"\nVideos in metadata: "
        f"{len(metadata)}"
    )

    results = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Processing videos"
    ):

        relative_path = str(
            row["file"]
        )

        category = str(
            row["category"]
        )

        identity = str(
            row["identity"]
        )

        # Metadata path is relative to project root
        video_path = os.path.join(
            PROJECT_ROOT,
            relative_path
        )

        # If that doesn't exist, try raw dataset path
        if not os.path.exists(
            video_path
        ):

            video_path = os.path.join(
                RAW_DATASET_DIR,
                relative_path
            )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            features = process_video(
                video_path
            )

        except Exception as e:

            features = None

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        if features is None:

            failed += 1

            continue

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        video_stem = os.path.splitext(
            os.path.basename(
                relative_path
            )
        )[0]

        split = str(
            row["split"]
        ) if "split" in row else ""

        result = {
            "file": relative_path,
            "video_name": video_stem,
            "category": category,
            "identity": identity,
            "split": split,
            "label": LABEL_MAP.get(
                category,
                -1
            )
        }

        result.update(
            features
        )

        results.append(
            result
        )

        successful += 1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("rPPG FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nVideos processed: "
        f"{successful}"
    )

    print(
        f"Videos failed/skipped: "
        f"{failed}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )

    if len(results_df) > 0:

        print(
            "\nFeature statistics:"
        )

        print(
            results_df[
                [
                    "heart_rate_estimate",
                    "pulse_consistency",
                    "temporal_signal_quality"
                ]
            ].describe()
        )

        if "split" in results_df.columns:

            print(
                "\nSplit distribution:"
            )

            print(
                results_df[
                    "split"
                ].value_counts()
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()