import os
import glob
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, periodogram


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

# ============================================================
# AUTOMATICALLY FIND A PROCESSED VIDEO
# ============================================================

frame_dirs = []

for root, dirs, files in os.walk(FRAMES_ROOT):
    jpg_files = [
        f for f in files
        if f.lower().endswith(".jpg")
    ]

    if len(jpg_files) >= 16:
        frame_dirs.append(root)

if not frame_dirs:
    raise RuntimeError(
        "No processed video folder containing 16 frames was found."
    )

FRAME_DIR = frame_dirs[0]

print("Automatically selected frame directory:")
print(FRAME_DIR)

# Your preprocessing sampled 16 frames
EXPECTED_FRAMES = 16

# Approximate frame rate
FPS = 25.0


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fs, low=0.7, high=4.0, order=3):
    """
    Keep frequencies corresponding approximately to
    42-240 BPM.

    0.7 Hz  = 42 BPM
    4.0 Hz  = 240 BPM
    """

    nyquist = 0.5 * fs

    low_normalized = low / nyquist
    high_normalized = high / nyquist

    b, a = butter(
        order,
        [low_normalized, high_normalized],
        btype="band"
    )

    filtered = filtfilt(b, a, signal)

    return filtered


# ============================================================
# FIND SKIN REGION
# ============================================================

def get_skin_signal(frame):
    """
    Estimate average RGB values from facial skin pixels.

    Uses HSV thresholds to select pixels that are
    likely to belong to skin.
    """

    # OpenCV reads BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Basic skin-color range
    lower = np.array([0, 20, 40], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # Remove very small regions
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    pixels = frame[mask > 0]

    if len(pixels) == 0:
        return None

    # BGR → RGB
    b = pixels[:, 0]
    g = pixels[:, 1]
    r = pixels[:, 2]

    return np.array([
        np.mean(r),
        np.mean(g),
        np.mean(b)
    ])


# ============================================================
# LOAD FRAMES
# ============================================================

print("=" * 70)
print("rPPG TEST")
print("=" * 70)

print("\nFrame directory:")
print(FRAME_DIR)

frame_paths = sorted(
    glob.glob(
        os.path.join(FRAME_DIR, "*.jpg")
    )
)

print("\nFrames found:", len(frame_paths))

if len(frame_paths) == 0:
    raise RuntimeError(
        "No frames found. Check SPLIT, CATEGORY, IDENTITY and VIDEO_NAME."
    )


# ============================================================
# EXTRACT RGB TEMPORAL SIGNAL
# ============================================================

rgb_signal = []

valid_frames = 0

for frame_path in frame_paths[:EXPECTED_FRAMES]:

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    signal = get_skin_signal(frame)

    if signal is None:
        continue

    rgb_signal.append(signal)

    valid_frames += 1


rgb_signal = np.array(rgb_signal)

print("\nValid frames:", valid_frames)

if len(rgb_signal) < 5:
    raise RuntimeError(
        "Not enough valid skin regions for rPPG."
    )


# ============================================================
# PRINT RGB SIGNAL
# ============================================================

print("\nRGB temporal signal shape:")
print(rgb_signal.shape)

print("\nFirst RGB samples:")

for i in range(min(5, len(rgb_signal))):
    print(
        f"Frame {i + 1}: "
        f"R={rgb_signal[i, 0]:.2f}, "
        f"G={rgb_signal[i, 1]:.2f}, "
        f"B={rgb_signal[i, 2]:.2f}"
    )


# ============================================================
# NORMALIZE RGB SIGNAL
# ============================================================

rgb_normalized = (
    rgb_signal - np.mean(rgb_signal, axis=0)
) / (
    np.std(rgb_signal, axis=0) + 1e-8
)


# ============================================================
# SIMPLE CHROMINANCE rPPG SIGNAL
# ============================================================

R = rgb_normalized[:, 0]
G = rgb_normalized[:, 1]
B = rgb_normalized[:, 2]

# Chrominance-based pulse estimate
X = 3 * R - 2 * G

Y = 1.5 * R + G - 1.5 * B

pulse_signal = X - (
    np.std(X) /
    (np.std(Y) + 1e-8)
) * Y


# ============================================================
# FILTER PULSE SIGNAL
# ============================================================

# ============================================================
# FILTER PULSE SIGNAL
# ============================================================

# 16 frames are too few for reliable filtfilt().
# For this initial test, skip filtering when the signal
# is too short.

if len(pulse_signal) > 21:

    filtered_pulse = bandpass_filter(
        pulse_signal,
        FPS
    )

else:

    print(
        "\nWARNING: Only "
        f"{len(pulse_signal)} frames available."
    )

    print(
        "Skipping band-pass filtering for this test."
    )

    filtered_pulse = pulse_signal.copy()# ============================================================
# FILTER PULSE SIGNAL
# ============================================================

# 16 frames are too few for reliable filtfilt().
# For this initial test, skip filtering when the signal
# is too short.

if len(pulse_signal) > 21:

    filtered_pulse = bandpass_filter(
        pulse_signal,
        FPS
    )

else:

    print(
        "\nWARNING: Only "
        f"{len(pulse_signal)} frames available."
    )

    print(
        "Skipping band-pass filtering for this test."
    )

    filtered_pulse = pulse_signal.copy()


# ============================================================
# SIGNAL QUALITY
# ============================================================

signal_std = np.std(filtered_pulse)

signal_mean = np.mean(
    np.abs(filtered_pulse)
)

if signal_mean > 1e-8:

    temporal_signal_quality = (
        signal_std / signal_mean
    )

else:

    temporal_signal_quality = 0.0


# ============================================================
# HEART RATE ESTIMATION
# ============================================================

if len(filtered_pulse) >= 8:

    frequencies, power = periodogram(
        filtered_pulse,
        fs=FPS
    )

    # Keep physiological frequency range
    valid = (
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    if np.any(valid):

        valid_frequencies = frequencies[valid]
        valid_power = power[valid]

        peak_frequency = valid_frequencies[
            np.argmax(valid_power)
        ]

        heart_rate = peak_frequency * 60.0

    else:

        heart_rate = 0.0

else:

    heart_rate = 0.0


# ============================================================
# PULSE CONSISTENCY
# ============================================================

if np.std(filtered_pulse) > 1e-8:

    normalized_pulse = (
        filtered_pulse -
        np.mean(filtered_pulse)
    ) / (
        np.std(filtered_pulse)
    )

    # Autocorrelation
    autocorrelation = np.correlate(
        normalized_pulse,
        normalized_pulse,
        mode="full"
    )

    autocorrelation = autocorrelation[
        len(autocorrelation) // 2:
    ]

    autocorrelation /= (
        autocorrelation[0] + 1e-8
    )

    # Ignore lag 0
    if len(autocorrelation) > 1:

        pulse_consistency = np.max(
            autocorrelation[1:]
        )

    else:

        pulse_consistency = 0.0

else:

    pulse_consistency = 0.0


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("rPPG RESULTS")
print("=" * 70)

print(
    f"\nHeart-rate estimate: "
    f"{heart_rate:.2f} BPM"
)

print(
    f"Pulse consistency: "
    f"{pulse_consistency:.4f}"
)

print(
    f"Temporal signal quality: "
    f"{temporal_signal_quality:.4f}"
)

print(
    f"Pulse signal mean: "
    f"{np.mean(filtered_pulse):.6f}"
)

print(
    f"Pulse signal std: "
    f"{np.std(filtered_pulse):.6f}"
)

print("\nFeature vector:")

print(
    np.array([
        heart_rate,
        pulse_consistency,
        temporal_signal_quality
    ])
)

print("\n" + "=" * 70)
print("rPPG TEST COMPLETE")
print("=" * 70)