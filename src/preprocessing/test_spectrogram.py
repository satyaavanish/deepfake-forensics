import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

AUDIO_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio",
    "train",
    "00281.wav"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "spectrogram_test"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 512

N_MELS = 128


# ============================================================
# LOAD AUDIO
# ============================================================

print("=" * 70)
print("LOG-MEL SPECTROGRAM TEST")
print("=" * 70)

print()
print("Audio:")
print(AUDIO_PATH)

if not os.path.exists(AUDIO_PATH):

    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_PATH}"
    )

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=SAMPLE_RATE,
    mono=True
)

print()
print(f"Sample rate: {sr}")
print(f"Audio samples: {len(audio)}")
print(
    f"Duration: {len(audio) / sr:.2f} seconds"
)


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)


print()
print("Mel spectrogram shape:")
print(mel.shape)


# ============================================================
# LOG-MEL SPECTROGRAM
# ============================================================

log_mel = librosa.power_to_db(
    mel,
    ref=np.max
)

print()
print("Log-Mel spectrogram shape:")
print(log_mel.shape)


# ============================================================
# SAVE NUMPY ARRAY
# ============================================================

npy_path = os.path.join(
    OUTPUT_DIR,
    "00281_logmel.npy"
)

np.save(
    npy_path,
    log_mel
)

print()
print("Saved:")
print(npy_path)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "00281_logmel.png"
)

plt.figure(
    figsize=(10, 4)
)

librosa.display.specshow(
    log_mel,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.title(
    "Log-Mel Spectrogram"
)

plt.tight_layout()

plt.savefig(
    png_path,
    dpi=150
)

plt.close()


print()
print("Visualization:")
print(png_path)

print()
print("=" * 70)
print("SPECTROGRAM TEST COMPLETE")
print("=" * 70)