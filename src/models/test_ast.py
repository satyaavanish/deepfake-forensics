import os
import torch
import librosa
from transformers import AutoFeatureExtractor, ASTModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio",
    "test"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AST AUDIO MODEL TEST")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # FIND AUDIO FILE
    # --------------------------------------------------------

    audio_files = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ]

    if len(audio_files) == 0:
        raise RuntimeError(
            "No WAV files found in: "
            + AUDIO_DIR
        )

    audio_file = audio_files[0]

    audio_path = os.path.join(
        AUDIO_DIR,
        audio_file
    )

    print()
    print("Audio file:")
    print(audio_path)

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    print()
    print("Loading audio...")

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    print(
        "Sample rate:",
        sample_rate
    )

    print(
        "Number of samples:",
        len(waveform)
    )

    print(
        "Duration:",
        f"{len(waveform) / sample_rate:.2f}",
        "seconds"
    )

    # --------------------------------------------------------
    # LOAD AST FEATURE EXTRACTOR
    # --------------------------------------------------------

    print()
    print("Loading AST feature extractor...")

    processor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # LOAD AST MODEL
    # --------------------------------------------------------

    print()
    print("Loading AST model...")

    model = ASTModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        "✓ AST model loaded"
    )

    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    print()
    print("Preparing audio for AST...")

    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs[
        "input_values"
    ]

    print(
        "AST input shape:",
        input_values.shape
    )

    input_values = input_values.to(
        DEVICE
    )

    # --------------------------------------------------------
    # AST FORWARD PASS
    # --------------------------------------------------------

    print()
    print("Running AST...")

    with torch.no_grad():

        outputs = model(
            input_values
        )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    print()
    print(
        "Last hidden state shape:"
    )

    print(
        outputs.last_hidden_state.shape
    )

    # Mean pooling
    audio_embedding = (
        outputs.last_hidden_state.mean(
            dim=1
        )
    )

    print()
    print(
        "Audio embedding shape:"
    )

    print(
        audio_embedding.shape
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AST TEST SUCCESSFUL")
    print("=" * 70)

    print()
    print("Audio pipeline:")

    print(
        "WAV → AST → Audio Embedding"
    )

    print()
    print(
        "Embedding dimension:",
        audio_embedding.shape[-1]
    )


if __name__ == "__main__":
    main()