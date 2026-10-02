import os
import torch
import librosa
import pandas as pd
from transformers import AutoFeatureExtractor, ASTModel
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "train.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "val.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test.csv"
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ast_embeddings.pt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

SAMPLE_RATE = 16000


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("AST AUDIO EMBEDDING EXTRACTION")
print("=" * 70)

print(
    "\nDevice:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET SPLITS
# ============================================================

print("\nLoading split files...")

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

metadata = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)

print(
    "\nTrain:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

print(
    "Total:",
    len(metadata)
)


if len(metadata) != 1006:

    raise ValueError(
        f"Expected 1006 videos, got {len(metadata)}"
    )


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {
    "fake_video_fake_audio": 1,
    "fake_video_real_audio": 1,
    "real_video_fake_audio": 0,
    "real_video_real_audio": 0
}


# ============================================================
# LOAD AST
# ============================================================

print(
    "\nLoading AST:"
)

print(
    MODEL_NAME
)

feature_extractor = (
    AutoFeatureExtractor
    .from_pretrained(
        MODEL_NAME
    )
)

model = (
    ASTModel
    .from_pretrained(
        MODEL_NAME
    )
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# FIND AUDIO FILE
# ============================================================

def find_audio_path(row):

    split = str(
        row["split"]
    )

    video_file = str(
        row["file"]
    )

    video_name = os.path.splitext(
        os.path.basename(video_file)
    )[0]

    audio_path = os.path.join(
        AUDIO_DIR,
        split,
        video_name + ".wav"
    )

    return audio_path


# ============================================================
# EXTRACT AUDIO EMBEDDING
# ============================================================

@torch.no_grad()
def extract_audio_embedding(
    audio_path
):

    if not os.path.isfile(
        audio_path
    ):

        return None

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    waveform, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(waveform) == 0:

        return None

    # --------------------------------------------------------
    # AST preprocessing
    # --------------------------------------------------------

    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = (
        inputs["input_values"]
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # AST forward
    # --------------------------------------------------------

    outputs = model(
        input_values=input_values
    )

    # --------------------------------------------------------
    # Mean pooling over AST tokens
    #
    # [1, tokens, 768]
    #        ↓
    # [1, 768]
    # --------------------------------------------------------

    audio_embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
    )

    return audio_embedding.cpu()


# ============================================================
# EXTRACTION
# ============================================================

embeddings = {}

successful = 0
failed = 0


print(
    "\nStarting extraction..."
)

print()


for _, row in tqdm(
    metadata.iterrows(),
    total=len(metadata),
    desc="AST embeddings"
):

    identity = str(
        row["identity"]
    )

    video_file = str(
        row["file"]
    )

    video_name = os.path.splitext(
        os.path.basename(video_file)
    )[0]

    sample_key = (
        identity
        + "||"
        + video_name
    )

    audio_path = (
        find_audio_path(row)
    )

    try:

        embedding = (
            extract_audio_embedding(
                audio_path
            )
        )

        if embedding is None:

            failed += 1

            print(
                "\nFailed:",
                sample_key
            )

            print(
                "Audio:",
                audio_path
            )

            continue

        embeddings[sample_key] = {

            "embedding": embedding,

            "identity": identity,

            "video": video_name,

            "split": str(
                row["split"]
            ),

            "category": str(
                row["category"]
            ),

            "label": LABEL_MAP[
                str(row["category"])
            ]
        }

        successful += 1

    except Exception as e:

        failed += 1

        print(
            "\nERROR:",
            sample_key
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaving AST embeddings..."
)

torch.save(
    embeddings,
    OUTPUT_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "AST EMBEDDING EXTRACTION COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nSuccessful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Total saved:",
    len(embeddings)
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# CHECK EMBEDDING SHAPE
# ============================================================

if len(embeddings) > 0:

    first_key = next(
        iter(embeddings)
    )

    first_embedding = (
        embeddings[
            first_key
        ]["embedding"]
    )

    print(
        "\nExample key:",
        first_key
    )

    print(
        "Embedding shape:",
        first_embedding.shape
    )

    print(
        "Embedding dtype:",
        first_embedding.dtype
    )


# ============================================================
# SPLIT CHECK
# ============================================================

split_counts = {}

for item in embeddings.values():

    split = item["split"]

    split_counts[split] = (
        split_counts.get(
            split,
            0
        ) + 1
    )


print(
    "\nSaved embeddings by split:"
)

for split, count in sorted(
    split_counts.items()
):

    print(
        f"{split}: {count}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

if successful == 1006:

    print(
        "\n✓ ALL 1006 VIDEOS HAVE AST EMBEDDINGS"
    )

elif successful > 0:

    print(
        "\nWARNING:"
    )

    print(
        "Some videos failed."
    )

else:

    print(
        "\nERROR:"
    )

    print(
        "No AST embeddings were generated."
    )


print(
    "\n" + "=" * 70
)