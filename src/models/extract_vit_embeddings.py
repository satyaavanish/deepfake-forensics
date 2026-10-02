import os
import torch
import pandas as pd
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
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

FRAMES_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "vit_embeddings.pt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("ViT VISUAL EMBEDDING EXTRACTION")
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
# LOAD TRAIN / VAL / TEST
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

# Add split information explicitly
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# Combine all splits
metadata = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


print(
    "\nTrain videos:",
    len(train_df)
)

print(
    "Validation videos:",
    len(val_df)
)

print(
    "Test videos:",
    len(test_df)
)

print(
    "Total videos:",
    len(metadata)
)


# ============================================================
# VERIFY EXPECTED COUNT
# ============================================================

if len(metadata) != 1006:

    raise ValueError(
        f"Expected 1006 videos, got {len(metadata)}"
    )

print(
    "✓ Dataset contains all 1006 videos"
)


# ============================================================
# LOAD ViT
# ============================================================

MODEL_NAME = (
    "google/vit-base-patch16-224-in21k"
)

print(
    "\nLoading ViT:"
)

print(
    MODEL_NAME
)

processor = (
    ViTImageProcessor
    .from_pretrained(MODEL_NAME)
)

model = (
    ViTModel
    .from_pretrained(MODEL_NAME)
)

model = model.to(
    DEVICE
)

model.eval()


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
# FIND FRAME DIRECTORY
# ============================================================

def find_frame_directory(row):

    split = str(
        row["split"]
    )

    category = str(
        row["category"]
    )

    identity = str(
        row["identity"]
    )

    video_file = str(
        row["file"]
    )

    video_name = os.path.splitext(
        os.path.basename(video_file)
    )[0]

    frame_dir = os.path.join(
        FRAMES_DIR,
        split,
        category,
        identity,
        video_name
    )

    return frame_dir


# ============================================================
# EXTRACT ONE VIDEO EMBEDDING
# ============================================================

@torch.no_grad()
def extract_video_embedding(
    frame_dir
):

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not os.path.isdir(
        frame_dir
    ):

        return None

    # --------------------------------------------------------
    # Find frames
    # --------------------------------------------------------

    frame_files = [
        f
        for f in os.listdir(frame_dir)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    frame_files = sorted(
        frame_files
    )

    if len(frame_files) == 0:

        return None

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = []

    for frame_file in frame_files:

        frame_path = os.path.join(
            frame_dir,
            frame_file
        )

        try:

            image = (
                Image.open(frame_path)
                .convert("RGB")
            )

            images.append(
                image
            )

        except Exception as e:

            print(
                "\nWarning: Could not read:",
                frame_path
            )

            print(
                "Reason:",
                e
            )

    if len(images) == 0:

        return None

    # --------------------------------------------------------
    # ViT preprocessing
    # --------------------------------------------------------

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # ViT forward pass
    # --------------------------------------------------------

    outputs = model(
        pixel_values=pixel_values
    )

    # --------------------------------------------------------
    # CLS embedding
    #
    # [frames, 197, 768]
    #        ↓
    # [frames, 768]
    # --------------------------------------------------------

    frame_embeddings = (
        outputs
        .last_hidden_state[:, 0, :]
    )

    # --------------------------------------------------------
    # Temporal mean pooling
    #
    # [frames, 768]
    #        ↓
    # [768]
    # --------------------------------------------------------

    video_embedding = (
        frame_embeddings.mean(
            dim=0
        )
    )

    return video_embedding.cpu()


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
    desc="ViT embeddings"
):

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Frame directory
    # --------------------------------------------------------

    frame_dir = (
        find_frame_directory(row)
    )

    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    try:

        embedding = (
            extract_video_embedding(
                frame_dir
            )
        )

        if embedding is None:

            failed += 1

            print(
                "\nFailed:",
                sample_key
            )

            print(
                "Frame directory:",
                frame_dir
            )

            continue

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

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
    "\nSaving embeddings..."
)

torch.save(
    embeddings,
    OUTPUT_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ViT EMBEDDING EXTRACTION COMPLETE"
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
# VERIFY EMBEDDING
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
# VERIFY SPLITS
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
        "\n✓ ALL 1006 VIDEOS HAVE ViT EMBEDDINGS"
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
        "No embeddings were generated."
    )


print(
    "\n" + "=" * 70
)