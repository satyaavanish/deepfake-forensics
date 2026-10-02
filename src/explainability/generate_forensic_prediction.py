import os
import sys
import torch
import pandas as pd
import numpy as np


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(PROJECT_ROOT)


# =========================================================
# IMPORT MODEL
# =========================================================

from src.fusion.fusion_model import CrossModalFusionModel


# =========================================================
# PATHS
# =========================================================

VIT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings",
    "vit_embeddings.pt"
)

AST_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings",
    "ast_embeddings.pt"
)

BEHAVIOR_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "behavior_features.csv"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cross_modal_fusion_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# LOAD VIT EMBEDDINGS
# =========================================================

print("\nLoading ViT embeddings...")

vit_data = torch.load(
    VIT_PATH,
    map_location="cpu"
)

print("✓ ViT embeddings loaded")
print("Number of samples:", len(vit_data))


# =========================================================
# LOAD AST EMBEDDINGS
# =========================================================

print("\nLoading AST embeddings...")

ast_data = torch.load(
    AST_PATH,
    map_location="cpu"
)

print("✓ AST embeddings loaded")
print("Number of samples:", len(ast_data))


# =========================================================
# LOAD BEHAVIORAL FEATURES
# =========================================================

print("\nLoading behavioral features...")

behavior_df = pd.read_csv(
    BEHAVIOR_PATH
)

print("✓ Behavioral features loaded")
print("Rows:", len(behavior_df))
print("Columns:", len(behavior_df.columns))


# =========================================================
# LOAD TRAINED FUSION MODEL
# =========================================================

print("\nLoading trained fusion model...")

model = CrossModalFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    num_heads=8
).to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print("✓ Fusion model loaded")


# =========================================================
# SELECT TEST DATA
# =========================================================

test_df = behavior_df[
    behavior_df["split"]
    .astype(str)
    .str.lower()
    == "test"
].copy()


if len(test_df) == 0:

    raise RuntimeError(
        "No test samples found."
    )


# =========================================================
# SELECT SAMPLE
# Prefer a sample with rPPG available
# =========================================================

available = test_df[
    test_df["rppg_available"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes"
        ]
    )
]


if len(available) > 0:

    sample = available.iloc[0]

else:

    sample = test_df.iloc[0]


sample_key = sample["sample_key"]


print("\nSelected sample:")
print("Sample key:", sample_key)
print("Identity:", sample["identity"])
print("Video:", sample["video"])
print("Category:", sample["category"])


# =========================================================
# CHECK EMBEDDINGS EXIST
# =========================================================

if sample_key not in vit_data:

    raise RuntimeError(
        f"Sample key not found in ViT embeddings: "
        f"{sample_key}"
    )


if sample_key not in ast_data:

    raise RuntimeError(
        f"Sample key not found in AST embeddings: "
        f"{sample_key}"
    )


# =========================================================
# EXTRACT ACTUAL EMBEDDING TENSORS
# =========================================================

visual_embedding = vit_data[
    sample_key
]["embedding"]

audio_embedding = ast_data[
    sample_key
]["embedding"]


# Make sure float
visual_embedding = visual_embedding.float()
audio_embedding = audio_embedding.float()


# =========================================================
# BEHAVIORAL FEATURES
# =========================================================

behavior_exclude = {
    "sample_key",
    "identity",
    "video",
    "category",
    "split",
    "label"
}


behavior_columns = [
    c
    for c in behavior_df.columns
    if c not in behavior_exclude
]


print("\nBehavior features:")
print(
    "Number of features:",
    len(behavior_columns)
)

print(behavior_columns)


# =========================================================
# VERIFY 22 FEATURES
# =========================================================

if len(behavior_columns) != 22:

    raise RuntimeError(
        f"Expected exactly 22 behavioral features "
        f"but found {len(behavior_columns)}:\n"
        f"{behavior_columns}"
    )


# =========================================================
# TRAINING DATA
# Used ONLY for preprocessing statistics
# =========================================================

train_df = behavior_df[
    behavior_df["split"]
    .astype(str)
    .str.lower()
    == "train"
].copy()


if len(train_df) == 0:

    raise RuntimeError(
        "No training samples found."
    )


# =========================================================
# CONVERT SAMPLE BEHAVIOR FEATURES
# =========================================================

behavior_row = sample[
    behavior_columns
].copy()


# ---------------------------------------------------------
# Convert rPPG availability to numeric
# ---------------------------------------------------------

if "rppg_available" in behavior_row.index:

    value = str(
        behavior_row[
            "rppg_available"
        ]
    ).strip().lower()


    if value in [
        "true",
        "1",
        "yes"
    ]:

        behavior_row[
            "rppg_available"
        ] = 1.0

    else:

        behavior_row[
            "rppg_available"
        ] = 0.0


# ---------------------------------------------------------
# Convert everything else to numeric
# ---------------------------------------------------------

behavior_row = pd.to_numeric(
    behavior_row,
    errors="coerce"
)


behavior_values = behavior_row.values.astype(
    float
)


# =========================================================
# PREPARE TRAINING BEHAVIOR DATA
# =========================================================

train_values = train_df[
    behavior_columns
].copy()


# ---------------------------------------------------------
# Convert rPPG availability
# ---------------------------------------------------------

if "rppg_available" in train_values.columns:

    train_values[
        "rppg_available"
    ] = (
        train_values[
            "rppg_available"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "1": 1.0,
                "0": 0.0
            }
        )
    )


# ---------------------------------------------------------
# Convert all columns to numeric
# ---------------------------------------------------------

train_values = train_values.apply(
    pd.to_numeric,
    errors="coerce"
)


# =========================================================
# TRAINING MEDIANS
# =========================================================

train_medians = train_values.median()


# =========================================================
# IMPUTE SAMPLE MISSING VALUES
# =========================================================

for i, column in enumerate(
    behavior_columns
):

    if np.isnan(
        behavior_values[i]
    ):

        behavior_values[i] = (
            train_medians[column]
        )


# =========================================================
# IMPUTE TRAINING DATA
# =========================================================

train_values = train_values.fillna(
    train_medians
)


# =========================================================
# TRAINING MEAN AND STD
# =========================================================

train_mean = train_values.mean()

train_std = train_values.std()


# Avoid division by zero
train_std = train_std.replace(
    0,
    1.0
)


# =========================================================
# STANDARDIZE BEHAVIOR
# =========================================================

behavior_embedding = torch.tensor(
    behavior_values,
    dtype=torch.float32
)


mean_tensor = torch.tensor(
    train_mean.values,
    dtype=torch.float32
)


std_tensor = torch.tensor(
    train_std.values,
    dtype=torch.float32
)


behavior_embedding = (
    behavior_embedding
    - mean_tensor
) / std_tensor


# =========================================================
# ADD BATCH DIMENSION
# =========================================================

visual_embedding = (
    visual_embedding
    .unsqueeze(0)
    .to(device)
)

audio_embedding = (
    audio_embedding
    .unsqueeze(0)
    .to(device)
)

behavior_embedding = (
    behavior_embedding
    .unsqueeze(0)
    .to(device)
)


# =========================================================
# PRINT INPUT SHAPES
# =========================================================

print("\nInput shapes:")

print(
    "Visual:",
    visual_embedding.shape
)

print(
    "Audio:",
    audio_embedding.shape
)

print(
    "Behavior:",
    behavior_embedding.shape
)


# =========================================================
# MODEL PREDICTION
# =========================================================

print("\nRunning fusion model...")


with torch.no_grad():

    output = model(
        visual_embedding,
        audio_embedding,
        behavior_embedding
    )


# =========================================================
# EXTRACT LOGITS
# =========================================================

print(
    "\nModel output type:",
    type(output)
)


if isinstance(output, dict):

    print(
        "Model output keys:",
        list(output.keys())
    )


    if "logits" in output:

        logits = output["logits"]

    elif "output" in output:

        logits = output["output"]

    elif "prediction" in output:

        logits = output["prediction"]

    else:

        raise RuntimeError(
            "Could not find logits in model output.\n"
            f"Available keys: {list(output.keys())}"
        )


elif isinstance(output, tuple):

    logits = output[0]


else:

    logits = output


# =========================================================
# VERIFY LOGITS
# =========================================================

if not torch.is_tensor(logits):

    raise RuntimeError(
        f"Logits is not a tensor. "
        f"Type: {type(logits)}"
    )


print(
    "Logits shape:",
    logits.shape
)


# =========================================================
# PROBABILITIES
# =========================================================

probabilities = torch.softmax(
    logits,
    dim=1
)


# =========================================================
# PREDICTED CLASS
# =========================================================

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()


# =========================================================
# CONFIDENCE
# =========================================================

confidence = probabilities[
    0,
    predicted_class
].item()


# =========================================================
# LABEL MAPPING
# =========================================================

# Project convention:
# 0 = REAL
# 1 = FAKE

label_names = {
    0: "REAL",
    1: "FAKE"
}


prediction = label_names[
    predicted_class
]


real_probability = probabilities[
    0,
    0
].item()


fake_probability = probabilities[
    0,
    1
].item()


# =========================================================
# DISPLAY FORENSIC RESULT
# =========================================================

print("\n" + "=" * 60)

print(
    "EXPLAINABLE MULTIMODAL DEEPFAKE FORENSICS"
)

print("=" * 60)

print(
    "Sample key  :",
    sample_key
)

print(
    "Identity    :",
    sample["identity"]
)

print(
    "Video       :",
    sample["video"]
)

print(
    "Category    :",
    sample["category"]
)

print()

print(
    "Prediction  :",
    prediction
)

print(
    "Confidence  :",
    f"{confidence * 100:.2f}%"
)

print()

print("Class probabilities:")

print(
    "REAL:",
    f"{real_probability * 100:.2f}%"
)

print(
    "FAKE:",
    f"{fake_probability * 100:.2f}%"
)

print("=" * 60)


# =========================================================
# SAVE RESULT
# =========================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "forensic_prediction.txt"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPLAINABLE MULTIMODAL "
        "DEEPFAKE FORENSICS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Sample key: {sample_key}\n"
    )

    f.write(
        f"Identity: {sample['identity']}\n"
    )

    f.write(
        f"Video: {sample['video']}\n"
    )

    f.write(
        f"Category: {sample['category']}\n\n"
    )

    f.write(
        f"Prediction: {prediction}\n"
    )

    f.write(
        f"Confidence: "
        f"{confidence * 100:.2f}%\n\n"
    )

    f.write(
        f"REAL probability: "
        f"{real_probability * 100:.2f}%\n"
    )

    f.write(
        f"FAKE probability: "
        f"{fake_probability * 100:.2f}%\n"
    )


print("\n✓ Prediction saved:")

print(output_file)