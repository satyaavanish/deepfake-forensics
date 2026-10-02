import os
import torch
import pandas as pd

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.fusion.fusion_model import CrossModalFusionModel


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


# ============================================================
# PATHS
# ============================================================

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

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "confidence_score.txt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print(
    "\nLoading ViT embeddings..."
)

vit_data = torch.load(
    VIT_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    "ViT samples:",
    len(vit_data)
)


print(
    "\nLoading AST embeddings..."
)

ast_data = torch.load(
    AST_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    "AST samples:",
    len(ast_data)
)


# ============================================================
# LOAD BEHAVIOR
# ============================================================

print(
    "\nLoading behavioral features..."
)

behavior_df = pd.read_csv(
    BEHAVIOR_PATH
)

print(
    "Behavior rows:",
    len(behavior_df)
)


# ============================================================
# BEHAVIOR FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "sample_key",
    "identity",
    "video",
    "category",
    "split",
    "label"
]

behavior_columns = [
    column
    for column in behavior_df.columns
    if column not in metadata_columns
]


print(
    "Behavior features:",
    len(behavior_columns)
)


# ============================================================
# SELECT SAMPLE
# ============================================================

# Same sample used in your current forensic report.

TARGET_SAMPLE = (
    "id00049||00118_fake"
)


if TARGET_SAMPLE not in vit_data:

    raise RuntimeError(
        f"Sample not found in ViT embeddings: "
        f"{TARGET_SAMPLE}"
    )


if TARGET_SAMPLE not in ast_data:

    raise RuntimeError(
        f"Sample not found in AST embeddings: "
        f"{TARGET_SAMPLE}"
    )


behavior_row = behavior_df[
    behavior_df["sample_key"].astype(str)
    == TARGET_SAMPLE
]


if len(behavior_row) == 0:

    raise RuntimeError(
        f"Sample not found in behavioral features: "
        f"{TARGET_SAMPLE}"
    )


behavior_row = behavior_row.iloc[0]


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

visual_embedding = vit_data[
    TARGET_SAMPLE
]["embedding"].float()


audio_embedding = ast_data[
    TARGET_SAMPLE
]["embedding"].float()


# ============================================================
# EXTRACT BEHAVIOR
# ============================================================

behavior_values = []

for column in behavior_columns:

    value = behavior_row[column]

    try:

        value = float(value)

    except:

        value = 0.0

    behavior_values.append(
        value
    )


behavior_tensor = torch.tensor(
    behavior_values,
    dtype=torch.float32
)


print(
    "\nInput shapes:"
)

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
    behavior_tensor.shape
)


# ============================================================
# HANDLE NaN
# ============================================================

behavior_tensor = torch.nan_to_num(
    behavior_tensor,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# BATCH DIMENSION
# ============================================================

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

behavior_tensor = (
    behavior_tensor
    .unsqueeze(0)
    .to(device)
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading fusion model..."
)

model = CrossModalFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    num_heads=8
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


# Handle checkpoints saved either as
# a state_dict or as a dictionary.

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    device
)

model.eval()

print(
    "✓ Fusion model loaded"
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nCalculating confidence..."
)

with torch.no_grad():

    output = model(
        visual_embedding,
        audio_embedding,
        behavior_tensor
    )


# ============================================================
# EXTRACT LOGITS
# ============================================================

if isinstance(
    output,
    dict
):

    logits = output[
        "logits"
    ]

elif isinstance(
    output,
    tuple
):

    logits = output[0]

else:

    logits = output


# ============================================================
# SOFTMAX
# ============================================================

probabilities = torch.softmax(
    logits,
    dim=1
)


real_probability = float(
    probabilities[
        0,
        0
    ].item()
)

fake_probability = float(
    probabilities[
        0,
        1
    ].item()
)


# ============================================================
# PREDICTION
# ============================================================

predicted_class = int(
    torch.argmax(
        probabilities,
        dim=1
    ).item()
)


if predicted_class == 0:

    prediction = "REAL"

    confidence = (
        real_probability
    )

else:

    prediction = "FAKE"

    confidence = (
        fake_probability
    )


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "CONFIDENCE SCORE"
)

print(
    "=" * 60
)

print(
    f"Sample: {TARGET_SAMPLE}"
)

print(
    f"P(REAL): {real_probability:.4f}"
)

print(
    f"P(FAKE): {fake_probability:.4f}"
)

print(
    f"Prediction: {prediction}"
)

print(
    f"Model confidence: "
    f"{confidence * 100:.2f}%"
)

print(
    "=" * 60
)


# ============================================================
# SAVE RESULT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MODEL CONFIDENCE SCORE\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Sample: {TARGET_SAMPLE}\n\n"
    )

    f.write(
        f"P(REAL): "
        f"{real_probability:.4f}\n"
    )

    f.write(
        f"P(FAKE): "
        f"{fake_probability:.4f}\n\n"
    )

    f.write(
        f"Prediction: "
        f"{prediction}\n"
    )

    f.write(
        f"Model confidence: "
        f"{confidence * 100:.2f}%\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "Model confidence represents the probability "
        "assigned by the trained classifier to its "
        "predicted class. It is not proof that the "
        "video is objectively real or fake.\n"
    )


print(
    "\n✓ CONFIDENCE SCORE GENERATED"
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)