import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORT MODEL
# ============================================================

from src.fusion.fusion_model import CrossModalFusionModel


# ============================================================
# PATHS
# ============================================================

VIT_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings",
    "vit_embeddings.pt"
)

AST_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "embeddings",
    "ast_embeddings.pt"
)

BEHAVIOR_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "behavior_features.csv"
)

CHECKPOINT_FILE = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cross_modal_fusion_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "fusion"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FUSION MODEL EVALUATION")
print("=" * 70)
print("Device:", device)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading ViT embeddings...")

vit_data = torch.load(
    VIT_FILE,
    map_location="cpu",
    weights_only=False
)

print("ViT samples:", len(vit_data))


print("\nLoading AST embeddings...")

ast_data = torch.load(
    AST_FILE,
    map_location="cpu",
    weights_only=False
)

print("AST samples:", len(ast_data))


# ============================================================
# LOAD BEHAVIOR
# ============================================================

print("\nLoading behavioral features...")

behavior_df = pd.read_csv(BEHAVIOR_FILE)

print("Behavior rows:", len(behavior_df))


# ============================================================
# CREATE SAMPLE KEY
# ============================================================

behavior_df["sample_key"] = (
    behavior_df["identity"].astype(str)
    + "||"
    + behavior_df["video"].astype(str)
)


# ============================================================
# BEHAVIOR FEATURES
# ============================================================

BEHAVIOR_FEATURES = [
    # Eye-blink features
    "blink_count",
    "blink_rate",
    "mean_blink_duration",
    "blink_interval_mean",
    "blink_interval_variance",
    "mean_ear",
    "ear_variance",

    # Lip + speech features
    "mean_mouth_opening",
    "mouth_opening_variance",
    "mean_lip_distance",
    "lip_distance_variance",
    "mouth_movement_velocity",
    "mouth_movement_frequency",
    "speech_activity_ratio",
    "lip_activity_ratio",
    "lip_speech_consistency",

    # rPPG features
    "heart_rate_estimate",
    "pulse_consistency",
    "temporal_signal_quality",
    "rppg_frames",
    "rppg_duration_seconds",
    "rppg_available",
]

print("\nBehavior features:", len(BEHAVIOR_FEATURES))


# ============================================================
# CONVERT NUMERIC FEATURES
# ============================================================

for column in BEHAVIOR_FEATURES:
    behavior_df[column] = pd.to_numeric(
        behavior_df[column],
        errors="coerce"
    )


# ============================================================
# MATCH SAMPLES
# ============================================================

records = []

for _, row in behavior_df.iterrows():

    sample_key = row["sample_key"]

    if sample_key not in vit_data:
        continue

    if sample_key not in ast_data:
        continue

    records.append({
        "sample_key": sample_key,
        "identity": row["identity"],
        "video": row["video"],
        "category": row["category"],
        "split": row["split"],

        "visual": vit_data[sample_key]["embedding"],
        "audio": ast_data[sample_key]["embedding"],

        "behavior": row[BEHAVIOR_FEATURES].values.astype(
            np.float32
        )
    })


print("\nMatched samples:", len(records))


# ============================================================
# TEST SET ONLY
# ============================================================

test_records = [
    r for r in records
    if r["split"] == "test"
]

print("Test samples:", len(test_records))


# ============================================================
# FINAL LABEL
#
# REAL-REAL = REAL
# EVERYTHING ELSE = FAKE
# ============================================================

def get_final_label(category):

    if category == "real_video_real_audio":
        return 0

    return 1


y_true = np.array([
    get_final_label(r["category"])
    for r in test_records
])


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print("\nTest category distribution:")
print("-" * 50)

category_counts = {}

for r in test_records:

    category = r["category"]

    if category not in category_counts:
        category_counts[category] = 0

    category_counts[category] += 1

for category, count in category_counts.items():

    label = get_final_label(category)

    label_name = "FAKE" if label == 1 else "REAL"

    print(
        f"{category:30s} "
        f"{count:3d} samples "
        f"→ {label_name}"
    )


# ============================================================
# TRAINING DATA FOR BEHAVIOR NORMALIZATION
# ============================================================

train_records = [
    r for r in records
    if r["split"] == "train"
]

train_behavior = np.array([
    r["behavior"]
    for r in train_records
], dtype=np.float32)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

train_medians = np.nanmedian(
    train_behavior,
    axis=0
)

for i in range(train_behavior.shape[1]):

    nan_mask = np.isnan(train_behavior[:, i])

    train_behavior[nan_mask, i] = train_medians[i]


# ============================================================
# MEAN / STD
# ============================================================

train_mean = np.mean(
    train_behavior,
    axis=0
)

train_std = np.std(
    train_behavior,
    axis=0
)

train_std[train_std < 1e-8] = 1.0


# ============================================================
# PREPARE TEST BEHAVIOR
# ============================================================

test_behavior = np.array([
    r["behavior"]
    for r in test_records
], dtype=np.float32)

for i in range(test_behavior.shape[1]):

    nan_mask = np.isnan(test_behavior[:, i])

    test_behavior[nan_mask, i] = train_medians[i]


# ============================================================
# STANDARDIZE
# ============================================================

test_behavior = (
    test_behavior - train_mean
) / train_std


# ============================================================
# PREPARE VISUAL / AUDIO
# ============================================================

visual = torch.stack([
    r["visual"].float()
    for r in test_records
])

audio = torch.stack([
    r["audio"].float()
    for r in test_records
])

behavior = torch.tensor(
    test_behavior,
    dtype=torch.float32
)


print("\nInput shapes:")
print("Visual :", visual.shape)
print("Audio  :", audio.shape)
print("Behavior:", behavior.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fusion model...")

model = CrossModalFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    behavior_projection_dim=128,
    num_heads=8,
    dropout=0.2
)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=False
)


# Handle either complete checkpoint or state_dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)
model.eval()

print("✓ Fusion model loaded")


# ============================================================
# INFERENCE
# ============================================================

visual = visual.to(device)
audio = audio.to(device)
behavior = behavior.to(device)

with torch.no_grad():

    output = model(
        visual,
        audio,
        behavior
    )

    logits = output["logits"]

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

y_pred = predictions.cpu().numpy()

fake_probabilities = (
    probabilities[:, 1]
    .cpu()
    .numpy()
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

if len(np.unique(y_true)) == 2:

    roc_auc = roc_auc_score(
        y_true,
        fake_probabilities
    )

else:

    roc_auc = float("nan")


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FUSION TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print("----------------")

print("                 Predicted")
print("                 REAL    FAKE")

print(
    f"Actual REAL     {cm[0,0]:5d}   {cm[0,1]:5d}"
)

print(
    f"Actual FAKE     {cm[1,0]:5d}   {cm[1,1]:5d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0
    )
)


# ============================================================
# PER-CATEGORY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-CATEGORY RESULTS")
print("=" * 70)

for category in sorted(category_counts.keys()):

    indices = [
        i
        for i, r in enumerate(test_records)
        if r["category"] == category
    ]

    category_true = y_true[indices]
    category_pred = y_pred[indices]

    category_accuracy = accuracy_score(
        category_true,
        category_pred
    )

    print(
        f"\n{category}"
    )

    print(
        f"Samples : {len(indices)}"
    )

    print(
        f"Target  : "
        f"{'FAKE' if get_final_label(category) else 'REAL'}"
    )

    print(
        f"Accuracy: {category_accuracy:.4f}"
    )

    print(
        "Predictions:",
        {
            "REAL": int(np.sum(category_pred == 0)),
            "FAKE": int(np.sum(category_pred == 1))
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "fusion_test_results_new_labels.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("FUSION MODEL TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("FINAL LABEL DEFINITION\n")
    f.write("----------------------\n")
    f.write("real_video_real_audio = REAL\n")
    f.write("all other categories = FAKE\n\n")

    f.write(f"Test samples: {len(test_records)}\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Category Distribution\n")
    f.write("---------------------\n")

    for category, count in category_counts.items():

        f.write(
            f"{category}: {count}\n"
        )

print("\n")
print("Results saved to:")
print(results_file)

print("\n✓ FUSION EVALUATION COMPLETE")