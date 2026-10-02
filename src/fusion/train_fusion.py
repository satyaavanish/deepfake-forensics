import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from fusion_model import CrossModalFusionModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

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

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "fusion"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "cross_modal_fusion_best.pth"
)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("CROSS-MODAL FUSION TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading ViT embeddings...")

vit_data = torch.load(
    VIT_FILE,
    map_location="cpu",
    weights_only=False
)

print(
    "ViT samples:",
    len(vit_data)
)

print("\nLoading AST embeddings...")

ast_data = torch.load(
    AST_FILE,
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

print("\nLoading behavioral features...")

behavior = pd.read_csv(
    BEHAVIOR_FILE
)

print(
    "Behavior rows:",
    len(behavior)
)


# ============================================================
# BEHAVIOR FEATURE COLUMNS
# ============================================================

BEHAVIOR_COLUMNS = [

    # Eye blink
    "blink_count",
    "blink_rate",
    "mean_blink_duration",
    "blink_interval_mean",
    "blink_interval_variance",
    "mean_ear",
    "ear_variance",

    # Lip / speech
    "mean_mouth_opening",
    "mouth_opening_variance",
    "mean_lip_distance",
    "lip_distance_variance",
    "mouth_movement_velocity",
    "mouth_movement_frequency",
    "speech_activity_ratio",
    "lip_activity_ratio",
    "lip_speech_consistency",

    # rPPG
    "heart_rate_estimate",
    "pulse_consistency",
    "temporal_signal_quality",
    "rppg_frames",
    "rppg_duration_seconds",

    # rPPG availability
    "rppg_available"
]


# ============================================================
# VERIFY BEHAVIOR COLUMNS
# ============================================================

for column in BEHAVIOR_COLUMNS:

    if column not in behavior.columns:

        raise ValueError(
            f"Missing behavior feature: {column}"
        )


print(
    "\nBehavior feature count:",
    len(BEHAVIOR_COLUMNS)
)


# ============================================================
# CREATE SAMPLE KEY
# ============================================================

behavior["sample_key"] = (
    behavior["identity"].astype(str)
    + "||"
    + behavior["video"].astype(str)
)


# ============================================================
# VERIFY BEHAVIOR KEYS
# ============================================================

if behavior["sample_key"].duplicated().any():

    raise ValueError(
        "Duplicate behavior sample keys found."
    )


# ============================================================
# BUILD COMMON DATASET
# ============================================================

print(
    "\nMatching ViT + AST + Behavior..."
)

records = []

missing_vit = 0
missing_ast = 0
missing_behavior = 0


for key, vit_item in vit_data.items():

    if key not in ast_data:
        missing_ast += 1
        continue

    behavior_rows = behavior[
        behavior["sample_key"] == key
    ]

    if len(behavior_rows) != 1:
        missing_behavior += 1
        continue

    behavior_row = behavior_rows.iloc[0]

    records.append({

        "sample_key": key,

        "vit": vit_item["embedding"],

        "ast": ast_data[key]["embedding"],

        "behavior": behavior_row[
            BEHAVIOR_COLUMNS
        ].values.astype(np.float32),

        "split": behavior_row["split"],

         "label": int(
    behavior_row["category"] != "real_video_real_audio"
),

        "identity": str(
            behavior_row["identity"]
        ),

        "category": str(
            behavior_row["category"]
        )
    })


print(
    "\nMatched samples:",
    len(records)
)

print(
    "Missing ViT:",
    missing_vit
)

print(
    "Missing AST:",
    missing_ast
)

print(
    "Missing behavior:",
    missing_behavior
)


# ============================================================
# EXPECT 1006
# ============================================================

if len(records) != 1006:

    raise ValueError(
        f"Expected 1006 matched samples, got {len(records)}"
    )

print(
    "✓ All 1006 samples matched"
)


# ============================================================
# SPLIT
# ============================================================

train_records = [
    r for r in records
    if r["split"] == "train"
]

val_records = [
    r for r in records
    if r["split"] == "val"
]

test_records = [
    r for r in records
    if r["split"] == "test"
]


print("\nSplit sizes:")

print(
    "Train:",
    len(train_records)
)

print(
    "Validation:",
    len(val_records)
)

print(
    "Test:",
    len(test_records)
)


# ============================================================
# BEHAVIOR NUMERIC ARRAYS
# ============================================================

train_behavior = np.stack(
    [
        r["behavior"]
        for r in train_records
    ]
)

val_behavior = np.stack(
    [
        r["behavior"]
        for r in val_records
    ]
)

test_behavior = np.stack(
    [
        r["behavior"]
        for r in test_records
    ]
)


# ============================================================
# HANDLE MISSING rPPG
# ============================================================

# rPPG is unavailable for 26 videos.
#
# We calculate the median for each rPPG feature using
# TRAINING DATA ONLY.
#
# This prevents validation/test information leakage.

RPPG_COLUMNS = [
    "heart_rate_estimate",
    "pulse_consistency",
    "temporal_signal_quality",
    "rppg_frames",
    "rppg_duration_seconds"
]

RPPG_INDICES = [
    BEHAVIOR_COLUMNS.index(
        column
    )
    for column in RPPG_COLUMNS
]


print(
    "\nHandling missing rPPG..."
)

for index, column in zip(
    RPPG_INDICES,
    RPPG_COLUMNS
):

    train_values = train_behavior[
        :, index
    ]

    valid_values = train_values[
        ~np.isnan(train_values)
    ]

    if len(valid_values) == 0:

        raise ValueError(
            f"No valid training values for {column}"
        )

    median_value = np.median(
        valid_values
    )

    train_behavior[
        np.isnan(
            train_behavior[:, index]
        ),
        index
    ] = median_value

    val_behavior[
        np.isnan(
            val_behavior[:, index]
        ),
        index
    ] = median_value

    test_behavior[
        np.isnan(
            test_behavior[:, index]
        ),
        index
    ] = median_value


# ============================================================
# STANDARDIZE BEHAVIOR
# ============================================================

print(
    "Standardizing behavioral features..."
)

scaler = StandardScaler()

train_behavior = scaler.fit_transform(
    train_behavior
)

val_behavior = scaler.transform(
    val_behavior
)

test_behavior = scaler.transform(
    test_behavior
)


# ============================================================
# UPDATE RECORDS
# ============================================================

for i, record in enumerate(
    train_records
):

    record["behavior"] = (
        train_behavior[i]
        .astype(np.float32)
    )


for i, record in enumerate(
    val_records
):

    record["behavior"] = (
        val_behavior[i]
        .astype(np.float32)
    )


for i, record in enumerate(
    test_records
):

    record["behavior"] = (
        test_behavior[i]
        .astype(np.float32)
    )


# ============================================================
# DATASET CLASS
# ============================================================

class FusionDataset(Dataset):

    def __init__(
        self,
        records
    ):

        self.records = records

    def __len__(self):

        return len(self.records)

    def __getitem__(
        self,
        index
    ):

        record = self.records[index]

        visual = record[
            "vit"
        ].float()

        audio = record[
            "ast"
        ].float()

        behavior = torch.tensor(
            record["behavior"],
            dtype=torch.float32
        )

        label = torch.tensor(
            record["label"],
            dtype=torch.long
        )

        return (
            visual,
            audio,
            behavior,
            label
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = FusionDataset(
    train_records
)

val_dataset = FusionDataset(
    val_records
)

test_dataset = FusionDataset(
    test_records
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LABEL COUNTS
# ============================================================

train_labels = np.array(
    [
        r["label"]
        for r in train_records
    ]
)

real_count = np.sum(
    train_labels == 0
)

fake_count = np.sum(
    train_labels == 1
)


print(
    "\nTraining class distribution:"
)

print(
    "REAL:",
    real_count
)

print(
    "FAKE:",
    fake_count
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Weight each class inversely to its frequency.

total = real_count + fake_count

real_weight = (
    total /
    (2.0 * real_count)
)

fake_weight = (
    total /
    (2.0 * fake_count)
)

class_weights = torch.tensor(
    [
        real_weight,
        fake_weight
    ],
    dtype=torch.float32,
    device=DEVICE
)


print(
    "\nClass weights:"
)

print(
    "REAL:",
    real_weight
)

print(
    "FAKE:",
    fake_weight
)


# ============================================================
# MODEL
# ============================================================

model = CrossModalFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    behavior_projection_dim=128,
    num_heads=8,
    dropout=0.2
)

model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    for visual, audio, behavior_x, labels in train_loader:

        visual = visual.to(
            DEVICE,
            non_blocking=True
        )

        audio = audio.to(
            DEVICE,
            non_blocking=True
        )

        behavior_x = behavior_x.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        output = model(
            visual,
            audio,
            behavior_x
        )

        logits = output[
            "logits"
        ]

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            labels.size(0)
        )

    epoch_loss = (
        running_loss /
        total_samples
    )

    epoch_accuracy = (
        correct /
        total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for visual, audio, behavior_x, labels in loader:

        visual = visual.to(
            DEVICE,
            non_blocking=True
        )

        audio = audio.to(
            DEVICE,
            non_blocking=True
        )

        behavior_x = behavior_x.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        output = model(
            visual,
            audio,
            behavior_x
        )

        logits = output[
            "logits"
        ]

        loss = criterion(
            logits,
            labels
        )

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            labels.size(0)
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        total_samples
    )

    epoch_accuracy = (
        correct /
        total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAIN
# ============================================================

best_val_accuracy = 0.0

print(
    "\n" + "=" * 70
)

print(
    "STARTING FUSION TRAINING"
)

print(
    "=" * 70
)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy, _, _ = (
        evaluate(
            val_loader
        )
    )

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "val_accuracy":
                    val_accuracy,

                "behavior_columns":
                    BEHAVIOR_COLUMNS,

                "behavior_scaler_mean":
                    scaler.mean_,

                "behavior_scaler_scale":
                    scaler.scale_
            },
            CHECKPOINT_FILE
        )

        print(
            "✓ Best model saved"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy, test_labels, test_predictions = (
    evaluate(
        test_loader
    )
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL FUSION TEST RESULT"
)

print(
    "=" * 70
)

print(
    "\nBest validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Test loss:",
    f"{test_loss:.4f}"
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)


print(
    "\nPrecision:",
    f"{precision:.4f}"
)

print(
    "Recall:",
    f"{recall:.4f}"
)

print(
    "F1:",
    f"{f1:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "fusion_training_results.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "CROSS-MODAL FUSION RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}\n"
    )

    f.write(
        f"Test Loss: "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{f1:.4f}\n"
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


print(
    "\nResults saved:"
)

print(
    results_file
)

print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT_FILE
)

print(
    "\n" + "=" * 70
)