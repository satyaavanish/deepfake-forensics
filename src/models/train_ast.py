import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, ASTModel
from tqdm import tqdm


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

# IMPORTANT:
# This is a separate checkpoint for the AUDIO detector.
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "ast_audio_best.pth"
)


# ============================================================
# AUDIO LABEL MAPPING
# ============================================================
#
# We are detecting AUDIO manipulation.
#
# fake_video_fake_audio  -> FAKE AUDIO
# fake_video_real_audio  -> REAL AUDIO
# real_video_fake_audio  -> FAKE AUDIO
# real_video_real_audio  -> REAL AUDIO
#
# 1 = FAKE AUDIO
# 0 = REAL AUDIO
# ============================================================

LABEL_MAP = {
    "fake_video_fake_audio": 1,
    "fake_video_real_audio": 0,
    "real_video_fake_audio": 1,
    "real_video_real_audio": 0
}


# ============================================================
# DATASET
# ============================================================

class FakeAVCelebAudioDataset(Dataset):

    def __init__(
        self,
        split,
        processor
    ):

        self.split = split

        self.processor = processor

        csv_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            f"{split}.csv"
        )

        self.df = pd.read_csv(
            csv_path
        )

        self.audio_dir = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "processed",
            "audio",
            split
        )

        self.samples = []

        for _, row in self.df.iterrows():

            category = str(
                row["category"]
            )

            if category not in LABEL_MAP:
                continue

            file_path = str(
                row["file"]
            )

            video_stem = os.path.splitext(
                os.path.basename(
                    file_path
                )
            )[0]

            audio_path = os.path.join(
                self.audio_dir,
                video_stem + ".wav"
            )

            if not os.path.exists(
                audio_path
            ):
                continue

            self.samples.append(
                {
                    "audio_path": audio_path,

                    "label": LABEL_MAP[
                        category
                    ],

                    "category": category,

                    "identity": str(
                        row["identity"]
                    ),

                    "video": video_stem
                }
            )

        print(
            f"{split.upper()} audio samples: "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load 16 kHz mono audio
        # ----------------------------------------------------

        waveform, _ = librosa.load(
            sample["audio_path"],
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # AST feature extraction
        # ----------------------------------------------------

        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs[
            "input_values"
        ].squeeze(0)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return (
            input_values,
            label
        )


# ============================================================
# AST AUDIO CLASSIFIER
# ============================================================

class ASTAudioClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Pretrained AST
        # ----------------------------------------------------

        self.ast = ASTModel.from_pretrained(
            MODEL_NAME
        )

        # ----------------------------------------------------
        # Freeze pretrained AST
        # ----------------------------------------------------

        for param in self.ast.parameters():

            param.requires_grad = False

        # ----------------------------------------------------
        # Audio classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                768,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                2
            )
        )

    def forward(
        self,
        input_values
    ):

        outputs = self.ast(
            input_values=input_values
        )

        # ----------------------------------------------------
        # AST hidden states
        # ----------------------------------------------------

        hidden_states = (
            outputs.last_hidden_state
        )

        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        audio_embedding = (
            hidden_states.mean(
                dim=1
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            audio_embedding
        )

        return logits


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    # Keep pretrained AST frozen
    model.ast.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(
        loader,
        desc="Training"
    )

    for input_values, labels in progress:

        input_values = input_values.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            input_values
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation"
        )

        for input_values, labels in progress:

            input_values = input_values.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                input_values
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AST AUDIO MANIPULATION CLASSIFIER"
    )

    print("=" * 70)

    print()

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Print label definition
    # --------------------------------------------------------

    print()

    print(
        "AUDIO LABEL DEFINITION:"
    )

    print(
        "fake_video_fake_audio -> FAKE AUDIO"
    )

    print(
        "fake_video_real_audio -> REAL AUDIO"
    )

    print(
        "real_video_fake_audio -> FAKE AUDIO"
    )

    print(
        "real_video_real_audio -> REAL AUDIO"
    )

    print()

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    print(
        "Loading AST feature extractor..."
    )

    processor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print()

    print(
        "Loading training dataset..."
    )

    train_dataset = FakeAVCelebAudioDataset(
        "train",
        processor
    )

    print()

    print(
        "Loading validation dataset..."
    )

    val_dataset = FakeAVCelebAudioDataset(
        "val",
        processor
    )

    print()

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading AST model..."
    )

    model = ASTAudioClassifier()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    train_labels = [
        sample["label"]
        for sample in train_dataset.samples
    ]

    real_count = train_labels.count(0)

    fake_count = train_labels.count(1)

    print()

    print(
        "Training class distribution:"
    )

    print(
        "REAL AUDIO:",
        real_count
    )

    print(
        "FAKE AUDIO:",
        fake_count
    )

    if real_count > 0:

        real_weight = (
            fake_count / real_count
        )

    else:

        real_weight = 1.0

    if fake_count > 0:

        fake_weight = 1.0

    else:

        fake_weight = 1.0

    class_weights = torch.tensor(
        [
            real_weight,
            fake_weight
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    print()

    print(
        "Class weights:",
        class_weights
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    print()

    print("=" * 70)

    print(
        "STARTING AUDIO CLASSIFIER TRAINING"
    )

    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()

        print("=" * 70)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss:          "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy:      "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss:     "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "model_name":
                        MODEL_NAME,

                    "label_map":
                        LABEL_MAP,

                    "sample_rate":
                        SAMPLE_RATE
                },
                CHECKPOINT_PATH
            )

            print()

            print(
                "✓ Best AUDIO AST model saved:"
            )

            print(
                CHECKPOINT_PATH
            )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "AST AUDIO TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print()

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()