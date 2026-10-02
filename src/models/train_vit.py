import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_NAME = "google/vit-base-patch16-224-in21k"

NUM_FRAMES = 16
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
# DATASET
# ============================================================

class FakeAVCelebDataset(Dataset):

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

        self.df = pd.read_csv(csv_path)

        self.frames_root = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "processed",
            "frames",
            split
        )

        self.samples = []

        for _, row in self.df.iterrows():

            category = row["category"]

            if category not in LABEL_MAP:
                continue

            file_path = str(row["file"])

            video_stem = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            identity = str(row["identity"])

            video_name = (
                f"{identity}_{video_stem}"
            )

            category_dir = os.path.join(
                self.frames_root,
                category,
                identity,
                video_stem
            )

            # Check that the frame directory exists
            if not os.path.exists(category_dir):
                continue

            frame_files = sorted([
                f for f in os.listdir(category_dir)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ])

            if len(frame_files) < NUM_FRAMES:
                continue

            frame_files = frame_files[:NUM_FRAMES]

            self.samples.append({
                "frame_dir": category_dir,
                "frames": frame_files,
                "label": LABEL_MAP[category]
            })

        print(
            f"{split.upper()} samples: {len(self.samples)}"
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        sample = self.samples[index]

        images = []

        for frame_file in sample["frames"]:

            frame_path = os.path.join(
                sample["frame_dir"],
                frame_file
            )

            image = Image.open(
                frame_path
            ).convert("RGB")

            images.append(image)

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"]

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return pixel_values, label


# ============================================================
# MODEL
# ============================================================

class ViTVideoClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.vit = ViTModel.from_pretrained(
            MODEL_NAME
        )

        # Freeze pretrained ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )


    def forward(self, pixel_values):

        # pixel_values:
        # [batch, 16, 3, 224, 224]

        batch_size = pixel_values.size(0)

        num_frames = pixel_values.size(1)

        # Combine batch and frame dimensions
        x = pixel_values.view(
            batch_size * num_frames,
            3,
            224,
            224
        )

        outputs = self.vit(
            pixel_values=x
        )

        # CLS token
        cls = outputs.last_hidden_state[:, 0, :]

        # [batch * frames, 768]
        cls = cls.view(
            batch_size,
            num_frames,
            768
        )

        # Temporal mean pooling
        video_embedding = cls.mean(
            dim=1
        )

        logits = self.classifier(
            video_embedding
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

    # Keep ViT frozen
    model.vit.eval()

    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training"
    )

    for pixel_values, labels in progress:

        pixel_values = pixel_values.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            pixel_values
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
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
            loss=loss.item()
        )

    return (
        total_loss / total,
        correct / total
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

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for pixel_values, labels in tqdm(
            loader,
            desc="Validation"
        ):

            pixel_values = pixel_values.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                pixel_values
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VIT DEEPFAKE CLASSIFIER")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Loading processor...")

    processor = ViTImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_dataset = FakeAVCelebDataset(
        "train",
        processor
    )

    val_dataset = FakeAVCelebDataset(
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
    print("Loading ViT model...")

    model = ViTVideoClassifier()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):

        print()
        print("=" * 70)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        print()
        print(
            f"Train Loss:       {train_loss:.4f}"
        )

        print(
            f"Train Accuracy:   {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss:  {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "vit_baseline_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy
                },
                checkpoint_path
            )

            print()
            print(
                "✓ Best model saved:"
            )

            print(
                checkpoint_path
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()