import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "vit_baseline_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "vit_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
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

class FakeAVCelebTestDataset(Dataset):

    def __init__(self, processor):

        self.processor = processor

        csv_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "test.csv"
        )

        self.df = pd.read_csv(csv_path)

        self.frames_root = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "processed",
            "frames",
            "test"
        )

        self.samples = []

        for _, row in self.df.iterrows():

            category = str(row["category"])

            if category not in LABEL_MAP:
                continue

            file_path = str(row["file"])

            video_stem = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            identity = str(row["identity"])

            frame_dir = os.path.join(
                self.frames_root,
                category,
                identity,
                video_stem
            )

            if not os.path.exists(frame_dir):
                continue

            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ])

            if len(frame_files) == 0:
                continue

            # If fewer than 16 frames exist,
            # repeat the last frame.
            if len(frame_files) < NUM_FRAMES:

                while len(frame_files) < NUM_FRAMES:
                    frame_files.append(
                        frame_files[-1]
                    )

            # Select exactly 16 frames
            if len(frame_files) > NUM_FRAMES:
                frame_files = frame_files[:NUM_FRAMES]

            self.samples.append({
                "frame_dir": frame_dir,
                "frames": frame_files,
                "label": LABEL_MAP[category],
                "category": category,
                "identity": identity,
                "video": video_stem
            })

        print(
            f"Test samples loaded: {len(self.samples)}"
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

        return (
            pixel_values,
            label
        )


# ============================================================
# MODEL
# ============================================================

class ViTVideoClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.vit = ViTModel.from_pretrained(
            MODEL_NAME
        )

        # Same architecture used during training
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )


    def forward(self, pixel_values):

        batch_size = pixel_values.size(0)

        num_frames = pixel_values.size(1)

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

        cls = cls.view(
            batch_size,
            num_frames,
            768
        )

        # Mean pooling across frames
        video_embedding = cls.mean(
            dim=1
        )

        logits = self.classifier(
            video_embedding
        )

        return logits


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("VIT TEST SET EVALUATION")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    print()
    print("Loading ViT processor...")

    processor = ViTImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    test_dataset = FakeAVCelebTestDataset(
        processor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = ViTVideoClassifier()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        "✓ Checkpoint loaded successfully"
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"]
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print()
    print("Running test evaluation...")

    with torch.no_grad():

        for pixel_values, labels in tqdm(
            test_loader,
            desc="Testing"
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

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Probability of FAKE
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # Convert to NumPy
    y_true = np.array(all_labels)

    y_pred = np.array(all_predictions)

    y_prob = np.array(all_probabilities)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # ROC-AUC
    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print()
    print("Confusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "                 REAL  FAKE"
    )
    print(
        f"Actual REAL      {cm[0][0]:4d}  {cm[0][1]:4d}"
    )
    print(
        f"Actual FAKE      {cm[1][0]:4d}  {cm[1][1]:4d}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("Classification Report")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = os.path.join(
        OUTPUT_DIR,
        "vit_test_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as f:

        f.write(
            "ViT Test Set Evaluation\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            f"Accuracy:  {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall:    {recall:.4f}\n"
        )

        f.write(
            f"F1-score:  {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC:   {roc_auc:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write("\n\n")

        f.write(
            classification_report(
                y_true,
                y_pred,
                target_names=[
                    "REAL",
                    "FAKE"
                ],
                zero_division=0
            )
        )

    print()
    print(
        "✓ Results saved to:"
    )

    print(
        results_path
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()