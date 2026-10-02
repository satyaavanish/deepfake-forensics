import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa

from transformers import (
    AutoFeatureExtractor,
    ASTModel
)

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
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000

BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "ast_baseline_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "ast_evaluation"
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
# TEST DATASET
# ============================================================

class FakeAVCelebAudioTestDataset(Dataset):

    def __init__(self, processor):

        self.processor = processor

        csv_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "test.csv"
        )

        self.df = pd.read_csv(
            csv_path
        )

        self.audio_dir = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "processed",
            "audio",
            "test"
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

            self.samples.append({
                "audio_path": audio_path,
                "label": LABEL_MAP[category],
                "category": category,
                "identity": str(
                    row["identity"]
                ),
                "video": video_stem
            })

        print(
            f"Test audio samples: "
            f"{len(self.samples)}"
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        sample = self.samples[index]

        waveform, _ = librosa.load(
            sample["audio_path"],
            sr=SAMPLE_RATE,
            mono=True
        )

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
# AST CLASSIFIER
# ============================================================

class ASTAudioClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.ast = ASTModel.from_pretrained(
            MODEL_NAME
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )


    def forward(
        self,
        input_values
    ):

        outputs = self.ast(
            input_values=input_values
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        audio_embedding = (
            hidden_states.mean(dim=1)
        )

        logits = self.classifier(
            audio_embedding
        )

        return logits


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AST TEST SET EVALUATION")
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
    # PROCESSOR
    # --------------------------------------------------------

    print()
    print(
        "Loading AST feature extractor..."
    )

    processor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print(
        "Loading test dataset..."
    )

    test_dataset = (
        FakeAVCelebAudioTestDataset(
            processor
        )
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
    print(
        "Loading AST model..."
    )

    model = ASTAudioClassifier()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "✓ AST checkpoint loaded"
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"]
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print()
    print(
        "Running test evaluation..."
    )

    all_labels = []

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        for input_values, labels in tqdm(
            test_loader,
            desc="Testing"
        ):

            input_values = input_values.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                input_values
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

    # --------------------------------------------------------
    # NUMPY ARRAYS
    # --------------------------------------------------------

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    y_prob = np.array(
        all_probabilities
    )

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

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

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
    print("AST TEST RESULTS")
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

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print()
    print("Confusion Matrix")

    print(
        "                 Predicted"
    )

    print(
        "                 REAL  FAKE"
    )

    print(
        f"Actual REAL      "
        f"{cm[0][0]:4d}  "
        f"{cm[0][1]:4d}"
    )

    print(
        f"Actual FAKE      "
        f"{cm[1][0]:4d}  "
        f"{cm[1][1]:4d}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )

    print()
    print(
        "Classification Report"
    )

    print()

    print(report)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = os.path.join(
        OUTPUT_DIR,
        "ast_test_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as f:

        f.write(
            "AST Test Set Evaluation\n"
        )

        f.write(
            "=======================\n\n"
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

        f.write(
            "\n\nClassification Report\n\n"
        )

        f.write(
            report
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
    print("AST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()