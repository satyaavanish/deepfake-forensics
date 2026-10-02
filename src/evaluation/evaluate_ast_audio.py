import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa

from transformers import AutoFeatureExtractor, ASTModel
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
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

SAMPLE_RATE = 16000

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "ast_audio_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "ast_audio_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# AUDIO LABELS
# ============================================================

LABEL_MAP = {

    "fake_video_fake_audio": 1,

    "fake_video_real_audio": 0,

    "real_video_fake_audio": 1,

    "real_video_real_audio": 0
}


# ============================================================
# AST MODEL
# ============================================================

class ASTAudioClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.ast = ASTModel.from_pretrained(
            MODEL_NAME
        )

        for param in self.ast.parameters():

            param.requires_grad = False

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

        hidden_states = (
            outputs.last_hidden_state
        )

        audio_embedding = (
            hidden_states.mean(
                dim=1
            )
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

    print(
        "AST AUDIO MANIPULATION TEST EVALUATION"
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


    # ========================================================
    # LOAD TEST CSV
    # ========================================================

    test_csv = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "test.csv"
    )

    test_df = pd.read_csv(
        test_csv
    )

    print()

    print(
        "Test samples in CSV:",
        len(test_df)
    )


    # ========================================================
    # AUDIO DIRECTORY
    # ========================================================

    audio_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "audio",
        "test"
    )


    # ========================================================
    # LOAD PROCESSOR
    # ========================================================

    print()

    print(
        "Loading AST feature extractor..."
    )

    processor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print(
        "Loading AST audio classifier..."
    )

    model = ASTAudioClassifier()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ Audio checkpoint loaded"
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    y_true = []

    y_pred = []

    y_probability = []

    categories = []

    identities = []

    videos = []

    skipped = 0


    print()

    print(
        "Evaluating test set..."
    )

    progress = tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="AST Test"
    )


    for _, row in progress:

        category = str(
            row["category"]
        )

        if category not in LABEL_MAP:

            skipped += 1

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
            audio_dir,
            video_stem + ".wav"
        )


        if not os.path.exists(
            audio_path
        ):

            skipped += 1

            continue


        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        waveform, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        # ----------------------------------------------------
        # AST INPUT
        # ----------------------------------------------------

        inputs = processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )


        input_values = (
            inputs["input_values"]
            .to(DEVICE)
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                input_values
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            fake_probability = (
                probabilities[0, 1]
                .item()
            )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        true_label = LABEL_MAP[
            category
        ]

        y_true.append(
            true_label
        )

        y_pred.append(
            prediction
        )

        y_probability.append(
            fake_probability
        )

        categories.append(
            category
        )

        identities.append(
            str(row["identity"])
        )

        videos.append(
            video_stem
        )


    # ========================================================
    # ARRAYS
    # ========================================================

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_probability = np.array(
        y_probability
    )


    print()

    print(
        "Evaluated samples:",
        len(y_true)
    )

    print(
        "Skipped:",
        skipped
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    if len(
        np.unique(y_true)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    else:

        roc_auc = float("nan")


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "AST AUDIO TEST RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print()

    print(
        "Confusion Matrix"
    )

    print(
        "----------------"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL     "
        f"{cm[0,0]:5d}   "
        f"{cm[0,1]:5d}"
    )

    print(
        f"Actual FAKE     "
        f"{cm[1,0]:5d}   "
        f"{cm[1,1]:5d}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()

    print(
        "Classification Report"
    )

    print(
        "---------------------"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "REAL AUDIO",
                "FAKE AUDIO"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # PER CATEGORY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "PER-CATEGORY AUDIO RESULTS"
    )

    print("=" * 70)


    results = []

    for category in sorted(
        set(categories)
    ):

        indices = [
            i
            for i, c in enumerate(
                categories
            )
            if c == category
        ]

        category_true = (
            y_true[indices]
        )

        category_pred = (
            y_pred[indices]
        )

        category_probability = (
            y_probability[indices]
        )

        category_accuracy = (
            accuracy_score(
                category_true,
                category_pred
            )
        )

        target = (
            "FAKE AUDIO"
            if LABEL_MAP[category] == 1
            else "REAL AUDIO"
        )

        print()

        print(
            category
        )

        print(
            "Samples:",
            len(indices)
        )

        print(
            "Target:",
            target
        )

        print(
            f"Accuracy: "
            f"{category_accuracy:.4f}"
        )

        print(
            "Predictions:",
            {
                "REAL AUDIO":
                    int(
                        np.sum(
                            category_pred == 0
                        )
                    ),

                "FAKE AUDIO":
                    int(
                        np.sum(
                            category_pred == 1
                        )
                    )
            }
        )

        print(
            f"Mean FAKE probability: "
            f"{category_probability.mean():.4f}"
        )

        results.append(
            {
                "category": category,
                "samples": len(indices),
                "target": target,
                "accuracy": category_accuracy,
                "mean_fake_probability":
                    category_probability.mean()
            }
        )


    # ========================================================
    # SAVE DETAILED PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "identity": identities,
            "video": videos,
            "category": categories,
            "true_label": y_true,
            "predicted_label": y_pred,
            "fake_audio_probability":
                y_probability
        }
    )

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "ast_audio_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )


    # ========================================================
    # SAVE CATEGORY RESULTS
    # ========================================================

    category_df = pd.DataFrame(
        results
    )

    category_path = os.path.join(
        OUTPUT_DIR,
        "ast_audio_category_results.csv"
    )

    category_df.to_csv(
        category_path,
        index=False
    )


    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    report_path = os.path.join(
        OUTPUT_DIR,
        "ast_audio_test_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "AST AUDIO MANIPULATION TEST RESULTS\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "LABEL DEFINITION\n"
        )

        f.write(
            "fake_video_fake_audio = FAKE AUDIO\n"
        )

        f.write(
            "fake_video_real_audio = REAL AUDIO\n"
        )

        f.write(
            "real_video_fake_audio = FAKE AUDIO\n"
        )

        f.write(
            "real_video_real_audio = REAL AUDIO\n\n"
        )

        f.write(
            f"Evaluated samples: "
            f"{len(y_true)}\n"
        )

        f.write(
            f"Skipped: {skipped}\n\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score : {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC  : {roc_auc:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\n"
        )

        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "REAL AUDIO",
                    "FAKE AUDIO"
                ],
                zero_division=0
            )
        )

        f.write(
            "\n\nPER-CATEGORY RESULTS\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        for result in results:

            f.write(
                f"\n{result['category']}\n"
            )

            f.write(
                f"Samples: "
                f"{result['samples']}\n"
            )

            f.write(
                f"Target: "
                f"{result['target']}\n"
            )

            f.write(
                f"Accuracy: "
                f"{result['accuracy']:.4f}\n"
            )

            f.write(
                f"Mean FAKE probability: "
                f"{result['mean_fake_probability']:.4f}\n"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "Prediction file:"
    )

    print(
        prediction_path
    )

    print()

    print(
        "Category results:"
    )

    print(
        category_path
    )

    print()

    print(
        "Report:"
    )

    print(
        report_path
    )

    print()

    print(
        "✓ AST AUDIO TEST EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()