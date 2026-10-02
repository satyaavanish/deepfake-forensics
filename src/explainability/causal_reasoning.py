import os
import sys
import numpy as np
import pandas as pd
import torch


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

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

BEHAVIOR_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "behavior_features.csv"
)

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

CONFIDENCE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "confidence_score.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "causal_reasoning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "causal_reasoning_report.txt"
)


# ============================================================
# TARGET SAMPLE
# ============================================================

TARGET_SAMPLE = "id00049||00118_fake"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(value, minimum, maximum):
    """
    Normalize a value to [0, 1].
    """

    if maximum == minimum:
        return 0.0

    value = (
        (value - minimum)
        / (maximum - minimum)
    )

    return float(
        np.clip(value, 0.0, 1.0)
    )


def evidence_level(score):
    """
    Convert evidence score into
    LOW / MEDIUM / HIGH.
    """

    if score < 0.33:
        return "LOW"

    elif score < 0.66:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# LOAD BEHAVIOR FEATURES
# ============================================================

print(
    "Loading behavioral features..."
)

behavior_df = pd.read_csv(
    BEHAVIOR_PATH
)


row = behavior_df[
    behavior_df["sample_key"].astype(str)
    == TARGET_SAMPLE
]


if len(row) == 0:

    raise RuntimeError(
        f"Sample not found: {TARGET_SAMPLE}"
    )


row = row.iloc[0]


# ============================================================
# BEHAVIORAL VALUES
# ============================================================

blink_count = float(
    row["blink_count"]
)

blink_rate = float(
    row["blink_rate"]
)

mean_ear = float(
    row["mean_ear"]
)

ear_variance = float(
    row["ear_variance"]
)

speech_activity = float(
    row["speech_activity_ratio"]
)

lip_activity = float(
    row["lip_activity_ratio"]
)

lip_speech_consistency = float(
    row["lip_speech_consistency"]
)

rppg_available = float(
    row["rppg_available"]
)

heart_rate = float(
    row["heart_rate_estimate"]
)

pulse_consistency = float(
    row["pulse_consistency"]
)

temporal_quality = float(
    row["temporal_signal_quality"]
)


# ============================================================
# 1. LIP-SYNC EVIDENCE
# ============================================================

# Higher inconsistency = stronger evidence.

lip_sync_anomaly = 1.0 - lip_speech_consistency

lip_sync_score = float(
    np.clip(
        lip_sync_anomaly,
        0.0,
        1.0
    )
)

lip_sync_level = evidence_level(
    lip_sync_score
)


# ============================================================
# 2. BLINK / BEHAVIOR EVIDENCE
# ============================================================

# Use blink-related features as supporting evidence.
#
# This is intentionally simple.
# It is NOT a medically validated blink abnormality detector.

blink_score = 0.0

if mean_ear < 0.18:
    blink_score += 0.4

if blink_rate < 0.10:
    blink_score += 0.2

if blink_rate > 1.00:
    blink_score += 0.3

if blink_count == 0:
    blink_score += 0.2

blink_score = float(
    np.clip(
        blink_score,
        0.0,
        1.0
    )
)

blink_level = evidence_level(
    blink_score
)


# ============================================================
# 3. rPPG EVIDENCE
# ============================================================

if rppg_available > 0.5:

    # Lower pulse consistency can indicate
    # weaker physiological consistency.

    rppg_score = 1.0 - pulse_consistency

    rppg_score = float(
        np.clip(
            rppg_score,
            0.0,
            1.0
        )
    )

else:

    # No rPPG evidence available.
    rppg_score = 0.0


rppg_level = evidence_level(
    rppg_score
)


# ============================================================
# 4. VISUAL EVIDENCE
# ============================================================

# Use the trained fusion model's final prediction
# as the primary visual/multimodal forensic signal.
#
# The actual visual heatmap is handled separately.

visual_score = 0.0

if os.path.exists(CONFIDENCE_PATH):

    with open(
        CONFIDENCE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        confidence_text = f.read()

    # Extract P(FAKE)

    for line in confidence_text.splitlines():

        if line.startswith(
            "P(FAKE):"
        ):

            try:

                fake_probability = float(
                    line.split(":")[1].strip()
                )

                visual_score = fake_probability

            except:

                pass


visual_level = evidence_level(
    visual_score
)


# ============================================================
# 5. AUDIO EVIDENCE
# ============================================================

# AST embeddings do not have a standalone
# fake/real probability in the current pipeline.
#
# Therefore do not fabricate an audio score.

audio_score = None

audio_level = "UNAVAILABLE"


# ============================================================
# EVIDENCE AGGREGATION
# ============================================================

available_scores = [
    visual_score,
    lip_sync_score,
    blink_score
]


if rppg_available > 0.5:

    available_scores.append(
        rppg_score
    )


overall_evidence_score = float(
    np.mean(
        available_scores
    )
)


# ============================================================
# FORENSIC REASONING
# ============================================================

# Conservative reasoning:
#
# High multimodal evidence -> FAKE
# Otherwise -> REAL
#
# This is an evidence aggregation rule,
# not a causal proof.

if overall_evidence_score >= 0.50:

    reasoning_output = "FAKE"

elif overall_evidence_score < 0.50:

    reasoning_output = "REAL"

else:

    reasoning_output = "UNCERTAIN"


# ============================================================
# SUPPORTING EVIDENCE COUNT
# ============================================================

evidence_items = [
    ("Visual manipulation", visual_score),
    ("Lip synchronization", lip_sync_score),
    ("Blink inconsistency", blink_score),
]

if rppg_available > 0.5:

    evidence_items.append(
        ("rPPG anomaly", rppg_score)
    )


high_count = sum(
    1
    for _, score in evidence_items
    if score >= 0.66
)

medium_count = sum(
    1
    for _, score in evidence_items
    if 0.33 <= score < 0.66
)

low_count = sum(
    1
    for _, score in evidence_items
    if score < 0.33
)


# ============================================================
# PRINT REPORT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CAUSAL / EVIDENCE REASONING"
)

print(
    "=" * 70
)

print(
    f"Sample: {TARGET_SAMPLE}"
)

print(
    "\nEvidence:"
)

print(
    f"Visual manipulation  → "
    f"{visual_level} "
    f"({visual_score:.4f})"
)

print(
    f"Audio manipulation   → "
    f"{audio_level}"
)

print(
    f"Lip synchronization  → "
    f"{lip_sync_level} "
    f"({lip_sync_score:.4f})"
)

print(
    f"Blink inconsistency  → "
    f"{blink_level} "
    f"({blink_score:.4f})"
)

print(
    f"rPPG anomaly         → "
    f"{rppg_level} "
    f"({rppg_score:.4f})"
)

print(
    "\nEvidence aggregation:"
)

print(
    f"Overall evidence score: "
    f"{overall_evidence_score:.4f}"
)

print(
    f"High evidence items: "
    f"{high_count}"
)

print(
    f"Medium evidence items: "
    f"{medium_count}"
)

print(
    f"Low evidence items: "
    f"{low_count}"
)

print(
    "\nForensic reasoning:"
)

print(
    f"Overall output → "
    f"{reasoning_output}"
)

print(
    "=" * 70
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAUSAL / EVIDENCE REASONING REPORT\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Sample: {TARGET_SAMPLE}\n\n"
    )

    f.write(
        "EVIDENCE\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        f"Visual manipulation → "
        f"{visual_level} "
        f"({visual_score:.4f})\n"
    )

    f.write(
        f"Audio manipulation → "
        f"{audio_level}\n"
    )

    f.write(
        f"Lip synchronization → "
        f"{lip_sync_level} "
        f"({lip_sync_score:.4f})\n"
    )

    f.write(
        f"Blink inconsistency → "
        f"{blink_level} "
        f"({blink_score:.4f})\n"
    )

    f.write(
        f"rPPG anomaly → "
        f"{rppg_level} "
        f"({rppg_score:.4f})\n\n"
    )

    f.write(
        "EVIDENCE AGGREGATION\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        f"Overall evidence score: "
        f"{overall_evidence_score:.4f}\n"
    )

    f.write(
        f"High evidence items: "
        f"{high_count}\n"
    )

    f.write(
        f"Medium evidence items: "
        f"{medium_count}\n"
    )

    f.write(
        f"Low evidence items: "
        f"{low_count}\n\n"
    )

    f.write(
        "FORENSIC REASONING\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        f"Overall output: "
        f"{reasoning_output}\n\n"
    )

    f.write(
        "LIMITATIONS\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        "This module is an evidence aggregation "
        "layer rather than a formal causal inference "
        "model. The evidence levels are heuristic "
        "descriptive categories and are not validated "
        "forensic thresholds.\n"
    )

    f.write(
        "Audio evidence is marked unavailable because "
        "the current AST pipeline does not produce a "
        "standalone audio manipulation probability.\n"
    )

    f.write(
        "rPPG is treated as supporting evidence only.\n"
    )


print(
    "\n✓ CAUSAL REASONING REPORT GENERATED"
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)