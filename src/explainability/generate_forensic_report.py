import os
import re
import pandas as pd
from datetime import timedelta


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

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "forensic_prediction.txt"
)

ANOMALY_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "anomaly_timestamps",
    "anomaly_timestamps.csv"
)

BEHAVIOR_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "behavior_features.csv"
)

HEATMAP_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "vit_heatmaps"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_forensic_report.txt"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def get_value(text, pattern, default="N/A"):

    match = re.search(
        pattern,
        text
    )

    if match:
        return match.group(1).strip()

    return default


def safe_float(value):

    try:
        return float(value)
    except:
        return None


def format_seconds(seconds):

    if seconds is None:
        return "N/A"

    minutes = int(seconds // 60)

    remaining = seconds - (
        minutes * 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:04.1f}"
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    PREDICTION_FILE,
    ANOMALY_FILE,
    BEHAVIOR_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD PREDICTION
# ============================================================

print("Loading prediction...")

with open(
    PREDICTION_FILE,
    "r",
    encoding="utf-8"
) as f:

    prediction_text = f.read()


sample_key = get_value(
    prediction_text,
    r"Sample key\s*:\s*(.+)"
)

identity = get_value(
    prediction_text,
    r"Identity\s*:\s*(.+)"
)

video = get_value(
    prediction_text,
    r"Video\s*:\s*(.+)"
)

category = get_value(
    prediction_text,
    r"Category\s*:\s*(.+)"
)

prediction = get_value(
    prediction_text,
    r"Prediction\s*:\s*(REAL|FAKE)"
)

real_confidence = get_value(
    prediction_text,
    r"REAL probability\s*:\s*([0-9.]+)%"
)

fake_confidence = get_value(
    prediction_text,
    r"FAKE probability\s*:\s*([0-9.]+)%"
)

# ============================================================
# LOAD BEHAVIOR FEATURES
# ============================================================

print("Loading behavioral features...")

behavior_df = pd.read_csv(
    BEHAVIOR_FILE
)


# Try sample_key first
behavior_row = behavior_df[
    behavior_df["sample_key"].astype(str)
    == str(sample_key)
]


# Fallback using identity + video
if len(behavior_row) == 0:

    behavior_row = behavior_df[
        (behavior_df["identity"].astype(str) == str(identity))
        &
        (behavior_df["video"].astype(str) == str(video))
    ]


if len(behavior_row) == 0:

    raise RuntimeError(
        "Could not find behavioral features "
        f"for sample: {sample_key}"
    )


behavior = behavior_row.iloc[0]


# ============================================================
# BEHAVIOR FEATURES
# ============================================================

def feature(name):

    if name not in behavior.index:
        return None

    return safe_float(
        behavior[name]
    )


blink_count = feature(
    "blink_count"
)

blink_rate = feature(
    "blink_rate"
)

mean_blink_duration = feature(
    "mean_blink_duration"
)

mean_ear = feature(
    "mean_ear"
)

ear_variance = feature(
    "ear_variance"
)

mean_mouth_opening = feature(
    "mean_mouth_opening"
)

mouth_opening_variance = feature(
    "mouth_opening_variance"
)

mean_lip_distance = feature(
    "mean_lip_distance"
)

lip_distance_variance = feature(
    "lip_distance_variance"
)

mouth_movement_velocity = feature(
    "mouth_movement_velocity"
)

mouth_movement_frequency = feature(
    "mouth_movement_frequency"
)

speech_activity_ratio = feature(
    "speech_activity_ratio"
)

lip_activity_ratio = feature(
    "lip_activity_ratio"
)

lip_speech_consistency = feature(
    "lip_speech_consistency"
)

heart_rate = feature(
    "heart_rate_estimate"
)

pulse_consistency = feature(
    "pulse_consistency"
)

temporal_signal_quality = feature(
    "temporal_signal_quality"
)

rppg_frames = feature(
    "rppg_frames"
)

rppg_duration = feature(
    "rppg_duration_seconds"
)

rppg_available = feature(
    "rppg_available"
)


# ============================================================
# LOAD ANOMALY TIMESTAMPS
# ============================================================

print("Loading anomaly timestamps...")

anomaly_df = pd.read_csv(
    ANOMALY_FILE
)


anomaly_df = anomaly_df.sort_values(
    "visual_anomaly_score",
    ascending=False
)


top_anomalies = anomaly_df.head(3)


# ============================================================
# FIND HEATMAP
# ============================================================
# ============================================================
# FIND HEATMAP
# ============================================================

heatmap_path = None

if os.path.exists(HEATMAP_DIR):

    possible_files = [
        f
        for f in os.listdir(HEATMAP_DIR)
        if f.lower().endswith(".jpg")
    ]

    for file_name in possible_files:

        if (
            str(identity) in file_name
            and str(video) in file_name
        ):

            heatmap_path = os.path.join(
                HEATMAP_DIR,
                file_name
            )

            break

# ============================================================
# GENERATE REPORT
# ============================================================

print("Generating final forensic report...")


report = []


report.append(
    "=" * 75
)

report.append(
    "EXPLAINABLE MULTIMODAL DEEPFAKE FORENSIC REPORT"
)

report.append(
    "=" * 75
)

report.append("")


# ============================================================
# SAMPLE INFORMATION
# ============================================================

report.append(
    "1. SAMPLE INFORMATION"
)

report.append(
    "-" * 75
)

report.append(
    f"Sample Key       : {sample_key}"
)

report.append(
    f"Identity         : {identity}"
)

report.append(
    f"Video            : {video}"
)

report.append(
    f"Dataset Category : {category}"
)

report.append("")


# ============================================================
# PREDICTION
# ============================================================

report.append(
    "2. FINAL PREDICTION"
)

report.append(
    "-" * 75
)

report.append(
    f"Prediction       : {prediction}"
)

report.append(
    f"REAL confidence  : {real_confidence}%"
)

report.append(
    f"FAKE confidence  : {fake_confidence}%"
)

report.append("")


# ============================================================
# VISUAL EVIDENCE
# ============================================================

report.append(
    "3. VISUAL EVIDENCE"
)

report.append(
    "-" * 75
)

report.append(
    "Visual model      : ViT-Base/16"
)

report.append(
    "Visual embedding  : 768-dimensional"
)

report.append(
    "Frames analyzed   : 16 sampled face frames"
)

report.append(
    "Visual analysis   : Frame-level embedding deviation"
)

if heatmap_path is not None:

    report.append(
        f"Attention heatmap : {heatmap_path}"
    )

else:

    report.append(
        "Attention heatmap : Not found"
    )

report.append("")


# ============================================================
# AUDIO EVIDENCE
# ============================================================

report.append(
    "4. AUDIO EVIDENCE"
)

report.append(
    "-" * 75
)

report.append(
    "Audio model       : AST"
)

report.append(
    "Audio embedding   : 768-dimensional"
)

report.append(
    "Audio input       : 16 kHz mono waveform"
)

report.append(
    "Audio evidence    : AST embedding available"
)

report.append(
    "Note              : The current explainability "
    "pipeline does not assign a standalone synthetic-audio "
    "probability to this sample."
)

report.append("")


# ============================================================
# LIP-SPEECH EVIDENCE
# ============================================================

report.append(
    "5. LIP-SPEECH EVIDENCE"
)

report.append(
    "-" * 75
)

if speech_activity_ratio is not None:

    report.append(
        f"Speech activity ratio   : "
        f"{speech_activity_ratio:.4f}"
    )

if lip_activity_ratio is not None:

    report.append(
        f"Lip activity ratio      : "
        f"{lip_activity_ratio:.4f}"
    )

if lip_speech_consistency is not None:

    report.append(
        f"Lip-speech consistency  : "
        f"{lip_speech_consistency:.4f}"
    )

if mean_mouth_opening is not None:

    report.append(
        f"Mean mouth opening      : "
        f"{mean_mouth_opening:.4f}"
    )

if mouth_opening_variance is not None:

    report.append(
        f"Mouth opening variance  : "
        f"{mouth_opening_variance:.4f}"
    )

if mouth_movement_velocity is not None:

    report.append(
        f"Mouth movement velocity : "
        f"{mouth_movement_velocity:.4f}"
    )

report.append("")


# ============================================================
# BLINK / BEHAVIOR
# ============================================================

report.append(
    "6. BEHAVIORAL EVIDENCE"
)

report.append(
    "-" * 75
)

if blink_count is not None:

    report.append(
        f"Blink count             : "
        f"{blink_count:.4f}"
    )

if blink_rate is not None:

    report.append(
        f"Blink rate              : "
        f"{blink_rate:.4f}"
    )

if mean_blink_duration is not None:

    report.append(
        f"Mean blink duration     : "
        f"{mean_blink_duration:.4f} sec"
    )

if mean_ear is not None:

    report.append(
        f"Mean EAR                : "
        f"{mean_ear:.4f}"
    )

if ear_variance is not None:

    report.append(
        f"EAR variance            : "
        f"{ear_variance:.6f}"
    )

report.append("")


# ============================================================
# rPPG
# ============================================================

report.append(
    "7. PHYSIOLOGICAL / rPPG EVIDENCE"
)

report.append(
    "-" * 75
)

if rppg_available is not None:

    if rppg_available == 1:

        report.append(
            "rPPG available         : YES"
        )

    else:

        report.append(
            "rPPG available         : NO"
        )


if heart_rate is not None:

    report.append(
        f"Heart-rate estimate    : "
        f"{heart_rate:.2f} BPM"
    )

if pulse_consistency is not None:

    report.append(
        f"Pulse consistency      : "
        f"{pulse_consistency:.4f}"
    )

if temporal_signal_quality is not None:

    report.append(
        f"Temporal signal quality: "
        f"{temporal_signal_quality:.4f}"
    )

if rppg_frames is not None:

    report.append(
        f"rPPG frames            : "
        f"{rppg_frames:.0f}"
    )

if rppg_duration is not None:

    report.append(
        f"rPPG duration          : "
        f"{rppg_duration:.2f} sec"
    )

report.append("")


# ============================================================
# ANOMALY TIMESTAMPS
# ============================================================

report.append(
    "8. SUSPICIOUS VISUAL TIMESTAMPS"
)

report.append(
    "-" * 75
)

for index, (_, row) in enumerate(
    top_anomalies.iterrows(),
    start=1
):

    timestamp = float(
        row["timestamp_seconds"]
    )

    score = float(
        row["visual_anomaly_score"]
    )

    frame_index = int(
        row["frame_index"]
    )

    report.append(
        f"{index}. "
        f"{format_seconds(timestamp)} "
        f"| Frame {frame_index:02d} "
        f"| Visual score {score:.4f}"
    )

report.append("")


# ============================================================
# INTERPRETATION
# ============================================================

report.append(
    "9. EVIDENCE INTERPRETATION"
)

report.append(
    "-" * 75
)

report.append(
    "The final prediction is produced by the trained "
    "cross-modal fusion model using visual, audio, and "
    "behavioral features."
)

report.append(
    "The visual timestamps identify frames whose ViT "
    "embeddings deviate most from the video's mean "
    "visual embedding."
)

report.append(
    "These scores are explainability indicators and "
    "should not be interpreted as direct proof of "
    "manipulation at a specific timestamp."
)

report.append(
    "Lip-speech, blink, and rPPG values are reported "
    "as measured features. They should be interpreted "
    "together with the multimodal prediction rather "
    "than independently."
)

report.append("")


# ============================================================
# LIMITATIONS
# ============================================================

report.append(
    "10. LIMITATIONS"
)

report.append(
    "-" * 75
)

report.append(
    "1. Only 16 uniformly sampled face frames are used "
    "for the current visual explainability analysis."
)

report.append(
    "2. Therefore, suspicious timestamps are approximate."
)

report.append(
    "3. Visual embedding deviation is not equivalent to "
    "a trained frame-level manipulation detector."
)

report.append(
    "4. The current audio explainability stage does not "
    "produce a standalone AST fake/real probability."
)

report.append(
    "5. rPPG is unavailable for videos that are too short "
    "for the required temporal analysis."
)

report.append(
    "6. Behavioral features provide supporting forensic "
    "evidence and should not be treated as standalone "
    "proof of a deepfake."
)

report.append("")


# ============================================================
# OUTPUT
# ============================================================

report.append(
    "=" * 75
)

report.append(
    "END OF FORENSIC REPORT"
)

report.append(
    "=" * 75
)


final_report = "\n".join(
    report
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        final_report
    )


# ============================================================
# PRINT
# ============================================================

print("\n")
print(final_report)

print("\n")
print(
    "✓ FINAL FORENSIC REPORT GENERATED"
)

print(
    "Report:"
)

print(
    REPORT_FILE
)