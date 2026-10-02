import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# SAMPLE
# =========================================================

SAMPLE_KEY = "id00049||00118_fake"

IDENTITY = "id00049"
VIDEO_NAME = "00118_fake"

# This is the corresponding category from your selected sample
CATEGORY = "real_video_fake_audio"


# =========================================================
# PATHS
# =========================================================

FRAMES_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames",
    "test",
    CATEGORY,
    IDENTITY,
    VIDEO_NAME
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "anomaly_timestamps"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# CHECK FRAMES
# =========================================================

if not os.path.exists(FRAMES_DIR):

    raise FileNotFoundError(
        f"Frame directory not found:\n{FRAMES_DIR}"
    )


frame_files = sorted(
    [
        f
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith(".jpg")
    ]
)


if len(frame_files) == 0:

    raise RuntimeError(
        "No processed frames found."
    )


print("\nFrames found:", len(frame_files))


# =========================================================
# LOAD METADATA
# =========================================================

metadata = pd.read_csv(
    METADATA_PATH
)

metadata["video_stem"] = (
    metadata["file"]
    .astype(str)
    .apply(
        lambda x: os.path.splitext(
            os.path.basename(x)
        )[0]
    )
)


row = metadata[
    metadata["video_stem"] == VIDEO_NAME
]


if len(row) == 0:

    raise RuntimeError(
        f"Video not found in metadata: {VIDEO_NAME}"
    )


row = row.iloc[0]

duration = float(
    row["duration_seconds"]
)

fps = float(
    row["fps"]
)


print("Video duration:", duration, "seconds")
print("Original FPS:", fps)


# =========================================================
# LOAD ViT
# =========================================================

print("\nLoading ViT...")

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

vit = ViTModel.from_pretrained(
    "google/vit-base-patch16-224-in21k"
).to(device)

vit.eval()

print("✓ ViT loaded")


# =========================================================
# PROCESS FRAMES
# =========================================================

print("\nCalculating frame-level visual scores...")


scores = []


for index, frame_file in enumerate(frame_files):

    frame_path = os.path.join(
        FRAMES_DIR,
        frame_file
    )


    image = Image.open(
        frame_path
    ).convert("RGB")


    inputs = processor(
        images=image,
        return_tensors="pt"
    )


    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        outputs = vit(
            **inputs
        )


    # -----------------------------------------------------
    # CLS embedding
    # -----------------------------------------------------

    cls_embedding = (
        outputs.last_hidden_state[
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )


    # -----------------------------------------------------
    # Simple visual anomaly score
    #
    # We measure how far this frame's embedding is from
    # the video's average visual embedding.
    # -----------------------------------------------------

    scores.append(
        cls_embedding
    )


# =========================================================
# EMBEDDING MATRIX
# =========================================================

embeddings = np.stack(
    scores
)


mean_embedding = np.mean(
    embeddings,
    axis=0
)


distances = np.linalg.norm(
    embeddings - mean_embedding,
    axis=1
)


# =========================================================
# NORMALIZE SCORES
# =========================================================

if np.max(distances) > np.min(distances):

    normalized_scores = (
        distances - np.min(distances)
    ) / (
        np.max(distances)
        - np.min(distances)
    )

else:

    normalized_scores = np.zeros(
        len(distances)
    )


# =========================================================
# TIMESTAMPS
# =========================================================

results = []


for i, frame_file in enumerate(
    frame_files
):

    # Uniformly sampled frames
    if len(frame_files) > 1:

        timestamp = (
            i
            / (len(frame_files) - 1)
            * duration
        )

    else:

        timestamp = 0.0


    results.append(
        {
            "frame": frame_file,
            "frame_index": i + 1,
            "timestamp_seconds": timestamp,
            "visual_anomaly_score":
                float(normalized_scores[i])
        }
    )


# =========================================================
# SORT BY ANOMALY SCORE
# =========================================================

results_sorted = sorted(
    results,
    key=lambda x:
        x["visual_anomaly_score"],
    reverse=True
)


# =========================================================
# SELECT TOP 3
# =========================================================

top_results = results_sorted[:3]


# =========================================================
# DISPLAY
# =========================================================

print("\n" + "=" * 60)

print(
    "VISUAL ANOMALY TIMESTAMPS"
)

print("=" * 60)


for item in top_results:

    seconds = item[
        "timestamp_seconds"
    ]

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds
        - minutes * 60
    )

    timestamp = (
        f"{minutes:02d}:"
        f"{remaining_seconds:04.1f}"
    )

    print(
        f"{timestamp}  "
        f"| Frame {item['frame_index']:02d} "
        f"| Score "
        f"{item['visual_anomaly_score']:.4f}"
    )


print("=" * 60)


# =========================================================
# SAVE CSV
# =========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "anomaly_timestamps.csv"
)


pd.DataFrame(
    results
).to_csv(
    csv_path,
    index=False
)


# =========================================================
# SAVE REPORT
# =========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "anomaly_timestamps.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VISUAL ANOMALY TIMESTAMP ANALYSIS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Sample key: {SAMPLE_KEY}\n"
    )

    f.write(
        f"Identity: {IDENTITY}\n"
    )

    f.write(
        f"Video: {VIDEO_NAME}\n"
    )

    f.write(
        f"Category: {CATEGORY}\n"
    )

    f.write(
        f"Duration: {duration:.2f} seconds\n"
    )

    f.write(
        f"Frames analyzed: {len(frame_files)}\n\n"
    )

    f.write(
        "Top suspicious timestamps:\n\n"
    )


    for item in top_results:

        seconds = item[
            "timestamp_seconds"
        ]

        minutes = int(
            seconds // 60
        )

        remaining_seconds = (
            seconds
            - minutes * 60
        )

        timestamp = (
            f"{minutes:02d}:"
            f"{remaining_seconds:04.1f}"
        )

        f.write(
            f"{timestamp} | "
            f"Frame {item['frame_index']:02d} | "
            f"Score "
            f"{item['visual_anomaly_score']:.4f}\n"
        )


    f.write(
        "\nNOTE:\n"
    )

    f.write(
        "These timestamps are approximate because "
        "the current pipeline analyzes 16 uniformly "
        "sampled face frames rather than every original "
        "video frame.\n"
    )

    f.write(
        "The visual anomaly score represents deviation "
        "of a frame embedding from the video's mean "
        "visual embedding. It is an explainability "
        "indicator, not proof of manipulation.\n"
    )


print("\n✓ Anomaly analysis complete")

print(
    "CSV:",
    csv_path
)

print(
    "Report:",
    report_path
)