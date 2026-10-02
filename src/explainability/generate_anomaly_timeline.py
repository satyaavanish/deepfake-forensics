import os
import numpy as np
import pandas as pd


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

ANOMALY_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "anomaly_timestamps"
)

ANOMALY_FILE = os.path.join(
    ANOMALY_DIR,
    "anomaly_timestamps.csv"
)

OUTPUT_CSV = os.path.join(
    ANOMALY_DIR,
    "anomaly_timeline.csv"
)

OUTPUT_TXT = os.path.join(
    ANOMALY_DIR,
    "anomaly_timeline.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SEGMENT_DURATION = 2.0


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(ANOMALY_FILE):

    raise FileNotFoundError(
        f"Anomaly timestamp file not found:\n"
        f"{ANOMALY_FILE}"
    )


# ============================================================
# LOAD FRAME-LEVEL SCORES
# ============================================================

print(
    "Loading frame-level anomaly scores..."
)

df = pd.read_csv(
    ANOMALY_FILE
)


required_columns = [
    "frame",
    "frame_index",
    "timestamp_seconds",
    "visual_anomaly_score"
]


for column in required_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Required column missing: {column}"
        )


df = df.sort_values(
    "timestamp_seconds"
).reset_index(
    drop=True
)


# ============================================================
# VIDEO DURATION
# ============================================================

video_duration = float(
    df["timestamp_seconds"].max()
)

# The last sampled frame can be at the end of
# the video, so use the last timestamp as the
# available temporal boundary.

if video_duration <= 0:

    video_duration = SEGMENT_DURATION


# ============================================================
# CREATE SEGMENTS
# ============================================================

segment_results = []


segment_start = 0.0


while segment_start < video_duration:

    segment_end = min(
        segment_start + SEGMENT_DURATION,
        video_duration
    )


    segment_frames = df[
        (
            df["timestamp_seconds"]
            >= segment_start
        )
        &
        (
            df["timestamp_seconds"]
            < segment_end
        )
    ]


    # Include the final frame if it lies exactly
    # on the final boundary.
    if (
        segment_end == video_duration
        and len(segment_frames) == 0
    ):

        segment_frames = df[
            df["timestamp_seconds"]
            >= segment_start
        ]


    if len(segment_frames) > 0:

        scores = (
            segment_frames[
                "visual_anomaly_score"
            ]
            .astype(float)
            .values
        )

        mean_score = float(
            np.mean(scores)
        )

        max_score = float(
            np.max(scores)
        )

        frame_count = len(
            segment_frames
        )

        highest_frame = int(
            segment_frames.loc[
                segment_frames[
                    "visual_anomaly_score"
                ].idxmax(),
                "frame_index"
            ]
        )

    else:

        mean_score = 0.0
        max_score = 0.0
        frame_count = 0
        highest_frame = -1


    segment_results.append(
        {
            "segment_start_seconds":
                segment_start,

            "segment_end_seconds":
                segment_end,

            "mean_anomaly_score":
                mean_score,

            "max_anomaly_score":
                max_score,

            "frames_in_segment":
                frame_count,

            "highest_score_frame":
                highest_frame
        }
    )


    segment_start += SEGMENT_DURATION


# ============================================================
# DATAFRAME
# ============================================================

timeline_df = pd.DataFrame(
    segment_results
)


# ============================================================
# CLASSIFY SEGMENTS
# ============================================================
#
# These are descriptive visualization levels,
# not clinically/statistically validated
# manipulation thresholds.
#
# 0.00 - <0.33 : Normal
# 0.33 - <0.66 : Suspicious
# >=0.66       : Highly suspicious
#
# ============================================================

def classify_score(score):

    if score < 0.33:

        return "Normal"

    elif score < 0.66:

        return "Suspicious"

    else:

        return "Highly Suspicious"


timeline_df[
    "anomaly_level"
] = timeline_df[
    "mean_anomaly_score"
].apply(
    classify_score
)


# ============================================================
# TIMESTAMP FORMAT
# ============================================================

def format_time(seconds):

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:04.1f}"
    )


timeline_df[
    "time_range"
] = timeline_df.apply(
    lambda row:
        f"{format_time(row['segment_start_seconds'])}"
        f"–"
        f"{format_time(row['segment_end_seconds'])}",
    axis=1
)


# ============================================================
# SAVE CSV
# ============================================================

timeline_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DISPLAY TIMELINE
# ============================================================

print("\n")
print("=" * 75)
print("TEMPORAL ANOMALY TIMELINE")
print("=" * 75)

for _, row in timeline_df.iterrows():

    score = float(
        row["mean_anomaly_score"]
    )

    level = row[
        "anomaly_level"
    ]

    # Convert score into a visual bar.
    bar_length = max(
        1,
        int(
            round(
                score * 30
            )
        )
    )

    bar = "█" * bar_length

    print(
        f"{row['time_range']:>17} "
        f"{bar:<30} "
        f"{score:.4f} "
        f"| {level}"
    )


# ============================================================
# MOST SUSPICIOUS SEGMENT
# ============================================================

highest_index = timeline_df[
    "mean_anomaly_score"
].idxmax()


highest_segment = timeline_df.loc[
    highest_index
]


print("\n")
print(
    "Most suspicious temporal segment:"
)

print(
    f"{highest_segment['time_range']} "
    f"| Score: "
    f"{highest_segment['mean_anomaly_score']:.4f}"
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TEMPORAL ANOMALY TIMELINE\n"
    )

    f.write(
        "=" * 75
        + "\n\n"
    )

    f.write(
        "Segment duration: "
        f"{SEGMENT_DURATION:.1f} seconds\n"
    )

    f.write(
        "Scoring method: mean frame-level "
        "visual embedding deviation\n\n"
    )


    for _, row in timeline_df.iterrows():

        score = float(
            row["mean_anomaly_score"]
        )

        bar_length = max(
            1,
            int(
                round(
                    score * 30
                )
            )
        )

        bar = "█" * bar_length


        f.write(
            f"{row['time_range']:>17} "
            f"{bar:<30} "
            f"{score:.4f} "
            f"| {row['anomaly_level']}\n"
        )


    f.write("\n")

    f.write(
        "Most suspicious temporal segment:\n"
    )

    f.write(
        f"{highest_segment['time_range']} "
        f"| Score: "
        f"{highest_segment['mean_anomaly_score']:.4f}\n"
    )

    f.write("\n")

    f.write(
        "IMPORTANT NOTE:\n"
    )

    f.write(
        "The anomaly score is derived from "
        "frame-level ViT embedding deviation "
        "within each temporal segment. It is an "
        "explainability indicator and is not a "
        "direct frame-level manipulation probability.\n"
    )

    f.write(
        "The Normal/Suspicious/Highly Suspicious "
        "labels are visualization categories based "
        "on the current score ranges and are not "
        "validated forensic thresholds.\n"
    )

    f.write(
        "The timeline is approximate because the "
        "current pipeline uses 16 uniformly sampled "
        "frames rather than every original video frame.\n"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print(
    "✓ TEMPORAL ANOMALY TIMELINE GENERATED"
)

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "Report:",
    OUTPUT_TXT
)