import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

EYE_FILE = os.path.join(
    FEATURE_DIR,
    "eye_blink_features.csv"
)

LIP_FILE = os.path.join(
    FEATURE_DIR,
    "lip_speech_features.csv"
)

RPPG_FILE = os.path.join(
    FEATURE_DIR,
    "rppg_features_final.csv"
)

OUTPUT_FILE = os.path.join(
    FEATURE_DIR,
    "behavior_features.csv"
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
# LOAD FEATURE FILES
# ============================================================

print("=" * 70)
print("MERGING BEHAVIORAL FEATURES")
print("=" * 70)

print("\nLoading feature files...")

eye = pd.read_csv(EYE_FILE)
lip = pd.read_csv(LIP_FILE)
rppg = pd.read_csv(RPPG_FILE)

print("\nEye:", eye.shape)
print("Lip:", lip.shape)
print("rPPG:", rppg.shape)


# ============================================================
# CREATE UNIQUE SAMPLE KEY
# ============================================================

def create_sample_key(df, video_column):

    required_columns = [
        "identity",
        video_column
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column '{column}'"
            )

    df["sample_key"] = (
        df["identity"].astype(str)
        + "||"
        + df[video_column].astype(str)
    )

    return df


# ============================================================
# CREATE KEYS
# ============================================================

# Eye and Lip files use column: video
eye = create_sample_key(
    eye,
    "video"
)

lip = create_sample_key(
    lip,
    "video"
)

# rPPG file uses column: video_name
rppg = create_sample_key(
    rppg,
    "video_name"
)


# ============================================================
# CHECK DUPLICATES BEFORE MERGING
# ============================================================

print("\nDuplicate sample keys BEFORE merge:")

eye_duplicates = eye["sample_key"].duplicated().sum()
lip_duplicates = lip["sample_key"].duplicated().sum()
rppg_duplicates = rppg["sample_key"].duplicated().sum()

print(
    "Eye:",
    eye_duplicates
)

print(
    "Lip:",
    lip_duplicates
)

print(
    "rPPG:",
    rppg_duplicates
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

eye = eye.drop_duplicates(
    subset=["sample_key"],
    keep="first"
)

lip = lip.drop_duplicates(
    subset=["sample_key"],
    keep="first"
)

rppg = rppg.drop_duplicates(
    subset=["sample_key"],
    keep="first"
)


print("\nAfter duplicate removal:")

print(
    "Eye:",
    len(eye)
)

print(
    "Lip:",
    len(lip)
)

print(
    "rPPG:",
    len(rppg)
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# Eye-blink features
# ------------------------------------------------------------

eye_features = [
    "blink_count",
    "blink_rate",
    "mean_blink_duration",
    "blink_interval_mean",
    "blink_interval_variance",
    "mean_ear",
    "ear_variance"
]


# ------------------------------------------------------------
# Lip + speech features
# ------------------------------------------------------------

lip_features = [
    "mean_mouth_opening",
    "mouth_opening_variance",
    "mean_lip_distance",
    "lip_distance_variance",
    "mouth_movement_velocity",
    "mouth_movement_frequency",
    "speech_activity_ratio",
    "lip_activity_ratio",
    "lip_speech_consistency"
]


# ------------------------------------------------------------
# rPPG features
# ------------------------------------------------------------

rppg_features = [
    "heart_rate_estimate",
    "pulse_consistency",
    "temporal_signal_quality",
    "rppg_frames",
    "rppg_duration_seconds"
]


# ============================================================
# VERIFY FEATURE COLUMNS
# ============================================================

print("\nChecking required feature columns...")


for column in eye_features:

    if column not in eye.columns:

        raise ValueError(
            f"Missing eye feature: {column}"
        )


for column in lip_features:

    if column not in lip.columns:

        raise ValueError(
            f"Missing lip feature: {column}"
        )


for column in rppg_features:

    if column not in rppg.columns:

        raise ValueError(
            f"Missing rPPG feature: {column}"
        )


print(
    "All required feature columns found."
)


# ============================================================
# SELECT COLUMNS
# ============================================================

eye_small = eye[
    [
        "sample_key",
        "identity",
        "video",
        "category",
        "split"
    ]
    + eye_features
].copy()


lip_small = lip[
    [
        "sample_key"
    ]
    + lip_features
].copy()


rppg_small = rppg[
    [
        "sample_key"
    ]
    + rppg_features
].copy()


# ============================================================
# MERGE EYE + LIP
# ============================================================

print("\nMerging eye + lip...")

behavior = pd.merge(
    eye_small,
    lip_small,
    on="sample_key",
    how="inner",
    validate="one_to_one"
)

print(
    "After eye + lip:",
    behavior.shape
)


# ============================================================
# MERGE rPPG
# ============================================================

print("\nMerging rPPG...")

behavior = pd.merge(
    behavior,
    rppg_small,
    on="sample_key",
    how="left",
    validate="one_to_one"
)

print(
    "After rPPG:",
    behavior.shape
)


# ============================================================
# rPPG AVAILABILITY
# ============================================================

behavior["rppg_available"] = (
    behavior[
        "heart_rate_estimate"
    ]
    .notna()
    .astype(int)
)


print("\nrPPG availability:")

print(
    behavior[
        "rppg_available"
    ].value_counts()
)


# ============================================================
# CREATE BINARY LABEL
# ============================================================

behavior["label"] = (
    behavior[
        "category"
    ].map(LABEL_MAP)
)


# ============================================================
# CHECK LABELS
# ============================================================

print("\nLabel distribution:")

print(
    behavior[
        "label"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# CHECK UNKNOWN LABELS
# ============================================================

unknown_labels = behavior[
    "label"
].isna().sum()

if unknown_labels > 0:

    print(
        "\nWARNING:",
        unknown_labels,
        "samples have no valid label."
    )

    print(
        behavior[
            behavior["label"].isna()
        ]["category"].value_counts()
    )

else:

    print(
        "\nAll samples have valid labels."
    )


# ============================================================
# CONVERT NON-rPPG FEATURES TO NUMERIC
# ============================================================

non_rppg_features = (
    eye_features
    + lip_features
)


for column in non_rppg_features:

    behavior[column] = pd.to_numeric(
        behavior[column],
        errors="coerce"
    )


# ============================================================
# HANDLE MISSING NON-rPPG FEATURES
# ============================================================

print(
    "\nChecking missing non-rPPG features..."
)

for column in non_rppg_features:

    missing = behavior[
        column
    ].isna().sum()

    if missing > 0:

        print(
            f"{column}: {missing} missing → filling with 0"
        )

        behavior[column] = (
            behavior[column]
            .fillna(0)
        )


# ============================================================
# rPPG MISSING VALUES
# ============================================================

print(
    "\nMissing rPPG values:"
)

rppg_missing = (
    behavior[
        rppg_features
    ]
    .isna()
    .sum()
)

print(
    rppg_missing
)


# IMPORTANT:
# We DO NOT fabricate rPPG values.
#
# The 26 short videos that could not produce rPPG
# measurements remain NaN.
#
# rppg_available indicates whether rPPG exists.


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

duplicate_count = (
    behavior[
        "sample_key"
    ]
    .duplicated()
    .sum()
)

print(
    "\nFinal duplicate sample keys:",
    duplicate_count
)


if duplicate_count != 0:

    raise ValueError(
        "Duplicate samples remain after merge!"
    )


# ============================================================
# FINAL SAMPLE COUNT
# ============================================================

print(
    "\nFinal behavior samples:",
    len(behavior)
)


if len(behavior) != 1006:

    print(
        "\nWARNING:"
        " Expected 1006 samples."
        f" Got {len(behavior)}."
    )

else:

    print(
        "✓ Correct sample count: 1006"
    )


# ============================================================
# CHECK SPLITS
# ============================================================

print(
    "\nSplit distribution:"
)

print(
    behavior[
        "split"
    ].value_counts()
)


# ============================================================
# SORT DATA
# ============================================================

behavior = behavior.sort_values(
    [
        "split",
        "identity",
        "video"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

behavior.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BEHAVIOR FEATURE MERGE COMPLETE")
print("=" * 70)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nRows:",
    len(behavior)
)

print(
    "Columns:",
    len(behavior.columns)
)

print(
    "\nFinal split distribution:"
)

print(
    behavior[
        "split"
    ].value_counts()
)

print(
    "\nFinal label distribution:"
)

print(
    behavior[
        "label"
    ].value_counts()
)

print(
    "\nFinal rPPG availability:"
)

print(
    behavior[
        "rppg_available"
    ].value_counts()
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)