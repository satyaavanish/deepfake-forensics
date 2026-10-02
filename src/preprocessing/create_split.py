from pathlib import Path
import pandas as pd
import random


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = Path("outputs/dataset_metadata.csv")
OUTPUT_DIR = Path("outputs")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("Identity-Aware Dataset Split")
print("=" * 70)

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

df = pd.read_csv(METADATA_PATH)

print(f"\nTotal videos: {len(df)}")
print(f"Total identities: {df['identity'].nunique()}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "file",
    "category",
    "identity",
    "manipulation",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# GET UNIQUE IDENTITIES
# ============================================================

identities = sorted(df["identity"].dropna().unique())

print("\nIdentities:")
print(identities)

print(f"\nNumber of identities: {len(identities)}")


# ============================================================
# SHUFFLE IDENTITIES
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(identities)


# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

num_identities = len(identities)

train_count = round(num_identities * TRAIN_RATIO)
val_count = round(num_identities * VAL_RATIO)

# Whatever remains goes to test
test_count = num_identities - train_count - val_count


# ============================================================
# SPLIT IDENTITIES
# ============================================================

train_ids = identities[:train_count]

val_ids = identities[
    train_count:train_count + val_count
]

test_ids = identities[
    train_count + val_count:
]


# ============================================================
# PRINT IDENTITY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("IDENTITY SPLIT")
print("=" * 70)

print(f"TRAIN identities: {len(train_ids)}")
print(f"VAL identities:   {len(val_ids)}")
print(f"TEST identities:  {len(test_ids)}")

print("\nTrain identities:")
print(train_ids)

print("\nValidation identities:")
print(val_ids)

print("\nTest identities:")
print(test_ids)


# ============================================================
# CREATE DATAFRAME SPLITS
# ============================================================

train_df = df[df["identity"].isin(train_ids)].copy()

val_df = df[df["identity"].isin(val_ids)].copy()

test_df = df[df["identity"].isin(test_ids)].copy()


# ============================================================
# SHUFFLE VIDEOS
# ============================================================

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# VERIFY TOTAL
# ============================================================

total_split_videos = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("\n" + "=" * 70)
print("VIDEO COUNTS")
print("=" * 70)

print(f"TRAIN videos: {len(train_df)}")
print(f"VAL videos:   {len(val_df)}")
print(f"TEST videos:  {len(test_df)}")

print(f"\nTotal split videos: {total_split_videos}")
print(f"Original videos:    {len(df)}")


if total_split_videos != len(df):
    raise RuntimeError(
        "ERROR: Split video count does not match original dataset!"
    )

print("\n✓ All videos accounted for.")


# ============================================================
# VERIFY IDENTITY LEAKAGE
# ============================================================

train_identity_set = set(train_df["identity"])
val_identity_set = set(val_df["identity"])
test_identity_set = set(test_df["identity"])


train_val_overlap = train_identity_set & val_identity_set
train_test_overlap = train_identity_set & test_identity_set
val_test_overlap = val_identity_set & test_identity_set


print("\n" + "=" * 70)
print("IDENTITY LEAKAGE CHECK")
print("=" * 70)

print(
    f"TRAIN ∩ VAL:  {len(train_val_overlap)}"
)

print(
    f"TRAIN ∩ TEST: {len(train_test_overlap)}"
)

print(
    f"VAL ∩ TEST:   {len(val_test_overlap)}"
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "ERROR: Identity leakage detected!"
    )

print("\n✓ No identity leakage detected.")


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY DISTRIBUTION")
print("=" * 70)

print("\nTRAIN:")
print(train_df["category"].value_counts())

print("\nVALIDATION:")
print(val_df["category"].value_counts())

print("\nTEST:")
print(test_df["category"].value_counts())


# ============================================================
# SAVE CSV FILES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "val.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"TRAIN: {train_path}")
print(f"VAL:   {val_path}")
print(f"TEST:  {test_path}")

print("\nFiles created successfully.")