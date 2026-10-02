import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RPPG_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "rppg_features.csv"
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "train.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "val.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "rppg_features_final.csv"
)


print("=" * 70)
print("FINALIZING rPPG FEATURES")
print("=" * 70)


# ============================================================
# LOAD RPPG
# ============================================================

rppg = pd.read_csv(
    RPPG_FILE
)

print(
    f"\nCurrent rPPG rows: {len(rppg)}"
)


# ============================================================
# LOAD SPLITS
# ============================================================

train = pd.read_csv(
    TRAIN_FILE
)

val = pd.read_csv(
    VAL_FILE
)

test = pd.read_csv(
    TEST_FILE
)


train["split"] = "train"
val["split"] = "val"
test["split"] = "test"


split_data = pd.concat(
    [
        train[["file", "split"]],
        val[["file", "split"]],
        test[["file", "split"]]
    ],
    ignore_index=True
)


# ============================================================
# REMOVE OLD SPLIT COLUMN
# ============================================================

if "split" in rppg.columns:

    rppg = rppg.drop(
        columns=["split"]
    )


# ============================================================
# MERGE SPLIT
# ============================================================

rppg = rppg.merge(
    split_data,
    on="file",
    how="left"
)


# ============================================================
# CHECK
# ============================================================

print("\nSplit distribution:")

print(
    rppg["split"].value_counts(
        dropna=False
    )
)


missing_split = rppg[
    rppg["split"].isna()
]

print(
    f"\nRows without split: "
    f"{len(missing_split)}"
)


# ============================================================
# SAVE
# ============================================================

rppg.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nFinal file saved to:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("rPPG FINALIZATION COMPLETE")
print("=" * 70)