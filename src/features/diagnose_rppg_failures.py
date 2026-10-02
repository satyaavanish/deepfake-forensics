import os
import cv2
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

RPPG_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features",
    "rppg_features.csv"
)


metadata = pd.read_csv(METADATA_FILE)
rppg = pd.read_csv(RPPG_FILE)

processed = set(
    rppg["file"].astype(str)
)

failed = metadata[
    ~metadata["file"].astype(str).isin(processed)
]

print("=" * 70)
print("RPPG FAILURE DIAGNOSIS")
print("=" * 70)

print(
    f"\nFailed videos: {len(failed)}"
)

print("\n")


for index, row in failed.iterrows():

    relative_path = str(
        row["file"]
    )

    video_path = os.path.join(
        PROJECT_ROOT,
        relative_path
    )

    print("-" * 70)

    print(
        "File:",
        relative_path
    )

    print(
        "Exists:",
        os.path.exists(video_path)
    )

    if not os.path.exists(video_path):
        print(
            "REASON: Video file does not exist"
        )
        continue

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "REASON: OpenCV cannot open video"
        )

        continue

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    print(
        f"Duration: {duration:.2f} sec"
    )

    # Try reading frames
    readable_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        readable_frames += 1

        if readable_frames >= 160:
            break

    cap.release()

    print(
        f"Readable frames: "
        f"{readable_frames}"
    )

    if readable_frames < 60:

        print(
            "REASON: Fewer than 60 readable frames"
        )

    else:

        print(
            "REASON: Enough frames; "
            "failure is likely inside rPPG processing"
        )


print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)