import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\siris\deepfake_forensics"

PROCESSED_FRAMES = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.21

# Number of consecutive sampled frames required
# to consider the eyes closed
MIN_BLINK_FRAMES = 2

# Your preprocessing sampled 16 frames/video
EXPECTED_FRAMES = 16


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ============================================================
# EYE LANDMARKS
# ============================================================

# Left eye
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

# Right eye
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# EAR FUNCTION
# ============================================================

def calculate_ear(landmarks, eye_indices, image_width, image_height):

    points = []

    for idx in eye_indices:

        landmark = landmarks[idx]

        x = landmark.x * image_width
        y = landmark.y * image_height

        points.append(np.array([x, y]))

    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return float(ear)


# ============================================================
# DETECT EAR FOR ONE FRAME
# ============================================================

def get_frame_ear(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0].landmark

    height, width = image.shape[:2]

    left_ear = calculate_ear(
        face_landmarks,
        LEFT_EYE,
        width,
        height
    )

    right_ear = calculate_ear(
        face_landmarks,
        RIGHT_EYE,
        width,
        height
    )

    average_ear = (
        left_ear + right_ear
    ) / 2.0

    return average_ear


# ============================================================
# DETECT BLINKS
# ============================================================

def detect_blinks(
    ears,
    video_duration
):

    if len(ears) == 0:
        return {
            "blink_count": 0,
            "blink_rate": 0.0,
            "mean_blink_duration": 0.0,
            "blink_interval_mean": 0.0,
            "blink_interval_variance": 0.0
        }

    # --------------------------------------------------------
    # Frames where eyes are considered closed
    # --------------------------------------------------------

    closed = [
        ear < EAR_THRESHOLD
        for ear in ears
    ]

    blink_events = []

    start = None

    for i, is_closed in enumerate(closed):

        if is_closed:

            if start is None:
                start = i

        else:

            if start is not None:

                end = i - 1

                duration_frames = (
                    end - start + 1
                )

                if duration_frames >= MIN_BLINK_FRAMES:

                    blink_events.append(
                        (
                            start,
                            end
                        )
                    )

                start = None

    # --------------------------------------------------------
    # Handle blink extending to final frame
    # --------------------------------------------------------

    if start is not None:

        end = len(closed) - 1

        duration_frames = (
            end - start + 1
        )

        if duration_frames >= MIN_BLINK_FRAMES:

            blink_events.append(
                (
                    start,
                    end
                )
            )

    blink_count = len(blink_events)

    # --------------------------------------------------------
    # Convert sampled-frame positions to seconds
    # --------------------------------------------------------

    if len(ears) > 1:

        sample_interval = (
            video_duration /
            (len(ears) - 1)
        )

    else:

        sample_interval = 0.0

    # --------------------------------------------------------
    # Blink durations
    # --------------------------------------------------------

    blink_durations = []

    for start, end in blink_events:

        duration = (
            end - start + 1
        ) * sample_interval

        blink_durations.append(
            duration
        )

    # --------------------------------------------------------
    # Blink centers
    # --------------------------------------------------------

    blink_centers = []

    for start, end in blink_events:

        center = (
            (start + end) / 2.0
        )

        center_time = (
            center * sample_interval
        )

        blink_centers.append(
            center_time
        )

    # --------------------------------------------------------
    # Blink intervals
    # --------------------------------------------------------

    blink_intervals = []

    for i in range(
        1,
        len(blink_centers)
    ):

        interval = (
            blink_centers[i]
            -
            blink_centers[i - 1]
        )

        blink_intervals.append(
            interval
        )

    # --------------------------------------------------------
    # Blink rate
    # --------------------------------------------------------

    if video_duration > 0:

        blink_rate = (
            blink_count /
            video_duration
        )

    else:

        blink_rate = 0.0

    # --------------------------------------------------------
    # Mean blink duration
    # --------------------------------------------------------

    if blink_durations:

        mean_blink_duration = float(
            np.mean(
                blink_durations
            )
        )

    else:

        mean_blink_duration = 0.0

    # --------------------------------------------------------
    # Blink interval statistics
    # --------------------------------------------------------

    if blink_intervals:

        blink_interval_mean = float(
            np.mean(
                blink_intervals
            )
        )

        blink_interval_variance = float(
            np.var(
                blink_intervals
            )
        )

    else:

        blink_interval_mean = 0.0

        blink_interval_variance = 0.0

    return {

        "blink_count":
            blink_count,

        "blink_rate":
            blink_rate,

        "mean_blink_duration":
            mean_blink_duration,

        "blink_interval_mean":
            blink_interval_mean,

        "blink_interval_variance":
            blink_interval_variance
    }


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    frame_directory,
    video_duration
):

    frame_files = [
        f
        for f in os.listdir(frame_directory)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    frame_files.sort()

    ears = []

    for frame_file in frame_files:

        frame_path = os.path.join(
            frame_directory,
            frame_file
        )

        image = cv2.imread(
            frame_path
        )

        if image is None:
            continue

        ear = get_frame_ear(
            image
        )

        if ear is not None:

            ears.append(
                ear
            )

    # --------------------------------------------------------
    # Blink features
    # --------------------------------------------------------

    blink_features = detect_blinks(
        ears,
        video_duration
    )

    # --------------------------------------------------------
    # EAR statistics
    # --------------------------------------------------------

    if ears:

        mean_ear = float(
            np.mean(ears)
        )

        ear_variance = float(
            np.var(ears)
        )

    else:

        mean_ear = 0.0
        ear_variance = 0.0

    return {

        "frames_detected":
            len(ears),

        "blink_count":
            blink_features[
                "blink_count"
            ],

        "blink_rate":
            blink_features[
                "blink_rate"
            ],

        "mean_blink_duration":
            blink_features[
                "mean_blink_duration"
            ],

        "blink_interval_mean":
            blink_features[
                "blink_interval_mean"
            ],

        "blink_interval_variance":
            blink_features[
                "blink_interval_variance"
            ],

        "mean_ear":
            mean_ear,

        "ear_variance":
            ear_variance
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL EYE BLINK FEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\nLoading metadata...")

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Videos in metadata: {len(metadata)}"
    )

    all_results = []

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for index, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Processing videos"
    ):

        file_path = str(
            row["file"]
        )

        identity = str(
            row["identity"]
        )

        category = str(
            row["category"]
        )

        duration = float(
            row["duration_seconds"]
        )

        # ----------------------------------------------------
        # Original video filename
        # ----------------------------------------------------

        video_stem = os.path.splitext(
            os.path.basename(
                file_path
            )
        )[0]

        # ----------------------------------------------------
        # Frames were stored as:
        #
        # processed/frames/
        #     split/
        #         category/
        #             identity/
        #                 video_name/
        # ----------------------------------------------------

        # Determine split
        split = None

        # Check train / val / test
        for possible_split in [
            "train",
            "val",
            "test"
        ]:

            candidate = os.path.join(
                PROCESSED_FRAMES,
                possible_split,
                category,
                identity,
                video_stem
            )

            if os.path.isdir(
                candidate
            ):

                split = possible_split
                frame_directory = candidate

                break

        # ----------------------------------------------------
        # Frames not found
        # ----------------------------------------------------

        if split is None:

            print(
                f"\nWARNING: Frames not found:"
                f" {video_stem}"
            )

            continue

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = process_video(
            frame_directory,
            duration
        )

        # ----------------------------------------------------
        # Save metadata + features
        # ----------------------------------------------------

        result = {

            "split":
                split,

            "identity":
                identity,

            "file":
                file_path,

            "category":
                category,

            "video":
                video_stem,

            "duration_seconds":
                duration,

            **features
        }

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "eye_blink_features.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EYE BLINK FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nVideos processed: "
        f"{len(results_df)}"
    )

    print(
        f"Output file:\n"
        f"{output_file}"
    )

    if len(results_df) > 0:

        print("\nFeature statistics:")

        print(
            results_df[
                [
                    "blink_count",
                    "blink_rate",
                    "mean_blink_duration",
                    "blink_interval_mean",
                    "blink_interval_variance",
                    "mean_ear",
                    "ear_variance"
                ]
            ].describe()
        )

        print("\nSplit distribution:")

        print(
            results_df[
                "split"
            ].value_counts()
        )

        print(
            "\nCategory distribution:"
        )

        print(
            results_df[
                "category"
            ].value_counts()
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

    face_mesh.close()