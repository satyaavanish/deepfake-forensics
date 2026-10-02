import os
import cv2
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\siris\deepfake_forensics"

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

AUDIO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio"
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
NUM_FRAMES = 16

SPEECH_THRESHOLD_RATIO = 0.20


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ============================================================
# MOUTH LANDMARKS
# ============================================================

UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# MOUTH FEATURES
# ============================================================

def get_mouth_measurements(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    height, width = image.shape[:2]

    upper = np.array([
        landmarks[UPPER_LIP].x * width,
        landmarks[UPPER_LIP].y * height
    ])

    lower = np.array([
        landmarks[LOWER_LIP].x * width,
        landmarks[LOWER_LIP].y * height
    ])

    left_corner = np.array([
        landmarks[LEFT_MOUTH_CORNER].x * width,
        landmarks[LEFT_MOUTH_CORNER].y * height
    ])

    right_corner = np.array([
        landmarks[RIGHT_MOUTH_CORNER].x * width,
        landmarks[RIGHT_MOUTH_CORNER].y * height
    ])

    mouth_opening = distance(
        upper,
        lower
    )

    lip_distance = distance(
        left_corner,
        right_corner
    )

    return mouth_opening, lip_distance


# ============================================================
# FIND FRAME DIRECTORY
# ============================================================

def find_frame_directory(
    split,
    category,
    identity,
    video_stem
):

    path = os.path.join(
        FRAMES_ROOT,
        split,
        category,
        identity,
        video_stem
    )

    if os.path.isdir(path):
        return path

    return None


# ============================================================
# EXTRACT MOUTH FEATURES
# ============================================================

def extract_mouth_features(
    frame_directory
):

    frame_files = [
        f
        for f in os.listdir(frame_directory)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    frame_files.sort()

    frame_files = frame_files[:NUM_FRAMES]

    mouth_openings = []
    lip_distances = []

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

        measurements = get_mouth_measurements(
            image
        )

        if measurements is None:
            continue

        mouth_opening, lip_distance = measurements

        mouth_openings.append(
            mouth_opening
        )

        lip_distances.append(
            lip_distance
        )

    return (
        mouth_openings,
        lip_distances
    )


# ============================================================
# SPEECH ACTIVITY
# ============================================================

def detect_speech_activity(
    audio_path
):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:

        return (
            np.zeros(NUM_FRAMES, dtype=bool),
            0.0
        )

    duration = len(audio) / sr

    segment_length = (
        len(audio) /
        NUM_FRAMES
    )

    energies = []

    for i in range(NUM_FRAMES):

        start = int(
            i * segment_length
        )

        end = int(
            (i + 1) * segment_length
        )

        segment = audio[start:end]

        if len(segment) == 0:

            energy = 0.0

        else:

            energy = np.sqrt(
                np.mean(
                    segment ** 2
                )
            )

        energies.append(
            energy
        )

    energies = np.array(
        energies
    )

    max_energy = np.max(
        energies
    )

    if max_energy > 0:

        threshold = (
            max_energy *
            SPEECH_THRESHOLD_RATIO
        )

    else:

        threshold = 0.0

    speech_active = (
        energies >= threshold
    )

    return (
        speech_active,
        duration
    )


# ============================================================
# LIP ACTIVITY
# ============================================================

def calculate_lip_activity(
    mouth_openings
):

    if len(mouth_openings) < 2:

        return np.zeros(
            len(mouth_openings),
            dtype=bool
        )

    mouth_openings = np.array(
        mouth_openings
    )

    movement = np.abs(
        np.diff(
            mouth_openings
        )
    )

    median_movement = np.median(
        movement
    )

    threshold = max(
        median_movement,
        0.5
    )

    lip_active = (
        movement >= threshold
    )

    lip_active = np.insert(
        lip_active,
        0,
        False
    )

    return lip_active


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    frame_directory,
    audio_path
):

    mouth_openings, lip_distances = (
        extract_mouth_features(
            frame_directory
        )
    )

    if len(mouth_openings) < 2:

        return {
            "frames_detected": len(
                mouth_openings
            ),
            "mean_mouth_opening": 0.0,
            "mouth_opening_variance": 0.0,
            "mean_lip_distance": 0.0,
            "lip_distance_variance": 0.0,
            "mouth_movement_velocity": 0.0,
            "mouth_movement_frequency": 0.0,
            "speech_activity_ratio": 0.0,
            "lip_activity_ratio": 0.0,
            "lip_speech_consistency": 0.0
        }

    # --------------------------------------------------------
    # Mouth statistics
    # --------------------------------------------------------

    mouth = np.array(
        mouth_openings
    )

    lips = np.array(
        lip_distances
    )

    mean_mouth_opening = float(
        np.mean(mouth)
    )

    mouth_opening_variance = float(
        np.var(mouth)
    )

    mean_lip_distance = float(
        np.mean(lips)
    )

    lip_distance_variance = float(
        np.var(lips)
    )

    # --------------------------------------------------------
    # Mouth movement
    # --------------------------------------------------------

    movement = np.abs(
        np.diff(mouth)
    )

    mouth_movement_velocity = float(
        np.mean(movement)
    )

    if len(movement) >= 2:

        signs = np.sign(
            np.diff(mouth)
        )

        direction_changes = np.sum(
            signs[1:] != signs[:-1]
        )

        mouth_movement_frequency = float(
            direction_changes /
            len(mouth)
        )

    else:

        mouth_movement_frequency = 0.0

    # --------------------------------------------------------
    # Speech
    # --------------------------------------------------------

    speech_active, duration = (
        detect_speech_activity(
            audio_path
        )
    )

    # --------------------------------------------------------
    # Lip activity
    # --------------------------------------------------------

    lip_active = calculate_lip_activity(
        mouth_openings
    )

    # Match lengths
    n = min(
        len(lip_active),
        len(speech_active)
    )

    lip_active = lip_active[:n]
    speech_active = speech_active[:n]

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    speech_activity_ratio = float(
        np.mean(
            speech_active
        )
    )

    lip_activity_ratio = float(
        np.mean(
            lip_active
        )
    )

    # --------------------------------------------------------
    # Lip-speech consistency
    # --------------------------------------------------------

    speech_frames = np.sum(
        speech_active
    )

    if speech_frames > 0:

        matching_frames = np.sum(
            speech_active &
            lip_active
        )

        lip_speech_consistency = float(
            matching_frames /
            speech_frames
        )

    else:

        lip_speech_consistency = 0.0

    return {
        "frames_detected": len(
            mouth_openings
        ),
        "mean_mouth_opening":
            mean_mouth_opening,
        "mouth_opening_variance":
            mouth_opening_variance,
        "mean_lip_distance":
            mean_lip_distance,
        "lip_distance_variance":
            lip_distance_variance,
        "mouth_movement_velocity":
            mouth_movement_velocity,
        "mouth_movement_frequency":
            mouth_movement_frequency,
        "speech_activity_ratio":
            speech_activity_ratio,
        "lip_activity_ratio":
            lip_activity_ratio,
        "lip_speech_consistency":
            lip_speech_consistency
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL LIP-SPEECH FEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"\nVideos in metadata: "
        f"{len(metadata)}"
    )

    results = []

    # --------------------------------------------------------
    # Process all videos
    # --------------------------------------------------------

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Processing videos"
    ):

        file_path = str(
            row["file"]
        )

        category = str(
            row["category"]
        )

        identity = str(
            row["identity"]
        )

        video_stem = os.path.splitext(
            os.path.basename(
                file_path
            )
        )[0]

        # ----------------------------------------------------
        # Find split
        # ----------------------------------------------------

        split = None
        frame_directory = None

        for possible_split in [
            "train",
            "val",
            "test"
        ]:

            candidate = find_frame_directory(
                possible_split,
                category,
                identity,
                video_stem
            )

            if candidate is not None:

                split = possible_split
                frame_directory = candidate

                break

        if frame_directory is None:

            continue

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        audio_path = os.path.join(
            AUDIO_ROOT,
            split,
            video_stem + ".wav"
        )

        if not os.path.exists(
            audio_path
        ):

            continue

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        features = process_video(
            frame_directory,
            audio_path
        )

        result = {
            "split": split,
            "identity": identity,
            "file": file_path,
            "category": category,
            "video": video_stem,
            **features
        }

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "lip_speech_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LIP-SPEECH FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nVideos processed: "
        f"{len(df)}"
    )

    print(
        f"\nOutput file:\n"
        f"{output_file}"
    )

    if len(df) > 0:

        print(
            "\nFeature statistics:"
        )

        print(
            df[
                [
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
            ].describe()
        )

        print(
            "\nSplit distribution:"
        )

        print(
            df["split"].value_counts()
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    finally:

        face_mesh.close()