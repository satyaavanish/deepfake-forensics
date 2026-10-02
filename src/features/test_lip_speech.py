import os
import cv2
import numpy as np
import librosa
import mediapipe as mp


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\siris\deepfake_forensics"

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames",
    "test"
)

AUDIO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "audio",
    "test"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# Number of visual frames
NUM_FRAMES = 16

# Audio energy threshold
# We use a relative threshold instead of a fixed amplitude.
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
# FIND TEST VIDEO
# ============================================================

def find_test_video():

    for root, dirs, files in os.walk(
        FRAMES_ROOT
    ):

        for directory in dirs:

            full_path = os.path.join(
                root,
                directory
            )

            images = [
                f for f in os.listdir(full_path)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            if len(images) >= NUM_FRAMES:

                return full_path

    return None


# ============================================================
# MOUTH MEASUREMENTS
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
# EXTRACT MOUTH MOVEMENT
# ============================================================

def extract_mouth_motion(
    video_folder
):

    frame_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    frame_files.sort()

    frame_files = frame_files[:NUM_FRAMES]

    mouth_openings = []
    lip_distances = []

    print("\n" + "=" * 70)
    print("MOUTH MOVEMENT")
    print("=" * 70)

    for i, frame_file in enumerate(
        frame_files
    ):

        frame_path = os.path.join(
            video_folder,
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

            print(
                f"Frame {i+1:02d}: "
                "Face not detected"
            )

            continue

        mouth_opening, lip_distance = measurements

        mouth_openings.append(
            mouth_opening
        )

        lip_distances.append(
            lip_distance
        )

        print(
            f"Frame {i+1:02d} | "
            f"Mouth opening: "
            f"{mouth_opening:.2f}"
        )

    return mouth_openings, lip_distances


# ============================================================
# AUDIO SPEECH ACTIVITY
# ============================================================

def detect_speech_activity(
    audio_path
):

    print("\n" + "=" * 70)
    print("AUDIO SPEECH ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = len(audio) / sr

    print(
        f"Audio duration: "
        f"{duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Divide audio into 16 segments
    #
    # This aligns the audio timeline approximately
    # with the 16 visual frames.
    # --------------------------------------------------------

    segment_length = len(audio) / NUM_FRAMES

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

    # --------------------------------------------------------
    # Relative threshold
    # --------------------------------------------------------

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

    print(
        f"Speech threshold: "
        f"{threshold:.6f}"
    )

    print()

    for i in range(NUM_FRAMES):

        state = (
            "SPEECH"
            if speech_active[i]
            else "SILENCE"
        )

        print(
            f"Segment {i+1:02d} | "
            f"Energy: {energies[i]:.6f} | "
            f"{state}"
        )

    return (
        speech_active,
        energies,
        duration
    )


# ============================================================
# LIP MOVEMENT ACTIVITY
# ============================================================

def calculate_lip_activity(
    mouth_openings
):

    if len(mouth_openings) < 2:

        return np.array([])

    mouth_openings = np.array(
        mouth_openings
    )

    # Frame-to-frame mouth movement
    movement = np.abs(
        np.diff(
            mouth_openings
        )
    )

    # Use median movement as adaptive threshold
    median_movement = np.median(
        movement
    )

    # Avoid zero threshold
    threshold = max(
        median_movement,
        0.5
    )

    lip_active = (
        movement >= threshold
    )

    # First frame has no previous frame
    lip_active = np.insert(
        lip_active,
        0,
        False
    )

    return lip_active


# ============================================================
# LIP-SPEECH CONSISTENCY
# ============================================================

def calculate_consistency(
    mouth_openings,
    speech_active
):

    lip_active = calculate_lip_activity(
        mouth_openings
    )

    if len(lip_active) == 0:

        return 0.0, lip_active

    # Match lengths
    n = min(
        len(lip_active),
        len(speech_active)
    )

    lip_active = lip_active[:n]
    speech_active = speech_active[:n]

    # --------------------------------------------------------
    # Basic temporal consistency
    #
    # During speech:
    #     mouth should generally move.
    # --------------------------------------------------------

    speech_frames = np.sum(
        speech_active
    )

    if speech_frames == 0:

        consistency = 0.0

    else:

        matching_frames = np.sum(
            speech_active &
            lip_active
        )

        consistency = (
            matching_frames /
            speech_frames
        )

    return (
        float(consistency),
        lip_active
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LIP-SPEECH CONSISTENCY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Find video
    # --------------------------------------------------------

    video_folder = find_test_video()

    if video_folder is None:

        print(
            "ERROR: Could not find "
            "a processed video."
        )

        return

    print(
        "\nSelected video:"
    )

    print(video_folder)

    video_name = os.path.basename(
        video_folder
    )

    print(
        f"\nVideo name: {video_name}"
    )

    # --------------------------------------------------------
    # Extract mouth movement
    # --------------------------------------------------------

    mouth_openings, lip_distances = (
        extract_mouth_motion(
            video_folder
        )
    )

    if len(mouth_openings) < 2:

        print(
            "\nERROR: Not enough "
            "valid mouth frames."
        )

        face_mesh.close()

        return

    # --------------------------------------------------------
    # Find corresponding audio
    #
    # Audio files use original video stem.
    # --------------------------------------------------------

    audio_path = os.path.join(
        AUDIO_ROOT,
        video_name + ".wav"
    )

    if not os.path.exists(
        audio_path
    ):

        print(
            "\nERROR: Audio file not found:"
        )

        print(audio_path)

        face_mesh.close()

        return

    print(
        "\nAudio file:"
    )

    print(audio_path)

    # --------------------------------------------------------
    # Speech detection
    # --------------------------------------------------------

    speech_active, energies, duration = (
        detect_speech_activity(
            audio_path
        )
    )

    # --------------------------------------------------------
    # Lip-speech consistency
    # --------------------------------------------------------

    consistency, lip_active = (
        calculate_consistency(
            mouth_openings,
            speech_active
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mouth_openings_np = np.array(
        mouth_openings
    )

    movement = np.abs(
        np.diff(
            mouth_openings_np
        )
    )

    mean_mouth_opening = np.mean(
        mouth_openings_np
    )

    mouth_variance = np.var(
        mouth_openings_np
    )

    mean_velocity = np.mean(
        movement
    )

    speech_ratio = np.mean(
        speech_active
    )

    lip_activity_ratio = np.mean(
        lip_active
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LIP-SPEECH CONSISTENCY RESULTS")
    print("=" * 70)

    print(
        f"\nMean mouth opening: "
        f"{mean_mouth_opening:.4f}"
    )

    print(
        f"Mouth opening variance: "
        f"{mouth_variance:.6f}"
    )

    print(
        f"Mouth movement velocity: "
        f"{mean_velocity:.4f}"
    )

    print(
        f"Speech activity ratio: "
        f"{speech_ratio:.4f}"
    )

    print(
        f"Lip activity ratio: "
        f"{lip_activity_ratio:.4f}"
    )

    print(
        f"Lip-speech consistency: "
        f"{consistency:.4f}"
    )

    # --------------------------------------------------------
    # Frame-by-frame alignment
    # --------------------------------------------------------

    print(
        "\nFrame/segment alignment:"
    )

    n = min(
        len(lip_active),
        len(speech_active)
    )

    for i in range(n):

        speech = (
            "SPEECH"
            if speech_active[i]
            else "SILENCE"
        )

        lips = (
            "MOVING"
            if lip_active[i]
            else "STILL"
        )

        print(
            f"{i+1:02d} | "
            f"{speech:<7} | "
            f"{lips}"
        )

    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    print(
        "\nFeature vector:"
    )

    print(
        "["
        f"{mean_mouth_opening:.4f}, "
        f"{mouth_variance:.6f}, "
        f"{mean_velocity:.4f}, "
        f"{speech_ratio:.4f}, "
        f"{lip_activity_ratio:.4f}, "
        f"{consistency:.4f}"
        "]"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LIP-SPEECH TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    face_mesh.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()