import os
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

# EAR threshold
EAR_THRESHOLD = 0.21

# Number of consecutive frames below threshold
MIN_BLINK_FRAMES = 2


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# MediaPipe Face Mesh eye landmark indices
LEFT_EYE = [
    33,   # outer corner
    160,  # upper
    158,  # upper
    133,  # inner corner
    153,  # lower
    144   # lower
]

RIGHT_EYE = [
    362,  # outer corner
    385,  # upper
    387,  # upper
    263,  # inner corner
    373,  # lower
    380   # lower
]


# ============================================================
# EAR FUNCTION
# ============================================================

def calculate_ear(landmarks, eye_indices):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        points.append(
            np.array([
                landmark.x,
                landmark.y
            ])
        )

    p1, p2, p3, p4, p5, p6 = points

    horizontal_distance = np.linalg.norm(
        p1 - p4
    )

    vertical_distance_1 = np.linalg.norm(
        p2 - p6
    )

    vertical_distance_2 = np.linalg.norm(
        p3 - p5
    )

    if horizontal_distance == 0:

        return 0.0

    ear = (
        vertical_distance_1
        +
        vertical_distance_2
    ) / (
        2.0 * horizontal_distance
    )

    return float(ear)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EYE BLINK ANALYSIS TEST")
    print("=" * 70)

    print()
    print("EAR threshold:", EAR_THRESHOLD)
    print(
        "Minimum blink frames:",
        MIN_BLINK_FRAMES
    )

    # --------------------------------------------------------
    # FIND ONE VIDEO FRAME DIRECTORY
    # --------------------------------------------------------

    selected_video_dir = None

    for split in ["test", "val", "train"]:

        split_dir = os.path.join(
            FRAMES_ROOT,
            split
        )

        if not os.path.exists(split_dir):
            continue

        found = False

        for category in os.listdir(split_dir):

            category_dir = os.path.join(
                split_dir,
                category
            )

            if not os.path.isdir(category_dir):
                continue

            for identity in os.listdir(
                category_dir
            ):

                identity_dir = os.path.join(
                    category_dir,
                    identity
                )

                if not os.path.isdir(
                    identity_dir
                ):
                    continue

                for video in os.listdir(
                    identity_dir
                ):

                    video_dir = os.path.join(
                        identity_dir,
                        video
                    )

                    if not os.path.isdir(
                        video_dir
                    ):
                        continue

                    frame_files = [
                        f for f in os.listdir(
                            video_dir
                        )
                        if f.lower().endswith(
                            (".jpg", ".jpeg", ".png")
                        )
                    ]

                    if len(frame_files) >= 16:

                        selected_video_dir = (
                            video_dir
                        )

                        found = True
                        break

                if found:
                    break

            if found:
                break

        if found:
            break

    if selected_video_dir is None:

        raise RuntimeError(
            "Could not find a video with "
            "processed face frames."
        )

    print()
    print("Selected video:")
    print(selected_video_dir)

    # --------------------------------------------------------
    # LOAD FRAMES
    # --------------------------------------------------------

    frame_files = sorted([
        f for f in os.listdir(
            selected_video_dir
        )
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    # Use exactly 16 frames
    frame_files = frame_files[:16]

    print()
    print(
        "Frames found:",
        len(frame_files)
    )

    # --------------------------------------------------------
    # MEDIAPIPE FACE MESH
    # --------------------------------------------------------

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    left_ears = []
    right_ears = []
    average_ears = []

    successful_frames = 0

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for frame_number, frame_file in enumerate(
        frame_files,
        start=1
    ):

        frame_path = os.path.join(
            selected_video_dir,
            frame_file
        )

        image = cv2.imread(
            frame_path
        )

        if image is None:

            print(
                f"Frame {frame_number}: "
                "could not read"
            )

            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result = face_mesh.process(
            rgb
        )

        if not result.multi_face_landmarks:

            print(
                f"Frame {frame_number}: "
                "no landmarks"
            )

            continue

        landmarks = (
            result.multi_face_landmarks[0]
            .landmark
        )

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        left_ears.append(
            left_ear
        )

        right_ears.append(
            right_ear
        )

        average_ears.append(
            average_ear
        )

        successful_frames += 1

        print(
            f"Frame {frame_number:02d} | "
            f"Left EAR: {left_ear:.4f} | "
            f"Right EAR: {right_ear:.4f} | "
            f"Average EAR: {average_ear:.4f}"
        )

    face_mesh.close()

    # --------------------------------------------------------
    # BLINK DETECTION
    # --------------------------------------------------------

    print()

    print(
        "Successfully processed:",
        successful_frames,
        "/",
        len(frame_files)
    )

    if len(average_ears) == 0:

        raise RuntimeError(
            "No facial landmarks detected."
        )

    blink_count = 0

    blink_frames = 0

    currently_blinking = False

    blink_lengths = []

    for ear in average_ears:

        if ear < EAR_THRESHOLD:

            blink_frames += 1

        else:

            if (
                blink_frames
                >= MIN_BLINK_FRAMES
            ):

                blink_count += 1

                blink_lengths.append(
                    blink_frames
                )

            blink_frames = 0

    # Handle blink at final frame
    if (
        blink_frames
        >= MIN_BLINK_FRAMES
    ):

        blink_count += 1

        blink_lengths.append(
            blink_frames
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    mean_ear = np.mean(
        average_ears
    )

    ear_variance = np.var(
        average_ears
    )

    if len(blink_lengths) > 0:

        mean_blink_duration = np.mean(
            blink_lengths
        )

    else:

        mean_blink_duration = 0.0

    # Since we have 16 sampled frames,
    # report blink count for these frames.
    blink_rate = (
        blink_count
        / len(average_ears)
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BLINK ANALYSIS RESULTS")
    print("=" * 70)

    print()
    print(
        f"Blink count:           {blink_count}"
    )

    print(
        f"Blink rate:            {blink_rate:.4f}"
    )

    print(
        f"Mean blink duration:   "
        f"{mean_blink_duration:.4f} frames"
    )

    print(
        f"Mean EAR:              {mean_ear:.4f}"
    )

    print(
        f"EAR variance:          "
        f"{ear_variance:.6f}"
    )

    print()
    print(
        "Feature vector:"
    )

    print(
        "["
        f"{blink_rate:.4f}, "
        f"{mean_blink_duration:.4f}, "
        f"{ear_variance:.6f}"
        "]"
    )

    print()
    print("=" * 70)
    print("EYE BLINK TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()