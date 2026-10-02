import os
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = r"C:\Users\siris\deepfake_forensics"

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames",
    "test"
)


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
# MOUTH LANDMARKS
# ============================================================

# Upper/lower lip center
UPPER_LIP = 13
LOWER_LIP = 14

# Left/right mouth corners
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# EXTRACT MOUTH FEATURES FROM ONE FRAME
# ============================================================

def extract_mouth_features(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = face_mesh.process(
        image_rgb
    )

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Convert landmarks to pixel coordinates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mouth opening
    # --------------------------------------------------------

    mouth_opening = distance(
        upper,
        lower
    )

    # --------------------------------------------------------
    # Lip distance / mouth width
    # --------------------------------------------------------

    lip_distance = distance(
        left_corner,
        right_corner
    )

    return {
        "mouth_opening": mouth_opening,
        "lip_distance": lip_distance
    }


# ============================================================
# FIND A TEST VIDEO
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

            if len(images) >= 16:

                return full_path

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LIP MOVEMENT ANALYSIS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Find video
    # --------------------------------------------------------

    video_folder = find_test_video()

    if video_folder is None:

        print(
            "ERROR: No processed video found."
        )

        return

    print(
        "\nSelected video:"
    )

    print(video_folder)

    # --------------------------------------------------------
    # Find frames
    # --------------------------------------------------------

    frame_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    frame_files.sort()

    print(
        f"\nFrames found: {len(frame_files)}"
    )

    # Use first 16 frames
    frame_files = frame_files[:16]

    mouth_openings = []
    lip_distances = []

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

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

            print(
                f"Frame {i+1:02d}: "
                "Could not read"
            )

            continue

        features = extract_mouth_features(
            image
        )

        if features is None:

            print(
                f"Frame {i+1:02d}: "
                "Face not detected"
            )

            continue

        mouth_opening = features[
            "mouth_opening"
        ]

        lip_distance = features[
            "lip_distance"
        ]

        mouth_openings.append(
            mouth_opening
        )

        lip_distances.append(
            lip_distance
        )

        print(
            f"Frame {i+1:02d} | "
            f"Mouth Opening: "
            f"{mouth_opening:.2f} | "
            f"Lip Distance: "
            f"{lip_distance:.2f}"
        )

    # ========================================================
    # MOVEMENT CALCULATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("LIP MOVEMENT RESULTS")
    print("=" * 70)

    if len(mouth_openings) < 2:

        print(
            "\nNot enough valid frames."
        )

        face_mesh.close()

        return

    # --------------------------------------------------------
    # Mouth opening statistics
    # --------------------------------------------------------

    mean_opening = np.mean(
        mouth_openings
    )

    opening_variance = np.var(
        mouth_openings
    )

    # --------------------------------------------------------
    # Lip distance statistics
    # --------------------------------------------------------

    mean_lip_distance = np.mean(
        lip_distances
    )

    lip_distance_variance = np.var(
        lip_distances
    )

    # --------------------------------------------------------
    # Mouth movement
    # --------------------------------------------------------

    mouth_movement = np.diff(
        mouth_openings
    )

    movement_velocity = np.mean(
        np.abs(
            mouth_movement
        )
    )

    # --------------------------------------------------------
    # Movement frequency
    #
    # Count direction changes in mouth
    # opening movement.
    # --------------------------------------------------------

    signs = np.sign(
        mouth_movement
    )

    direction_changes = np.sum(
        signs[1:] != signs[:-1]
    )

    movement_frequency = (
        direction_changes /
        len(mouth_openings)
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nMean mouth opening: "
        f"{mean_opening:.4f}"
    )

    print(
        f"Mouth opening variance: "
        f"{opening_variance:.6f}"
    )

    print(
        f"Mean lip distance: "
        f"{mean_lip_distance:.4f}"
    )

    print(
        f"Lip distance variance: "
        f"{lip_distance_variance:.6f}"
    )

    print(
        f"Mouth movement velocity: "
        f"{movement_velocity:.4f}"
    )

    print(
        f"Mouth movement frequency: "
        f"{movement_frequency:.4f}"
    )

    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    print("\nFeature vector:")

    print(
        "["
        f"{mean_opening:.4f}, "
        f"{mean_lip_distance:.4f}, "
        f"{movement_velocity:.4f}, "
        f"{movement_frequency:.4f}"
        "]"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LIP MOVEMENT TEST COMPLETE"
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