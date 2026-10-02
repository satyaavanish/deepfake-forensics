from pathlib import Path
import cv2
import csv
import re
import subprocess
import json
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("dataset/raw/FakeAVCeleb")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUTPUT_DIR / "dataset_metadata.csv"


# Your actual folder names
CATEGORY_MAP = {
    "reala udio-real video": "real_video_real_audio",
    "realaudio-realvideo": "real_video_real_audio",

    "reak video -fake auio": "real_video_fake_audio",
    "realvideo-fakeaudio": "real_video_fake_audio",

    "fakevidoe-realaudio": "fake_video_real_audio",
    "fakevideo-realaudio": "fake_video_real_audio",

    "fakevide-fake audi": "fake_video_fake_audio",
    "fakevideo-fakeaudio": "fake_video_fake_audio",
}


# ============================================================
# MANIPULATION DETECTION
# ============================================================

def detect_manipulation(filename):
    """
    Detect manipulation method from the video filename.
    """

    name = filename.lower()

    # Wav2Lip / WavToLip
    if "wavtolip" in name or "wav2lip" in name:
        return "Wav2Lip"

    # FaceSwap
    if "faceswap" in name or "face_swap" in name:
        return "FaceSwap"

    # FSGAN
    if "fsgan" in name:
        return "FSGAN"

    # SV2TTS
    if "sv2tts" in name or "tts" in name:
        return "SV2TTS"

    return "Unknown"


# ============================================================
# AUDIO DETECTION USING FFPROBE
# ============================================================

def check_audio(video_path):
    """
    Uses ffprobe to determine whether the MP4 contains an audio stream.

    Returns:
        True  -> audio exists
        False -> no audio
        None  -> ffprobe unavailable / error
    """

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        streams = data.get("streams", [])

        return len(streams) > 0

    except Exception:
        return None


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "duration": None,
            "fps": None,
            "width": None,
            "height": None,
            "frames": None
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps and fps > 0:
        duration = frames / fps
    else:
        duration = None

    cap.release()

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": int(frames) if frames else None
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FakeAVCeleb Dataset Exploration")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print()
        print("ERROR: Dataset directory not found:")
        print(DATASET_ROOT)
        print()
        print("Run this script from the project root:")
        print("C:\\Users\\siris\\deepfake_forensics")
        return

    video_files = list(DATASET_ROOT.rglob("*.mp4"))

    print()
    print(f"Dataset path: {DATASET_ROOT}")
    print(f"Total videos: {len(video_files)}")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    category_counts = Counter()
    manipulation_counts = Counter()

    resolution_counts = Counter()

    audio_counts = Counter()

    durations = []
    fps_values = []

    metadata_rows = []

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for index, video_path in enumerate(video_files, start=1):

        print(
            f"\rProcessing {index}/{len(video_files)}: "
            f"{video_path.name[:50]}",
            end=""
        )

        # Folder directly containing video
        relative_parts = video_path.relative_to(DATASET_ROOT).parts

        if len(relative_parts) >= 2:
            category_folder = relative_parts[0]
        else:
            category_folder = "unknown"

        category = CATEGORY_MAP.get(
            category_folder,
            category_folder
        )

        # Identity folder
        identity = (
            relative_parts[1]
            if len(relative_parts) >= 2
            else "unknown"
        )

        # Detect manipulation
        manipulation = detect_manipulation(
            video_path.name
        )

        # Video information
        info = get_video_info(video_path)

        # Audio
        audio = check_audio(video_path)

        # Counters
        category_counts[category] += 1
        manipulation_counts[manipulation] += 1

        if info["duration"] is not None:
            durations.append(info["duration"])

        if info["fps"] is not None and info["fps"] > 0:
            fps_values.append(info["fps"])

        if info["width"] and info["height"]:
            resolution = f"{info['width']}x{info['height']}"
            resolution_counts[resolution] += 1
        else:
            resolution = "Unknown"

        if audio is True:
            audio_counts["With audio"] += 1
        elif audio is False:
            audio_counts["Without audio"] += 1
        else:
            audio_counts["Unknown"] += 1

        metadata_rows.append({
            "file": str(video_path),
            "category": category,
            "identity": identity,
            "manipulation": manipulation,
            "duration_seconds": info["duration"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "frames": info["frames"],
            "audio": audio
        })

    print()
    print()

    # ========================================================
    # REPORT
    # ========================================================

    print("=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)

    for category, count in category_counts.items():
        print(f"{category:30} {count}")

    print()

    print("=" * 70)
    print("MANIPULATION METHODS")
    print("=" * 70)

    for method, count in manipulation_counts.items():
        print(f"{method:30} {count}")

    print()

    print("=" * 70)
    print("VIDEO STATISTICS")
    print("=" * 70)

    if durations:
        print(
            f"Average duration: "
            f"{sum(durations) / len(durations):.2f} seconds"
        )

        print(
            f"Minimum duration: "
            f"{min(durations):.2f} seconds"
        )

        print(
            f"Maximum duration: "
            f"{max(durations):.2f} seconds"
        )

    if fps_values:
        print(
            f"Average FPS: "
            f"{sum(fps_values) / len(fps_values):.2f}"
        )

    print()

    print("Resolutions:")

    for resolution, count in resolution_counts.most_common():
        print(f"    {resolution}: {count}")

    print()

    print("=" * 70)
    print("AUDIO")
    print("=" * 70)

    for audio_type, count in audio_counts.items():
        print(f"{audio_type:30} {count}")

    print()

    print("=" * 70)
    print("IDENTITIES")
    print("=" * 70)

    identities = Counter(
        row["identity"]
        for row in metadata_rows
    )

    print(f"Unique identities: {len(identities)}")

    print()

    print("=" * 70)
    print("SAVING METADATA")
    print("=" * 70)

    fieldnames = [
        "file",
        "category",
        "identity",
        "manipulation",
        "duration_seconds",
        "fps",
        "width",
        "height",
        "frames",
        "audio"
    ]

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Metadata saved to:")
    print(METADATA_FILE)

    print()
    print("=" * 70)
    print("DATASET EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()