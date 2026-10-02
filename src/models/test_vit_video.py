import os
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

MODEL_NAME = "google/vit-base-patch16-224-in21k"

NUM_FRAMES = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FIND ONE VIDEO FOLDER
# ============================================================

video_folder = None

for root, dirs, files in os.walk(FRAMES_ROOT):

    image_files = [
        f for f in files
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if len(image_files) >= NUM_FRAMES:

        video_folder = root
        break


if video_folder is None:

    raise FileNotFoundError(
        "Could not find a video folder containing "
        f"at least {NUM_FRAMES} frames."
    )


# ============================================================
# GET FRAMES
# ============================================================

frame_files = sorted([
    f for f in os.listdir(video_folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

frame_files = frame_files[:NUM_FRAMES]

frame_paths = [
    os.path.join(video_folder, f)
    for f in frame_files
]


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("VIT VIDEO TEST")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Video folder:")
print(video_folder)

print()
print("Number of frames:", len(frame_paths))

print()
print("Frames:")

for path in frame_paths:

    print(
        " ",
        os.path.basename(path)
    )


# ============================================================
# LOAD IMAGES
# ============================================================

images = []

for path in frame_paths:

    image = Image.open(
        path
    ).convert("RGB")

    images.append(image)


print()
print("Loaded images:", len(images))

print(
    "Image size:",
    images[0].size
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print()
print("Loading ViT processor...")

processor = ViTImageProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading pretrained ViT...")

model = ViTModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PROCESS ALL 16 FRAMES
# ============================================================

inputs = processor(
    images=images,
    return_tensors="pt"
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}


print()
print("Pixel values:")
print(
    inputs["pixel_values"].shape
)


# ============================================================
# VIT FORWARD PASS
# ============================================================

print()
print("Running 16 frames through ViT...")

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ============================================================
# EXTRACT CLS EMBEDDINGS
# ============================================================

last_hidden_state = outputs.last_hidden_state

# CLS token from every frame
cls_embeddings = last_hidden_state[:, 0, :]


# ============================================================
# TEMPORAL MEAN POOLING
# ============================================================

video_embedding = cls_embeddings.mean(
    dim=0,
    keepdim=True
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("VIT VIDEO RESULTS")
print("=" * 70)

print()
print("Last hidden state:")
print(
    last_hidden_state.shape
)

print()
print("CLS embeddings:")
print(
    cls_embeddings.shape
)

print()
print("Video embedding after mean pooling:")
print(
    video_embedding.shape
)

print()
print(
    "Expected video embedding:",
    "[1, 768]"
)

print()
print("=" * 70)
print("VIT VIDEO TEST SUCCESSFUL")
print("=" * 70)