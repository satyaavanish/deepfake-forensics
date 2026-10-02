import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTModel
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

FRAME_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "frames"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "explainability",
    "vit_heatmaps"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# TARGET SAMPLE
# ============================================================

TARGET_SPLIT = "test"

TARGET_CATEGORY = "real_video_fake_audio"

TARGET_IDENTITY = "id00049"

TARGET_VIDEO = "00118_fake"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "google/vit-base-patch16-224-in21k"
)

print(
    "\nLoading ViT..."
)

processor = ViTImageProcessor.from_pretrained(
    MODEL_NAME
)

model = ViTModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "✓ ViT loaded"
)


# ============================================================
# FIND TARGET VIDEO
# ============================================================

video_path = os.path.join(
    FRAME_ROOT,
    TARGET_SPLIT,
    TARGET_CATEGORY,
    TARGET_IDENTITY,
    TARGET_VIDEO
)


print(
    "\nTarget sample:"
)

print(
    "Split:",
    TARGET_SPLIT
)

print(
    "Category:",
    TARGET_CATEGORY
)

print(
    "Identity:",
    TARGET_IDENTITY
)

print(
    "Video:",
    TARGET_VIDEO
)

print(
    "Path:",
    video_path
)


if not os.path.exists(video_path):

    raise FileNotFoundError(
        "\nTarget video frame directory was not found:\n"
        + video_path
    )


# ============================================================
# FIND FRAMES
# ============================================================

frames = sorted(
    [
        f
        for f in os.listdir(
            video_path
        )
        if f.lower().endswith(
            ".jpg"
        )
    ]
)


if len(frames) == 0:

    raise RuntimeError(
        "No processed frames found for target video."
    )


print(
    "Frames:",
    len(frames)
)


# ============================================================
# PROCESS FIRST FRAME
# ============================================================

frame_file = frames[0]

frame_path = os.path.join(
    video_path,
    frame_file
)

print(
    "\nProcessing:",
    frame_file
)


image = Image.open(
    frame_path
).convert(
    "RGB"
)


# ============================================================
# PREPARE IMAGE
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt"
)

pixel_values = inputs[
    "pixel_values"
].to(
    DEVICE
)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    outputs = model(
        pixel_values=pixel_values
    )


print(
    "\nHidden state:",
    outputs.last_hidden_state.shape
)

print(
    "Number of attention layers:",
    len(outputs.attentions)
)


# ============================================================
# GET LAST-LAYER ATTENTION
# ============================================================

last_attention = outputs.attentions[-1]

print(
    "Last attention shape:",
    last_attention.shape
)


# ============================================================
# CLS TOKEN ATTENTION
# ============================================================

# CLS token = token 0
#
# We want:
#
# CLS -> patch attention
#
# Exclude CLS itself.

cls_attention = last_attention[
    0,
    :,
    0,
    1:
]


print(
    "CLS attention shape:",
    cls_attention.shape
)


# ============================================================
# AVERAGE HEADS
# ============================================================

attention_map = cls_attention.mean(
    dim=0
)


attention_map = (
    attention_map
    .detach()
    .cpu()
    .numpy()
)


print(
    "Attention vector:",
    attention_map.shape
)


# ============================================================
# CONVERT PATCHES TO 2D
# ============================================================

num_patches = len(
    attention_map
)

grid_size = int(
    np.sqrt(
        num_patches
    )
)


if grid_size * grid_size != num_patches:

    raise RuntimeError(
        f"Cannot form square attention map "
        f"from {num_patches} patches."
    )


attention_map = attention_map.reshape(
    grid_size,
    grid_size
)


# ============================================================
# NORMALIZE
# ============================================================

attention_map -= (
    attention_map.min()
)

max_value = (
    attention_map.max()
)

if max_value > 0:

    attention_map /= max_value


# ============================================================
# RESIZE TO IMAGE SIZE
# ============================================================

image_array = np.array(
    image
)

height, width = (
    image_array.shape[:2]
)

heatmap = cv2.resize(
    attention_map,
    (
        width,
        height
    ),
    interpolation=cv2.INTER_CUBIC
)


# ============================================================
# CREATE COLOR HEATMAP
# ============================================================

heatmap_uint8 = (
    heatmap * 255
).astype(
    np.uint8
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

original_bgr = cv2.cvtColor(
    image_array,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    original_bgr,
    0.55,
    heatmap_color,
    0.45,
    0
)


# ============================================================
# OUTPUT FILE
# ============================================================

safe_name = (
    f"{TARGET_SPLIT}_"
    f"{TARGET_IDENTITY}_"
    f"{TARGET_VIDEO}_"
    f"{frame_file}"
)


output_path = os.path.join(
    OUTPUT_DIR,
    safe_name
)


cv2.imwrite(
    output_path,
    overlay
)


# ============================================================
# SAVE RAW ATTENTION
# ============================================================

raw_attention_path = os.path.join(
    OUTPUT_DIR,
    safe_name.replace(
        ".jpg",
        "_attention.npy"
    )
)

np.save(
    raw_attention_path,
    heatmap
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis(
    "off"
)

plt.title(
    "ViT Attention Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# FINISHED
# ============================================================

print(
    "\n✓ VIT HEATMAP GENERATED"
)

print(
    "Heatmap:",
    output_path
)

print(
    "Raw attention:",
    raw_attention_path
)