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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FIND ONE FACE FRAME
# ============================================================

image_path = None

for root, dirs, files in os.walk(FRAMES_ROOT):

    for file in files:

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                root,
                file
            )

            break

    if image_path:
        break


if image_path is None:

    raise FileNotFoundError(
        "No face frame found in dataset/processed/frames"
    )


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("VIT SINGLE IMAGE TEST")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    image_path
).convert("RGB")

print()
print("Original image size:", image.size)


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
# PROCESS IMAGE
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}


print()
print("Pixel values shape:")
print(inputs["pixel_values"].shape)


# ============================================================
# FORWARD PASS
# ============================================================

print()
print("Running ViT...")

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ============================================================
# EXTRACT EMBEDDING
# ============================================================

last_hidden_state = outputs.last_hidden_state

cls_embedding = last_hidden_state[:, 0, :]

pooler_output = outputs.pooler_output


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("VIT RESULTS")
print("=" * 70)

print()
print("Last hidden state:")
print(last_hidden_state.shape)

print()
print("CLS embedding:")
print(cls_embedding.shape)

print()
print("Pooler output:")
print(
    pooler_output.shape
    if pooler_output is not None
    else None
)

print()
print("=" * 70)
print("VIT TEST SUCCESSFUL")
print("=" * 70)