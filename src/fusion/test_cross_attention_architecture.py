import torch

from cross_attention_architecture import (
    CrossAttentionClassifier
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# CREATE MODEL
# ============================================================

model = CrossAttentionClassifier(
    visual_dim=768,
    audio_dim=768,
    common_dim=512,
    num_heads=8,
    dropout=0.2
)

model = model.to(
    device
)

model.eval()


# ============================================================
# CREATE TEST INPUTS
# ============================================================

batch_size = 4

visual = torch.randn(
    batch_size,
    768
).to(device)

audio = torch.randn(
    batch_size,
    768
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(
        visual,
        audio
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "\nVisual:",
    visual.shape
)

print(
    "Audio:",
    audio.shape
)

print(
    "Fused:",
    output["fused"].shape
)

print(
    "Logit:",
    output["logit"].shape
)

print(
    "Probability:",
    output["probability"].shape
)

print(
    "Attention:",
    output["attention"].shape
)


# ============================================================
# CHECK PROBABILITY
# ============================================================

print(
    "\nPredicted probabilities:"
)

print(
    output["probability"].squeeze(1)
)


# ============================================================
# CHECK REAL / FAKE
# ============================================================

predictions = (
    output["probability"] >= 0.5
).long()


print(
    "\nPredictions:"
)

print(
    predictions.squeeze(1)
)


# ============================================================
# TEST
# ============================================================

assert output["fused"].shape == (
    batch_size,
    128
)

assert output["logit"].shape == (
    batch_size,
    1
)

assert output["probability"].shape == (
    batch_size,
    1
)

assert torch.all(
    output["probability"] >= 0
)

assert torch.all(
    output["probability"] <= 1
)


print(
    "\n✓ CROSS-ATTENTION ARCHITECTURE TEST PASSED"
)