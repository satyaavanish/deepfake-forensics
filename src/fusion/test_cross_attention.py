import torch

from cross_attention import (
    VisualAudioCrossAttention
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2

VISUAL_DIM = 768
AUDIO_DIM = 768

BEHAVIOR_DIM = 10


# ============================================================
# Create model
# ============================================================

model = VisualAudioCrossAttention(
    visual_dim=VISUAL_DIM,
    audio_dim=AUDIO_DIM,
    common_dim=512,
    num_heads=8
)


# ============================================================
# Dummy inputs
# ============================================================

visual_embedding = torch.randn(
    BATCH_SIZE,
    VISUAL_DIM
)

audio_embedding = torch.randn(
    BATCH_SIZE,
    AUDIO_DIM
)

behavior_features = torch.randn(
    BATCH_SIZE,
    BEHAVIOR_DIM
)


# ============================================================
# Forward pass
# ============================================================

fused, attention = model(
    visual_embedding,
    audio_embedding,
    behavior_features
)


# ============================================================
# Results
# ============================================================

print("=" * 70)
print("CROSS-ATTENTION FUSION TEST")
print("=" * 70)

print(
    "\nVisual embedding:",
    visual_embedding.shape
)

print(
    "Audio embedding:",
    audio_embedding.shape
)

print(
    "Behavior features:",
    behavior_features.shape
)

print(
    "\nFused embedding:",
    fused.shape
)

print(
    "Attention weights:",
    attention.shape
)

print("\nExpected:")

print(
    "Visual:   [2, 768]"
)

print(
    "Audio:    [2, 768]"
)

print(
    "Behavior: [2, 10]"
)

print(
    "Fusion:   [2, 512]"
)

print(
    "\n" + "=" * 70
)
print("TEST COMPLETE")
print("=" * 70)