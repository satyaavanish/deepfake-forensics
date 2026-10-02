import torch

from fusion_model import CrossModalFusionModel


print("=" * 70)
print("REAL CROSS-MODAL FUSION MODEL TEST")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)


# ============================================================
# MODEL
# ============================================================

model = CrossModalFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    behavior_projection_dim=128,
    num_heads=8,
    dropout=0.2
)

model = model.to(device)


# ============================================================
# TEST INPUTS
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

behavior = torch.randn(
    batch_size,
    22
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(
        visual,
        audio,
        behavior
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "\nVisual input:",
    visual.shape
)

print(
    "Audio input:",
    audio.shape
)

print(
    "Behavior input:",
    behavior.shape
)

print(
    "\nFused embedding:",
    output["fused_embedding"].shape
)

print(
    "Logits:",
    output["logits"].shape
)

print(
    "Visual attention:",
    output["visual_attention"].shape
)

print(
    "Audio attention:",
    output["audio_attention"].shape
)

print(
    "Behavior attention:",
    output["behavior_attention"].shape
)


# ============================================================
# EXPECTED
# ============================================================

print("\nExpected:")

print(
    "Visual:       [4, 768]"
)

print(
    "Audio:        [4, 768]"
)

print(
    "Behavior:     [4, 22]"
)

print(
    "Fusion:       [4, 512]"
)

print(
    "Logits:       [4, 2]"
)


# ============================================================
# SUCCESS
# ============================================================

if (
    output["fused_embedding"].shape
    == (4, 512)
    and
    output["logits"].shape
    == (4, 2)
):

    print(
        "\n✓ REAL FUSION MODEL TEST PASSED"
    )

else:

    print(
        "\n✗ TEST FAILED"
    )


print(
    "\n" + "=" * 70
)