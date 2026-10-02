import torch
import torch.nn as nn


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# BEHAVIORAL FUSION MODEL
# ============================================================

class BehavioralFusionModel(nn.Module):

    def __init__(
        self,
        visual_dim=768,
        audio_dim=768,
        behavior_dim=22,
        common_dim=512,
        behavior_dim_out=128,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()

        # ----------------------------------------------------
        # ViT: 768 -> 512
        # ----------------------------------------------------

        self.visual_projection = nn.Linear(
            visual_dim,
            common_dim
        )

        # ----------------------------------------------------
        # AST: 768 -> 512
        # ----------------------------------------------------

        self.audio_projection = nn.Linear(
            audio_dim,
            common_dim
        )

        # ----------------------------------------------------
        # Behavior:
        #
        # 22 -> 64 -> 128
        # ----------------------------------------------------

        self.behavior_projection = nn.Sequential(

            nn.Linear(
                behavior_dim,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                behavior_dim_out
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # Behavior 128 -> 512
        #
        # This allows behavior to participate in the same
        # fusion space as visual/audio.
        # ----------------------------------------------------

        self.behavior_to_common = nn.Linear(
            behavior_dim_out,
            common_dim
        )

        # ----------------------------------------------------
        # Cross Attention
        #
        # Visual = Query
        # Audio  = Key + Value
        # ----------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Fusion
        #
        # Visual 512
        # Audio  512
        # Behavior 512
        #
        # Total = 1536
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                common_dim * 3,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                2
            )
        )

    def forward(
        self,
        visual_embedding,
        audio_embedding,
        behavior_features
    ):

        # ====================================================
        # 1. PROJECT VISUAL
        # ====================================================

        visual = self.visual_projection(
            visual_embedding
        )

        # Shape:
        # [B, 768] -> [B, 512]


        # ====================================================
        # 2. PROJECT AUDIO
        # ====================================================

        audio = self.audio_projection(
            audio_embedding
        )

        # Shape:
        # [B, 768] -> [B, 512]


        # ====================================================
        # 3. PROJECT BEHAVIOR
        # ====================================================

        behavior = self.behavior_projection(
            behavior_features
        )

        # Shape:
        # [B, 22] -> [B, 128]


        # ====================================================
        # 4. BEHAVIOR -> COMMON SPACE
        # ====================================================

        behavior_common = (
            self.behavior_to_common(
                behavior
            )
        )

        # Shape:
        # [B, 128] -> [B, 512]


        # ====================================================
        # 5. CREATE VISUAL/AUDIO TOKENS
        # ====================================================

        visual_token = (
            visual.unsqueeze(1)
        )

        audio_token = (
            audio.unsqueeze(1)
        )

        # ====================================================
        # 6. CROSS ATTENTION
        #
        # Q = Visual
        # K = Audio
        # V = Audio
        # ====================================================

        attended_visual, attention_weights = (
            self.cross_attention(
                query=visual_token,
                key=audio_token,
                value=audio_token
            )
        )

        attended_visual = (
            attended_visual.squeeze(1)
        )

        # ====================================================
        # 7. FUSION
        #
        # Visual 512
        # Audio 512
        # Behavior 512
        # ====================================================

        combined = torch.cat(
            [
                attended_visual,
                audio,
                behavior_common
            ],
            dim=1
        )

        # [B, 1536]

        fused = self.fusion(
            combined
        )

        # [B, 512]


        # ====================================================
        # 8. CLASSIFIER
        # ====================================================

        logits = self.classifier(
            fused
        )

        # [B, 2]

        return {
            "visual": visual,
            "audio": audio,
            "behavior": behavior,
            "behavior_common": behavior_common,
            "fused": fused,
            "logits": logits,
            "attention": attention_weights
        }


# ============================================================
# CREATE MODEL
# ============================================================

model = BehavioralFusionModel(
    visual_dim=768,
    audio_dim=768,
    behavior_dim=22,
    common_dim=512,
    behavior_dim_out=128,
    num_heads=8,
    dropout=0.2
)

model = model.to(device)

model.eval()


# ============================================================
# TEST INPUT
# ============================================================

batch_size = 4

visual_embedding = torch.randn(
    batch_size,
    768
).to(device)

audio_embedding = torch.randn(
    batch_size,
    768
).to(device)

behavior_features = torch.randn(
    batch_size,
    22
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(
        visual_embedding,
        audio_embedding,
        behavior_features
    )


# ============================================================
# PRINT SHAPES
# ============================================================

print("\nInput shapes:")

print(
    "ViT:",
    visual_embedding.shape
)

print(
    "AST:",
    audio_embedding.shape
)

print(
    "Behavior:",
    behavior_features.shape
)


print("\nProjected shapes:")

print(
    "ViT ->",
    output["visual"].shape
)

print(
    "AST ->",
    output["audio"].shape
)

print(
    "Behavior ->",
    output["behavior"].shape
)

print(
    "Behavior common ->",
    output["behavior_common"].shape
)


print("\nFusion:")

print(
    "Fused vector ->",
    output["fused"].shape
)

print(
    "Logits ->",
    output["logits"].shape
)

print(
    "Attention ->",
    output["attention"].shape
)


# ============================================================
# PREDICTION
# ============================================================

probabilities = torch.softmax(
    output["logits"],
    dim=1
)

predictions = torch.argmax(
    probabilities,
    dim=1
)


print("\nProbabilities:")

print(
    probabilities
)

print("\nPredictions:")

print(
    predictions
)


# ============================================================
# ASSERTIONS
# ============================================================

assert output["visual"].shape == (
    batch_size,
    512
)

assert output["audio"].shape == (
    batch_size,
    512
)

assert output["behavior"].shape == (
    batch_size,
    128
)

assert output["behavior_common"].shape == (
    batch_size,
    512
)

assert output["fused"].shape == (
    batch_size,
    512
)

assert output["logits"].shape == (
    batch_size,
    2
)


print(
    "\n✓ BEHAVIORAL FUSION TEST PASSED"
)