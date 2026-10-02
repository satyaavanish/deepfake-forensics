import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Cross-attention:

    Query  = visual tokens
    Key    = audio tokens
    Value  = audio tokens

    This allows the visual representation to attend
    to relevant audio information.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            ),

            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        visual_tokens,
        audio_tokens
    ):

        # ----------------------------------------------------
        # Cross attention
        # ----------------------------------------------------

        attended_audio, attention_weights = (
            self.attention(
                query=visual_tokens,
                key=audio_tokens,
                value=audio_tokens
            )
        )

        # Residual connection
        x = self.norm1(
            visual_tokens +
            attended_audio
        )

        # Feed-forward network
        ff = self.feed_forward(
            x
        )

        x = self.norm2(
            x + ff
        )

        return x, attention_weights


class VisualAudioCrossAttention(nn.Module):

    def __init__(
        self,
        visual_dim=768,
        audio_dim=768,
        common_dim=512,
        num_heads=8
    ):
        super().__init__()

        # ----------------------------------------------------
        # Project ViT → 512
        # ----------------------------------------------------

        self.visual_projection = nn.Linear(
            visual_dim,
            common_dim
        )

        # ----------------------------------------------------
        # Project AST → 512
        # ----------------------------------------------------

        self.audio_projection = nn.Linear(
            audio_dim,
            common_dim
        )

        # ----------------------------------------------------
        # Cross attention
        # ----------------------------------------------------

        self.cross_attention = CrossAttention(
            embed_dim=common_dim,
            num_heads=num_heads
        )

        # ----------------------------------------------------
        # Behavior → 128
        # ----------------------------------------------------

        self.behavior_projection = nn.Sequential(

            nn.Linear(
                10,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                128
            )
        )

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                common_dim + 128,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.2
            )
        )

    def forward(
        self,
        visual_embedding,
        audio_embedding,
        behavior_features
    ):

        # ====================================================
        # Visual projection
        # ====================================================

        visual = self.visual_projection(
            visual_embedding
        )

        # ====================================================
        # Audio projection
        # ====================================================

        audio = self.audio_projection(
            audio_embedding
        )

        # ====================================================
        # Convert embeddings into tokens
        # ====================================================

        visual_tokens = visual.unsqueeze(
            1
        )

        audio_tokens = audio.unsqueeze(
            1
        )

        # ====================================================
        # Visual ↔ Audio cross attention
        # ====================================================

        attended_visual, attention_weights = (
            self.cross_attention(
                visual_tokens,
                audio_tokens
            )
        )

        # Remove token dimension
        attended_visual = (
            attended_visual.squeeze(1)
        )

        # ====================================================
        # Behavior projection
        # ====================================================

        behavior = self.behavior_projection(
            behavior_features
        )

        # ====================================================
        # Combine
        # ====================================================

        combined = torch.cat(
            [
                attended_visual,
                behavior
            ],
            dim=1
        )

        # ====================================================
        # Fusion vector
        # ====================================================

        fused = self.fusion(
            combined
        )

        return fused, attention_weights