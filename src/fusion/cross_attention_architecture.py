import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Visual embedding -> Query
    Audio embedding  -> Key + Value

    Attention(Q,K,V)
    = softmax(QK^T / sqrt(d)) V
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
            )
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        visual_tokens,
        audio_tokens
    ):

        # ----------------------------------------------------
        # Query = Visual
        # Key   = Audio
        # Value = Audio
        # ----------------------------------------------------

        attended_visual, attention_weights = (
            self.attention(
                query=visual_tokens,
                key=audio_tokens,
                value=audio_tokens
            )
        )

        # Residual connection
        x = self.norm1(
            visual_tokens +
            self.dropout(
                attended_visual
            )
        )

        # Feed-forward network
        ff = self.feed_forward(
            x
        )

        # Second residual connection
        x = self.norm2(
            x +
            self.dropout(ff)
        )

        return (
            x,
            attention_weights
        )


class CrossAttentionClassifier(nn.Module):

    def __init__(
        self,
        visual_dim=768,
        audio_dim=768,
        common_dim=512,
        num_heads=8,
        dropout=0.2
    ):

        super().__init__()

        # ====================================================
        # Visual embedding -> common dimension
        # ====================================================

        self.visual_projection = nn.Linear(
            visual_dim,
            common_dim
        )

        # ====================================================
        # Audio embedding -> common dimension
        # ====================================================

        self.audio_projection = nn.Linear(
            audio_dim,
            common_dim
        )

        # ====================================================
        # Cross Attention
        # ====================================================

        self.cross_attention = (
            CrossAttentionBlock(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=0.1
            )
        )

        # ====================================================
        # Fusion
        # ====================================================

        self.fusion = nn.Sequential(

            nn.Linear(
                common_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ====================================================
        # Binary classifier
        # ====================================================
        #
        # IMPORTANT:
        # One output because we use sigmoid.
        #
        # 0 -> REAL
        # 1 -> FAKE
        # ====================================================

        self.classifier = nn.Linear(
            128,
            1
        )

    def forward(
        self,
        visual_embedding,
        audio_embedding
    ):

        # ====================================================
        # Project embeddings
        # ====================================================

        visual = self.visual_projection(
            visual_embedding
        )

        audio = self.audio_projection(
            audio_embedding
        )

        # ====================================================
        # Convert to tokens
        # ====================================================

        visual_tokens = (
            visual.unsqueeze(1)
        )

        audio_tokens = (
            audio.unsqueeze(1)
        )

        # ====================================================
        # Cross Attention
        #
        # Q = Visual
        # K = Audio
        # V = Audio
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
        # Fully connected fusion
        # ====================================================

        fused = self.fusion(
            attended_visual
        )

        # ====================================================
        # Classification logit
        # ====================================================

        logit = self.classifier(
            fused
        )

        # ====================================================
        # Sigmoid probability
        # ====================================================

        probability = torch.sigmoid(
            logit
        )

        return {
            "logit": logit,
            "probability": probability,
            "fused": fused,
            "attention": attention_weights
        }