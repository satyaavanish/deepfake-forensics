import torch
import torch.nn as nn


# ============================================================
# CROSS ATTENTION BLOCK
# ============================================================

class CrossAttentionBlock(nn.Module):

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
            ),

            nn.Dropout(
                dropout
            )
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        query,
        key,
        value
    ):

        attended, attention_weights = (
            self.attention(
                query=query,
                key=key,
                value=value
            )
        )

        x = self.norm1(
            query + attended
        )

        ff = self.feed_forward(
            x
        )

        x = self.norm2(
            x + ff
        )

        return x, attention_weights


# ============================================================
# CROSS-MODAL FUSION MODEL
# ============================================================

class CrossModalFusionModel(nn.Module):

    def __init__(
        self,
        visual_dim=768,
        audio_dim=768,
        behavior_dim=22,
        common_dim=512,
        behavior_projection_dim=128,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()

        # ----------------------------------------------------
        # Visual: 768 -> 512
        # ----------------------------------------------------

        self.visual_projection = nn.Sequential(

            nn.Linear(
                visual_dim,
                common_dim
            ),

            nn.LayerNorm(
                common_dim
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # Audio: 768 -> 512
        # ----------------------------------------------------

        self.audio_projection = nn.Sequential(

            nn.Linear(
                audio_dim,
                common_dim
            ),

            nn.LayerNorm(
                common_dim
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # Behavior: 23 -> 128
        #
        # 22 behavioral features
        # + rPPG availability
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
                behavior_projection_dim
            ),

            nn.LayerNorm(
                behavior_projection_dim
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # Behavior 128 -> 512
        #
        # Allows behavior to participate in the same
        # cross-modal representation space.
        # ----------------------------------------------------

        self.behavior_to_common = nn.Linear(
            behavior_projection_dim,
            common_dim
        )


        # ----------------------------------------------------
        # Visual attends to Audio
        # ----------------------------------------------------

        self.visual_to_audio = (
            CrossAttentionBlock(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        )


        # ----------------------------------------------------
        # Audio attends to Visual
        #
        # Bidirectional interaction.
        # ----------------------------------------------------

        self.audio_to_visual = (
            CrossAttentionBlock(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        )


        # ----------------------------------------------------
        # Behavior attends to fused visual/audio
        # ----------------------------------------------------

        self.behavior_attention = (
            CrossAttentionBlock(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        )


        # ----------------------------------------------------
        # Final fusion
        #
        # visual 512
        # audio 512
        # behavior 512
        #
        # concatenated = 1536
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                common_dim * 3,
                512
            ),

            nn.LayerNorm(
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


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        visual_embedding,
        audio_embedding,
        behavior_features
    ):

        # ----------------------------------------------------
        # Project modalities
        # ----------------------------------------------------

        visual = self.visual_projection(
            visual_embedding
        )

        audio = self.audio_projection(
            audio_embedding
        )

        behavior = self.behavior_projection(
            behavior_features
        )

        # ----------------------------------------------------
        # Convert behavior to common dimension
        # ----------------------------------------------------

        behavior_common = (
            self.behavior_to_common(
                behavior
            )
        )

        # ----------------------------------------------------
        # Convert vectors into tokens
        #
        # [B, 512]
        #      ↓
        # [B, 1, 512]
        # ----------------------------------------------------

        visual_token = (
            visual.unsqueeze(1)
        )

        audio_token = (
            audio.unsqueeze(1)
        )

        behavior_token = (
            behavior_common.unsqueeze(1)
        )

        # ----------------------------------------------------
        # Visual -> Audio
        # ----------------------------------------------------

        visual_attended, visual_attention = (
            self.visual_to_audio(
                query=visual_token,
                key=audio_token,
                value=audio_token
            )
        )

        # ----------------------------------------------------
        # Audio -> Visual
        # ----------------------------------------------------

        audio_attended, audio_attention = (
            self.audio_to_visual(
                query=audio_token,
                key=visual_token,
                value=visual_token
            )
        )

        # ----------------------------------------------------
        # Behavior -> Visual/Audio
        #
        # Use the two attended modalities as context.
        # ----------------------------------------------------

        visual_audio_context = torch.cat(
            [
                visual_attended,
                audio_attended
            ],
            dim=1
        )

        behavior_attended, behavior_attention = (
            self.behavior_attention(
                query=behavior_token,
                key=visual_audio_context,
                value=visual_audio_context
            )
        )

        # ----------------------------------------------------
        # Remove token dimension
        # ----------------------------------------------------

        visual_vector = (
            visual_attended
            .squeeze(1)
        )

        audio_vector = (
            audio_attended
            .squeeze(1)
        )

        behavior_vector = (
            behavior_attended
            .squeeze(1)
        )

        # ----------------------------------------------------
        # Final concatenation
        # ----------------------------------------------------

        combined = torch.cat(
            [
                visual_vector,
                audio_vector,
                behavior_vector
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused = self.fusion(
            combined
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            fused
        )

        return {
            "logits": logits,
            "fused_embedding": fused,
            "visual_attention": visual_attention,
            "audio_attention": audio_attention,
            "behavior_attention": behavior_attention
        }