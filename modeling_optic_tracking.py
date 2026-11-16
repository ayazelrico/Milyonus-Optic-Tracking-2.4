"""
Milyonus Optic Tracking Model Architecture
Ultra-detailed multi-task model for real-time facial optic tracking.
Supports 478-point landmarks, 3D pose, blendshapes, focus score, multi-face tracking.
Designed to be trained on top of or replace MediaPipe graphs for full ownership.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_optic_tracking import MilyonusOpticTrackingConfig


class PatchEmbed(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W -> B, embed, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers=4, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class MilyonusOpticTrackingModel(PreTrainedModel):
    config_class = MilyonusOpticTrackingConfig

    def __init__(self, config: MilyonusOpticTrackingConfig):
        super().__init__(config)
        self.config = config

        # Backbone: hybrid EfficientNet-style + ViT patches
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            embed_dim=config.hidden_size,
        )

        # Spatial transformer (ViT-style)
        self.spatial_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                batch_first=True,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.spatial_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Temporal module for multi-frame tracking
        if config.use_temporal_transformer:
            self.temporal = TemporalTransformer(config.hidden_size, num_layers=4)

        # Multi-task heads
        self.landmark_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_landmarks * 3),  # x,y,z
        )

        self.blendshape_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_blendshapes),
        )

        self.pose_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 6),  # yaw, pitch, roll, tx, ty, tz
        ) if config.pose_head else None

        self.focus_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ) if config.focus_score_head else None

        self.expression_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, config.num_expression_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Multi-face tracker embedding
        self.tracker_embed = nn.Linear(config.hidden_size, 128) if config.tracker_head else None

        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_sequence: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # pixel_values: B, C, H, W
        x = self.patch_embed(pixel_values)

        for layer in self.spatial_layers:
            x = layer(x)
        x = self.spatial_norm(x)
        pooled = x.mean(dim=1)  # global pool

        outputs = {
            "landmarks": self.landmark_head(pooled).view(-1, self.config.num_landmarks, 3),
            "blendshapes": self.blendshape_head(pooled),
            "expressions": self.expression_head(pooled),
            "confidence": self.confidence_head(pooled),
        }

        if self.pose_head is not None:
            outputs["pose"] = self.pose_head(pooled)

        if self.focus_head is not None:
            outputs["focus_score"] = self.focus_head(pooled)

        if self.tracker_embed is not None:
            outputs["track_embedding"] = self.tracker_embed(pooled)

        if temporal_sequence is not None and self.temporal is not None:
            # temporal_sequence: B, T, C
            temp_out = self.temporal(temporal_sequence)
            outputs["temporal_features"] = temp_out

        if return_dict:
            return outputs
        return tuple(outputs.values())


# For AutoModel registration (optional)
MilyonusOpticTrackingModel.register_for_auto_class("AutoModel")
