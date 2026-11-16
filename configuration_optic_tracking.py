"""
Milyonus Optic Tracking Configuration
"""
from transformers import PretrainedConfig


class MilyonusOpticTrackingConfig(PretrainedConfig):
    model_type = "milyonus_optic_tracking"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=256,
        patch_size=16,
        num_channels=3,
        num_landmarks=478,
        num_blendshapes=52,
        num_expression_classes=8,
        max_faces=4,
        temporal_window=16,
        focus_score_head=True,
        pose_head=True,
        tracker_head=True,
        use_temporal_transformer=True,
        backbone="efficientnet_b3_plus_vit",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_landmarks = num_landmarks
        self.num_blendshapes = num_blendshapes
        self.num_expression_classes = num_expression_classes
        self.max_faces = max_faces
        self.temporal_window = temporal_window
        self.focus_score_head = focus_score_head
        self.pose_head = pose_head
        self.tracker_head = tracker_head
        self.use_temporal_transformer = use_temporal_transformer
        self.backbone = backbone
