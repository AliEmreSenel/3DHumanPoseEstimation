class ModelConfig:
    """Configuration class for model architecture parameters"""

    def __init__(self, model_type, **kwargs):
        """Initialize model configuration with default values and provided kwargs."""
        if model_type == "cnn":
            self.get_cnn_config(**kwargs)
        elif model_type == "transformer":
            self.get_transformer_config(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_transformer_config(self, **kwargs):
        # Task-specific
        self.num_joints = kwargs.get("num_joints", 17)
        self.heatmap_sigma = kwargs.get("heatmap_sigma", 2.0)

        # Image Stream (ViT Backbone)
        self.vit_model_name = kwargs.get("vit_model_name", "vit_base_patch16_384")
        self.vit_pretrained = kwargs.get("vit_pretrained", True)
        self.vit_freeze_backbone = kwargs.get("vit_freeze_backbone", False)
        self.image_size = kwargs.get("image_size", (512, 512))
        self.image_in_channels = kwargs.get("image_in_channels", 4)

        # Heatmap Stream
        self.heatmap_size = kwargs.get("heatmap_size", 64)
        self.heatmap_patch_size = kwargs.get("heatmap_patch_size", 16)
        self.heatmap_in_channels = kwargs.get("heatmap_in_channels", self.num_joints)

        # Shared Transformer/Fusion parameters
        # transformer_embed_dim will be set by ViT (e.g., 768 for vit_base).
        self.transformer_heads = kwargs.get("transformer_heads", 16)
        self.transformer_mlp_ratio = kwargs.get("transformer_mlp_ratio", 4.0)
        self.transformer_dropout_rate = kwargs.get("transformer_dropout_rate", 0.1)
        self.transformer_attention_dropout_rate = kwargs.get(
            "transformer_attention_dropout_rate", 0.1
        )

        # Cross-Modal Fusion Config
        self.num_cross_modal_layers = kwargs.get("num_cross_modal_layers", 2)

        # Final Encoder (after fusion)
        self.final_encoder_depth = kwargs.get("final_encoder_depth", 4)

        self.activation = kwargs.get("activation", "gelu")

        # Head
        self.regression_hidden_dims = kwargs.get(
            "regression_hidden_dims", (1024, 512, 256)
        )
        self.regression_dropout = kwargs.get("regression_dropout", 0.25)

        # This will be determined by the chosen vit_model_name below
        self.transformer_embed_dim = kwargs.get("transformer_embed_dim", 768)

    def get_cnn_config(self, **kwargs):

        # Input parameters
        self.image_size = kwargs.get("image_size", (500, 500))

        self.in_channels = kwargs.get("in_channels", 3 + 1 + 17)
        self.num_joints = kwargs.get("num_joints", 17)

        # Heatmap parameters
        self.heatmap_size = kwargs.get("heatmap_size", 500)
        self.heatmap_sigma = kwargs.get("heatmap_sigma", 10.0)

        # Initial convolution
        self.initial_channels = kwargs.get("initial_channels", 64)
        self.initial_kernel_size = kwargs.get("initial_kernel_size", 5)
        self.initial_stride = kwargs.get("initial_stride", 2)

        # Backbone stages configuration
        self.stage_channels = kwargs.get("stage_channels", [128, 256, 512])
        self.stage_depths = kwargs.get("stage_depths", [3, 4, 5])
        self.stage_strides = kwargs.get("stage_strides", [2, 2, 2])
        self.stage_expand_ratios = kwargs.get("stage_expand_ratios", [1, 3, 6])

        # Block configurations
        self.use_se_blocks = kwargs.get("use_se_blocks", True)
        self.se_reduction = kwargs.get("se_reduction", 16)
        self.use_dual_path_blocks = kwargs.get("use_dual_path_blocks", True)

        # Global feature parameters
        self.global_pool_size = kwargs.get("global_pool_size", 8)
        self.global_feature_dim = kwargs.get("global_feature_dim", 1024)

        # Regression head parameters
        self.regression_dims = kwargs.get("regression_dims", [1024, 512])
        self.regression_dropout = kwargs.get("regression_dropout", 0.2)

        # Activation and normalization
        self.activation = kwargs.get("activation", "silu")
        self.normalization = kwargs.get("normalization", "batch")

        # Residual parameters
        self.residual_scale = kwargs.get("residual_scale", 1.0)

        # Depthwise separable convolution parameters
        self.depthwise_kernel_size = kwargs.get("depthwise_kernel_size", 3)

    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not callable(value) and not key.startswith("__")
        }
