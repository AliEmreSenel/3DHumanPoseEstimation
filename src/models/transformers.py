import torch
import torch.nn as nn
from utils import get_activation
from .common import GaussianHeatmapGenerator, PoseRegressionHead
import timm


class PatchEmbedding(nn.Module):  # Used for Heatmaps
    def __init__(self, img_size_h, img_size_w, patch_size, in_chans, embed_dim):
        super().__init__()
        if img_size_h % patch_size != 0 or img_size_w % patch_size != 0:
            raise ValueError(
                f"Image dims ({img_size_h}x{img_size_w}) must be divisible by patch size ({patch_size})."
            )
        self.num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout_rate,
        attention_dropout_rate,
        activation="gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_out)
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class CrossModalFusionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout_rate,
        attention_dropout_rate,
        activation="gelu",
    ):
        super().__init__()
        self.norm_img_q = nn.LayerNorm(embed_dim)
        self.norm_hm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn_img_to_hm = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True
        )
        self.dropout_img = nn.Dropout(dropout_rate)
        self.norm_hm_q = nn.LayerNorm(embed_dim)
        self.norm_img_kv = nn.LayerNorm(embed_dim)
        self.cross_attn_hm_to_img = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True
        )
        self.dropout_hm = nn.Dropout(dropout_rate)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.norm_img_mlp = nn.LayerNorm(embed_dim)
        self.mlp_img = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm_hm_mlp = nn.LayerNorm(embed_dim)
        self.mlp_hm = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x_img, x_hm):
        img_q = self.norm_img_q(x_img)
        hm_kv = self.norm_hm_kv(x_hm)
        img_ca_out, _ = self.cross_attn_img_to_hm(query=img_q, key=hm_kv, value=hm_kv)
        x_img = x_img + self.dropout_img(img_ca_out)
        hm_q = self.norm_hm_q(x_hm)
        img_kv = self.norm_img_kv(x_img)
        hm_ca_out, _ = self.cross_attn_hm_to_img(query=hm_q, key=img_kv, value=img_kv)
        x_hm = x_hm + self.dropout_hm(hm_ca_out)
        x_img = x_img + self.mlp_img(self.norm_img_mlp(x_img))
        x_hm = x_hm + self.mlp_hm(self.norm_hm_mlp(x_hm))
        return x_img, x_hm


class TransformerPoseEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()

        try:
            # Get ViT's default config to extract embed_dim and native patch_size
            _vit_default_cfg = timm.create_model(
                config.vit_model_name, pretrained=False
            ).default_cfg
            config.transformer_embed_dim = _vit_default_cfg["embed_dim"]
            _vit_native_patch_size = _vit_default_cfg[
                "patch_size"
            ]  # tuple e.g. (16,16)

            if (
                config.image_size[0] % _vit_native_patch_size[0] != 0
                or config.image_size[1] % _vit_native_patch_size[1] != 0
            ):
                print(
                    f"Warning: Configured image_size {config.image_size} is not perfectly divisible by ViT's native patch size {_vit_native_patch_size}. Timm will handle this via padding/interpolation, but for optimal patch alignment, consider adjusting image_size if issues arise or if exact patch counts are critical."
                )
            print(
                f"Using ViT model: {config.vit_model_name} with determined embed_dim: {config.transformer_embed_dim} and target image_size: {config.image_size}"
            )
        except Exception as e:
            print(
                f"Could not determine embed_dim/patch_size from {config.vit_model_name}. Error: {e}"
            )
            print(
                f"Ensure {config.vit_model_name} is a valid timm model name. Using default embed_dim from config: {config.transformer_embed_dim}"
            )
        self.config = config

        # --- Image/Depth Stream (Pre-trained ViT) ---
        self.vit_backbone = timm.create_model(
            config.vit_model_name,
            pretrained=config.vit_pretrained,
            num_classes=0,
            img_size=config.image_size,
        )
        orig_patch_embed_proj = self.vit_backbone.patch_embed.proj
        new_in_channels = config.image_in_channels
        orig_in_channels = orig_patch_embed_proj.in_channels

        if new_in_channels != orig_in_channels:
            print(
                f"Adapting pre-trained ViT patch_embed from {orig_in_channels} to {new_in_channels} channels."
            )
            new_weights = torch.zeros_like(
                orig_patch_embed_proj.weight.data.repeat(
                    1,
                    (new_in_channels + orig_in_channels - 1) // orig_in_channels,
                    1,
                    1,
                )
            )[:, :new_in_channels, :, :]

            if new_in_channels > orig_in_channels:
                new_weights[:, :orig_in_channels, :, :] = (
                    orig_patch_embed_proj.weight.data.clone()
                )
                for i in range(orig_in_channels, new_in_channels):
                    new_weights[:, i : i + 1, :, :] = (
                        orig_patch_embed_proj.weight.data.mean(dim=1, keepdim=True)
                    )  # Average existing channels for new ones
            else:
                new_weights = orig_patch_embed_proj.weight.data.mean(
                    dim=1, keepdim=True
                ).repeat(
                    1, new_in_channels, 1, 1
                )  # Average all to new_in_channels

            self.vit_backbone.patch_embed.proj = nn.Conv2d(
                new_in_channels,
                orig_patch_embed_proj.out_channels,
                kernel_size=orig_patch_embed_proj.kernel_size,
                stride=orig_patch_embed_proj.stride,
                padding=orig_patch_embed_proj.padding,
                bias=(orig_patch_embed_proj.bias is not None),
            )
            self.vit_backbone.patch_embed.proj.weight.data = new_weights
            if orig_patch_embed_proj.bias is not None:
                self.vit_backbone.patch_embed.proj.bias.data = (
                    orig_patch_embed_proj.bias.data.clone()
                )

        if config.vit_freeze_backbone:
            print(
                "Freezing ViT backbone weights (note: adapted patch_embed.proj may remain trainable depending on exact timm model)."
            )
            for param_name, param in self.vit_backbone.named_parameters():
                # Ensure the adapted conv layer remains trainable if we modified it
                if not (
                    param_name.startswith("patch_embed.proj")
                    and new_in_channels != orig_in_channels
                ):
                    param.requires_grad = False

        # --- Heatmap Stream ---
        self.heatmap_generator = GaussianHeatmapGenerator(
            config.num_joints, config.heatmap_size, config.heatmap_sigma
        )
        self.heatmap_patch_embed = PatchEmbedding(
            config.heatmap_size,
            config.heatmap_size,
            config.heatmap_patch_size,
            config.heatmap_in_channels,
            config.transformer_embed_dim,
        )
        self.pos_embed_hm = nn.Parameter(
            torch.zeros(
                1, self.heatmap_patch_embed.num_patches, config.transformer_embed_dim
            )
        )

        # --- Cross-Modal Fusion ---
        self.cross_modal_fusion_layers = nn.ModuleList(
            [
                CrossModalFusionBlock(
                    config.transformer_embed_dim,
                    config.transformer_heads,
                    config.transformer_mlp_ratio,
                    config.transformer_dropout_rate,
                    config.transformer_attention_dropout_rate,
                    config.activation,
                )
                for _ in range(config.num_cross_modal_layers)
            ]
        )

        # --- Final Encoder & Aggregation ---
        self.final_cls_token = nn.Parameter(
            torch.zeros(1, 1, config.transformer_embed_dim)
        )
        num_img_patches_from_vit = self.vit_backbone.patch_embed.num_patches
        num_total_final_tokens = (
            1 + num_img_patches_from_vit + self.heatmap_patch_embed.num_patches
        )
        self.final_pos_embed = nn.Parameter(
            torch.zeros(1, num_total_final_tokens, config.transformer_embed_dim)
        )
        self.pos_drop = nn.Dropout(config.transformer_dropout_rate)
        self.final_pos_drop = nn.Dropout(config.transformer_dropout_rate)

        self.final_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.transformer_embed_dim,
                    config.transformer_heads,
                    config.transformer_mlp_ratio,
                    config.transformer_dropout_rate,
                    config.transformer_attention_dropout_rate,
                    config.activation,
                )
                for _ in range(config.final_encoder_depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.transformer_embed_dim)
        self.pose_head = PoseRegressionHead(
            config.transformer_embed_dim,
            config.num_joints,
            config.regression_hidden_dims,
            config.regression_dropout,
            config.activation,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed_hm, std=0.02)
        nn.init.trunc_normal_(self.final_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.final_cls_token, std=0.02)
        self.heatmap_patch_embed.apply(self._init_weights_for_linear)
        self.cross_modal_fusion_layers.apply(self._init_weights_for_linear)
        self.final_encoder.apply(self._init_weights_for_linear)
        self.pose_head.apply(self._init_weights_for_linear)
        self.norm_out.apply(self._init_weights_for_linear)

    def _init_weights_for_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image, depth, keypoints_2d):
        B = image.shape[0]
        img_depth_input = torch.cat(
            [image, depth], dim=1
        )  # Ensure pre-resized to config.image_size

        img_tokens_from_vit_features = self.vit_backbone.forward_features(
            img_depth_input
        )

        if hasattr(
            self.vit_backbone, "cls_token"
        ) and img_tokens_from_vit_features.shape[
            1
        ] == self.vit_backbone.patch_embed.num_patches + getattr(
            self.vit_backbone, "num_prefix_tokens", 1
        ):  # num_prefix_tokens usually 1 for cls
            num_prefix_tokens = getattr(self.vit_backbone, "num_prefix_tokens", 1)
            img_tokens_from_vit = img_tokens_from_vit_features[:, num_prefix_tokens:]
        else:
            img_tokens_from_vit = img_tokens_from_vit_features

        heatmaps = self.heatmap_generator(keypoints_2d)
        hm_tokens = self.heatmap_patch_embed(heatmaps)
        hm_tokens = hm_tokens + self.pos_embed_hm.expand(B, -1, -1)

        fused_img_tokens, fused_hm_tokens = img_tokens_from_vit, hm_tokens
        for fusion_block in self.cross_modal_fusion_layers:
            fused_img_tokens, fused_hm_tokens = fusion_block(
                fused_img_tokens, fused_hm_tokens
            )

        final_cls_tokens_expanded = self.final_cls_token.expand(B, -1, -1)
        final_tokens = torch.cat(
            (final_cls_tokens_expanded, fused_img_tokens, fused_hm_tokens), dim=1
        )

        pos_embed_to_add = self.final_pos_embed.expand(B, -1, -1)

        final_tokens = final_tokens + pos_embed_to_add
        final_tokens = self.final_pos_drop(final_tokens)

        for block in self.final_encoder:
            final_tokens = block(final_tokens)

        final_cls_output = self.norm_out(final_tokens[:, 0])
        pose_3d = self.pose_head(final_cls_output)
        return pose_3d
