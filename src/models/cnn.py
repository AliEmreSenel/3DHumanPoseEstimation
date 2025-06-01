import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import get_activation, get_normalization
from .common import GaussianHeatmapGenerator, PoseRegressionHead


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=16, activation="silu"):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            get_activation(activation),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """Efficient Channel Attention block"""

    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    """Coordinate Attention module"""

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_w = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # Pool along height and width axes
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x)  # [n, c, 1, w]

        # Join the pooled features
        x_cat = torch.cat([x_h.transpose(2, 3), x_w], dim=3)  # [n, c, 1, h+w]

        # Process with shared MLP
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        # Split the features
        x_h, x_w = torch.split(x_cat, [h, w], dim=3)
        x_h = x_h.transpose(2, 3)  # [n, mid, h, 1]

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()  # [n, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [n, c, 1, w]

        # Apply attention maps
        out = identity * a_h * a_w

        return out


class ConvBnAct(nn.Module):
    """Basic Conv-BatchNorm-Activation block with configurable parameters"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        activation="silu",
        normalization="batch",
        dilation=1,
    ):
        super(ConvBnAct, self).__init__()
        # Auto-calculate padding if not specified
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.norm = get_normalization(normalization, out_channels)
        self.act = get_activation(activation) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with configurable parameters"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=False,
        activation="silu",
        normalization="batch",
    ):
        super(DepthwiseSeparableConv, self).__init__()
        # Auto-calculate padding if not specified
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.depthwise = ConvBnAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias,
            activation=activation,
            normalization=normalization,
        )
        self.pointwise = ConvBnAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted Residual Block with configurable parameters"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=6,
        use_se=True,
        se_reduction=16,
        activation="silu",
        normalization="batch",
        residual_scale=1.0,
        attention_type=None,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        self.residual_scale = residual_scale

        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(
                ConvBnAct(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    activation=activation,
                    normalization=normalization,
                )
            )

        # Depthwise
        layers.append(
            ConvBnAct(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                activation=activation,
                normalization=normalization,
            )
        )

        # Attention mechanism
        if attention_type == "se" or (use_se and attention_type is None):
            layers.append(
                SEBlock(hidden_dim, reduction=se_reduction, activation=activation)
            )
        elif attention_type == "eca":
            layers.append(ECABlock(hidden_dim))
        elif attention_type == "coord":
            layers.append(CoordAttention(hidden_dim, hidden_dim))

        # Projection
        layers.append(
            ConvBnAct(
                hidden_dim,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=None,
                normalization=normalization,
            )
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x) * self.residual_scale
        else:
            return self.conv(x)


class DualPathBlock(nn.Module):
    """Dual path block combining residual and dense connections with configurable parameters"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="silu",
        normalization="batch",
        residual_scale=1.0,
        attention_type=None,
    ):
        super(DualPathBlock, self).__init__()
        self.residual_scale = residual_scale

        # Residual path
        self.residual_path = nn.Sequential(
            ConvBnAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=activation,
                normalization=normalization,
            ),
            DepthwiseSeparableConv(
                out_channels,
                out_channels,
                stride=stride,
                activation=activation,
                normalization=normalization,
            ),
            ConvBnAct(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=None,
                normalization=normalization,
            ),
        )
        # Dense path
        dense_channels = out_channels // 2
        self.dense_path = nn.Sequential(
            ConvBnAct(
                in_channels,
                dense_channels,
                kernel_size=1,
                padding=0,
                activation=activation,
                normalization=normalization,
            ),
            DepthwiseSeparableConv(
                dense_channels,
                dense_channels,
                stride=stride,
                activation=activation,
                normalization=normalization,
            ),
        )

        # Attention mechanism
        self.attention = None
        if attention_type == "se":
            self.attention = SEBlock(out_channels, reduction=16, activation=activation)
        elif attention_type == "eca":
            self.attention = ECABlock(out_channels)
        elif attention_type == "coord":
            self.attention = CoordAttention(out_channels, out_channels)

        # Fusion
        self.fusion = ConvBnAct(
            out_channels + dense_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            activation=activation,
            normalization=normalization,
        )

        # Shortcut for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBnAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=None,
                normalization=normalization,
            )

    def forward(self, x):
        res_path = self.residual_path(x)
        dense_path = self.dense_path(x)

        # Residual connection
        res_path = res_path + self.shortcut(x) * self.residual_scale

        # Concatenate dense path
        combined = torch.cat([res_path, dense_path], dim=1)

        # Fusion
        out = self.fusion(combined)

        # Apply attention if specified
        if self.attention is not None:
            out = self.attention(out)

        return out


class WASPModule(nn.Module):
    """
    Weighted Atrous Spatial Pyramid module for multi-scale feature extraction
    with learnable weights for each branch.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations=(1, 6, 12, 18),
        activation="silu",
        normalization="batch",
    ):
        super(WASPModule, self).__init__()

        self.conv1x1 = ConvBnAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            activation=activation,
            normalization=normalization,
        )

        # Atrous (dilated) convolution branches
        self.atrous_branches = nn.ModuleList()
        for dilation in dilations:
            self.atrous_branches.append(
                ConvBnAct(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                )
            )

        # Global context branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=activation,
                normalization=normalization,
            ),
        )

        # Learnable weights for each branch
        num_branches = len(dilations) + 2  # dilated branches + 1x1 conv + global branch
        self.weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        self.softmax = nn.Softmax(dim=0)

        # Final fusion
        self.fusion = ConvBnAct(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x):
        # Get normalized weights
        weights = self.softmax(self.weights)

        # 1x1 convolution branch
        out_1x1 = self.conv1x1(x) * weights[0]

        # Atrous convolution branches
        atrous_outputs = []
        for i, branch in enumerate(self.atrous_branches):
            atrous_outputs.append(branch(x) * weights[i + 1])

        # Global context branch
        global_out = self.global_branch(x)
        global_out = F.interpolate(
            global_out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        global_out = global_out * weights[-1]

        # Combine all branches
        combined = out_1x1
        for out in atrous_outputs:
            combined = combined + out
        combined = combined + global_out

        # Final fusion
        output = self.fusion(combined)

        return output


class CNNPoseEstimation(nn.Module):
    """
    Enhanced CNN-based 3D human pose estimation model with configurable architecture
    """

    def __init__(self, config):
        super(CNNPoseEstimation, self).__init__()

        self.config = config

        self.conv1 = nn.Sequential(
            ConvBnAct(
                config.in_channels,
                config.initial_channels,
                kernel_size=config.initial_kernel_size,
                stride=config.initial_stride,
                activation=config.activation,
                normalization=config.normalization,
            ),
            ConvBnAct(
                config.initial_channels,
                config.initial_channels,
                kernel_size=3,
                stride=1,
                activation=config.activation,
                normalization=config.normalization,
            ),
        )

        # Add Gaussian heatmap generator
        self.heatmap_generator = GaussianHeatmapGenerator(
            num_joints=config.num_joints,
            heatmap_size=config.heatmap_size,
            sigma=config.heatmap_sigma,
        )

        # Build backbone stages
        self.stages = nn.ModuleList()
        in_channels = config.initial_channels

        for i in range(len(config.stage_channels)):
            stage = []
            out_channels = config.stage_channels[i]
            depth = config.stage_depths[i]
            stride = config.stage_strides[i]
            expand_ratio = config.stage_expand_ratios[i]

            # First block with stride
            if i >= 2 and config.use_dual_path_blocks:
                stage.append(
                    DualPathBlock(
                        in_channels,
                        out_channels,
                        stride=stride,
                        activation=config.activation,
                        normalization=config.normalization,
                        residual_scale=config.residual_scale,
                        attention_type="coord" if i >= 2 else "se",
                    )
                )
            else:
                stage.append(
                    InvertedResidual(
                        in_channels,
                        out_channels,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        use_se=config.use_se_blocks,
                        se_reduction=config.se_reduction,
                        activation=config.activation,
                        normalization=config.normalization,
                        residual_scale=config.residual_scale,
                        attention_type="coord" if i >= 2 else "se",
                    )
                )

            # Remaining blocks
            for j in range(1, depth):
                if i >= 2 and config.use_dual_path_blocks and j % 2 == 0:
                    stage.append(
                        DualPathBlock(
                            out_channels,
                            out_channels,
                            stride=1,
                            activation=config.activation,
                            normalization=config.normalization,
                            residual_scale=config.residual_scale,
                            attention_type="coord" if i >= 2 else "se",
                        )
                    )
                else:
                    stage.append(
                        InvertedResidual(
                            out_channels,
                            out_channels,
                            stride=1,
                            expand_ratio=expand_ratio,
                            use_se=config.use_se_blocks,
                            se_reduction=config.se_reduction,
                            activation=config.activation,
                            normalization=config.normalization,
                            residual_scale=config.residual_scale,
                            attention_type="eca" if j % 2 == 0 else "se",
                        )
                    )

            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

        # WASP module for multi-scale feature extraction
        self.wasp = WASPModule(
            config.stage_channels[-1],
            config.stage_channels[-1],
            dilations=(1, 6, 12, 18),
            activation=config.activation,
            normalization=config.normalization,
        )

        # Global feature aggregation with multi-scale pooling
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(config.global_pool_size),
            ConvBnAct(
                config.stage_channels[-1],
                config.global_feature_dim,
                kernel_size=1,
                padding=0,
                activation=config.activation,
                normalization=config.normalization,
            ),
            ECABlock(config.global_feature_dim),
            nn.AdaptiveAvgPool2d(1),
        )

        # Enhanced pose regression head
        self.pose_head = PoseRegressionHead(
            config.global_feature_dim,
            config.num_joints,
            hidden_dims=config.regression_dims,
            dropout=config.regression_dropout,
            activation=config.activation,
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, depth, keypoints_2d):

        # Generate heatmaps from 2D keypoints
        heatmaps = self.heatmap_generator(keypoints_2d)

        # Concatenate RGB and depth along channel dimension
        # image: [B, 3, H, W], depth: [B, 1, H, W] -> combined: [B, 4, H, W]
        x = torch.cat([image, depth, heatmaps], dim=1)

        # Extract features through backbone
        x = self.conv1(x)

        for stage in self.stages:
            x = stage(x)

        # Apply WASP module for multi-scale feature extraction
        x = self.wasp(x)

        # Global feature aggregation
        global_features = self.global_features(x)

        # Pose regression
        joints_3d = self.pose_head(global_features)

        return joints_3d
