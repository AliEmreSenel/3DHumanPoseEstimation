import torch
import torch.nn as nn
from utils import get_activation


class GaussianHeatmapGenerator(nn.Module):
    """
    Generates Gaussian heatmaps from 2D keypoints for improved pose estimation.
    """

    def __init__(self, num_joints, heatmap_size=64, sigma=2.0):
        super(GaussianHeatmapGenerator, self).__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Create coordinate grid once
        coords = torch.arange(heatmap_size, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("x_grid", x_grid.clone())
        self.register_buffer("y_grid", y_grid.clone())

    def forward(self, keypoints_2d):
        """
        Generate heatmaps from normalized 2D keypoints.

        Args:
            keypoints_2d: [batch_size, num_joints, 2] tensor with normalized coordinates (0-1)

        Returns:
            heatmaps: [batch_size, num_joints, heatmap_size, heatmap_size] tensor
        """

        # Scale keypoints to heatmap size
        keypoints_scaled = keypoints_2d * (self.heatmap_size - 1)  # [B, J, 2]
        mu_x = keypoints_scaled[..., 0].unsqueeze(-1).unsqueeze(-1)  # [B, J, 1, 1]
        mu_y = keypoints_scaled[..., 1].unsqueeze(-1).unsqueeze(-1)  # [B, J, 1, 1]

        # Compute distance squared
        dist_sq = (self.x_grid - mu_x) ** 2 + (
            self.y_grid - mu_y
        ) ** 2  # broadcast to [B, J, H, W]

        # Apply Gaussian
        heatmaps = torch.exp(-dist_sq / (2 * self.sigma**2))

        # Zero out heatmaps for invalid keypoints (e.g., negative coordinates)
        valid = (keypoints_2d > 0).all(dim=-1)  # [B, J]
        heatmaps *= valid.unsqueeze(-1).unsqueeze(-1)

        return heatmaps



class PoseRegressionHead(nn.Module):
    def __init__(
        self,
        in_features,
        num_joints,
        hidden_dims=(512, 256),
        dropout=0.2,
        activation="gelu",
    ):
        super(PoseRegressionHead, self).__init__()
        self.num_joints = num_joints
        layers = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    *[
                        nn.Linear(prev_dim, hidden_dim),
                        get_activation(activation),
                        nn.Dropout(dropout),
                    ]
                )
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_joints * 3))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, in_channels]

        pose = self.decoder(x)
        pose = pose.view(-1, self.num_joints, 3)

        return pose  # [B, num_joints, 3]
