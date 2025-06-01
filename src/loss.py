import torch
import torch.nn as nn
from config import (
    L1_LOSS_WEIGHT,
    MSE_LOSS_WEIGHT,
    INTER_JOINT_LOSS_WEIGHT,
    ABS_ROOT_LOSS_WEIGHT,
)


class ComprehensivePoseLoss(nn.Module):

    def __init__(
        self,
        l1_weight=L1_LOSS_WEIGHT,
        mse_weight=MSE_LOSS_WEIGHT,
        inter_joint_loss_weight=INTER_JOINT_LOSS_WEIGHT,
        abs_root_loss_weight=ABS_ROOT_LOSS_WEIGHT,
    ):
        super().__init__()
        self.mse_criterion = nn.MSELoss(reduction="mean")
        self.l1_criterion = nn.L1Loss(reduction="mean")

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.inter_joint_loss_weight = inter_joint_loss_weight
        self.abs_root_loss_weight = abs_root_loss_weight

    def inter_joint_distance_loss(self, pred_joints, gt_joints):
        """
        Compares every unique pair of joints in pred vs. gt.
        pred_joints, gt_joints: [B, J, 3]
        """
        _, J, _ = pred_joints.shape

        pred_diffs = pred_joints[:, :, None, :] - pred_joints[:, None, :, :]
        gt_diffs = gt_joints[:, :, None, :] - gt_joints[:, None, :, :]

        pred_dists = pred_diffs.norm(dim=3)
        gt_dists = gt_diffs.norm(dim=3)

        idx_i, idx_j = torch.triu_indices(J, J, offset=1)
        pred_pairs = pred_dists[:, idx_i, idx_j]  # [B, num_pairs]
        gt_pairs = gt_dists[:, idx_i, idx_j]

        rel_err = torch.abs(pred_pairs - gt_pairs)
        return rel_err.mean()

    def abs_root_distance_loss(self, pred_joints, gt_joints):
        """
        Computes the absolute distance of the root joint (index 0) between
        predicted and ground truth joints.
        pred_joints, gt_joints: [B, J, 3]
        """
        return torch.abs(pred_joints[:, 0, :] - gt_joints[:, 0, :]).mean()

    def forward(self, pred_joints, gt_joints):
        """
        Combines all loss components with their respective weights.

        Args:
            pred_joints: Predicted 3D joint positions [B, J, 3]
            gt_joints:    Ground truth 3D joint positions [B, J, 3]
            is_sequence:  Whether to apply temporal terms (not shown here)
        """
        mse_loss = self.mse_criterion(pred_joints, gt_joints)
        l1_loss = self.l1_criterion(pred_joints, gt_joints)
        inter_joint_loss = self.inter_joint_distance_loss(pred_joints, gt_joints)
        abs_root_loss = self.abs_root_distance_loss(pred_joints, gt_joints)

        total_loss = (
            self.mse_weight * mse_loss
            + self.l1_weight * l1_loss
            + self.inter_joint_loss_weight * inter_joint_loss
            + self.abs_root_loss_weight * abs_root_loss
        )

        loss_components = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "inter_joint_loss": inter_joint_loss,
            "abs_root_loss": abs_root_loss,
            "total_loss": total_loss,
        }
        return total_loss, loss_components
