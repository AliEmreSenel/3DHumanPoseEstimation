import numpy as np
import torch.nn as nn
import torch


def world_to_camera_coords(joints_world, R, t):
    """
    Transform joints from world coordinates to camera coordinates - vectorized version
    """
    R = np.array(R)
    t = np.array(t).reshape(3, 1)

    # Convert joints to numpy array for vectorized operations
    joints_world = np.array(joints_world)

    # Reshape for matrix multiplication
    joints_world_reshaped = joints_world.reshape(-1, 3, 1)

    # Vectorized transformation
    joints_camera = np.matmul(R, joints_world_reshaped) + t

    # Reshape back to original format
    return joints_camera.reshape(-1, 3)


def camera_to_pixel_coords(joints_camera, f, c):
    """
    Project 3D camera coordinates to 2D pixel coordinates

    Args:
        joints_camera: (N, 3) array of joint positions in camera coordinates
        f: (2,) focal length (fx, fy)
        c: (2,) principal point (cx, cy)

    Returns:
        joints_pixel: (N, 2) array of joint positions in pixel coordinates
    """
    joints_pixel = []

    for joint in joints_camera:
        x, y, z = joint
        px = (x * f[0] / z) + c[0]
        py = (y * f[1] / z) + c[1]
        joints_pixel.append([px, py])

    return np.array(joints_pixel)


def normalize_depth(depth_map, depth_min, depth_max):
    """Normalize depth map to range [0, 1]"""
    return (depth_map - depth_min) / (depth_max - depth_min)


# Helper functions for metrics (MPJPE and PA-MPJPE)
def compute_mpjpe(predicted_joints, ground_truth_joints):
    """
    Computes Mean Per Joint Position Error (MPJPE).
    Inputs:
    - predicted_joints: (N, num_joints, 3) tensor
    - ground_truth_joints: (N, num_joints, 3) tensor
    Output:
    - MPJPE (scalar tensor)
    """
    assert (
        predicted_joints.shape == ground_truth_joints.shape
    ), f"Shape mismatch: pred {predicted_joints.shape}, gt {ground_truth_joints.shape}"
    errors = torch.linalg.norm(predicted_joints - ground_truth_joints, dim=2)
    mpjpe_per_sample = errors.mean(dim=1)
    return mpjpe_per_sample.mean()


def compute_pa_mpjpe(predicted_joints, ground_truth_joints):
    """
    Computes Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) with corrections.
    """
    assert (
        predicted_joints.shape == ground_truth_joints.shape
    ), f"Shape mismatch: pred {predicted_joints.shape}, gt {ground_truth_joints.shape}"

    batch_size = predicted_joints.shape[0]
    num_joints = predicted_joints.shape[1]
    num_dims = predicted_joints.shape[2]
    device = predicted_joints.device
    pa_mpjpe_errors = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        pred_sample = predicted_joints[i]  # Shape: num_joints x num_dims
        gt_sample = ground_truth_joints[i]  # Shape: num_joints x num_dims

        # 1. Center the point clouds
        mu_pred = pred_sample.mean(dim=0, keepdim=True)  # Shape: 1 x num_dims
        mu_gt = gt_sample.mean(dim=0, keepdim=True)  # Shape: 1 x num_dims
        pred_centered = pred_sample - mu_pred  # Shape: num_joints x num_dims
        gt_centered = gt_sample - mu_gt  # Shape: num_joints x num_dims

        # 2. Compute the covariance matrix M = P_c^T G_c
        # pred_centered.T is num_dims x num_joints
        # gt_centered is num_joints x num_dims
        # svd_matrix is num_dims x num_dims
        svd_matrix = pred_centered.transpose(0, 1) @ gt_centered

        try:
            # 3. SVD of M: M = U_svd * diag(S_diag) * Vt_svd
            # U_svd: num_dims x num_dims
            # S_diag: num_dims
            # Vt_svd: num_dims x num_dims (this is V^T)
            U_svd, S_diag, Vt_svd = torch.linalg.svd(svd_matrix)
        except torch.linalg.LinAlgError:
            # If SVD fails, fall back to non-aligned MPJPE for this sample
            pa_mpjpe_errors[i] = torch.linalg.norm(
                pred_sample - gt_sample, dim=1
            ).mean()
            # logger.warning(f"SVD failed for sample {i} in PA-MPJPE. Using non-aligned error.")
            continue

        # 4. Calculate optimal rotation R = V * U_svd^T
        # V = Vt_svd^T
        # R_candidate is num_dims x num_dims
        R_candidate = Vt_svd.transpose(0, 1) @ U_svd.transpose(0, 1)

        # 5. Handle reflections: ensure det(R) > 0
        # The determinant of R_candidate tells us if a reflection is present.
        det_R = torch.linalg.det(R_candidate)

        R_final = R_candidate
        S_for_scale = S_diag.clone()  # Use a copy for modification

        if det_R < 0:
            # If determinant is negative, we have a reflection.
            # Modify V by flipping the sign of the column corresponding to the smallest singular value.
            # This is equivalent to flipping the sign of the last row of Vt_svd before transposing.
            Vt_svd_corrected = Vt_svd.clone()
            Vt_svd_corrected[-1, :] *= -1  # num_dims is at least 1. S_diag is sorted.
            R_final = Vt_svd_corrected.transpose(0, 1) @ U_svd.transpose(0, 1)

            # For scale calculation, the effective sum of singular values changes.
            # s = trace(Sigma * D) / trace(P_c^T * P_c)
            # where D is diag(1, ..., 1, det(V*U^T)).
            # So, if det < 0, the last singular value is effectively negated in the sum.
            if S_for_scale.numel() > 0:  # Check if S_for_scale is not empty
                S_for_scale[-1] *= -1

        # 6. Calculate optimal scale s
        # var_pred_centered is trace(P_c^T * P_c)
        var_pred_centered = torch.sum(pred_centered**2)

        # sum_S is trace(Sigma * D)
        sum_S_effective = torch.sum(S_for_scale)

        s = (
            sum_S_effective / var_pred_centered
            if var_pred_centered > 1e-9
            else torch.tensor(1.0, device=device)
        )

        # 7. Align the predicted pose
        # pred_aligned_centered = s * P_c * R
        pred_aligned_centered = s * pred_centered @ R_final
        pred_aligned = pred_aligned_centered + mu_gt

        # Compute error for this sample
        sample_error = torch.linalg.norm(pred_aligned - gt_sample, dim=1).mean()
        pa_mpjpe_errors[i] = sample_error

    return pa_mpjpe_errors.mean()


def get_activation(name):
    """Get activation function by name"""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == "mish":
        return nn.Mish(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_normalization(name, channels):
    """Get normalization layer by name"""
    if name == "batch":
        return nn.BatchNorm2d(channels)
    elif name == "instance":
        return nn.InstanceNorm2d(channels)
    elif name == "layer":
        return nn.GroupNorm(1, channels)
    elif name == "group":
        return nn.GroupNorm(min(32, channels), channels)
    else:
        return nn.BatchNorm2d(channels)
