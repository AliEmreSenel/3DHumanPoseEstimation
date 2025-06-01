import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from config import CONNECTIONS
import torch


def visualize_3d_pose(joints_3d, title="3D Pose"):
    """
    Create a 3D visualization of the pose with safeguards against NaN/Inf values

    Args:
        joints_3d: (17, 3) array of joint positions
        title: Title for the plot

    Returns:
        fig: matplotlib figure
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2] = (
        joints_3d[:, 0].copy(),
        joints_3d[:, 2].copy(),
        -joints_3d[:, 1].copy(),
    )


    # Check for NaN/Inf values
    has_nan = np.isnan(joints_3d).any() or np.isinf(joints_3d).any()
    if not has_nan:

        # Plot joints
        ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], color="red", s=50)

        # Plot connections
        for start, end in CONNECTIONS:
            xs = [joints_3d[start, 0], joints_3d[end, 0]]
            ys = [joints_3d[start, 1], joints_3d[end, 1]]
            zs = [joints_3d[start, 2], joints_3d[end, 2]]

            # Check if this specific connection has NaN/Inf
            if not (
                np.isnan(xs).any()
                or np.isnan(ys).any()
                or np.isnan(zs).any()
                or np.isinf(xs).any()
                or np.isinf(ys).any()
                or np.isinf(zs).any()
            ):
                ax.plot(xs, ys, zs, color="blue", linewidth=2)
    else:
        ax.text(
            0,
            0,
            0,
            "Invalid pose\n(NaN/Inf values)",
            ha="center",
            va="center",
            fontsize=12,
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio with safety checks
    if not has_nan:
        try:
            # Calculate ranges with safety checks
            x_range = joints_3d[:, 0].max() - joints_3d[:, 0].min()
            y_range = joints_3d[:, 1].max() - joints_3d[:, 1].min()
            z_range = joints_3d[:, 2].max() - joints_3d[:, 2].min()

            # Check if ranges are valid
            if (
                np.isnan(x_range)
                or np.isnan(y_range)
                or np.isnan(z_range)
                or np.isinf(x_range)
                or np.isinf(y_range)
                or np.isinf(z_range)
                or x_range <= 0
                or y_range <= 0
                or z_range <= 0
            ):
                # Use default range
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
            else:
                max_range = np.array([x_range, y_range, z_range]).max() / 2.0
                mid_x = (joints_3d[:, 0].max() + joints_3d[:, 0].min()) * 0.5
                mid_y = (joints_3d[:, 1].max() + joints_3d[:, 1].min()) * 0.5
                mid_z = (joints_3d[:, 2].max() + joints_3d[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

        except Exception as e:
            print(f"Error setting pose limits: {e}")
            # Use default range
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
    else:
        # Use default range for invalid data
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    return fig


def visualize_comparison(image, pred_joints, gt_joints, title="Pose Comparison"):
    """
    Create a visualization comparing the predicted and ground truth poses
    with safeguards against NaN/Inf values

    Args:
        image: RGB image tensor
        pred_joints: (17, 3) array of predicted joint positions
        gt_joints: (17, 3) array of ground truth joint positions
        title: Title for the plot

    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=(20, 10))

    # Convert image tensor to numpy array for visualization
    if isinstance(image, torch.Tensor):
        # Denormalize the image
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)

    # Plot the image
    ax1 = fig.add_subplot(131)
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Check for NaN/Inf values in predicted joints
    has_nan_pred = np.isnan(pred_joints).any() or np.isinf(pred_joints).any()

    # Swap the Y and Z axes for visualization
    pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2] = (
        pred_joints[:, 0].copy(),
        pred_joints[:, 2].copy(),
        -pred_joints[:, 1].copy(),
    )

    gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2] = (
        gt_joints[:, 0].copy(),
        gt_joints[:, 2].copy(),
        -gt_joints[:, 1].copy(),
    )

    # Plot the predicted 3D pose
    ax2 = fig.add_subplot(132, projection="3d")

    if not has_nan_pred:
        ax2.scatter(
            pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], color="red", s=50
        )
        for start, end in CONNECTIONS:
            xs = [pred_joints[start, 0], pred_joints[end, 0]]
            ys = [pred_joints[start, 1], pred_joints[end, 1]]
            zs = [pred_joints[start, 2], pred_joints[end, 2]]
            # Check if this specific connection has NaN/Inf
            if not (
                np.isnan(xs).any()
                or np.isnan(ys).any()
                or np.isnan(zs).any()
                or np.isinf(xs).any()
                or np.isinf(ys).any()
                or np.isinf(zs).any()
            ):
                ax2.plot(xs, ys, zs, color="blue", linewidth=2)
    else:
        ax2.text(
            0,
            0,
            0,
            "Invalid prediction\n(NaN/Inf values)",
            ha="center",
            va="center",
            fontsize=12,
        )

    ax2.set_title("Predicted Pose")

    # Set equal aspect ratio for predicted pose with safety checks
    if not has_nan_pred:
        try:
            # Calculate ranges with safety checks
            x_range = pred_joints[:, 0].max() - pred_joints[:, 0].min()
            y_range = pred_joints[:, 1].max() - pred_joints[:, 1].min()
            z_range = pred_joints[:, 2].max() - pred_joints[:, 2].min()

            # Check if ranges are valid
            if (
                np.isnan(x_range)
                or np.isnan(y_range)
                or np.isnan(z_range)
                or np.isinf(x_range)
                or np.isinf(y_range)
                or np.isinf(z_range)
                or x_range <= 0
                or y_range <= 0
                or z_range <= 0
            ):
                # Use default range
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_zlim(-1, 1)
            else:
                max_range_pred = np.array([x_range, y_range, z_range]).max() / 2.0
                mid_x_pred = (pred_joints[:, 0].max() + pred_joints[:, 0].min()) * 0.5
                mid_y_pred = (pred_joints[:, 1].max() + pred_joints[:, 1].min()) * 0.5
                mid_z_pred = (pred_joints[:, 2].max() + pred_joints[:, 2].min()) * 0.5

                ax2.set_xlim(mid_x_pred - max_range_pred, mid_x_pred + max_range_pred)
                ax2.set_ylim(mid_y_pred - max_range_pred, mid_y_pred + max_range_pred)
                ax2.set_zlim(mid_z_pred - max_range_pred, mid_z_pred + max_range_pred)
        except Exception as e:
            print(f"Error setting predicted pose limits: {e}")
            # Use default range
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(-1, 1)
            ax2.set_zlim(-1, 1)
    else:
        # Use default range for invalid data
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)

    # Check for NaN/Inf values in ground truth joints
    has_nan_gt = np.isnan(gt_joints).any() or np.isinf(gt_joints).any()

    # Plot the ground truth 3D pose
    ax3 = fig.add_subplot(133, projection="3d")

    if not has_nan_gt:
        ax3.scatter(
            gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2], color="green", s=50
        )
        for start, end in CONNECTIONS:
            xs = [gt_joints[start, 0], gt_joints[end, 0]]
            ys = [gt_joints[start, 1], gt_joints[end, 1]]
            zs = [gt_joints[start, 2], gt_joints[end, 2]]
            # Check if this specific connection has NaN/Inf
            if not (
                np.isnan(xs).any()
                or np.isnan(ys).any()
                or np.isnan(zs).any()
                or np.isinf(xs).any()
                or np.isinf(ys).any()
                or np.isinf(zs).any()
            ):
                ax3.plot(xs, ys, zs, color="blue", linewidth=2)
    else:
        ax3.text(
            0,
            0,
            0,
            "Invalid ground truth\n(NaN/Inf values)",
            ha="center",
            va="center",
            fontsize=12,
        )

    ax3.set_title("Ground Truth Pose")

    # Set equal aspect ratio for ground truth pose with safety checks
    if not has_nan_gt:
        try:
            # Calculate ranges with safety checks
            x_range = gt_joints[:, 0].max() - gt_joints[:, 0].min()
            y_range = gt_joints[:, 1].max() - gt_joints[:, 1].min()
            z_range = gt_joints[:, 2].max() - gt_joints[:, 2].min()

            # Check if ranges are valid
            if (
                np.isnan(x_range)
                or np.isnan(y_range)
                or np.isnan(z_range)
                or np.isinf(x_range)
                or np.isinf(y_range)
                or np.isinf(z_range)
                or x_range <= 0
                or y_range <= 0
                or z_range <= 0
            ):
                # Use default range
                ax3.set_xlim(-1, 1)
                ax3.set_ylim(-1, 1)
                ax3.set_zlim(-1, 1)
            else:
                max_range_gt = np.array([x_range, y_range, z_range]).max() / 2.0
                mid_x_gt = (gt_joints[:, 0].max() + gt_joints[:, 0].min()) * 0.5
                mid_y_gt = (gt_joints[:, 1].max() + gt_joints[:, 1].min()) * 0.5
                mid_z_gt = (gt_joints[:, 2].max() + gt_joints[:, 2].min()) * 0.5

                ax3.set_xlim(mid_x_gt - max_range_gt, mid_x_gt + max_range_gt)
                ax3.set_ylim(mid_y_gt - max_range_gt, mid_y_gt + max_range_gt)
                ax3.set_zlim(mid_z_gt - max_range_gt, mid_z_gt + max_range_gt)
        except Exception as e:
            print(f"Error setting ground truth pose limits: {e}")
            # Use default range
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.set_zlim(-1, 1)
    else:
        # Use default range for invalid data
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def fig_to_image(fig):
    """Convert a matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    return img
