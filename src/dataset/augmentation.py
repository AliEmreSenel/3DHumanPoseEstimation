import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple
import math


class PoseAugmentor:
    """
    Augmentation class for 3D human pose estimation.
    Applies consistent transformations to images, depth maps, 2D keypoints, and 3D joint positions.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-30, 30),  # Rotation range in degrees
        flip_prob: float = 0.5,  # Probability of horizontal flipping
        scale_range: Tuple[float, float] = (0.8, 1.2),  # Scale range
        translate_range: Tuple[float, float] = (
            -0.1,
            0.1,
        ),  # Translation range as fraction of image size
        brightness_range: Tuple[float, float] = (
            0.8,
            1.2,
        ),  # Brightness adjustment range
        contrast_range: Tuple[float, float] = (0.8, 1.2),  # Contrast adjustment range
        enable_rotation: bool = True,
        enable_flip: bool = True,
        enable_scale: bool = True,
        enable_translate: bool = True,
        enable_color: bool = True,
    ):
        """
        Initialize the augmentor with specified ranges and enabled transformations.
        """
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        self.enable_rotation = enable_rotation
        self.enable_flip = enable_flip
        self.enable_scale = enable_scale
        self.enable_translate = enable_translate
        self.enable_color = enable_color

    def _create_rotation_matrix(self, angle_degrees: float) -> np.ndarray:
        """
        Create a 3D rotation matrix for rotation around the Y-axis (vertical).

        Args:
            angle_degrees: Rotation angle in degrees

        Returns:
            3x3 rotation matrix
        """
        angle_rad = math.radians(angle_degrees)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Rotation around Y-axis (vertical axis)
        rotation_matrix = np.array(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )

        return rotation_matrix

    def _rotate_3d_joints(
        self, joints_3d: np.ndarray, rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Rotate 3D joint positions using the given rotation matrix.

        Args:
            joints_3d: [num_joints, 3] array of 3D joint positions
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Rotated 3D joint positions
        """
        # Apply rotation to each joint
        rotated_joints = np.dot(joints_3d, rotation_matrix.T)
        return rotated_joints

    def _project_3d_to_2d(
        self, joints_3d: np.ndarray, camera_params: Dict
    ) -> np.ndarray:
        """
        Project 3D joint positions to 2D pixel coordinates.

        Args:
            joints_3d: [num_joints, 3] array of 3D joint positions in camera coordinates
            camera_params: Camera parameters including focal length and principal point

        Returns:
            [num_joints, 2] array of 2D pixel coordinates
        """
        f = camera_params["f"]  # Focal length [fx, fy]
        c = camera_params["c"]  # Principal point [cx, cy]

        # Project each joint
        joints_2d = np.zeros((joints_3d.shape[0], 2))
        for i, joint in enumerate(joints_3d):
            x, y, z = joint
            if z > 0:  # Avoid division by zero or negative z
                px = (x * f[0] / z) + c[0]
                py = (y * f[1] / z) + c[1]
                joints_2d[i] = [px, py]
            else:
                # If joint is behind camera, use a fallback value
                joints_2d[i] = [-1, -1]  # Invalid coordinates

        return joints_2d

    def _normalize_2d_keypoints(
        self, keypoints_2d: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Normalize 2D keypoints to [0, 1] range.

        Args:
            keypoints_2d: [num_joints, 2] array of 2D pixel coordinates
            image_size: (width, height) of the image

        Returns:
            Normalized 2D keypoints
        """
        width, height = image_size
        normalized = keypoints_2d.copy()
        normalized[:, 0] = normalized[:, 0] / width
        normalized[:, 1] = normalized[:, 1] / height
        return normalized

    def _flip_horizontal(
        self,
        image: Image.Image,
        depth: Image.Image,
        joints_3d: np.ndarray,
        keypoints_2d: np.ndarray,
        symmetric_joints: List[Tuple[int, int]],
    ) -> Tuple:
        """
        Flip image, depth, 3D joints, and 2D keypoints horizontally.

        Args:
            image: RGB image
            depth: Depth image
            joints_3d: 3D joint positions
            keypoints_2d: 2D keypoint coordinates (normalized [0,1])
            symmetric_joints: List of symmetric joint pairs (left, right)

        Returns:
            Tuple of flipped (image, depth, joints_3d, keypoints_2d)
        """
        # Flip images
        flipped_image = TF.hflip(image)
        flipped_depth = TF.hflip(depth)

        # Flip 3D joints (negate x-coordinate)
        flipped_joints_3d = joints_3d.copy()
        flipped_joints_3d[:, 0] = -flipped_joints_3d[:, 0]

        # Flip 2D keypoints (invert x-coordinate)
        flipped_keypoints_2d = keypoints_2d.copy()
        flipped_keypoints_2d[:, 0] = 1.0 - flipped_keypoints_2d[:, 0]

        # Swap symmetric joints
        for left_idx, right_idx in symmetric_joints:
            flipped_joints_3d[[left_idx, right_idx]] = flipped_joints_3d[
                [right_idx, left_idx]
            ]
            flipped_keypoints_2d[[left_idx, right_idx]] = flipped_keypoints_2d[
                [right_idx, left_idx]
            ]

        return flipped_image, flipped_depth, flipped_joints_3d, flipped_keypoints_2d

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply augmentations to a sample.

        Args:
            sample: Dictionary containing image, depth, keypoints_2d, joints_3d, etc.

        Returns:
            Augmented sample
        """
        # Make a copy of the sample to avoid modifying the original
        augmented = sample.copy()

        # Convert tensors to numpy/PIL for augmentation
        if isinstance(sample["image"], torch.Tensor):
            image = TF.to_pil_image(sample["image"])
        else:
            image = sample["image"]

        if isinstance(sample["depth"], torch.Tensor):
            # Convert single-channel depth to PIL
            depth_np = sample["depth"].squeeze(0).numpy()
            depth = Image.fromarray((depth_np * 255).astype(np.uint8))
        else:
            depth = sample["depth"]

        if isinstance(sample["joints_3d"], torch.Tensor):
            joints_3d = sample["joints_3d"].numpy()
        else:
            joints_3d = sample["joints_3d"]

        if isinstance(sample["keypoints_2d"], torch.Tensor):
            keypoints_2d = sample["keypoints_2d"].numpy()
        else:
            keypoints_2d = sample["keypoints_2d"]

        camera_params = sample["camera_params"]
        image_size = (image.width, image.height)

        # Apply horizontal flipping with probability
        if self.enable_flip and np.random.random() < self.flip_prob:
            # Get symmetric joints from config or use default
            symmetric_joints = sample.get(
                "symmetric_joints",
                [
                    (1, 4),  # Right/Left Hip
                    (2, 5),  # Right/Left Knee
                    (3, 6),  # Right/Left Ankle
                    (11, 14),  # Left/Right Shoulder
                    (12, 15),  # Left/Right Elbow
                    (13, 16),  # Left/Right Wrist
                ],
            )

            image, depth, joints_3d, keypoints_2d = self._flip_horizontal(
                image, depth, joints_3d, keypoints_2d, symmetric_joints
            )

        # Apply rotation
        if self.enable_rotation:
            # Sample rotation angle
            angle_degrees = np.random.uniform(
                self.rotation_range[0], self.rotation_range[1]
            )

            # Create rotation matrix for 3D joints
            rotation_matrix = self._create_rotation_matrix(angle_degrees)

            # Rotate 3D joints
            joints_3d = self._rotate_3d_joints(joints_3d, rotation_matrix)

            # Rotate images
            image = TF.rotate(
                image, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR
            )
            depth = TF.rotate(
                depth, angle_degrees, interpolation=TF.InterpolationMode.NEAREST
            )

            # Recalculate 2D keypoints from rotated 3D joints
            keypoints_2d_pixels = self._project_3d_to_2d(joints_3d, camera_params)
            keypoints_2d = self._normalize_2d_keypoints(keypoints_2d_pixels, image_size)

        # Apply scaling
        if self.enable_scale:
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

            # Scale images
            new_size = (
                int(image.width * scale_factor),
                int(image.height * scale_factor),
            )
            image = TF.resize(
                image, new_size, interpolation=TF.InterpolationMode.BILINEAR
            )
            depth = TF.resize(
                depth, new_size, interpolation=TF.InterpolationMode.NEAREST
            )

            # No need to adjust 3D joints for scaling
            # But we need to update camera parameters for 2D projection
            scaled_camera_params = camera_params.copy()
            scaled_camera_params["f"] = [f * scale_factor for f in camera_params["f"]]
            scaled_camera_params["c"] = [c * scale_factor for c in camera_params["c"]]

            # Recalculate 2D keypoints with scaled camera parameters
            keypoints_2d_pixels = self._project_3d_to_2d(
                joints_3d, scaled_camera_params
            )
            keypoints_2d = self._normalize_2d_keypoints(
                keypoints_2d_pixels, (new_size[0], new_size[1])
            )

            # Update camera parameters in the sample
            augmented["camera_params"] = scaled_camera_params

        # Apply translation
        if self.enable_translate:
            # Sample translation offsets as fraction of image size
            tx = (
                np.random.uniform(self.translate_range[0], self.translate_range[1])
                * image.width
            )
            ty = (
                np.random.uniform(self.translate_range[0], self.translate_range[1])
                * image.height
            )

            # Translate images
            image = TF.affine(image, angle=0, translate=[tx, ty], scale=1.0, shear=0)
            depth = TF.affine(depth, angle=0, translate=[tx, ty], scale=1.0, shear=0)

            # Adjust 2D keypoints for translation
            keypoints_2d_unnorm = keypoints_2d.copy()
            keypoints_2d_unnorm[:, 0] *= image.width
            keypoints_2d_unnorm[:, 1] *= image.height

            keypoints_2d_unnorm[:, 0] += tx
            keypoints_2d_unnorm[:, 1] += ty

            # Renormalize
            keypoints_2d = self._normalize_2d_keypoints(
                keypoints_2d_unnorm, (image.width, image.height)
            )

        # Apply color augmentation (only to RGB image)
        if self.enable_color:
            # Brightness adjustment
            brightness_factor = np.random.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            image = TF.adjust_brightness(image, brightness_factor)

            # Contrast adjustment
            contrast_factor = np.random.uniform(
                self.contrast_range[0], self.contrast_range[1]
            )
            image = TF.adjust_contrast(image, contrast_factor)

        # Convert back to tensors
        augmented["image"] = TF.to_tensor(image)

        # Convert depth back to tensor with proper scaling
        depth_np = np.array(depth).astype(np.float32) / 255.0
        augmented["depth"] = torch.from_numpy(depth_np).unsqueeze(0)

        augmented["joints_3d"] = torch.from_numpy(joints_3d.astype(np.float32))
        augmented["keypoints_2d"] = torch.from_numpy(keypoints_2d.astype(np.float32))

        return augmented
