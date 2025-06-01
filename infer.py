import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import argparse
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import io

from models.cnn import CNNPoseEstimation
from models.transformers import TransformerPoseEstimation
from model_config import ModelConfig

from visualize import visualize_3d_pose, fig_to_image
from config import DEVICE, NUM_JOINTS as MODEL_NUM_JOINTS

from ultralytics import YOLO
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Inference")

# Suppress ultralytics logs, similar to preprocess.py
logging.getLogger("ultralytics").setLevel(logging.ERROR)

VIZ_THUMBNAIL_SIZE = (500, 500)  # Size for each panel in the combined viz
MODEL_INPUT_SIZE = (
    500,
    500,
)  # Default input size for the model, can be overridden by model config

# Define transformations for the input image
image_transform = transforms.Compose(
    [
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
    ]
)

depth_transform = transforms.Compose(
    [
        transforms.Resize(MODEL_INPUT_SIZE),
    ]
)


def load_pose_model(checkpoint_path, model_type, num_joints=MODEL_NUM_JOINTS):
    """Loads the pre-trained 3D pose estimation model."""
    global MODEL_INPUT_SIZE
    logger.info(f"Loading 3D pose model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model_type = checkpoint.get("model_type", model_type)
        model_args = checkpoint.get("model_args", {})

        # Determine the actual state dictionary
        actual_model_state_dict = checkpoint.get("model_state_dict")
        if actual_model_state_dict is None:
            # If 'model_state_dict' key is missing, assume the checkpoint IS the state_dict
            actual_model_state_dict = checkpoint
            if not isinstance(actual_model_state_dict, dict):
                # Ensure it's a dictionary; otherwise, the format is unexpected.
                raise ValueError(
                    "Checkpoint file does not appear to be a valid state_dict or contain a 'model_state_dict' key."
                )

        # Remove "module." prefix which is often added by DataParallel or DDP
        final_state_dict = {
            k.replace("module.", ""): v for k, v in actual_model_state_dict.items()
        }

        model_cfg = ModelConfig(model_type=model_type, **model_args)

        MODEL_INPUT_SIZE = model_cfg.image_size

        image_transform.transforms[0] = transforms.Resize(MODEL_INPUT_SIZE)
        depth_transform.transforms[0] = transforms.Resize(MODEL_INPUT_SIZE)

        if model_type == "transformer":
            pose_model = TransformerPoseEstimation(config=model_cfg)
        elif model_type == "cnn":
            pose_model = CNNPoseEstimation(config=model_cfg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            pose_model.load_state_dict(final_state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(
                f"Failed to load state_dict strictly (error: {e}). Trying with strict=False."
            )
            pose_model.load_state_dict(final_state_dict, strict=False)

        logger.info("3D Pose Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}.")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

    pose_model.to(DEVICE)
    pose_model.eval()
    return pose_model


def load_preprocessing_models(yolo_path):
    """Loads YOLO and DepthPro models."""
    logger.info("Loading preprocessing models (YOLO and DepthPro)...")
    try:
        yolo_model = YOLO(
            yolo_path
        )  # verbose=False is handled in the call to yolo_model
        logger.info(f"YOLO pose model loaded from {yolo_path}.")
        depth_processor = DepthProImageProcessorFast.from_pretrained(
            "apple/DepthPro-hf"
        )
        depth_model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf"
        ).to(DEVICE)
        depth_model.eval()
        logger.info("DepthPro model loaded.")
    except Exception as e:
        logger.error(f"Error loading preprocessing models: {e}")
        raise
    return yolo_model, depth_processor, depth_model


def get_2d_keypoints(yolo_model, image_pil, confidence_threshold=0.3):
    """
    Estimates 2D keypoints using the YOLO model, using .xy and .conf attributes.
    Returns a tensor of shape (1, MODEL_NUM_JOINTS, 3) with (norm_x, norm_y, confidence).
    Padded/truncated to MODEL_NUM_JOINTS. Invalid/padded keypoints have conf=0.
    """
    results = yolo_model(image_pil, device=DEVICE, verbose=False)
    processed_persons_keypoints = []
    img_w, img_h = image_pil.size

    for res_idx, res in enumerate(results):
        if res.keypoints is None:
            continue

        kpts_xy = res.keypoints.xy  # Pixel coordinates
        kpts_conf = res.keypoints.conf  # Confidences

        if (
            kpts_xy is None
            or kpts_conf is None
            or kpts_xy.nelement() == 0
            or kpts_conf.nelement() == 0
        ):
            continue

        kpts_xy = kpts_xy.to(DEVICE)
        kpts_conf = kpts_conf.to(DEVICE)

        if kpts_xy.ndim == 2:
            kpts_xy = kpts_xy.unsqueeze(0)
        if kpts_conf.ndim == 1:
            kpts_conf = kpts_conf.unsqueeze(0)

        if (
            kpts_xy.shape[0] != kpts_conf.shape[0]
            or kpts_xy.shape[1] != kpts_conf.shape[1]
        ):
            logger.warning(
                f"Shape mismatch between keypoints xy ({kpts_xy.shape}) and conf ({kpts_conf.shape}). Skipping person."
            )
            continue

        num_persons = kpts_xy.shape[0]

        for person_idx in range(num_persons):
            person_xy_tensor = kpts_xy[person_idx]
            person_conf_tensor = kpts_conf[person_idx]

            person_result_kpts_with_conf = torch.zeros(
                (MODEL_NUM_JOINTS, 3), device=DEVICE
            )

            num_detected_kpts_yolo = person_xy_tensor.shape[0]
            valid_kpt_idx = 0
            for yolo_kpt_idx in range(num_detected_kpts_yolo):
                if valid_kpt_idx >= MODEL_NUM_JOINTS:
                    break

                x_pixel, y_pixel = person_xy_tensor[yolo_kpt_idx]
                conf = person_conf_tensor[yolo_kpt_idx]

                norm_x = x_pixel / img_w
                norm_y = y_pixel / img_h
                person_result_kpts_with_conf[valid_kpt_idx, 0] = norm_x
                person_result_kpts_with_conf[valid_kpt_idx, 1] = norm_y
                person_result_kpts_with_conf[valid_kpt_idx, 2] = conf
                valid_kpt_idx += 1

            processed_persons_keypoints.append(person_result_kpts_with_conf)
            if processed_persons_keypoints:  # Take the first valid person
                return processed_persons_keypoints[0].unsqueeze(0).to(DEVICE)

    if not processed_persons_keypoints:
        logger.warning(
            "No 2D keypoints detected or none passed the confidence threshold for any person."
        )
        return torch.zeros(
            (1, MODEL_NUM_JOINTS, 3), device=DEVICE  # Return zeroed tensor
        )

    # Should have returned in the loop if a person was processed
    return processed_persons_keypoints[0].unsqueeze(0).to(DEVICE)


def get_depth_map(depth_processor, depth_model, image_pil):
    """Estimates depth map using DepthPro. Output is normalized to [0,1]."""
    img_w, img_h = image_pil.size
    inputs = depth_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    # Post-process to get the depth map, typically in a relative scale
    post_processed = depth_processor.post_process_depth_estimation(
        outputs, target_sizes=[(img_h, img_w)]  # Ensure original image dimensions
    )
    depth_tensor = post_processed[0]["predicted_depth"]  # This is [H, W]

    return depth_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)


def create_depth_viz(depth_tensor_normalized_0_1):
    """Creates a colored PIL image from a depth tensor"""
    # Squeeze batch and channel dimensions, convert to numpy
    depth_numpy = depth_tensor_normalized_0_1.squeeze().cpu().numpy()

    plt.figure(figsize=(5, 5))  # Matplotlib figure for visualization
    plt.imshow(depth_numpy, cmap="viridis")  # Use a colormap
    plt.axis("off")  # No axes for a clean image
    plt.tight_layout(pad=0)  # Remove padding

    buf = io.BytesIO()  # In-memory buffer
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    depth_pil = Image.open(buf)  # Create PIL image from buffer
    plt.close()  # Close the figure to free memory

    depth_pil.thumbnail(VIZ_THUMBNAIL_SIZE)  # Resize for consistent viz panel size
    return depth_pil


def create_2d_kpts_viz(image_pil, keypoints_normalized_with_conf):
    """
    Creates a PIL image with 2D keypoints overlaid.
    keypoints_normalized_with_conf: Tensor of shape (1, MODEL_NUM_JOINTS, 3) -> norm_x, norm_y, conf
    """
    img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv2.shape[:2]

    # Squeeze batch dimension, convert to numpy: Shape [MODEL_NUM_JOINTS, 3]
    kpts_data = keypoints_normalized_with_conf.squeeze(0).cpu().numpy()

    # Denormalize x, y coordinates to pixel values
    kpts_pixels = np.zeros((kpts_data.shape[0], 2), dtype=int)
    kpts_pixels[:, 0] = (kpts_data[:, 0] * w).astype(int)
    kpts_pixels[:, 1] = (kpts_data[:, 1] * h).astype(int)
    confidences = kpts_data[:, 2]

    # Draw keypoints
    for i in range(len(kpts_pixels)):
        conf = confidences[i]
        if conf > 0:  # Draw only if confidence is positive (valid keypoint)
            x, y = kpts_pixels[i, 0], kpts_pixels[i, 1]
            cv2.circle(img_cv2, (x, y), 5, (0, 0, 255), -1)  # Red dots

    # Draw connections (bones)
    # CONNECTIONS is assumed to be a list of tuples (start_idx, end_idx)
    # defined according to MODEL_NUM_JOINTS indexing
    for start_idx, end_idx in CONNECTIONS:
        # Ensure indices are within bounds for the current keypoints array
        if start_idx < len(kpts_pixels) and end_idx < len(kpts_pixels):
            start_conf = confidences[start_idx]
            end_conf = confidences[end_idx]

            # Draw line only if both connected keypoints are valid
            if start_conf > 0 and end_conf > 0:
                start_point = tuple(kpts_pixels[start_idx])
                end_point = tuple(kpts_pixels[end_idx])
                cv2.line(img_cv2, start_point, end_point, (0, 255, 0), 2)  # Green lines

    img_2d_kpts_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    img_2d_kpts_pil.thumbnail(VIZ_THUMBNAIL_SIZE)
    return img_2d_kpts_pil


def preprocess_input(
    image_pil, yolo_model, depth_processor, depth_model, yolo_conf_thresh=0.3
):
    """
    Preprocesses a single image.
    Returns:
        transformed_image: Tensor for the 3D model's image input.
        transformed_depth: Tensor for the 3D model's depth input (normalized [0,1] and resized).
        keypoints_2d_for_model: Tensor (Batch, MODEL_NUM_JOINTS, 2) for 3D model (norm_x, norm_y).
        keypoints_2d_for_viz: Tensor (Batch, MODEL_NUM_JOINTS, 3) for visualization (norm_x, norm_y, conf).
        depth_map_for_viz: Raw depth map (normalized [0,1], original size) for visualization.
    """
    try:
        keypoints_2d_with_conf = get_2d_keypoints(
            yolo_model, image_pil.copy(), confidence_threshold=yolo_conf_thresh
        )

        if (
            keypoints_2d_with_conf is None
        ):  # Should not happen with current get_2d_keypoints
            logger.error("get_2d_keypoints returned None, which is unexpected.")
            return None, None, None, None, None

        if not torch.any(
            keypoints_2d_with_conf[:, :, 2] > 0
        ):  # Check if any valid kpts found
            logger.warning(
                "No keypoints passed the threshold during 2D pose estimation."
            )
            # Continue processing, model might handle zeroed keypoints.

        keypoints_2d_for_model = keypoints_2d_with_conf[:, :, :2].clone()

        # depth_map_for_viz is normalized [0,1] and at original image resolution (but B,C,H,W)
        depth_map_for_viz = get_depth_map(
            depth_processor, depth_model, image_pil.copy()
        )
        if depth_map_for_viz is None:  # Should not happen
            logger.error("get_depth_map returned None, which is unexpected.")
            return None, None, None, None, None

        transformed_image = image_transform(image_pil.copy()).unsqueeze(0).to(DEVICE)

        # transformed_depth is the [0,1] normalized depth map, resized to model input size
        transformed_depth = torch.nn.functional.interpolate(
            depth_map_for_viz,
            size=MODEL_INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        ).to(DEVICE)

        return (
            transformed_image,
            transformed_depth,
            keypoints_2d_for_model,
            keypoints_2d_with_conf,  # For 2D kpts visualization
            depth_map_for_viz,  # For depth map visualization
        )

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return None, None, None, None, None


def run_inference(pose_model, image_tensor, depth_tensor, keypoints_2d_tensor):
    """Runs inference. keypoints_2d_tensor is (Batch, NumJoints, 2)"""
    try:
        with torch.no_grad():
            predicted_joints_3d_batch = pose_model(
                image_tensor, depth_tensor, keypoints_2d_tensor
            )
        return predicted_joints_3d_batch[0].cpu().numpy()  # Return first item in batch
    except Exception as e:
        logger.error(f"Error during model inference: {e}", exc_info=True)
        return None


def main(args):
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        pose_3d_model = load_pose_model(
            args.checkpoint_path, args.model_type, args.num_joints
        )
        yolo_model, depth_processor, depth_model = load_preprocessing_models(
            args.yolo_model_path
        )
    except Exception as e:
        logger.error(f"Failed to load models: {e}. Exiting.")
        return

    input_folder_path = Path(args.input_folder)
    image_files = sorted(
        [
            f
            for f in input_folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )

    if not image_files:
        logger.warning(f"No images found in {args.input_folder}")
        return

    for image_file_path in image_files:
        logger.info(f"Processing image: {image_file_path.name}")
        try:
            img_pil = Image.open(image_file_path).convert("RGB")
        except Exception as e:
            logger.error(f"Could not open image {image_file_path.name}: {e}")
            continue

        processed_data = preprocess_input(
            img_pil,
            yolo_model,
            depth_processor,
            depth_model,
            yolo_conf_thresh=args.yolo_confidence_threshold,
        )

        if processed_data[0] is None:  # Check if preprocessing failed
            logger.warning(
                f"Skipping {image_file_path.name} due to preprocessing failure."
            )
            continue

        (
            image_tensor,
            depth_tensor_for_model,  # Renamed for clarity
            keypoints_2d_for_model,
            keypoints_2d_for_viz,
            depth_map_for_viz,  # Renamed for clarity
        ) = processed_data

        predicted_joints = run_inference(
            pose_3d_model, image_tensor, depth_tensor_for_model, keypoints_2d_for_model
        )

        if predicted_joints is None:
            logger.warning(f"Skipping {image_file_path.name} due to inference failure.")
            continue

        npy_output_path = output_path / f"{image_file_path.stem}_pred_joints3d.npy"
        np.save(npy_output_path, predicted_joints)
        logger.info(f"Saved predicted 3D joints to {npy_output_path}")

        if args.visualize:
            try:
                img_orig_viz = img_pil.copy()
                img_orig_viz.thumbnail(VIZ_THUMBNAIL_SIZE)

                img_2d_viz = create_2d_kpts_viz(
                    img_pil.copy(), keypoints_2d_for_viz.clone()
                )

                # Use the depth_map_for_viz (normalized [0,1], original aspect ratio)
                img_depth_viz = create_depth_viz(depth_map_for_viz.clone())

                fig_3d = visualize_3d_pose(
                    predicted_joints.copy(), title="Predicted 3D Pose"
                )
                img_3d_viz = fig_to_image(
                    fig_3d
                )  # Converts matplotlib fig to PIL Image
                img_3d_viz.thumbnail(VIZ_THUMBNAIL_SIZE)
                plt.close(fig_3d)  # Close figure after converting

                w_thumb, h_thumb = VIZ_THUMBNAIL_SIZE
                combined_img = Image.new(
                    "RGB",
                    (w_thumb * 2, h_thumb * 2),
                    (255, 255, 255),  # White background
                )
                combined_img.paste(img_orig_viz, (0, 0))
                combined_img.paste(img_2d_viz, (w_thumb, 0))
                combined_img.paste(img_depth_viz, (0, h_thumb))
                combined_img.paste(img_3d_viz, (w_thumb, h_thumb))

                combined_output_path = (
                    output_path / f"{image_file_path.stem}_combined_viz.png"
                )
                combined_img.save(combined_output_path)
                logger.info(f"Saved combined visualization to {combined_output_path}")

            except Exception as e:
                logger.error(
                    f"Failed to create or save visualization for {image_file_path.name}: {e}",
                    exc_info=True,
                )

    logger.info("Inference processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D Human Pose Estimation Inference"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cnn", "transformer"],
        help="Model type: 'cnn' or 'transformer'",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="inference_output",
        help="Folder to save results.",
    )
    parser.add_argument(
        "--yolo_model_path",
        type=str,
        default="yolo11x-pose.pt",
        help="Path to the YOLO pose model (e.g. yolo11x-pose.pt).",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=MODEL_NUM_JOINTS,  # From config.py
        help="Number of joints expected by the 3D pose model.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable saving of visualizations."
    )
    parser.add_argument(
        "--yolo_confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for YOLO 2D keypoint detection.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
