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

# Import from your project's modules
from models.cnn import CNNPoseEstimation
from models.transformers import TransformerPoseEstimation
from model_config import ModelConfig

from visualize import visualize_3d_pose, fig_to_image, CONNECTIONS
from config import DEVICE, NUM_JOINTS as MODEL_NUM_JOINTS

# For preprocessing (Using models from preprocess.py)
from ultralytics import YOLO
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Inference")

# Define the expected input size for the model
MODEL_INPUT_SIZE = (500, 500)
VIZ_THUMBNAIL_SIZE = (400, 400)  # Size for each panel in the combined viz

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


def load_pose_model(checkpoint_path, num_joints=MODEL_NUM_JOINTS):
    """Loads the pre-trained 3D pose estimation model."""
    logger.info(f"Loading 3D pose model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model_type = checkpoint.get("model_type", "cnn").lower()
        model_args = checkpoint.get("model_args", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_cfg = ModelConfig(model_type=model_type, **model_args)
        if model_type == "transformer":
            pose_model = TransformerPoseEstimation(config=model_cfg)
        elif model_type == "cnn":
            pose_model = CNNPoseEstimation(config=model_cfg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        pose_model.load_state_dict(new_state_dict)

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
        yolo_model = YOLO(yolo_path)
        # yolo_model.to(DEVICE) # YOLO model handles device internally based on input
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
    Estimates 2D keypoints using the YOLO model.
    Returns a tensor of shape (1, MODEL_NUM_JOINTS, 3) with (norm_x, norm_y, confidence).
    Padded/truncated to MODEL_NUM_JOINTS. Invalid/padded keypoints have conf=0.
    """
    # YOLO model runs on CPU or GPU based on its own setup or input tensor device
    # Forcing device here for results if yolo_model itself doesn't handle it well with PIL/numpy inputs
    results = yolo_model(image_pil, device=DEVICE, verbose=False)

    processed_persons_keypoints = []

    img_w, img_h = image_pil.size

    for res_idx, res in enumerate(results):
        if res.keypoints is None or len(res.keypoints.data) == 0:
            continue

        # res.keypoints.data for Ultralytics YOLO is typically (num_persons, num_keypoints_yolo, 3) -> x, y, conf
        # If single image and single person, it might be (num_keypoints_yolo, 3)

        keypoints_data_for_image = res.keypoints.data.to(
            DEVICE
        )  # Ensure tensor is on the correct device

        # Handle if keypoints_data_for_image is (num_kpts, 3) vs (num_persons, num_kpts, 3)
        if keypoints_data_for_image.ndim == 2:  # Single person detected, add batch dim
            keypoints_data_for_image = keypoints_data_for_image.unsqueeze(0)

        for person_idx in range(keypoints_data_for_image.shape[0]):
            kpts_tensor = keypoints_data_for_image[
                person_idx
            ]  # Shape: (num_keypoints_yolo, 3) [x, y, conf]

            person_result_kpts_with_conf = torch.zeros(
                (MODEL_NUM_JOINTS, 3), device=DEVICE
            )  # norm_x, norm_y, conf

            num_detected_kpts_yolo = kpts_tensor.shape[0]

            valid_kpt_idx = 0
            for yolo_kpt_idx in range(num_detected_kpts_yolo):
                if valid_kpt_idx >= MODEL_NUM_JOINTS:
                    break  # Filled all target joint slots

                x_pixel, y_pixel, conf = kpts_tensor[yolo_kpt_idx]

                if conf >= confidence_threshold:
                    norm_x = x_pixel / img_w
                    norm_y = y_pixel / img_h
                    person_result_kpts_with_conf[valid_kpt_idx, 0] = norm_x
                    person_result_kpts_with_conf[valid_kpt_idx, 1] = norm_y
                    person_result_kpts_with_conf[valid_kpt_idx, 2] = conf
                    valid_kpt_idx += 1

            processed_persons_keypoints.append(person_result_kpts_with_conf)
            # For this script, we only take the first person with valid keypoints
            if processed_persons_keypoints:  # Take the first person processed
                return processed_persons_keypoints[0].unsqueeze(0).to(DEVICE)

    if not processed_persons_keypoints:
        logger.warning(
            "No 2D keypoints detected or none passed the confidence threshold."
        )
        return torch.zeros(
            (1, MODEL_NUM_JOINTS, 3), device=DEVICE
        )  # Return zeroed (norm_x, norm_y, conf=0)

    # Should have returned within the loop for the first person
    # Fallback, though ideally not reached if logic above is correct
    return processed_persons_keypoints[0].unsqueeze(0).to(DEVICE)


def get_depth_map(depth_processor, depth_model, image_pil):
    """Estimates depth map using DepthPro."""
    img_w, img_h = image_pil.size
    inputs = depth_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    post_processed = depth_processor.post_process_depth_estimation(
        outputs, target_sizes=[(img_h, img_w)]
    )
    depth_tensor = post_processed[0]["predicted_depth"]
    dmin, dmax = depth_tensor.min(), depth_tensor.max()
    depth_normalized = (
        (depth_tensor - dmin) / (dmax - dmin)
        if dmax > dmin
        else torch.zeros_like(depth_tensor)
    )
    return depth_normalized.unsqueeze(0).unsqueeze(0).to(DEVICE)


def create_depth_viz(depth_tensor):
    """Creates a colored PIL image from a depth tensor."""
    depth_numpy = depth_tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(depth_numpy, cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    depth_pil = Image.open(buf)
    plt.close()
    depth_pil.thumbnail(VIZ_THUMBNAIL_SIZE)
    return depth_pil


def create_2d_kpts_viz(image_pil, keypoints_normalized_with_conf):
    """
    Creates a PIL image with 2D keypoints overlaid.
    keypoints_normalized_with_conf: Tensor of shape (1, MODEL_NUM_JOINTS, 3) -> norm_x, norm_y, conf
    """
    img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv2.shape[:2]

    kpts_data = (
        keypoints_normalized_with_conf.squeeze(0).cpu().numpy()
    )  # Shape [MODEL_NUM_JOINTS, 3]

    # Denormalize x, y coordinates
    kpts_pixels = np.zeros((kpts_data.shape[0], 2), dtype=int)
    kpts_pixels[:, 0] = (kpts_data[:, 0] * w).astype(int)
    kpts_pixels[:, 1] = (kpts_data[:, 1] * h).astype(int)

    confidences = kpts_data[:, 2]

    # Draw keypoints
    for i in range(len(kpts_pixels)):
        conf = confidences[i]
        if (
            conf > 0
        ):  # Draw only if confidence is greater than 0 (i.e., it passed threshold and is not padding)
            x, y = kpts_pixels[i, 0], kpts_pixels[i, 1]
            cv2.circle(img_cv2, (x, y), 5, (0, 0, 255), -1)  # Red dots

    # Draw connections
    for start_idx, end_idx in CONNECTIONS:
        if start_idx < len(kpts_pixels) and end_idx < len(kpts_pixels):
            start_conf = confidences[start_idx]
            end_conf = confidences[end_idx]

            # Draw line only if both connected keypoints are valid (had conf > 0)
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
        transformed_depth: Tensor for the 3D model's depth input.
        keypoints_2d_for_model: Tensor (Batch, MODEL_NUM_JOINTS, 2) for 3D model's keypoint input (norm_x, norm_y).
        keypoints_2d_for_viz: Tensor (Batch, MODEL_NUM_JOINTS, 3) for visualization (norm_x, norm_y, conf).
        depth_map_raw: Raw depth map for visualization.
    """
    try:
        # keypoints_2d_with_conf shape: (1, MODEL_NUM_JOINTS, 3) -> (norm_x, norm_y, conf)
        keypoints_2d_with_conf = get_2d_keypoints(
            yolo_model, image_pil.copy(), confidence_threshold=yolo_conf_thresh
        )

        if (
            keypoints_2d_with_conf is None
        ):  # Should not happen due to get_2d_keypoints returning zeros
            logger.error("get_2d_keypoints returned None, which is unexpected.")
            return None, None, None, None, None

        # Check if any keypoints were actually found (any confidence > 0)
        if not torch.any(keypoints_2d_with_conf[:, :, 2] > 0):
            logger.warning("No keypoints passed the threshold during preprocessing.")

        # Keypoints for the model (x, y only)
        keypoints_2d_for_model = keypoints_2d_with_conf[
            :, :, :2
        ].clone()  # Shape: (1, MODEL_NUM_JOINTS, 2)

        depth_map_raw = get_depth_map(depth_processor, depth_model, image_pil.copy())
        if depth_map_raw is None:
            return None, None, None, None, None

        transformed_image = image_transform(image_pil.copy()).unsqueeze(0).to(DEVICE)
        # Ensure depth_map_raw is [B, C, H, W] before interpolate if it's not already
        # get_depth_map returns [1, 1, H, W] which is correct
        transformed_depth = torch.nn.functional.interpolate(
            depth_map_raw, size=MODEL_INPUT_SIZE, mode="bilinear", align_corners=False
        ).to(DEVICE)

        return (
            transformed_image,
            transformed_depth,
            keypoints_2d_for_model,
            keypoints_2d_with_conf,
            depth_map_raw,
        )

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return None, None, None, None, None


def run_inference(pose_model, image_tensor, depth_tensor, keypoints_2d_tensor):
    """Runs inference. keypoints_2d_tensor is (Batch, NumJoints, 2)"""
    try:
        with torch.no_grad():
            # keypoints_2d_tensor should be (Batch, NumJoints, 2) for the model
            predicted_joints_3d_batch = pose_model(
                image_tensor, depth_tensor, keypoints_2d_tensor
            )
        return predicted_joints_3d_batch[0].cpu().numpy()
    except Exception as e:
        logger.error(f"Error during model inference: {e}", exc_info=True)
        return None


def main(args):
    """Main inference loop with enhanced visualization."""
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        pose_3d_model = load_pose_model(args.checkpoint_path, args.num_joints)
        yolo_model, depth_processor, depth_model = load_preprocessing_models(
            args.yolo_model_path
        )
    except Exception as e:
        logger.error(f"Failed to load models: {e}. Exiting.")
        return

    input_folder_path = Path(args.input_folder)
    image_files = [
        f
        for f in input_folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

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

        # Unpack the 5 return values from preprocess_input
        processed_data = preprocess_input(
            img_pil,
            yolo_model,
            depth_processor,
            depth_model,
            yolo_conf_thresh=args.yolo_confidence_threshold,
        )

        if (
            processed_data[0] is None
        ):  # Check based on first element, assuming all are None on failure
            logger.warning(
                f"Skipping {image_file_path.name} due to preprocessing failure."
            )
            continue

        (
            image_tensor,
            depth_tensor,
            keypoints_2d_for_model,
            keypoints_2d_for_viz,
            depth_raw_for_viz,
        ) = processed_data

        predicted_joints = run_inference(
            pose_3d_model, image_tensor, depth_tensor, keypoints_2d_for_model
        )

        if predicted_joints is None:
            logger.warning(f"Skipping {image_file_path.name} due to inference failure.")
            continue

        npy_output_path = output_path / f"{image_file_path.stem}_pred_joints3d.npy"
        np.save(npy_output_path, predicted_joints)
        logger.info(f"Saved predicted 3D joints to {npy_output_path}")

        if args.visualize:
            try:
                # 1. Original Image
                img_orig_viz = img_pil.copy()
                img_orig_viz.thumbnail(VIZ_THUMBNAIL_SIZE)

                # 2. 2D Keypoints Image (use keypoints_2d_for_viz)
                img_2d_viz = create_2d_kpts_viz(
                    img_pil.copy(), keypoints_2d_for_viz.clone()
                )

                # 3. Depth Map Image
                img_depth_viz = create_depth_viz(depth_raw_for_viz.clone())

                # 4. 3D Pose Image
                fig_3d = visualize_3d_pose(
                    predicted_joints.copy(), title="Predicted 3D Pose"
                )
                img_3d_viz = fig_to_image(fig_3d)
                img_3d_viz.thumbnail(VIZ_THUMBNAIL_SIZE)
                plt.close(fig_3d)

                # 5. Create 2x2 Combined Image
                w_thumb, h_thumb = VIZ_THUMBNAIL_SIZE  # Corrected variable name
                combined_img = Image.new(
                    "RGB", (w_thumb * 2, h_thumb * 2), (255, 255, 255)
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
                    f"Failed to visualize for {image_file_path.name}: {e}",
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
        "--output_folder",
        type=str,
        default="inference_output",
        help="Folder to save results.",
    )
    parser.add_argument(
        "--yolo_model_path",
        type=str,
        default="yolov8n-pose.pt",
        help="Path to the YOLO pose model (e.g., yolov8n-pose.pt).",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=MODEL_NUM_JOINTS,
        help="Number of joints expected by the model.",
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
