import torch
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from torch.jit import trace
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import logging
from pathlib import Path
from config import (
    RANDOM_SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    LOG_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    WEIGHT_DECAY,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PERSISTENT_WORKERS,
    EVAL_INTERVAL as EVAL_STEPS,
    CHECKPOINT_PREFIX,
    MODEL_TYPE,
)

from loss import ComprehensivePoseLoss

from dataset.chunked_dataset import StreamingChunkedDataset
from models.cnn import CNNPoseEstimation
from models.transformers import TransformerPoseEstimation
from model_config import ModelConfig
from train import train_model

# Import the custom collator
from dataset.collator import Human36MCollator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Training")

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train on streaming Human3.6M dataset")
    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="Directory containing chunked dataset",
    )
    parser.add_argument(
        "--train-chunks", type=int, nargs="+", help="Chunk indices to use for training"
    )
    parser.add_argument(
        "--val-chunks", type=int, nargs="+", help="Chunk indices to use for validation"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache extracted chunks",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Explicit checkpoint path to load"
    )
    parser.add_argument(
        "--start-step", type=int, help="Global step index to resume from"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "transformer"],
        help="Model type: 'cnn' or 'transformer'",
    )

    args = parser.parse_args()

    # Configuration overrides
    batch_size = BATCH_SIZE
    start_step = 0

    # Prepare cache dir
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using cache directory: {cache_dir}")

    # TensorBoard writer
    log_dir = LOG_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs: {log_dir}")

    model_type = args.model_type.lower() if args.model_type else MODEL_TYPE

    # Load checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = None

    optimizer = None
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model_type = ckpt.get("model_type", model_type)

        model_args = ckpt.get("model_args", {})
        print(model_args)
        model_cfg = ModelConfig(model_type=model_type, **model_args)
        if model_type == "transformer":
            model = TransformerPoseEstimation(config=model_cfg).to(DEVICE)
        elif model_type == "cnn":
            model = CNNPoseEstimation(config=model_cfg).to(DEVICE)
        else:
            raise ValueError(f"Unsupported model type in checkpoint: {model_type}")
        try:
            model.load_state_dict(ckpt["model_state_dict"])

            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except RuntimeError:
            if "pose_head.decoder.3.bias" in ckpt["model_state_dict"]:
                ckpt["model_state_dict"].pop("pose_head.decoder.3.bias")
                ckpt["model_state_dict"].pop("pose_head.decoder.3.weight")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        logger.info(f"Resumed at step {start_step}")

        start_step = ckpt.get("global_step", start_step)
    else:
        model_cfg = ModelConfig(model_type=model_type)
        if model_type == "transformer":
            model = TransformerPoseEstimation(config=model_cfg).to(DEVICE)
        elif model_type == "cnn":
            model = CNNPoseEstimation(config=model_cfg).to(DEVICE)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        if ckpt_path:
            logger.warning(f"Checkpoint not found: {ckpt_path}, training from scratch.")

    start_step = args.start_step if args.start_step is not None else start_step

    print(f"Device: {DEVICE}")
    print(f"Model type: {args.model_type}")
    print(f"Effective batch size: {batch_size * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Resume from step: {start_step}")

    # Loss
    criterion = ComprehensivePoseLoss()

    # Data transforms
    transform = transforms.Compose(
        [transforms.Resize(model_cfg.image_size)]
    )

    train_ds = StreamingChunkedDataset(
        "train",
        chunks_dir=args.chunks_dir,
        chunk_indices=args.train_chunks,
        transform=transform,
        cache_dir=cache_dir,
        shuffle=True,
        shuffle_chunks=True,
    )
    train_ds.training = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=Human36MCollator(),
    )

    val_ds = StreamingChunkedDataset(
        "test",
        chunks_dir=args.chunks_dir,
        chunk_indices=args.val_chunks,
        transform=transform,
        cache_dir=cache_dir,
        shuffle=True,
        shuffle_chunks=True,
    )
    val_ds.training = False
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=Human36MCollator(),
    )

    dummy_image = torch.randn(1, 3, *model_cfg.image_size).to(DEVICE)
    dummy_depth = torch.randn(1, 1, *model_cfg.image_size).to(DEVICE)
    dummy_keypoints = torch.randn(1, 17, 2).to(DEVICE)

    try:
        traced_model = trace(model, (dummy_image, dummy_depth, dummy_keypoints))
        writer.add_graph(traced_model, (dummy_image, dummy_depth, dummy_keypoints))
        logger.info("Model graph added to TensorBoard")
    except Exception as e:
        logger.error(f"Could not add model graph to TensorBoard: {e}")

    # Training
    model, _, last_step = train_model(
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        writer=writer,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        start_step=start_step,
        eval_interval_steps=EVAL_STEPS,
        checkpoint_prefix=CHECKPOINT_PREFIX,
    )

    logger.info(f"Training complete at step {last_step}")
    writer.close()


main()
