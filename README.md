# 3D Human Pose Estimation Project

## Project Authors:

| Lorenzo Calda | Alberto Paolo Lolli | Mariano Masiello | Luca Ricci | Ali Emre Senel |
| :-----------: | :-----------------: | :--------------: | :--------: | :------------: |
|    3194670    |       3224481       |     3200991      |  3218444   |    3221337     |

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Project Structure](#project-structure)
4.  [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Clone Repository](#clone-repository)
    - [Conda Environment Setup](#conda-environment-setup)
    - [Install Project in Editable Mode](#install-project-in-editable-mode)
    - [Rclone Setup (for `dataset_split.py`)](#rclone-setup)
    - [Download Pretrained Models (for Preprocessing and Inference)](#download-pretrained-models)
    - [Dataset Preparation (Human3.6M)](#dataset-preparation)
    - [Configure `src/config.py`](#configure-srcconfigpy)
5.  [Dataset Processing Workflow](#dataset-processing-workflow)
    - [0. Per-Frame Preprocessing (`preprocess.py`)](#0-per-frame-preprocessing-preprocesspy)
    - [1. Initial Chunking (`dataset_chunker.py`)](#1-initial-chunking-dataset_chunkerpy)
    - [2. Splitting and Remote Re-chunking (`dataset_split.py`)](#2-splitting-and-remote-re-chunking-dataset_splitpy)
    - [3. Shuffling and Local Re-chunking (`dataset_rechunker.py`)](#3-shuffling-and-local-re-chunking-dataset_rechunkerpy)
6.  [Training the Model (`main.py`)](#training-the-model-mainpy)
    - [Configuration for Training](#configuration-for-training)
    - [Running Training](#running-training)
    - [TensorBoard Logging](#tensorboard-logging)
7.  [Running Inference (`infer.py`)](#running-inference-inferpy)
    - [Running Inference Script](#running-inference-script)
    - [Visualization](#visualization)
8.  [Core Components](#core-components)
    - [Configuration Files](#configuration-files)
    - [Models](#models)
    - [Dataset Handling](#dataset-handling)
    - [Loss Function](#loss-function)
    - [Training Logic](#training-logic)
    - [Utilities and Visualization](#utilities-and-visualization)
    - [Data Augmentation](#data-augmentation)

## Overview

This project implements a pipeline for 3D human pose estimation from RGB images. It includes scripts for dataset processing, model training (CNN and Transformer architectures), and inference. The project is designed for large datasets like Human3.6M, providing tools for preprocessing, chunking, filtering, and distributing data, which is necessary due to dataset size. Data management, especially for large intermediate datasets, can be handled using `rclone` with cloud storage providers like OneDrive.

## Features

- **Dataset Processing**:
  - Per-frame preprocessing (`preprocess.py`) for depth maps and 2D keypoints using YOLO (`yolo11x-pose`) and DepthPro.
  - Initial chunking (`dataset_chunker.py`) of large datasets using preprocessed data.
  - Dataset filtering and splitting (`dataset_split.py`) by subject IDs, with `rclone` for remote storage (e.g., OneDrive).
  - Data re-chunking to specified sizes for streaming.
  - Sample shuffling (`dataset_rechunker.py`) across chunks.
- **Model Training**:
  - Supports CNN (`src/models/cnn.py`) and Transformer (`src/models/transformers.py`) architectures.
  - Training script (`main.py`) with:
    - Checkpoint resumption.
    - Gradient accumulation.
    - TensorBoard logging.
    - Hyperparameter customization via `src/config.py`.
    - Customizable loss function (`src/loss.py`) combining MSE, L1, inter-joint distance, and absolute root joint losses.
- **Inference Pipeline (`infer.py`)**:
  - End-to-end inference from an image.
  - Utilizes YOLO (`yolo11x-pose`) for 2D keypoint detection.
  - Employs DepthPro for depth map estimation.
  - 3D joint location prediction.
  - Output of predictions as `.npy` files.
  - Optional visualization: original image, 2D keypoints, depth map, 3D pose, and a combined panel.
- **Configuration**:
  - Centralized settings for paths, hyperparameters, and models (`src/config.py`, `src/model_config.py`).
- **Data Augmentation**:
  - `src/dataset/augmentation.py` provides `PoseAugmentor` for applying consistent transformations to images, depth maps, 2D keypoints, and 3D joints.

## Project Structure

```
.
├── dataset_chunker.py          # Script for initial dataset chunking
├── dataset_rechunker.py        # Script for shuffling and re-chunking local datasets
├── dataset_split.py            # Script for filtering, splitting, and re-chunking using rclone
├── enviroment-specfile.txt     # Conda environment specification
├── infer.py                    # Script for running inference
├── main.py                     # Main script for training models
├── preprocess.py               # Script for per-frame pose & depth estimation
│
└── src/                        # Source directory for core modules
    ├── config.py               # Global configuration
    ├── model_config.py         # Model-specific configuration
    ├── loss.py                 # Custom loss functions
    ├── train.py                # Training loop and functions
    ├── visualize.py            # Visualization utilities
    ├── utils.py                # General utility functions
    │
    ├── dataset/
    │   ├── augmentation.py     # Data augmentation logic
    │   ├── chunked_dataset.py  # Streaming chunked dataset loader
    │   └── collator.py         # Custom collator
    │
    └── models/
        ├── common.py           # Common model components (heatmaps, regression head)
        ├── cnn.py              # CNN pose estimation model
        └── transformers.py     # Transformer pose estimation model
│
└── README.md                   # This file
```

## Setup and Installation

### Prerequisites

- **Git**
- **Conda** (Miniconda or Anaconda recommended)
- **Python** (as per `enviroment-specfile.txt`, e.g., 3.12)
- **Pip**
- **Rclone** (Optional, for `dataset_split.py` with cloud storage like OneDrive)

### Clone Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### Conda Environment Setup

The project uses a Conda environment defined in `enviroment-specfile.txt` (primarily for `linux-64`).

1.  **Create Environment**:
    ```bash
    conda create --name humanpose3d --file enviroment-specfile.txt
    ```
2.  **Activate Environment**:
    ```bash
    conda activate humanpose3d
    ```
3.  **Key Dependencies**: `python=3.12`, `pytorch=2.4.0` (CUDA 12.4), `torchvision`, `ultralytics`, `transformers`, `numpy`, `opencv-python`, `matplotlib`, `tensorboard`, `timm`.

### Install Project in Editable Mode

To ensure that the modules within the `src` directory (like `config`, `models`, `dataset`) are correctly imported by scripts in the root directory (e.g., `main.py`, `infer.py`), install the project in editable mode. This is typical for projects structured with a `src` layout and managed by `pyproject.toml`.

After activating your Conda environment, navigate to the root of the cloned repository and run:

```bash
pip install -e .
```

This command tells pip to install the project from the current directory (`.`) in editable mode (`-e`), meaning changes to your source files are immediately reflected without needing to reinstall.

### Rclone Setup

(If using `dataset_split.py` for managing data on remote/cloud storage like OneDrive)

1.  **Install Rclone**: See [rclone.org/install/](https://rclone.org/install/).
    On Debian/Ubuntu-based systems, you might install it via the system's package manager:
    ```bash
    sudo apt update
    sudo apt install rclone
    ```
    Alternatively, use the script from their website:
    ```bash
    sudo -v ; curl https://rclone.org/install.sh | sudo bash
    ```
2.  **Configure Rclone for OneDrive (or other provider)**:
    Run `rclone config` and follow the prompts. You will create a "remote" (e.g., named `myonedrive`).
    Example for OneDrive:
    ```bash
    rclone config
    # n) New remote
    # name> myonedrive
    # Storage> onedrive (or its number)
    # ... follow authentication steps ...
    ```
    The `dataset_split.py` script will use the remote name you define (e.g., `myonedrive:path/to/data`).

### Download Pretrained Models (for Preprocessing and Inference)

1.  **YOLO Pose Model**:

    - This project primarily uses `yolo11x-pose.pt`.
    - `preprocess.py` defaults to `yolo11x-pose`.
    - `infer.py` also uses `yolo11x-pose.pt` by default (change via `--yolo_model_path` if needed).
    - Ensure `yolo11x-pose.pt` (or the model you intend to use) is available. Ultralytics will download it if the path is just the name. See [Ultralytics YOLO](https://ultralytics.com/).

2.  **DepthPro Model**: (`apple/DepthPro-hf`) Downloaded automatically by `transformers` library on first use.

3.  **3D Pose Estimation Model Checkpoint**: Required for `infer.py` (`--checkpoint_path`) and optional for `main.py` (resuming). Train this using `main.py` or use a provided checkpoint.

### Dataset Preparation (Human3.6M)

1.  **Acquire Human3.6M Dataset**: Obtain image data, camera parameters, and 3D joint annotations.
2.  **Organize Raw Data**: Structure raw RGB images for `preprocess.py` (e.g., `input_base_dir/subject_action_camera/frame_xxxx.jpg`). Annotations for `dataset_chunker.py` should be placed as per `ANNOTATIONS_PATH` in `src/config.py`.

### Configure `src/config.py`

**Review and update `src/config.py` before running scripts.**

Key paths in `src/config.py`:

- `IMAGES_PATH`: Path to original Human3.6M RGB images.
- `PROCESSED_PATH`: Path to output of `preprocess.py` (depth maps, metadata JSONs).
- `ANNOTATIONS_PATH`: Path to Human3.6M annotation files (camera params, 3D joints).
- `LOG_DIR`: Path for TensorBoard logs and training checkpoints.
  Adjust other parameters like `DEVICE`, `LEARNING_RATE`, `BATCH_SIZE`, `MODEL_TYPE`, loss weights, and augmentation settings as needed.

## Dataset Processing Workflow

The large size of datasets like Human3.6M requires a multi-step processing workflow.

### 0. Per-Frame Preprocessing (`preprocess.py`)

This script generates inputs for chunking and training.

- **Purpose**: Perform 2D pose (YOLO `yolo11x-pose`) and depth (DepthPro) estimation for each image.
- **Operations**:
  - Iterates through image subfolders.
  - Detects 2D keypoints.
  - Estimates depth maps.
  - Saves depth images (`*_depth.png`) and metadata JSONs (`*.json` with keypoints, image/depth dimensions, depth min/max).
- **Usage**:
  ```bash
  python preprocess.py <input_base_directory> <output_base_directory>
  ```
  - YOLO model path is `yolo11x-pose` (hardcoded in `preprocess.py`).
- **Example**:
  ```bash
  python preprocess.py /path/to/h36m_raw_images /path/to/h36m_processed_frames
  ```
  The output `/path/to/h36m_processed_frames` becomes `PROCESSED_PATH` for `dataset_chunker.py`.

### 1. Initial Chunking (`dataset_chunker.py`)

Uses outputs from `preprocess.py` and original Human3.6M annotations to create initial `.tar` chunks. These chunks can be stored locally or on a cloud service like OneDrive using `rclone` (manual copy step if `dataset_chunker.py` outputs locally).

- **Purpose**: Consolidate frame data into larger chunks.
- **Operations**:
  - Loads data for specified subjects.
  - Reads depth maps/metadata from `PROCESSED_PATH`.
  - Reads RGB images from `IMAGES_PATH`.
  - Loads 3D joints/camera params from `ANNOTATIONS_PATH`.
  - Transforms 3D joint coordinates to camera space.
  - Packs samples into chunks.
- **Usage**:
  ```bash
  python dataset_chunker.py --subjects <S1> <S2> ... --output <output_chunk_dir> [options]
  ```
- **Example (Output to local, then optionally rclone copy to OneDrive)**:

  ```bash
  # 1. Generate chunks locally
  python dataset_chunker.py --subjects 1 5 6 --output ./data/h36m_initial_chunks --chunk-size 5000

  # 2. (Optional) Copy local chunks to OneDrive using rclone
  # rclone copy ./data/h36m_initial_chunks myonedrive:datasets/h36m_initial_chunks -P
  ```

### 2. Splitting and Remote Re-chunking (`dataset_split.py`)

Filters dataset chunks (from local path or `rclone` remote like OneDrive), splits them into train/test sets by subject ID, and re-chunks them to a new destination (local or `rclone` remote).

- **Purpose**: Create train/test splits, manage data on local or remote storage.
- **Operations**:
  - Lists/downloads original chunks.
  - Extracts samples, filters by subject ID.
  - Caches valid samples locally.
  - Re-chunks and uploads to destination.
- **Usage**:
  ```bash
  python dataset_split.py --rclone-input-prefix <source_path_or_remote:path> \
                          --rclone-destination-prefix <dest_path_or_remote:path> \
                          --train-subjects <S1,S2,...> \
                          --test-subjects <S3,S4,...> \
                          [options]
  ```
- **Example (OneDrive to OneDrive)**:
  ```bash
  python dataset_split.py \
      --rclone-input-prefix myonedrive:datasets/h36m_initial_chunks \
      --rclone-destination-prefix myonedrive:datasets/h36m_train_test_split \
      --temp-extraction-dir /mnt/local_ssd/temp_extract \
      --staging-dir ./staging_area_split \
      --train-subjects S1,S5,S6,S7,S8 \
      --test-subjects S9,S11 \
      --new-chunk-size 2000
  ```

### 3. Shuffling and Local Re-chunking (`dataset_rechunker.py`)

Unpacks local dataset chunks (e.g., downloaded train set from OneDrive), validates samples, shuffles all valid samples, and repacks them into new local chunks.

- **Purpose**: Improve training data distribution by shuffling.
- **Operations**:
  - Reads `.tar` chunks.
  - Extracts and validates samples.
  - Collects, shuffles, and repacks samples.
- **Usage**:
  ```bash
  python dataset_rechunker.py --input-dir <input_chunks_for_shuffling> \
                              --output-dir <output_shuffled_chunks> \
                              [options]
  ```
- **Example (After downloading train split from OneDrive to local)**:
  `bash
  # Assume train split is downloaded to ./data/h36m_train_test_split/train
  python dataset_rechunker.py \
   --input-dir ./data/h36m_train_test_split/train \
   --output-dir ./data/h36m_shuffled_train_chunks \
   --chunk-size 8000 \
   --keep-extracted-originals
  `The output (e.g.,`./data/h36m_shuffled_train_chunks`) is then used as `--chunks-dir`for`main.py`.

## Training the Model (`main.py`)

Script for training the 3D pose estimation model.

### Configuration for Training

1.  **`src/config.py`**: Set `DEVICE`, `LEARNING_RATE`, `BATCH_SIZE`, `LOG_DIR`, `MODEL_TYPE`, loss weights, augmentation flags.
2.  **`src/model_config.py`**: Adjust architecture parameters.
3.  **Dataset**: Ensure `--chunks-dir` points to final processed and chunked data.

### Running Training

- **Usage**:
  ```bash
  python main.py --chunks-dir <dataset_root_with_train_val_chunks> \
                 --train-chunks <idx1> <idx2> ... \
                 --val-chunks <idxA> <idxB> ... \
                 [options]
  ```
  - `--chunks-dir`: Base directory of chunked data. `StreamingChunkedDataset` expects subdirectories like `train/` and `test/` (or `val/`) within this path.
  - `--train-chunks`: Numeric indices of training chunk files (e.g., `0 1 2` for `train_0000.tar.gz`, etc. in `<chunks-dir>/train/`).
  - `--val-chunks`: Numeric indices of validation chunk files.
- **Example**:
  If `./data/final_dataset/` contains `train/train_0000.tar.gz`, `test/test_0000.tar.gz`, etc.
  ```bash
  python main.py \
      --chunks-dir ./data/final_dataset \
      --train-chunks 0 1 2 3 4 \
      --val-chunks 0 1 \
      --model-type transformer \
      --cache-dir ./dataset_cache
  ```

### TensorBoard Logging

Training progress is logged to TensorBoard.

- Logs are in `LOG_DIR/YYYYMMDD-HHMMSS/` (from `src/config.py`).
- View: `tensorboard --logdir <path_to_LOG_DIR>`
- Shows total loss, individual loss components (MSE, L1, inter-joint, root), and metrics (MPJPE, PA-MPJPE). Validation previews are also logged.

[Validation Previews](images/VALIDATION.md)

## Running Inference (`infer.py`)

Performs 3D human pose estimation on a folder of images.

### Running Inference Script

- **Prerequisites**: Trained 3D model checkpoint, YOLO pose model (`yolo11x-pose.pt` or other).
- **Usage**:
  ```bash
  python infer.py --checkpoint_path <path/to/3D_model.pth> \
                  --input_folder <path/to/input_images> \
                  --model_type <cnn, transformer> \
                  [options]
  ```
- **Example**:
  ```bash
  python infer.py \
      --checkpoint_path ./training_logs/transformer_run1/checkpoints/best_model.pth \
      --input_folder ./sample_test_images/ \
      --output_folder ./inference_results/ \
      --yolo_model_path yolo11x-pose.pt \
      --visualize
  ```

### Visualization

If `--visualize` is used, `infer.py` saves a 2x2 combined image:

1.  Original Image
2.  2D Keypoints Overlay
3.  Estimated Depth Map
4.  Rendered 3D Pose

[Inference Samples](images/INFERENCE.md)

## Core Components

### Configuration Files

- **`src/config.py`**: Global settings (paths, hyperparameters, loss weights, augmentation flags). **Customize for your setup.**
- **`src/model_config.py`**: `ModelConfig` dataclass for architecture parameters.

### Models (`src/models/`)

- **`common.py`**: `GaussianHeatmapGenerator`, `PoseRegressionHead`.
- **`cnn.py`**: `CNNPoseEstimation` model.
- **`transformers.py`**: `TransformerPoseEstimation` model.
  Both main models take image, depth, and 2D keypoints as input.

### Dataset Handling (`src/dataset/`)

- **`chunked_dataset.py`**: `StreamingChunkedDataset` for loading from `.tar` chunks.
- **`collator.py`**: `Human36MCollator` for batching samples.

### Loss Function (`src/loss.py`)

- **`ComprehensivePoseLoss`**: Combines MSE, L1, inter-joint distance, and absolute root joint losses. Weights configurable in `src/config.py`.

### Training Logic (`src/train.py`)

- **`train_model` function**: Manages training/validation loop, logging, checkpoints.

### Utilities and Visualization

- **`src/utils.py`**: Coordinate transformations, depth normalization, evaluation metrics (MPJPE, PA-MPJPE), model layer helpers.
- **`src/visualize.py`**: `visualize_comparison` (for validation previews), `fig_to_image`.

### Data Augmentation (`src/dataset/augmentation.py`)

- **`PoseAugmentor`**: Applies transformations (flip, rotation, scale, translate, color jitter) to image, depth, 2D keypoints, and 3D joints.
