import torch
from pathlib import Path

RANDOM_SEED = 42

NUM_JOINTS = 17
BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 10
EVAL_INTERVAL = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "cnn"  # Options: "cnn", "transformer"

# Loss parameters
INTER_JOINT_LOSS_WEIGHT = 100
ABS_ROOT_LOSS_WEIGHT = 1
L1_LOSS_WEIGHT = 1
MSE_LOSS_WEIGHT = 1

# Optimizer parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

NUM_WORKERS = 1  # Increased from 1
PREFETCH_FACTOR = 1  # Increased from 1
PERSISTENT_WORKERS = False

USE_AUGMENTATION = False  # Whether to use data augmentation
ROTATION_RANGE = (-30, 30)  # Rotation range in degrees
FLIP_PROB = 0.5  # Probability of horizontal flipping
SCALE_RANGE = (0.8, 1.2)  # Scale range
TRANSLATE_RANGE = (-0.1, 0.1)  # Translation range as fraction of image size
BRIGHTNESS_RANGE = (0.8, 1.2)  # Brightness adjustment range
CONTRAST_RANGE = (0.8, 1.2)  # Contrast adjustment range

BASE_PATH = Path("/mnt/data/AI/Human3.6m")
IMAGES_PATH = BASE_PATH / "images"
PROCESSED_PATH = BASE_PATH / "processed"
ANNOTATIONS_PATH = BASE_PATH / "rannotations"

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("./dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

CHECKPOINT_PREFIX = "model_epoch_"

# Define joint connectivity for visualization
# Joint indices for Human3.6M:
#  0: Pelvis, 1: Right Hip, 2: Right Knee, 3: Right Ankle,
#  4: Left Hip, 5: Left Knee, 6: Left Ankle,
#  7: Spine, 8: Thorax, 9: Neck, 10: Head,
#  11: Left Shoulder, 12: Left Elbow, 13: Left Wrist,
#  14: Right Shoulder, 15: Right Elbow, 16: Right Wrist
CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),  # Right leg
    (0, 4),
    (4, 5),
    (5, 6),  # Left leg
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # Spine to head
    (8, 11),
    (11, 12),
    (12, 13),  # Left arm
    (8, 14),
    (14, 15),
    (15, 16),  # Right arm
]
