import json
import torch
import numpy as np
import pickle
import tarfile
import threading
import queue
import time
from pathlib import Path
from torch.utils.data import IterableDataset
import tempfile
import shutil
from tqdm import tqdm
import logging
from torchvision import transforms
import concurrent.futures

import torchvision.io

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChunkedDataset")

default_transform = transforms.Compose(
    [
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ]
)


def _extract_keypoints_to_tensor(keypoints_data, sample_identifier_for_log):
    """
    Helper function to robustly extract and convert keypoints to a tensor.
    Processes only the first person's keypoints if multiple are present.
    """
    if (
        isinstance(keypoints_data, list)
        and len(keypoints_data) > 0
        and isinstance(keypoints_data[0], list)
    ):
        if len(keypoints_data) > 1:
            logger.warning(
                f"Sample {sample_identifier_for_log} has multiple poses ({len(keypoints_data)}), using the first pose only."
            )
        person_keypoints = keypoints_data[0]
    elif isinstance(keypoints_data, list) and all(
        isinstance(kp, dict) for kp in keypoints_data
    ):
        person_keypoints = keypoints_data
    else:
        return torch.empty((0, 2), dtype=torch.float32)

    if not person_keypoints:
        logger.warning(
            f"No keypoints found for the first person in {sample_identifier_for_log}."
        )
        return torch.empty((0, 2), dtype=torch.float32)

    kp_tuples = []
    for kp in person_keypoints:
        kp_tuples.append((float(kp["x"]), float(kp["y"])))

    return torch.tensor(kp_tuples, dtype=torch.float32)


def process_sample(sample, chunk_dir_str, transform):
    """
    Processes a single sample.

    Args:
        sample (dict): Dictionary containing sample information.
        chunk_dir_str (str): String path to the directory containing chunk data.
        transform (callable): A function/transform to apply to images.
                              It will receive a float32 tensor in [0,1] range.

    Returns:
        dict or None: A dictionary with processed tensors and metadata, or None if processing fails.
    """
    sample_id_for_log = sample.get(
        "image_file", sample.get("frame_idx", "UNKNOWN_SAMPLE")
    )
    try:
        chunk_dir = Path(chunk_dir_str)

        # 1. Load RGB Image
        image_file = sample.get("image_file")

        image_path = chunk_dir / image_file
        if not image_path.is_file():
            logger.error(
                f"RGB image file not found at {image_path} for sample {sample_id_for_log}."
            )
            return None

        try:
            # Load RGB image as a uint8 tensor [0, 255]
            image_data_tensor = torchvision.io.read_image(
                str(image_path), mode=torchvision.io.ImageReadMode.RGB
            )
            # Convert to float32 tensor and normalize to [0, 1] range
            image_data_tensor = image_data_tensor.float() / 255.0
            # Apply further transformations
            image_tensor = transform(image_data_tensor)
        except Exception as e:
            logger.error(f"Failed to load or transform RGB image {image_path}: {e}")
            return None

        # 2. Load Depth Image
        depth_file = sample.get("depth_file")
        depth_tensor_transformed = None
        depth_path = chunk_dir / depth_file
        if not depth_path.is_file():
            logger.error(
                f"Depth file {depth_file} (path: {depth_path}) not found during read."
            )
            return None

        try:
            # Load depth image as a uint8 tensor [0, 255]
            depth_data_tensor = torchvision.io.read_image(
                str(depth_path), mode=torchvision.io.ImageReadMode.GRAY
            )
            # Convert to float32 tensor and normalize to [0, 1] range
            depth_data_tensor = depth_data_tensor.float() / 255.0
            # Apply further transformations
            depth_tensor_transformed = transform(depth_data_tensor)
        except Exception as e:
            logger.error(f"Failed to load or transform depth image {depth_path}: {e}")
            return None

        # 3. Load Metadata
        metadata = {}
        metadata_file = sample.get("metadata_file")
        metadata_path = chunk_dir / metadata_file
        if metadata_path.is_file():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding JSON from {metadata_path} for {sample_id_for_log}: {e}"
                )
                return None
            except Exception as e:
                logger.error(
                    f"Error reading metadata file {metadata_path} for {sample_id_for_log}: {e}"
                )
                return None
        else:
            logger.error(
                f"Metadata file {metadata_file} (path: {metadata_path}) not found for {sample_id_for_log}."
            )
            return None

        # 4. Process Depth Tensor (Scaling and Channel Adjustment)
        depth_min_val = float(sample.get("depth_min", metadata.get("depth_min", 0.0)))
        depth_max_val = float(sample.get("depth_max", metadata.get("depth_max", 1.0)))

        final_depth_tensor = (
            depth_tensor_transformed * (depth_max_val - depth_min_val) + depth_min_val
        )

        if final_depth_tensor.ndim == 2:
            final_depth_tensor = final_depth_tensor.unsqueeze(0)

        # 5. Process Keypoints
        keypoints_data_source = sample.get("keypoints", metadata.get("keypoints"))

        keypoints_2d = _extract_keypoints_to_tensor(
            keypoints_data_source, sample_id_for_log
        )

        if keypoints_2d.shape[0] == 0:
            logger.warning(
                f"No valid keypoints processed for {sample_id_for_log}. Adopting similar behavior to original code (skip)."
            )
            return None

        # 6. Normalize Keypoints
        img_size_list = sample.get(
            "image_size",
            metadata.get(
                "image_size",
                [float(image_tensor.shape[2]), float(image_tensor.shape[1])],
            ),
        )  # Use actual W, H from loaded image_tensor

        image_width, image_height = float(img_size_list[0]), float(img_size_list[1])
        image_size_tensor = torch.tensor(
            [image_width, image_height], dtype=torch.float32
        )

        keypoints_2d_normalized = keypoints_2d.clone()
        keypoints_2d_normalized[:, 0] /= image_width
        keypoints_2d_normalized[:, 1] /= image_height

        # 7. Process 3D Joints
        joints_3d_list = sample.get("joints_3d")

        joints_3d = torch.tensor(joints_3d_list, dtype=torch.float32)

        root_position = joints_3d[0, :].clone()
        joints_3d_root_relative = joints_3d - root_position

        # 8. Construct output dictionary
        num_actual_keypoints = keypoints_2d.shape[0]
        expected_num_joints = 17
        if (
            num_actual_keypoints != expected_num_joints
            and keypoints_data_source is not None
        ):
            logger.warning(
                f"Expected {expected_num_joints} keypoints but found {num_actual_keypoints} for {sample_id_for_log}."
            )

        output = {
            "image": image_tensor,
            "depth": final_depth_tensor,
            "keypoints_2d": keypoints_2d_normalized,
            "joints_3d": joints_3d_root_relative,
            "camera_params": sample.get("camera_params"),
            "image_path": image_file,
            "action": sample.get("action"),
            "subaction": sample.get("subaction"),
            "image_size": image_size_tensor,  # derived from loaded image or metadata
            "frame_idx": sample.get("frame_idx"),
            "num_joints": expected_num_joints,
        }
        return output

    except FileNotFoundError as e:
        logger.error(f"General FileNotFoundError for sample {sample_id_for_log}: {e}")
        return None
    except IOError as e:
        logger.error(f"General IOError for sample {sample_id_for_log}: {e}")
        return None
    except Exception:
        logger.exception(
            f"Unexpected critical exception while processing sample {sample_id_for_log}: {sample}"
        )
        return None


def preprocess_samples_multithreaded(
    samples, chunk_dir, chunk_id, transform, max_workers=12
):
    preprocessed_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_sample, sample, chunk_dir, transform)
            for sample in samples
        ]
        for f in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Preprocessing chunk {chunk_id}",
        ):
            result = f.result()
            if result is not None:
                preprocessed_samples.append(result)
    return preprocessed_samples


class ChunkPrefetcher:
    """
    Handles prefetching of chunks in a background thread.
    """

    def __init__(
        self, chunks_dir, chunk_files, cache_dir, max_chunks_in_memory=2, transform=None
    ):
        """
        Initialize the chunk prefetcher.

        Args:
            chunks_dir: Directory containing chunk files
            chunk_files: List of chunk file paths
            cache_dir: Directory to extract chunks to
            max_chunks_in_memory: Maximum number of chunks to keep in memory
        """
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = [Path(f) for f in chunk_files]
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.max_chunks_in_memory = max_chunks_in_memory

        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.chunk_queue = queue.Queue(maxsize=max_chunks_in_memory)
        self.chunks_in_memory = set()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.prefetcher_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.prefetcher_thread.start()

    def _prefetch_worker(self):
        """Background thread that prefetches and preprocesses chunks."""
        chunk_idx = 0
        while not self.stop_event.is_set():
            # Check if we need to prefetch more chunks
            if self.chunk_queue.qsize() < self.max_chunks_in_memory and chunk_idx < len(
                self.chunk_files
            ):
                # Get next chunk to prefetch
                chunk_file = self.chunk_files[chunk_idx]

                # Extract chunk ID from filename (handle multiple extensions)
                filename = chunk_file.name
                # Find the chunk ID pattern (e.g., "chunk_0001")
                import re

                match = re.search(r"chunk_(\d+)", filename)
                if match:
                    chunk_id = int(match.group(1))
                else:
                    # Fallback if pattern not found
                    chunk_id = 0
                    logger.warning(
                        f"Could not extract chunk ID from filename: {filename}"
                    )

                try:
                    # Extract chunk
                    chunk_dir = self.cache_dir / f"chunk_{chunk_id:06d}"

                    # Check if already extracted
                    if not (chunk_dir / "samples.pkl").exists():
                        logger.info(f"Prefetching chunk {chunk_id} from {chunk_file}")
                        with tarfile.open(chunk_file, "r:*") as tar:
                            # Extract everything at once
                            chunk_dir.mkdir(exist_ok=True, parents=True)
                            logger.info(f"Extracting all files for chunk {chunk_id}")
                            tar.extractall(path=self.cache_dir)

                    # Load samples
                    samples_file = chunk_dir / "samples.pkl"
                    with open(samples_file, "rb") as f:
                        samples = pickle.load(f)

                    # Preprocess all samples in the chunk to avoid CPU bottlenecks
                    logger.info(
                        f"Preprocessing all {len(samples)} samples in chunk {chunk_id}"
                    )
                    preprocessed_samples = preprocess_samples_multithreaded(
                        samples, chunk_dir, chunk_id, self.transform
                    )

                    with self.lock:
                        self.chunks_in_memory.add(chunk_id)

                    self.chunk_queue.put((chunk_id, preprocessed_samples, chunk_dir))
                    logger.info(
                        f"Chunk {chunk_id} prefetched, preprocessed, and added to queue"
                    )

                    chunk_idx += 1

                except Exception as e:
                    logger.error(f"Error prefetching chunk {chunk_id}: {e}")
                    chunk_idx += 1

            time.sleep(0.1)

    def get_next_chunk(self):
        """Get the next prefetched chunk."""
        try:
            return self.chunk_queue.get(
                timeout=300
            )  # If loading a new chunk takes too long, log a warning and return None
        except queue.Empty:
            logger.warning("Timeout waiting for next chunk")
            return None

    def release_chunk(self, chunk_id):
        """Release a chunk from memory."""
        with self.lock:
            if chunk_id in self.chunks_in_memory:
                self.chunks_in_memory.remove(chunk_id)
                logger.info(f"Released chunk {chunk_id} from memory")

    def cleanup(self):
        """Clean up resources."""
        self.stop_event.set()
        if self.prefetcher_thread.is_alive():
            self.prefetcher_thread.join(timeout=5)

        # Clear queue
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break


class StreamingChunkedDataset(IterableDataset):
    """
    Dataset that streams data from chunks, prefetching the next chunk
    while processing the current one.
    """

    def __init__(
        self,
        prefix,
        chunks_dir,
        chunk_indices=None,
        transform=default_transform,
        use_augmentation=False,
        augmentation_config=None,
        cache_dir=None,
        max_chunks_in_memory=2,
        shuffle=True,
        shuffle_chunks=True,
    ):
        """
        Initialize the streaming chunked dataset.

        Args:
            chunks_dir: Directory containing the chunked archives
            chunk_indices: List of chunk indices to use (if None, use all)
            transform: Optional transform to apply to the images
            use_augmentation: Whether to use data augmentation
            augmentation_config: Optional configuration for augmentation
            cache_dir: Directory to extract chunks to (if None, use temp dir)
            max_chunks_in_memory: Maximum number of chunks to keep in memory
            shuffle: Whether to shuffle samples within chunks
            shuffle_chunks: Whether to shuffle the order of chunks
        """
        self.chunks_dir = Path(chunks_dir + "/" + prefix)
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config
        self.max_chunks_in_memory = max_chunks_in_memory
        self.shuffle = shuffle
        self.shuffle_chunks = shuffle_chunks
        self.training = False  # For augmentation
        self.dataset_name = "Human3.6M"
        self.num_joints = 17  # Human3.6M has 17 joints

        # Find all chunk files
        chunk_files = sorted(list(self.chunks_dir.glob("*.tar.*")))

        if chunk_indices is not None:
            # Filter chunks by indices
            filtered_chunk_files = []
            for idx in chunk_indices:
                pattern = f"{idx:06d}.tar."
                matching_files = [f for f in chunk_files if pattern in str(f)]
                filtered_chunk_files.extend(matching_files)
            chunk_files = filtered_chunk_files

        self.chunk_files = chunk_files
        logger.info(f"Found {len(self.chunk_files)} chunk files")

        # Initialize augmentors if needed
        if self.use_augmentation:
            from augmentation import PoseAugmentor

            # Use provided config or default
            config = augmentation_config or {}
            self.augmentor = PoseAugmentor(**config)

        # Load or create cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir / prefix)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.using_temp_dir = False
        else:
            self.cache_dir = Path(tempfile.mkdtemp() + "/" + prefix)
            self.using_temp_dir = True

        # Shuffle chunks if needed
        if self.shuffle_chunks:
            import random

            random.shuffle(self.chunk_files)

        # Estimate the total number of samples for __len__
        self.estimated_length = 1000 * len(self.chunk_files)

    def __len__(self):
        """Return the estimated length of the dataset."""
        return self.estimated_length

    def __del__(self):
        """Clean up temporary directory if used."""
        if (
            hasattr(self, "using_temp_dir")
            and self.using_temp_dir
            and hasattr(self, "cache_dir")
        ):
            try:
                shutil.rmtree(self.cache_dir)
            except:
                pass

    def __iter__(self):
        """Return an iterator over the dataset."""
        # Create a worker-specific prefetcher
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Divide chunks among workers
        per_worker = int(np.ceil(len(self.chunk_files) / num_workers))
        chunk_files_for_worker = self.chunk_files[
            worker_id * per_worker : (worker_id + 1) * per_worker
        ]

        if not chunk_files_for_worker:
            logger.warning(f"Worker {worker_id} has no chunks to process")
            return iter([])  # Return empty iterator

        # Create worker-specific cache directory
        worker_cache_dir = self.cache_dir / f"worker_{worker_id}"
        worker_cache_dir.mkdir(exist_ok=True, parents=True)

        # Create prefetcher for this worker
        prefetcher = ChunkPrefetcher(
            self.chunks_dir,
            chunk_files_for_worker,
            worker_cache_dir,
            max_chunks_in_memory=self.max_chunks_in_memory,
            transform=self.transform,
        )

        # Return iterator
        return StreamingChunkedIterator(
            prefetcher,
            self.transform,
            self.use_augmentation,
            self.training,
            self.shuffle,
            self.augmentor if hasattr(self, "augmentor") else None,
        )


class StreamingChunkedIterator:
    """Iterator for the StreamingChunkedDataset."""

    def __init__(
        self,
        prefetcher,
        transform,
        use_augmentation,
        training,
        shuffle,
        augmentor=None,
    ):
        self.prefetcher = prefetcher
        self.use_augmentation = use_augmentation
        self.training = training
        self.shuffle = shuffle
        self.augmentor = augmentor

        # Current chunk data
        self.current_chunk_id = None
        self.current_samples = None
        self.current_chunk_dir = None

        # Sample indices for current chunk
        self.sample_indices = []
        self.sample_idx = 0

        # Get first chunk
        self._load_next_chunk()

    def __iter__(self):
        return self

    def __next__(self):
        """Get next sample."""
        # Return a single sample
        if self.sample_idx >= len(self.sample_indices):
            # Need to load next chunk
            if not self._load_next_chunk():
                raise StopIteration

        # Get sample
        sample_idx = self.sample_indices[self.sample_idx]
        result = self.current_samples[sample_idx]
        self.sample_idx += 1

        # Apply augmentation if enabled
        if self.use_augmentation and self.training and self.augmentor:
            result = self.augmentor(result)

        return result

    def _load_next_chunk(self):
        """Load the next chunk from the prefetcher."""
        # Release current chunk if any
        if self.current_chunk_id is not None:
            self.prefetcher.release_chunk(self.current_chunk_id)

        # Get next chunk
        chunk_data = self.prefetcher.get_next_chunk()
        if chunk_data is None:
            logger.info("All chunks done!")
            return False

        self.current_chunk_id, self.current_samples, self.current_chunk_dir = chunk_data
        logger.info(
            f"Loaded chunk {self.current_chunk_id} with {len(self.current_samples)} samples"
        )

        # Reset indices
        self.sample_indices = list(range(len(self.current_samples)))
        if self.shuffle:
            import random

            random.shuffle(self.sample_indices)
        self.sample_idx = 0
        return True
