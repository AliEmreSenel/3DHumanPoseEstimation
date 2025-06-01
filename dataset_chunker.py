import os
import json
import pickle
import tarfile
import shutil
import tempfile
import signal
import time
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from config import (
    IMAGES_PATH,
    PROCESSED_PATH,
    ANNOTATIONS_PATH,
)
from utils import world_to_camera_coords


class ProgressFileObject:
    def __init__(self, file_obj, progress_bar, total_size):
        self.file_obj = file_obj
        self.progress_bar = progress_bar
        self.total_size = total_size
        self.current = 0

    def write(self, data):
        data_len = len(data)
        self.current += data_len
        self.progress_bar.update(data_len)
        return self.file_obj.write(data)

    def read(self, size=None):
        if size:
            data = self.file_obj.read(size)
        else:
            data = self.file_obj.read()
        data_len = len(data)
        self.current += data_len
        self.progress_bar.update(data_len)
        return data

    def tell(self):
        return self.file_obj.tell()

    def seek(self, offset, whence=0):
        return self.file_obj.seek(offset, whence)

    def close(self):
        return self.file_obj.close()


class Human36MChunker:
    def __init__(
        self,
        subject_ids,
        output_dir,
        temp_dir=None,
        chunk_size=10000,
        compression="gz",  # Options: gz, bz2, xz
        include_images=True,
        include_depth=True,
        include_metadata=True,
        resume=False,
    ):
        """
        Process Human3.6M dataset into manageable chunks for easier upload and storage.

        Args:
            subject_ids: List of subject IDs to process
            output_dir: Directory to save the chunks (OneDrive)
            temp_dir: Local directory for temporary files (defaults to system temp)
            chunk_size: Number of samples per chunk
            compression: Compression type for tar archives
            include_images: Whether to include RGB images
            include_depth: Whether to include depth images
            include_metadata: Whether to include metadata files
            resume: Whether to resume from a previous run
        """
        self.subject_ids = subject_ids
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.compression = compression
        self.include_images = include_images
        self.include_depth = include_depth
        self.include_metadata = include_metadata
        self.resume = resume

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create temporary directory for processing (local)
        if temp_dir:
            self.temp_dir = Path(temp_dir) / "human36m_temp"
            self.temp_dir.mkdir(exist_ok=True, parents=True)
            self.using_system_temp = False
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="human36m_"))
            self.using_system_temp = True

        print(f"Using temporary directory: {self.temp_dir}")

        # Initialize sample list
        self.samples = []
        self.new_samples = []  # Track new samples separately

        # Initialize state
        self.state_file = self.output_dir / "chunker_state.json"

        # Try to load samples first if resuming
        if resume:
            samples_loaded = self._load_samples_file()
            if samples_loaded:
                # Rebuild state from samples if state file is corrupted
                self.state = (
                    self._rebuild_state_from_samples()
                    if not self._load_state()
                    else self.state
                )
            else:
                # If no samples file, try to load state or initialize new state
                self.state = (
                    self._load_state()
                    if self.state_file.exists()
                    else self._init_state()
                )
        else:
            # Not resuming, initialize new state
            self.state = self._init_state()

        # Set up signal handlers for graceful pause
        self._setup_signal_handlers()

        # Flag to indicate if processing should be paused
        self.paused = False

    def _init_state(self):
        """Initialize a new state."""
        return {
            "subjects_processed": [],
            "chunks_created": [],
            "chunks_uploaded": [],
            "current_subject": None,
            "current_chunk": None,
            "total_samples": 0,
            "processed_files": {},  # Track processed files by subject
            "last_chunk_index": -1,  # Track the last chunk index used
            "config": {
                "chunk_size": self.chunk_size,
                "compression": self.compression,
                "include_images": self.include_images,
                "include_depth": self.include_depth,
                "include_metadata": self.include_metadata,
                "subject_ids": self.subject_ids,
            },
        }

    def _rebuild_state_from_samples(self):
        """Rebuild state information from samples.pkl when state file is corrupted."""
        print("Rebuilding state information from samples.pkl...")

        # Start with a fresh state
        state = self._init_state()

        # Extract information from samples
        processed_files = {}
        subjects_processed = set()

        # Find existing chunks in the output directory
        chunks_created = []
        chunks_uploaded = []
        last_chunk_index = -1

        for chunk_file in self.output_dir.glob("human36m_chunk_*.tar.*"):
            chunk_name = chunk_file.name
            chunks_created.append(chunk_name)
            chunks_uploaded.append(
                chunk_name
            )  # Assume all existing chunks are uploaded

            # Extract chunk index
            try:
                idx = int(chunk_name.split("_")[2].split(".")[0])
                if idx > last_chunk_index:
                    last_chunk_index = idx
            except (IndexError, ValueError):
                continue

        # Process samples to extract subject and file information
        for sample in self.samples:
            subject = sample.get("subject")
            if subject:
                subjects_processed.add(subject)

                # Create a unique file ID
                file_id = f"{sample.get('cam_idx')}_{sample.get('action')}_{sample.get('subaction')}_{sample.get('frame_idx')}"

                if str(subject) not in processed_files:
                    processed_files[str(subject)] = []

                processed_files[str(subject)].append(file_id)

        # Update state with extracted information
        state["subjects_processed"] = list(subjects_processed)
        state["processed_files"] = processed_files
        state["chunks_created"] = chunks_created
        state["chunks_uploaded"] = chunks_uploaded
        state["last_chunk_index"] = last_chunk_index
        state["total_samples"] = len(self.samples)

        print("Rebuilt state from samples.pkl:")
        print(f"  Subjects processed: {state['subjects_processed']}")
        print(f"  Total samples: {state['total_samples']}")
        print(f"  Chunks created/uploaded: {len(state['chunks_created'])}")
        print(f"  Last chunk index: {state['last_chunk_index']}")

        return state

    def _load_state(self):
        """Load state from file."""
        if not self.state_file.exists():
            print("No state file found. Will try to rebuild from samples.pkl.")
            return None

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Verify config matches
            config = state.get("config", {})
            if (
                config.get("chunk_size") != self.chunk_size
                or config.get("compression") != self.compression
                or config.get("include_images") != self.include_images
                or config.get("include_depth") != self.include_depth
                or config.get("include_metadata") != self.include_metadata
            ):
                print(
                    "Warning: Configuration in state file doesn't match current settings."
                )
                print("Current settings will be used, but this may cause issues.")

            # Update config with current settings
            state["config"] = {
                "chunk_size": self.chunk_size,
                "compression": self.compression,
                "include_images": self.include_images,
                "include_depth": self.include_depth,
                "include_metadata": self.include_metadata,
                "subject_ids": self.subject_ids,
            }

            print("Resuming from previous state:")
            print(f"  Subjects processed: {state['subjects_processed']}")
            print(f"  Chunks created: {len(state['chunks_created'])}")
            print(f"  Chunks uploaded: {len(state['chunks_uploaded'])}")

            self.state = state
            return state
        except Exception as e:
            print(f"Error loading state file: {e}")
            print("Will try to rebuild from samples.pkl.")
            return None

    def _save_state(self):
        """Save current state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful pause."""
        # Handle Ctrl+C (SIGINT)
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Handle SIGTERM
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signals."""
        if self.paused:
            print("\nAlready paused. Press Ctrl+C again to exit.")
            # Restore original handler so another Ctrl+C will exit
            signal.signal(signal.SIGINT, self.original_sigint)
            return

        print("\nPausing processing. Please wait for current operation to complete...")
        self.paused = True

    def _check_pause(self):
        """Check if processing should be paused."""
        if self.paused:
            self._save_state()
            print("\nProcessing paused. Current state has been saved.")
            print("To resume, run the same command with --resume")
            sys.exit(0)

    def _load_samples_file(self):
        """Load samples from file if available."""
        samples_file = self.output_dir / "samples.pkl"
        if samples_file.exists() and self.resume:
            try:
                with open(samples_file, "rb") as f:
                    self.samples = pickle.load(f)
                print(
                    f"Loaded {len(self.samples)} existing samples from {samples_file}"
                )
                return True
            except Exception as e:
                print(f"Error loading samples file: {e}")
                return False
        return False

    def _save_samples_file(self):
        """Save samples to file for resuming."""
        samples_file = self.output_dir / "samples.pkl"

        # Combine existing samples with new samples
        all_samples = self.samples + self.new_samples

        with open(samples_file, "wb") as f:
            pickle.dump(all_samples, f)
        print(
            f"Saved {len(all_samples)} samples to {samples_file} ({len(self.samples)} existing + {len(self.new_samples)} new)"
        )

        # Update the samples list to include all samples
        self.samples = all_samples

    def process(self):
        """Process the dataset and create chunks."""
        print(f"Processing subjects: {self.subject_ids}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Output directory (OneDrive): {self.output_dir}")
        print(f"Local temp directory: {self.temp_dir}")
        print(f"Compression: {self.compression}")
        print(f"Resume: {self.resume}")

        try:
            # Initialize new samples list
            self.new_samples = []

            # Load and process data for each subject
            for subject_id in self.subject_ids:
                # Even if subject was processed before, we'll check for new files
                self.state["current_subject"] = subject_id
                self._save_state()

                self._load_subject_data(subject_id)

                # Mark subject as processed if it wasn't already
                if subject_id not in self.state["subjects_processed"]:
                    self.state["subjects_processed"].append(subject_id)

                self.state["current_subject"] = None
                self._save_state()

                # Check if should pause
                self._check_pause()

            # Update total samples count
            total_samples = len(self.samples) + len(self.new_samples)
            self.state["total_samples"] = total_samples
            self._save_state()

            # Save combined samples for debugging/resuming
            self._save_samples_file()

            # Create chunks for new samples only
            if len(self.new_samples) > 0:
                print(f"Creating chunks for {len(self.new_samples)} new samples...")
                self._create_chunks()
            else:
                print("No new samples found. No new chunks needed.")

        finally:
            # Clean up temporary directory
            if self.using_system_temp:
                print(f"Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            else:
                print(f"Temporary files remain in: {self.temp_dir}")
                print("You may want to clean this up manually to free disk space.")

        print(f"Processing complete. Total samples: {len(self.samples)}")

    def _load_subject_data(self, subject_id):
        """Load data for a specific subject."""
        print(f"Loading data for subject {subject_id}...")

        # Check if we've processed this subject before
        is_resume_subject = (
            self.resume and subject_id in self.state["subjects_processed"]
        )

        # Get previously processed files for this subject if resuming
        processed_files = set(
            self.state.get("processed_files", {}).get(str(subject_id), [])
        )

        # Load annotations
        data_file = ANNOTATIONS_PATH / f"Human36M_subject{subject_id}_data.json"
        camera_file = ANNOTATIONS_PATH / f"Human36M_subject{subject_id}_camera.json"
        joint_file = ANNOTATIONS_PATH / f"Human36M_subject{subject_id}_joint_3d.json"

        # Load all data at once
        with open(data_file, "r") as f:
            data_info = json.load(f)

        with open(camera_file, "r") as f:
            camera_info = json.load(f)

        with open(joint_file, "r") as f:
            joint_info = json.load(f)

        # Create lookup dictionaries
        annotation_lookup = {ann["image_id"]: ann for ann in data_info["annotations"]}

        # Pre-check which files exist
        depth_files = {}
        metadata_files = {}

        # Gather all potential file paths
        potential_paths = []
        for img_data in data_info["images"]:
            file_name = img_data["file_name"]
            folder_name = os.path.dirname(file_name)
            base_name = os.path.basename(file_name).split(".")[0]
            depth_file = PROCESSED_PATH / folder_name / f"{base_name}_depth.png"
            metadata_file = PROCESSED_PATH / folder_name / f"{base_name}.json"
            potential_paths.append(
                (img_data["id"], str(depth_file), str(metadata_file))
            )

        # Check file existence
        print(f"Checking file existence for {len(potential_paths)} potential files...")
        for img_id, depth_path, metadata_path in tqdm(
            potential_paths, desc="Checking files"
        ):
            if os.path.exists(depth_path):
                depth_files[img_id] = depth_path
            if os.path.exists(metadata_path):
                metadata_files[img_id] = metadata_path

            # Check for pause periodically
            if len(depth_files) % 1000 == 0:
                self._check_pause()

        # Process images
        image_data_list = []
        new_files_count = 0

        for img_data in data_info["images"]:
            image_id = img_data["id"]

            # Skip if files don't exist
            if image_id not in depth_files or image_id not in metadata_files:
                continue

            # Skip if no annotation
            if image_id not in annotation_lookup:
                continue

            # Skip if already processed and we're resuming
            file_id = f"{image_id}_{img_data['cam_idx']}_{img_data['frame_idx']}"
            if is_resume_subject and file_id in processed_files:
                continue

            # This is a new file
            if is_resume_subject:
                new_files_count += 1

            # Add to processing list
            image_data_list.append(
                {
                    "image_id": image_id,
                    "file_name": img_data["file_name"],
                    "subject": img_data["subject"],
                    "action_idx": img_data["action_idx"],
                    "subaction_idx": img_data["subaction_idx"],
                    "cam_idx": img_data["cam_idx"],
                    "frame_idx": img_data["frame_idx"],
                    "annotation": annotation_lookup[image_id],
                    "depth_file": depth_files[image_id],
                    "metadata_file": metadata_files[image_id],
                    "file_id": file_id,  # Store the unique file ID
                }
            )

        if is_resume_subject:
            print(
                f"Found {new_files_count} new files for previously processed subject {subject_id}"
            )

        print(
            f"Found {len(image_data_list)} valid samples to process for subject {subject_id}"
        )

        if len(image_data_list) == 0:
            # No new files to process
            return

        # Process in parallel
        cpu_count = multiprocessing.cpu_count()
        optimal_batch_size = max(100, len(image_data_list) // (cpu_count * 2))
        batches = [
            image_data_list[i : i + optimal_batch_size]
            for i in range(0, len(image_data_list), optimal_batch_size)
        ]

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_image_batch, batch, camera_info, joint_info
                )
                futures.append(future)

            # Collect results
            all_samples = []
            processed_file_ids = []

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing subject {subject_id}",
            ):
                batch_samples, batch_file_ids = future.result()
                if batch_samples:
                    all_samples.extend(batch_samples)
                    processed_file_ids.extend(batch_file_ids)

                # Check for pause periodically
                self._check_pause()

            # Add to new samples list
            self.new_samples.extend(all_samples)

            # Update processed files in state
            if str(subject_id) not in self.state.get("processed_files", {}):
                self.state.setdefault("processed_files", {})[str(subject_id)] = []

            self.state["processed_files"][str(subject_id)].extend(processed_file_ids)

        print(f"Loaded {len(all_samples)} new samples for subject {subject_id}")

    def _process_image_batch(self, image_batch, camera_info, joint_info):
        """Process a batch of images in parallel."""
        batch_samples = []
        batch_file_ids = []

        for img_data in image_batch:
            try:
                # Get camera parameters
                camera_params = camera_info[str(img_data["cam_idx"])]
                R = camera_params["R"]
                t = camera_params["t"]
                f = camera_params["f"]
                c = camera_params["c"]

                # Get 3D joint positions
                action_idx = img_data["action_idx"]
                subaction_idx = img_data["subaction_idx"]
                frame_idx = img_data["frame_idx"]

                try:
                    joints_world = joint_info[str(action_idx)][str(subaction_idx)][
                        str(frame_idx)
                    ]
                except KeyError:
                    continue

                # Convert to camera coordinates
                joints_camera = world_to_camera_coords(joints_world, R, t)

                # Store sample data
                batch_samples.append(
                    {
                        "image_file": str(IMAGES_PATH / img_data["file_name"]),
                        "depth_file": img_data["depth_file"],
                        "metadata_file": img_data["metadata_file"],
                        "joints_3d": joints_camera,
                        "camera_params": {"R": R, "t": t, "f": f, "c": c},
                        "bbox": img_data["annotation"]["bbox"],
                        "action": action_idx,
                        "subaction": subaction_idx,
                        "frame_idx": frame_idx,
                        "cam_idx": img_data["cam_idx"],
                        "subject": img_data["subject"],
                    }
                )

                # Store the file ID
                batch_file_ids.append(img_data["file_id"])

            except Exception:
                # Skip problematic samples
                continue

        return batch_samples, batch_file_ids

    def _get_directory_size(self, directory):
        """Calculate the total size of a directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def _create_chunks(self):
        """Create chunks of the dataset."""
        # Determine the starting chunk index
        if self.state["chunks_created"]:
            # Extract the highest chunk index from existing chunk filenames
            chunk_indices = []
            for chunk_name in self.state["chunks_created"]:
                try:
                    # Extract the index from filenames like "human36m_chunk_0042.tar.gz"
                    idx = int(chunk_name.split("_")[2].split(".")[0])
                    chunk_indices.append(idx)
                except (IndexError, ValueError):
                    continue

            start_chunk_idx = max(chunk_indices) + 1 if chunk_indices else 0
        else:
            # If no chunks created yet, check for existing chunk files in the output directory
            chunk_indices = []
            for chunk_file in self.output_dir.glob("human36m_chunk_*.tar.*"):
                try:
                    idx = int(chunk_file.name.split("_")[2].split(".")[0])
                    chunk_indices.append(idx)
                except (IndexError, ValueError):
                    continue

            start_chunk_idx = max(chunk_indices) + 1 if chunk_indices else 0

        print(f"Starting from chunk index {start_chunk_idx} (after last created chunk)")

        # Calculate number of chunks needed for new samples only
        num_samples = len(self.new_samples)
        num_chunks = (num_samples + self.chunk_size - 1) // self.chunk_size

        if num_samples == 0:
            print("No new samples to process. Skipping chunk creation.")
            return

        print(
            f"Creating {num_chunks} new chunks starting from index {start_chunk_idx}..."
        )

        for i in range(num_chunks):
            chunk_idx = start_chunk_idx + i

            # Skip already uploaded chunks
            chunk_filename = f"human36m_chunk_{chunk_idx:04d}.tar.{self.compression}"
            if chunk_filename in self.state["chunks_uploaded"]:
                print(
                    f"Chunk {i+1}/{num_chunks} (index {chunk_idx}) already uploaded. Skipping."
                )
                continue

            # Skip already created chunks if they exist in OneDrive
            if (
                chunk_filename in self.state["chunks_created"]
                and (self.output_dir / chunk_filename).exists()
            ):
                print(
                    f"Chunk {i+1}/{num_chunks} (index {chunk_idx}) already created. Uploading..."
                )
                self._upload_chunk(chunk_idx, num_chunks)
                continue

            # Update state
            self.state["current_chunk"] = chunk_idx
            self._save_state()

            # Get samples for this chunk from new samples only
            start_sample_idx = i * self.chunk_size
            end_sample_idx = min((i + 1) * self.chunk_size, num_samples)

            chunk_samples = self.new_samples[start_sample_idx:end_sample_idx]

            # Create chunk directory in local temp
            chunk_dir = self.temp_dir / f"chunk_{chunk_idx:04d}"
            chunk_dir.mkdir(exist_ok=True)

            # Create data directory
            data_dir = chunk_dir / "data"
            data_dir.mkdir(exist_ok=True)

            # Process samples
            processed_samples = []

            for j, sample in enumerate(
                tqdm(
                    chunk_samples,
                    desc=f"Processing chunk {i+1}/{num_chunks} (index {chunk_idx})",
                )
            ):
                sample_id = f"{j:06d}"
                sample_dir = data_dir / sample_id
                sample_dir.mkdir(exist_ok=True)

                # Copy files
                processed_sample = self._process_sample(sample, sample_dir, sample_id)
                processed_samples.append(processed_sample)

                # Check for pause periodically
                if j % 100 == 0:
                    self._check_pause()

            # Save sample metadata
            with open(chunk_dir / "samples.pkl", "wb") as f:
                pickle.dump(processed_samples, f)

            # Create archive in local temp first
            local_archive_path = (
                self.temp_dir / f"human36m_chunk_{chunk_idx:04d}.tar.{self.compression}"
            )

            # Calculate total size for progress bar
            total_size = self._get_directory_size(chunk_dir)

            # Create archive with progress bar (locally)
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Creating archive {i+1}/{num_chunks} (index {chunk_idx})",
            ) as pbar:
                with open(local_archive_path, "wb") as f:
                    wrapped_file = ProgressFileObject(f, pbar, total_size)
                    with tarfile.open(
                        fileobj=wrapped_file, mode=f"w:{self.compression}"
                    ) as tar:
                        for root, dirs, files in os.walk(chunk_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.join(
                                    f"chunk_{chunk_idx:04d}",
                                    os.path.relpath(file_path, chunk_dir),
                                )
                                tar.add(file_path, arcname=arcname)

                                # Check for pause periodically
                                if pbar.n % (100 * 1024 * 1024) == 0:  # Every 100MB
                                    self._check_pause()

            # Clean up chunk directory to free space
            shutil.rmtree(chunk_dir)

            # Get final archive size
            archive_size = os.path.getsize(local_archive_path)
            print(
                f"Created local archive: {local_archive_path} ({archive_size / (1024*1024):.2f} MB)"
            )

            # Mark chunk as created
            self.state["chunks_created"].append(chunk_filename)
            self.state["last_chunk_index"] = chunk_idx
            self._save_state()

            # Upload to OneDrive
            self._upload_chunk(chunk_idx, num_chunks, local_archive_path)

            # Check if should pause
            self._check_pause()

    def _upload_chunk(self, chunk_idx, num_chunks, local_archive_path=None):
        """Upload a chunk to OneDrive."""
        chunk_filename = f"human36m_chunk_{chunk_idx:04d}.tar.{self.compression}"
        onedrive_archive_path = self.output_dir / chunk_filename

        # If local archive path not provided, check if it exists in temp dir
        if local_archive_path is None:
            local_archive_path = self.temp_dir / chunk_filename
            if not local_archive_path.exists():
                print(f"Local archive not found: {local_archive_path}")
                return False

        # Get archive size
        archive_size = os.path.getsize(local_archive_path)

        # Copy to OneDrive with progress and pause support
        with tqdm(
            total=archive_size,
            unit="B",
            unit_scale=True,
            desc=f"Copying to OneDrive {chunk_idx+1}/{num_chunks}",
        ) as pbar:
            try:
                with open(local_archive_path, "rb") as src_file:
                    with open(onedrive_archive_path, "wb") as dst_file:
                        while True:
                            buffer = src_file.read(8 * 1024 * 1024)  # 8MB buffer
                            if not buffer:
                                break
                            dst_file.write(buffer)
                            pbar.update(len(buffer))

                            # Check for pause
                            self._check_pause()

                            # Add a small delay to avoid OneDrive rate limits
                            time.sleep(0.01)
            except Exception as e:
                print(f"Error copying to OneDrive: {e}")
                return False

        # Remove local archive to free space
        os.remove(local_archive_path)

        # Mark chunk as uploaded
        self.state["chunks_uploaded"].append(chunk_filename)
        self.state["current_chunk"] = None
        self._save_state()

        print(
            f"Copied chunk {chunk_idx+1}/{num_chunks} to OneDrive: {onedrive_archive_path}"
        )
        return True

    def _process_sample(self, sample, sample_dir, sample_id):
        """Process a single sample and copy files to the chunk directory."""
        processed_sample = {
            "sample_id": sample_id,
            "joints_3d": sample["joints_3d"],
            "camera_params": sample["camera_params"],
            "bbox": sample["bbox"],
            "action": sample["action"],
            "subaction": sample["subaction"],
            "frame_idx": sample["frame_idx"],
            "cam_idx": sample["cam_idx"],
            "subject": sample["subject"],
        }

        # Copy image if included
        if self.include_images:
            image_path = sample["image_file"]
            image_ext = os.path.splitext(image_path)[1]
            dest_image_path = sample_dir / f"image{image_ext}"

            try:
                shutil.copy(image_path, dest_image_path)
                processed_sample["image_file"] = str(
                    dest_image_path.relative_to(sample_dir.parent.parent)
                )
            except Exception:
                # If copy fails, don't include the image
                processed_sample["image_file"] = None

        # Copy depth if included
        if self.include_depth:
            depth_path = sample["depth_file"]
            depth_ext = os.path.splitext(depth_path)[1]
            dest_depth_path = sample_dir / f"depth{depth_ext}"

            try:
                shutil.copy(depth_path, dest_depth_path)
                processed_sample["depth_file"] = str(
                    dest_depth_path.relative_to(sample_dir.parent.parent)
                )
            except Exception:
                # If copy fails, don't include the depth
                processed_sample["depth_file"] = None

        # Copy metadata if included
        if self.include_metadata:
            metadata_path = sample["metadata_file"]
            dest_metadata_path = sample_dir / "metadata.json"

            try:
                shutil.copy(metadata_path, dest_metadata_path)
                processed_sample["metadata_file"] = str(
                    dest_metadata_path.relative_to(sample_dir.parent.parent)
                )

                # Load metadata to extract keypoints
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                processed_sample["keypoints"] = metadata.get("keypoints", [])
                processed_sample["image_size"] = metadata.get("image_size", [])
                processed_sample["depth_min"] = metadata.get("depth_min", 0)
                processed_sample["depth_max"] = metadata.get("depth_max", 1)
            except Exception:
                # If copy fails, don't include the metadata
                processed_sample["metadata_file"] = None

        return processed_sample


def main():
    parser = argparse.ArgumentParser(
        description="Process Human3.6M dataset into chunks"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 5, 6, 7, 8, 9, 11],
        help="Subject IDs to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./chunked_dataset",
        help="Output directory for chunks (OneDrive)",
    )
    parser.add_argument(
        "--temp",
        type=str,
        default=None,
        help="Local temporary directory (defaults to system temp)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of samples per chunk"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gz",
        choices=["gz", "bz2", "xz"],
        help="Compression type for tar archives",
    )
    parser.add_argument(
        "--no-images",
        action="store_false",
        dest="include_images",
        help="Don't include RGB images in chunks",
    )
    parser.add_argument(
        "--no-depth",
        action="store_false",
        dest="include_depth",
        help="Don't include depth images in chunks",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_false",
        dest="include_metadata",
        help="Don't include metadata files in chunks",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run"
    )

    args = parser.parse_args()

    chunker = Human36MChunker(
        subject_ids=args.subjects,
        output_dir=args.output,
        temp_dir=args.temp,
        chunk_size=args.chunk_size,
        compression=args.compression,
        include_images=args.include_images,
        include_depth=args.include_depth,
        include_metadata=args.include_metadata,
        resume=args.resume,
    )

    chunker.process()


if __name__ == "__main__":
    main()
