import os
import json
import pickle
import tarfile
import shutil
import tempfile
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import concurrent.futures
import threading # For locks if ever needed, though current design avoids shared list appends

class DatasetShuffler:
    """
    Unpacks dataset chunks, validates samples, shuffles them, and repacks them into new chunks.
    Uses multi-threading for unpacking/validation and repacking.
    Focuses on keeping images and their metadata together, handling potentially invalid data
    (e.g., missing or 0-byte depth images/metadata JSONs), and managing memory.
    Can optionally keep extracted original chunks to speed up subsequent runs.
    Allows separate directories for extracted originals and new chunk build areas.
    """
    def __init__(self, input_dir, output_dir,
                 extracted_originals_dir_path, # Path for storing/finding extracted original chunks
                 new_chunks_build_dir_path,    # Path for temporarily building new chunks
                 repack_chunk_size=10000, output_compression="gz",
                 keep_extracted_originals=False, num_workers=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extracted_originals_dir = Path(extracted_originals_dir_path)
        self.new_chunks_build_dir = Path(new_chunks_build_dir_path)
        self.repack_chunk_size = repack_chunk_size
        self.output_compression = output_compression
        self.keep_extracted_originals = keep_extracted_originals
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_originals_dir.mkdir(parents=True, exist_ok=True)
        self.new_chunks_build_dir.mkdir(parents=True, exist_ok=True)

        self.invalid_data_log_entries = [] # Collect log entries from threads

    def _unpack_chunk(self, chunk_path, extract_to_dir):
        """Unpacks a single tar archive."""
        local_invalid_log = []
        try:
            extract_to_dir.mkdir(parents=True, exist_ok=True)
            # print(f"Unpacking {chunk_path.name} to {extract_to_dir}...") # tqdm will show progress
            with tarfile.open(chunk_path, "r:*") as tar:
                tar.extractall(path=extract_to_dir)
            return True, local_invalid_log
        except tarfile.ReadError as e:
            error_msg = f"Failed to read/unpack chunk (ReadError): {chunk_path.name} - Error: {e}"
            local_invalid_log.append(error_msg)
            return False, local_invalid_log
        except Exception as e:
            error_msg = f"An unexpected error occurred while unpacking {chunk_path.name}: {e}"
            local_invalid_log.append(error_msg)
            return False, local_invalid_log

    def _load_metadata_from_extracted_chunk(self, chunk_extract_path):
        """Loads samples.pkl from an extracted chunk's content."""
        local_invalid_log = []
        extracted_content_dirs = [d for d in chunk_extract_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        if not extracted_content_dirs:
            error_msg = f"Could not locate chunk content root (e.g., 'chunk_XXXX' folder) in {chunk_extract_path}."
            local_invalid_log.append(error_msg)
            return None, None, local_invalid_log
        
        data_base_path = extracted_content_dirs[0]
        metadata_path = data_base_path / "samples.pkl"
        if not metadata_path.exists():
            error_msg = f"samples.pkl not found at expected location: {metadata_path}."
            local_invalid_log.append(error_msg)
            return None, data_base_path, local_invalid_log
        try:
            with open(metadata_path, "rb") as f:
                samples_data = pickle.load(f)
            if not isinstance(samples_data, list):
                error_msg = f"Metadata file {metadata_path} does not contain a list as expected."
                local_invalid_log.append(error_msg)
                return None, data_base_path, local_invalid_log
            return samples_data, data_base_path, local_invalid_log
        except Exception as e:
            error_msg = f"Error loading or parsing metadata from {metadata_path}: {e}"
            local_invalid_log.append(error_msg)
            return None, data_base_path, local_invalid_log

    def _validate_sample_and_get_files(self, sample_metadata, data_base_path, original_chunk_name_for_logging):
        """Validates a single sample: checks file existence and non-zero size for essential files."""
        local_invalid_log = []
        if not isinstance(sample_metadata, dict):
            local_invalid_log.append(f"Invalid sample format (not a dict) in chunk {original_chunk_name_for_logging}.")
            return False, [], local_invalid_log

        is_sample_valid = True
        files_to_copy_details = []

        img_rel_path_str = sample_metadata.get("image_file")
        if not img_rel_path_str:
            local_invalid_log.append(f"INVALID SAMPLE: 'image_file' key missing or path empty in metadata for a sample in chunk {original_chunk_name_for_logging}. Sample excluded.")
            is_sample_valid = False
        else:
            img_abs_path = data_base_path / img_rel_path_str
            if not img_abs_path.exists():
                local_invalid_log.append(f"INVALID SAMPLE: Missing image_file: {img_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            elif img_abs_path.stat().st_size == 0:
                local_invalid_log.append(f"INVALID SAMPLE: image_file is 0 bytes: {img_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            else:
                files_to_copy_details.append({"type": "image", "src_abs_path": img_abs_path, "original_rel_path": img_rel_path_str})

        depth_rel_path_str = sample_metadata.get("depth_file")
        if depth_rel_path_str:
            depth_abs_path = data_base_path / depth_rel_path_str
            if not depth_abs_path.exists():
                local_invalid_log.append(f"INVALID SAMPLE: Missing expected depth_file: {depth_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            elif depth_abs_path.stat().st_size == 0:
                local_invalid_log.append(f"INVALID SAMPLE: Expected depth_file is 0 bytes: {depth_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            else:
                files_to_copy_details.append({"type": "depth", "src_abs_path": depth_abs_path, "original_rel_path": depth_rel_path_str})

        meta_json_rel_path_str = sample_metadata.get("metadata_file")
        if not meta_json_rel_path_str:
            local_invalid_log.append(f"INVALID SAMPLE: 'metadata_file' (JSON) key missing or path empty in metadata for a sample in chunk {original_chunk_name_for_logging}. Sample excluded.")
            is_sample_valid = False
        else:
            meta_json_abs_path = data_base_path / meta_json_rel_path_str
            if not meta_json_abs_path.exists():
                local_invalid_log.append(f"INVALID SAMPLE: Missing mandatory metadata_file (JSON): {meta_json_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            elif meta_json_abs_path.stat().st_size == 0:
                local_invalid_log.append(f"INVALID SAMPLE: Mandatory metadata_file (JSON) is 0 bytes: {meta_json_abs_path} (in chunk {original_chunk_name_for_logging}). Sample excluded.")
                is_sample_valid = False
            else:
                files_to_copy_details.append({"type": "metadata_json", "src_abs_path": meta_json_abs_path, "original_rel_path": meta_json_rel_path_str})
        
        if not is_sample_valid:
            return False, [], local_invalid_log
        return True, files_to_copy_details, local_invalid_log

    def _process_one_original_chunk(self, chunk_path):
        """Worker function to process a single original chunk."""
        chunk_valid_samples = []
        chunk_invalid_log = []

        base_name = chunk_path.name
        for ext in [".tar.gz", ".tar.bz2", ".tar.xz", ".tar"]:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        chunk_specific_extract_dir = self.extracted_originals_dir / f"{base_name}_content"

        samples_data_list = None
        data_base_path = None

        if self.keep_extracted_originals and chunk_specific_extract_dir.is_dir():
            # print(f"Found existing directory for {chunk_path.name}. Attempting to load...")
            loaded_samples_data, loaded_data_base_path, log_entries = self._load_metadata_from_extracted_chunk(chunk_specific_extract_dir)
            chunk_invalid_log.extend(log_entries)
            if loaded_samples_data is not None and loaded_data_base_path is not None:
                # print(f"Successfully loaded metadata from existing extraction for {chunk_path.name}.")
                samples_data_list = loaded_samples_data
                data_base_path = loaded_data_base_path
            else:
                # print(f"Failed to load from existing directory {chunk_specific_extract_dir}, will re-unpack.")
                if chunk_specific_extract_dir.exists(): shutil.rmtree(chunk_specific_extract_dir)
        
        if samples_data_list is None:
            if chunk_specific_extract_dir.exists(): shutil.rmtree(chunk_specific_extract_dir)
            
            unpacked_ok, log_entries = self._unpack_chunk(chunk_path, chunk_specific_extract_dir)
            chunk_invalid_log.extend(log_entries)
            if not unpacked_ok:
                if chunk_specific_extract_dir.exists(): shutil.rmtree(chunk_specific_extract_dir)
                return [], chunk_invalid_log # Return empty samples and logs for this failed chunk
            
            samples_data_list, data_base_path, log_entries = self._load_metadata_from_extracted_chunk(chunk_specific_extract_dir)
            chunk_invalid_log.extend(log_entries)

        if samples_data_list is None:
            if chunk_specific_extract_dir.exists() and not self.keep_extracted_originals:
                shutil.rmtree(chunk_specific_extract_dir)
            return [], chunk_invalid_log

        for sample_meta in samples_data_list:
            is_valid, files_to_copy, log_entries = self._validate_sample_and_get_files(sample_meta, data_base_path, chunk_path.name)
            chunk_invalid_log.extend(log_entries)
            if is_valid:
                chunk_valid_samples.append({
                    "original_metadata": sample_meta,
                    "files_to_copy": files_to_copy,
                    "original_chunk_name": chunk_path.name,
                })
        return chunk_valid_samples, chunk_invalid_log

    def collect_all_valid_samples(self):
        """Collects valid samples from all original chunks using multiple threads."""
        all_valid_samples_details = []
        
        potential_chunk_paths = []
        for ext_pattern in ["*.tar", "*.tar.gz", "*.tar.bz2", "*.tar.xz"]:
            potential_chunk_paths.extend(self.input_dir.glob(ext_pattern))
        original_chunk_paths = sorted(list(set(p for p in potential_chunk_paths if p.is_file())))

        if not original_chunk_paths:
            msg = f"No .tar[.compression] chunks found in {self.input_dir}"
            print(msg)
            self.invalid_data_log_entries.append(msg)
            return []

        print(f"Found {len(original_chunk_paths)} potential chunk archives. Processing with {self.num_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_one_original_chunk, chunk_path) for chunk_path in original_chunk_paths]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Original Chunks"):
                try:
                    chunk_samples, chunk_logs = future.result()
                    all_valid_samples_details.extend(chunk_samples)
                    self.invalid_data_log_entries.extend(chunk_logs)
                except Exception as exc:
                    log_msg = f"A worker for original chunk processing generated an exception: {exc}"
                    print(log_msg) # Also print directly for immediate visibility
                    self.invalid_data_log_entries.append(log_msg)
        return all_valid_samples_details

    def _process_one_new_chunk(self, chunk_idx, samples_for_this_chunk, total_new_chunks):
        """Worker function to build and archive one new chunk."""
        local_invalid_log = []
        new_chunk_numeric_id_str = f"{chunk_idx:04d}"
        
        tar_extension = f".tar.{self.output_compression}" if self.output_compression else ".tar"
        new_tar_filename = f"shuffled_dataset_chunk_{new_chunk_numeric_id_str}{tar_extension}"
        
        # Physical directory on disk for building this specific new chunk's content
        current_new_chunk_build_dir = self.new_chunks_build_dir / f"build_chunk_{new_chunk_numeric_id_str}"
        current_new_chunk_data_dir = current_new_chunk_build_dir / "data"
        
        if current_new_chunk_build_dir.exists(): shutil.rmtree(current_new_chunk_build_dir) # Clean from any prior attempt
        current_new_chunk_build_dir.mkdir(parents=True)
        current_new_chunk_data_dir.mkdir(parents=True)

        new_chunk_sample_metadata_list = []
        
        # tqdm description for this specific worker can be tricky with outer tqdm.
        # Consider logging start/end or using a shared progress mechanism if detailed per-thread progress is vital.
        # For now, the outer loop's tqdm will cover overall chunk creation.

        for sample_j, original_sample_detail in enumerate(samples_for_this_chunk):
            new_sample_id_in_chunk_str = f"{sample_j:06d}"
            new_sample_files_destination_dir = current_new_chunk_data_dir / new_sample_id_in_chunk_str
            new_sample_files_destination_dir.mkdir()
            repacked_sample_metadata = original_sample_detail["original_metadata"].copy()
            
            for file_to_copy_info in original_sample_detail["files_to_copy"]:
                src_abs_path = file_to_copy_info["src_abs_path"]
                original_relative_path = Path(file_to_copy_info["original_rel_path"])
                new_path_for_pkl = Path("data") / new_sample_id_in_chunk_str / original_relative_path.name
                dest_abs_path_for_file = new_sample_files_destination_dir / original_relative_path.name
                try:
                    shutil.copy2(src_abs_path, dest_abs_path_for_file)
                    file_type_key = file_to_copy_info["type"]
                    if file_type_key == "image": repacked_sample_metadata["image_file"] = str(new_path_for_pkl)
                    elif file_type_key == "depth": repacked_sample_metadata["depth_file"] = str(new_path_for_pkl)
                    elif file_type_key == "metadata_json": repacked_sample_metadata["metadata_file"] = str(new_path_for_pkl)
                except FileNotFoundError:
                    crit_error = f"CRITICAL ERROR: Source file {src_abs_path} not found during repacking for new chunk {new_tar_filename}."
                    local_invalid_log.append(crit_error)
                except Exception as e:
                    copy_error = f"Error copying file {src_abs_path} to {dest_abs_path_for_file} for {new_tar_filename}: {e}"
                    local_invalid_log.append(copy_error)
            repacked_sample_metadata["sample_id"] = new_sample_id_in_chunk_str
            new_chunk_sample_metadata_list.append(repacked_sample_metadata)

        new_samples_pkl_path = current_new_chunk_build_dir / "samples.pkl"
        with open(new_samples_pkl_path, "wb") as f: pickle.dump(new_chunk_sample_metadata_list, f)

        final_tar_archive_path = self.output_dir / new_tar_filename
        # print(f"Creating new archive: {final_tar_archive_path}") # tqdm will show overall progress
        
        archive_internal_root_name = f"chunk_{new_chunk_numeric_id_str}"
        try:
            with tarfile.open(final_tar_archive_path, f"w:{self.output_compression}") as tar:
                tar.add(new_samples_pkl_path, arcname=Path(archive_internal_root_name) / "samples.pkl")
                tar.add(current_new_chunk_data_dir, arcname=Path(archive_internal_root_name) / "data")
        except Exception as e:
            tar_error = f"Error creating tar archive {final_tar_archive_path}: {e}"
            local_invalid_log.append(tar_error)
        
        if current_new_chunk_build_dir.exists(): shutil.rmtree(current_new_chunk_build_dir)
        return local_invalid_log


    def shuffle_and_repack_samples(self, all_samples_details):
        """Shuffles samples and repacks them into new tar archives using multiple threads."""
        if not all_samples_details:
            print("No valid samples collected. Nothing to shuffle or repack.")
            return

        random.shuffle(all_samples_details)
        print(f"Shuffled {len(all_samples_details)} valid samples.")

        num_new_chunks = (len(all_samples_details) + self.repack_chunk_size - 1) // self.repack_chunk_size
        
        print(f"Repacking into {num_new_chunks} new chunks using {self.num_workers} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(num_new_chunks):
                start_idx = i * self.repack_chunk_size
                end_idx = min((i + 1) * self.repack_chunk_size, len(all_samples_details))
                samples_for_this_chunk = all_samples_details[start_idx:end_idx]
                futures.append(executor.submit(self._process_one_new_chunk, i, samples_for_this_chunk, num_new_chunks))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Repacking New Chunks"):
                try:
                    chunk_logs = future.result()
                    self.invalid_data_log_entries.extend(chunk_logs)
                except Exception as exc:
                    log_msg = f"A worker for new chunk repacking generated an exception: {exc}"
                    print(log_msg) # Also print directly
                    self.invalid_data_log_entries.append(log_msg)
        
        # Clean up the root build directory for new chunks, as individual build dirs are cleaned by workers
        # This is more of a safeguard if a worker failed before cleaning its own dir.
        if self.new_chunks_build_dir.exists():
             # Check if it's empty or contains only unexpected leftovers
            if not any(self.new_chunks_build_dir.iterdir()):
                # print(f"Cleaning up empty new chunks build directory: {self.new_chunks_build_dir}") # Usually done by TemporaryDirectory
                pass # If it was a system temp, it's handled by TemporaryDirectory context manager in main.
                     # If user-provided, we generally don't delete the root they gave.
            else: # Contains unexpected leftovers
                # print(f"Warning: New chunks build directory {self.new_chunks_build_dir} is not empty after processing. Manual check may be needed.")
                pass


    def run(self):
        """ Orchestrates the entire shuffling and repacking process. """
        print(f"Starting dataset shuffling and repacking process with {self.num_workers} worker(s).")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Extracted originals directory: {self.extracted_originals_dir} (Keep: {self.keep_extracted_originals})")
        print(f"New chunks build directory: {self.new_chunks_build_dir}")


        all_valid_samples = self.collect_all_valid_samples()
        
        if all_valid_samples:
            self.shuffle_and_repack_samples(all_valid_samples)
            print(f"\nShuffling and repacking complete. New chunks saved to: {self.output_dir}")
        else:
            print("\nNo valid samples collected after processing input chunks. Process aborted.")

        if self.invalid_data_log_entries:
            print("\n--- Issues and Invalid Data Log ---")
            for log_entry in self.invalid_data_log_entries: print(log_entry)
            log_file_path = self.output_dir / "shuffler_issues_report.txt"
            try:
                with open(log_file_path, "w") as f:
                    f.write("Dataset Shuffler - Issues and Invalid Data Report\n" + "="*50 + "\n")
                    for log_entry in self.invalid_data_log_entries: f.write(log_entry + "\n")
                print(f"Full log of issues saved to: {log_file_path}")
            except Exception as e: print(f"Error writing issues log: {e}")
        else:
            print("\nNo issues or invalid data encountered during the process.")
        
        # Cleanup for extracted_originals_dir if not kept and if it was managed by this class (not user-provided and meant to persist)
        # The main function's TemporaryDirectory context manager will handle cleanup if system temp was used.
        # If user provided extracted_originals_dir and not keeping, we assume they want to manage it.
        # The logic in main handles whether the root temp dirs are auto-cleaned.
        # This class just uses the paths it's given.
        if not self.keep_extracted_originals:
            # If self.extracted_originals_dir was a system-generated temp, it's cleaned by main().
            # If it was user-provided, we don't delete it. We only clean its *contents* if we created them and are not keeping.
            # The current design has workers clean their specific chunk_specific_extract_dir if not keeping.
            # So, self.extracted_originals_dir might be empty or contain kept extractions.
            is_system_temp_for_originals = not getattr(self, '_user_provided_extracted_originals_dir', True)
            if is_system_temp_for_originals and self.extracted_originals_dir.exists():
                 # print(f"System temp for originals {self.extracted_originals_dir} will be cleaned by main context manager.")
                 pass
            elif not is_system_temp_for_originals and self.extracted_originals_dir.exists():
                 # print(f"User provided extracted originals dir {self.extracted_originals_dir}. Not deleting root. Contents might be present if --keep-extracted-originals was not used for all runs.")
                 pass


def main():
    parser = argparse.ArgumentParser(
        description="Unpack, validate, shuffle, and repack dataset chunks with multi-threading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, type=str, help="Directory containing original .tar chunks.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save new shuffled chunks.")
    
    parser.add_argument("--extracted-originals-dir", type=str, default=None, 
                        help="Directory for storing/finding extracted original chunks. If --keep-extracted-originals is set and this is not provided, 'retained_extracted_originals' in output-dir is used.")
    parser.add_argument("--new-chunks-build-dir", type=str, default=None,
                        help="Directory for temporarily building new chunks. If None, a system temp dir is created and removed.")
    
    parser.add_argument("--chunk-size", type=int, default=10000, help="Samples per new repacked chunk.")
    parser.add_argument("--output-compression", type=str, default="gz", choices=["gz", "bz2", "xz", "tar"], help="Compression for output .tar ('tar' for no compression).")
    parser.add_argument("--keep-extracted-originals", action="store_true", help="Keep extracted original chunks in 'extracted-originals-dir' to speed up subsequent runs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker threads. Defaults to CPU count.")
    
    args = parser.parse_args()

    actual_output_compression = args.output_compression if args.output_compression != "tar" else ""

    # Manage temporary/specified directories
    extracted_originals_temp_obj = None
    new_chunks_build_temp_obj = None

    # Determine path for extracted originals
    user_provided_extracted_originals_dir = False
    if args.extracted_originals_dir:
        extracted_originals_path = Path(args.extracted_originals_dir)
        user_provided_extracted_originals_dir = True
        print(f"Using user-provided directory for extracted originals: {extracted_originals_path}")
    elif args.keep_extracted_originals: # Not provided, but keeping
        extracted_originals_path = Path(args.output_dir) / "retained_extracted_originals"
        print(f"Keeping extracted originals: Using default path in output directory: {extracted_originals_path}")
        # This directory will be created by the class and not cleaned by a temp manager here.
    else: # Not provided and not keeping, use system temp
        extracted_originals_temp_obj = tempfile.TemporaryDirectory(prefix="shuffler_originals_")
        extracted_originals_path = Path(extracted_originals_temp_obj.name)
        print(f"Using system temporary directory for extracted originals: {extracted_originals_path}")

    # Determine path for new chunks build area
    user_provided_new_chunks_build_dir = False
    if args.new_chunks_build_dir:
        new_chunks_build_path = Path(args.new_chunks_build_dir)
        user_provided_new_chunks_build_dir = True
        print(f"Using user-provided directory for new chunks build area: {new_chunks_build_path}")
    else: # Not provided, use system temp
        new_chunks_build_temp_obj = tempfile.TemporaryDirectory(prefix="shuffler_build_")
        new_chunks_build_path = Path(new_chunks_build_temp_obj.name)
        print(f"Using system temporary directory for new chunks build area: {new_chunks_build_path}")

    # Ensure paths exist before passing to class (class also does this, but good practice)
    extracted_originals_path.mkdir(parents=True, exist_ok=True)
    new_chunks_build_path.mkdir(parents=True, exist_ok=True)

    try:
        shuffler = DatasetShuffler(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            extracted_originals_dir_path=extracted_originals_path,
            new_chunks_build_dir_path=new_chunks_build_path,
            repack_chunk_size=args.chunk_size,
            output_compression=actual_output_compression,
            keep_extracted_originals=args.keep_extracted_originals,
            num_workers=args.num_workers
        )
        # Pass info about user-provided paths for more nuanced cleanup messages in Shuffler if needed
        shuffler._user_provided_extracted_originals_dir = user_provided_extracted_originals_dir
        shuffler._user_provided_new_chunks_build_dir = user_provided_new_chunks_build_dir

        shuffler.run()
    except Exception as e:
        print(f"An unhandled error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if extracted_originals_temp_obj:
            print(f"Cleaning up system temporary directory for extracted originals: {extracted_originals_temp_obj.name}")
            extracted_originals_temp_obj.cleanup()
        if new_chunks_build_temp_obj:
            print(f"Cleaning up system temporary directory for new chunks build area: {new_chunks_build_temp_obj.name}")
            new_chunks_build_temp_obj.cleanup()

        if user_provided_extracted_originals_dir:
            print(f"User-provided directory for extracted originals ({extracted_originals_path}) was used.")
            if args.keep_extracted_originals:
                print("  It was configured to keep extracted originals.")
            else:
                print("  Its contents (if created by this script and not kept) should have been cleaned by the script.")
        
        if user_provided_new_chunks_build_dir:
            print(f"User-provided directory for new chunks build area ({new_chunks_build_path}) was used.")
            print("  Its contents should have been cleaned by the script after building each chunk.")

if __name__ == "__main__":
    main()
