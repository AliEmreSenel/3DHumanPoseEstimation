import os
import json
import pickle
import tarfile
import shutil
import tempfile
from pathlib import Path
import random
import argparse
from tqdm.auto import tqdm
import time
import uuid # For unique IDs for cached samples
import subprocess # For running rclone

class DatasetFiltererAndRechunker:
    def __init__(self, rclone_input_prefix, rclone_destination_prefix,
                 temp_extraction_dir_path,
                 staging_dir_path,
                 train_subject_ids, test_subject_ids,
                 current_user, current_timestamp_utc, # For logging
                 new_chunk_size=1000, output_compression="gz",
                 subject_id_key="subject_id",
                 rclone_common_flags=None):

        self.rclone_input_prefix = rclone_input_prefix.rstrip("/")
        self.rclone_destination_prefix = rclone_destination_prefix.rstrip("/")
        
        self.temp_extraction_dir = Path(temp_extraction_dir_path)
        self.staging_dir = Path(staging_dir_path)

        self.train_subject_ids = set(map(str, train_subject_ids))
        self.test_subject_ids = set(map(str, test_subject_ids))
        self.subject_id_key = subject_id_key

        self.new_chunk_size = new_chunk_size
        self.output_compression = output_compression if output_compression != "tar" else ""
        
        self.rclone_base_command_parts = ['rclone']
        if rclone_common_flags:
            self.rclone_base_command_parts.extend(rclone_common_flags)
        else: 
            self.rclone_base_command_parts.extend(['--retries', '3', '--low-level-retries', '5'])

        self.pending_files_cache_dir = self.staging_dir / "pending_files_cache"
        self.pending_train_cache_dir = self.pending_files_cache_dir / "train"
        self.pending_test_cache_dir = self.pending_files_cache_dir / "test"
        self.train_staging_build_dir = self.staging_dir / "train_build_active"
        self.test_staging_build_dir = self.staging_dir / "test_build_active"

        self.resume_state_file = self.staging_dir / "resume_state.json"
        self.pending_train_samples_file = self.staging_dir / "pending_train_samples.pkl"
        self.pending_test_samples_file = self.staging_dir / "pending_test_samples.pkl"
        self.log_file = self.staging_dir / "filter_rechunk_issues_report.txt"

        self._setup_directories()
        self.invalid_data_log_entries = [
            f"===== Log Session Start: {current_timestamp_utc} (UTC) =====",
            f"User: {current_user}",
            f"Rclone Input: {self.rclone_input_prefix}",
            f"Rclone Output Destination: {self.rclone_destination_prefix}",
            f"Local Temp Extraction Dir: {self.temp_extraction_dir}",
            f"Local Staging Dir: {self.staging_dir}",
            f"======================================================="
        ]

        self.pending_train_samples = []
        self.pending_test_samples = []
        self.next_train_chunk_idx_to_create = 0
        self.next_test_chunk_idx_to_create = 0
        self.original_chunk_process_start_idx = 0
        
        self._load_resume_state()

    def _setup_directories(self):
        self.temp_extraction_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.pending_files_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pending_train_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pending_test_cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_staging_build_dir.mkdir(parents=True, exist_ok=True)
        self.test_staging_build_dir.mkdir(parents=True, exist_ok=True)

    def _execute_rclone_command(self, command_specific_parts, operation_description, capture_stream_output=False):
        full_command = self.rclone_base_command_parts + command_specific_parts
        log_msg = f"Executing rclone for {operation_description}: {' '.join(full_command)}"
        print(log_msg); self.invalid_data_log_entries.append(log_msg)
        
        stdout_text = ""
        stderr_text = ""

        try:
            if capture_stream_output:
                # Capture output for parsing (e.g., lsjson)
                result = subprocess.run(full_command, capture_output=True, text=True, check=False)
                stdout_text = result.stdout
                stderr_text = result.stderr
            else:
                # Let rclone's output (like -P progress) go directly to terminal
                # This is crucial for progress display and avoiding pipe deadlocks
                result = subprocess.run(full_command, text=True, check=False) 
                # stdout and stderr are not captured here, they go to terminal

            if result.returncode == 0:
                self.invalid_data_log_entries.append(f"Rclone {operation_description} SUCCESSFUL.")
                if capture_stream_output and stdout_text: # Log stdout only if it was captured
                    self.invalid_data_log_entries.append(f"Rclone stdout: {stdout_text.strip()}")
                return True, stdout_text # Return captured stdout or empty string
            else:
                err_msg_parts = [f"Rclone {operation_description} FAILED (code {result.returncode})."]
                if capture_stream_output:
                    if stderr_text: err_msg_parts.append(f"  Stderr: {stderr_text.strip()}")
                    if stdout_text: err_msg_parts.append(f"  Stdout: {stdout_text.strip()}") # stdout might also contain error info
                else:
                    err_msg_parts.append("  (Rclone output was directed to terminal, check above for messages)")
                
                err_msg = "\n".join(err_msg_parts)
                print(err_msg); self.invalid_data_log_entries.append(err_msg)
                return False, stderr_text if capture_stream_output else "" # Return captured stderr or empty string
        except FileNotFoundError:
            err_msg = "Rclone command not found. Ensure rclone is installed and in PATH."
            print(err_msg); self.invalid_data_log_entries.append(err_msg)
            return False, err_msg # Return the error message as the second value
        except Exception as e:
            err_msg = f"Exception during rclone {operation_description}: {e}"
            print(err_msg); self.invalid_data_log_entries.append(err_msg)
            return False, str(e) # Return the error message

    def _list_remote_rclone_input_chunks(self):
        command_parts = ['lsjson', '--files-only', self.rclone_input_prefix]
        # Capture output for lsjson because we need to parse it
        success, output_str = self._execute_rclone_command(command_parts, "listing input chunks", capture_stream_output=True)
        
        if not success:
            self.invalid_data_log_entries.append("CRITICAL: Failed to list input chunks from rclone. Aborting.")
            return None

        remote_files = []
        try:
            items = json.loads(output_str) # output_str is captured stdout from _execute_rclone_command
            valid_extensions = ('.tar', '.tar.gz', '.tar.bz2', '.tar.xz')
            for item in items:
                if isinstance(item, dict) and 'Path' in item and item['Path'].endswith(valid_extensions):
                    full_remote_path = f"{self.rclone_input_prefix}/{item['Path']}"
                    remote_files.append(full_remote_path)
            
            if not remote_files:
                 self.invalid_data_log_entries.append(f"No valid chunk files (.tar*) found in {self.rclone_input_prefix}")
            else:
                 self.invalid_data_log_entries.append(f"Found {len(remote_files)} potential input chunks via rclone lsjson.")
            return sorted(list(set(remote_files)))
        except json.JSONDecodeError as e:
            self.invalid_data_log_entries.append(f"Error decoding rclone lsjson output: {e}. Output was: {output_str[:500]}")
            return None
        except Exception as e:
            self.invalid_data_log_entries.append(f"Unexpected error processing rclone lsjson output: {e}")
            return None

    def _load_resume_state(self): # (Identical to v3/v4)
        if self.resume_state_file.exists():
            try:
                with open(self.resume_state_file, 'r') as f: state = json.load(f)
                self.original_chunk_process_start_idx = state.get("next_original_chunk_to_process_idx", 0)
                self.next_train_chunk_idx_to_create = state.get("next_train_chunk_idx_to_create", 0)
                self.next_test_chunk_idx_to_create = state.get("next_test_chunk_idx_to_create", 0)
                if self.pending_train_samples_file.exists():
                    with open(self.pending_train_samples_file, 'rb') as f: self.pending_train_samples = pickle.load(f)
                    self.invalid_data_log_entries.append(f"Loaded {len(self.pending_train_samples)} pending train samples.")
                if self.pending_test_samples_file.exists():
                    with open(self.pending_test_samples_file, 'rb') as f: self.pending_test_samples = pickle.load(f)
                    self.invalid_data_log_entries.append(f"Loaded {len(self.pending_test_samples)} pending test samples.")
                self.invalid_data_log_entries.append(f"Resume: Original Idx {self.original_chunk_process_start_idx}, TrainChunkNext {self.next_train_chunk_idx_to_create}, TestChunkNext {self.next_test_chunk_idx_to_create}")
            except Exception as e:
                self.invalid_data_log_entries.append(f"Error loading resume state: {e}. Starting fresh."); self._reset_resume_indices()
        else: self.invalid_data_log_entries.append("No resume state file. Starting fresh."); self._reset_resume_indices()

    def _reset_resume_indices(self): # (Identical to v3/v4)
        self.original_chunk_process_start_idx = 0; self.next_train_chunk_idx_to_create = 0
        self.next_test_chunk_idx_to_create = 0; self.pending_train_samples = []; self.pending_test_samples = []

    def _save_resume_state(self, current_original_chunk_idx_processed_or_next_to_process): # (Identical to v3/v4)
        state = {"next_original_chunk_to_process_idx": current_original_chunk_idx_processed_or_next_to_process,
                 "next_train_chunk_idx_to_create": self.next_train_chunk_idx_to_create,
                 "next_test_chunk_idx_to_create": self.next_test_chunk_idx_to_create, "timestamp": time.time()}
        try:
            with open(self.resume_state_file, 'w') as f: json.dump(state, f, indent=2)
            if self.pending_train_samples:
                with open(self.pending_train_samples_file, 'wb') as f: pickle.dump(self.pending_train_samples, f)
            elif self.pending_train_samples_file.exists(): self.pending_train_samples_file.unlink()
            if self.pending_test_samples:
                with open(self.pending_test_samples_file, 'wb') as f: pickle.dump(self.pending_test_samples, f)
            elif self.pending_test_samples_file.exists(): self.pending_test_samples_file.unlink()
        except Exception as e: self.invalid_data_log_entries.append(f"Error saving resume state: {e}")

    def _cleanup_resume_state_and_cache(self): # (Identical to v3/v4)
        if self.resume_state_file.exists(): self.resume_state_file.unlink(missing_ok=True)
        if self.pending_train_samples_file.exists(): self.pending_train_samples_file.unlink(missing_ok=True)
        if self.pending_test_samples_file.exists(): self.pending_test_samples_file.unlink(missing_ok=True)
        if self.pending_files_cache_dir.exists():
            shutil.rmtree(self.pending_files_cache_dir)
            self.pending_files_cache_dir.mkdir(parents=True, exist_ok=True) 
            self.pending_train_cache_dir.mkdir(parents=True, exist_ok=True)
            self.pending_test_cache_dir.mkdir(parents=True, exist_ok=True)
        self.invalid_data_log_entries.append("Resume state and cache cleaned after successful run.")

    def _rclone_copyto_for_final_chunk_upload(self, local_source_path: Path, chunk_type: str, remote_filename: str):
        remote_destination_path = f"{self.rclone_destination_prefix}/{chunk_type}/{remote_filename}"
        # -P for progress, copyto command
        copy_specific_flags = ['copyto', '-P'] 
        command_parts = copy_specific_flags + [str(local_source_path), remote_destination_path]
        # Do not capture stream output for uploads with progress
        success, _ = self._execute_rclone_command(command_parts, f"uploading {remote_filename}", capture_stream_output=False)
        return success

    def _copy_and_extract_single_original_chunk(self, remote_original_chunk_full_path: str, target_extraction_base_dir: Path):
        local_log = []
        original_chunk_filename = remote_original_chunk_full_path.split('/')[-1]
        local_temp_tar_path = target_extraction_base_dir / original_chunk_filename
        
        chunk_base_name_for_subdir = original_chunk_filename
        for ext in [".tar.gz", ".tar.bz2", ".tar.xz", ".tar"]:
            if chunk_base_name_for_subdir.endswith(ext): chunk_base_name_for_subdir = chunk_base_name_for_subdir[:-len(ext)]; break
        chunk_specific_extracted_content_dir = target_extraction_base_dir / f"{chunk_base_name_for_subdir}_content"

        if chunk_specific_extracted_content_dir.exists(): shutil.rmtree(chunk_specific_extracted_content_dir)
        if local_temp_tar_path.exists(): local_temp_tar_path.unlink()
        chunk_specific_extracted_content_dir.mkdir(parents=True)

        # Download original chunk from rclone remote to local_temp_tar_path
        # -P for progress, copyto command
        download_specific_flags = ['copyto', '-P'] 
        download_command_parts = download_specific_flags + [remote_original_chunk_full_path, str(local_temp_tar_path)]
        # Do not capture stream output for downloads with progress
        download_success, _ = self._execute_rclone_command(download_command_parts, f"downloading {original_chunk_filename}", capture_stream_output=False)

        if not download_success:
            local_log.append(f"Failed to download {original_chunk_filename} from rclone. Skipping chunk.")
            if local_temp_tar_path.exists(): local_temp_tar_path.unlink()
            if chunk_specific_extracted_content_dir.exists(): shutil.rmtree(chunk_specific_extracted_content_dir)
            return None, local_log

        try:
            extract_desc = f"Extracting {original_chunk_filename} (now local)"
            with tqdm(total=local_temp_tar_path.stat().st_size, unit='B', unit_scale=True, desc=extract_desc, leave=False) as pbar:
                with tarfile.open(local_temp_tar_path, "r:*") as tar:
                    members = tar.getmembers()
                    for member in members:
                        tar.extract(member, path=chunk_specific_extracted_content_dir)
                        pbar.update(member.size if member.isfile() else 1024)
                pbar.n = pbar.total
            local_log.append(f"Successfully extracted {original_chunk_filename} to {chunk_specific_extracted_content_dir}")
            return chunk_specific_extracted_content_dir, local_log
        except (tarfile.ReadError, Exception) as e:
            local_log.append(f"Failed to unpack locally downloaded {original_chunk_filename}: {e}")
            return None, local_log
        finally:
            if local_temp_tar_path.exists(): local_temp_tar_path.unlink()

    def _load_metadata_from_extracted_chunk(self, chunk_extract_path): # Identical
        local_invalid_log = []; metadata_path = chunk_extract_path / "samples.pkl"; data_base_path_for_samples = chunk_extract_path 
        if not metadata_path.exists():
            potential_content_dirs = [d for d in chunk_extract_path.iterdir() if d.is_dir()]
            if potential_content_dirs: assumed_content_root = potential_content_dirs[0]; metadata_path = assumed_content_root / "samples.pkl"; data_base_path_for_samples = assumed_content_root
            if not metadata_path.exists(): local_invalid_log.append(f"samples.pkl not found in {chunk_extract_path} or subdirs."); return None, None, local_invalid_log
        try:
            with open(metadata_path, "rb") as f: samples_data = pickle.load(f)
            if not isinstance(samples_data, list): local_invalid_log.append(f"Metadata {metadata_path} not a list."); return None, data_base_path_for_samples, local_invalid_log
            return samples_data, data_base_path_for_samples, local_invalid_log
        except Exception as e: local_invalid_log.append(f"Error loading metadata {metadata_path}: {e}"); return None, data_base_path_for_samples, local_invalid_log

    def _validate_sample_and_get_files(self, sample_metadata, data_base_path, original_chunk_name_for_logging): # Identical
        local_invalid_log = [];
        if not isinstance(sample_metadata, dict): local_invalid_log.append(f"Invalid sample format in {original_chunk_name_for_logging}."); return False, [], local_invalid_log, None
        is_sample_valid = True; files_to_copy_details = []; subject_id = str(sample_metadata.get(self.subject_id_key, "UNKNOWN_SUBJECT"))
        for file_key, file_type, is_mandatory in [("image_file", "image", True), ("depth_file", "depth", False), ("metadata_file", "metadata_json", True)]:
            rel_path_str = sample_metadata.get(file_key)
            if not rel_path_str:
                if is_mandatory: local_invalid_log.append(f"INVALID (S:{subject_id}): Mand. '{file_key}' missing in {original_chunk_name_for_logging}."); is_sample_valid = False
                continue
            abs_path = data_base_path / rel_path_str
            if not abs_path.exists() or abs_path.stat().st_size == 0: local_invalid_log.append(f"INVALID (S:{subject_id}): File {abs_path} for '{file_key}' missing/0 bytes (from {original_chunk_name_for_logging})."); is_sample_valid = False
            else: files_to_copy_details.append({"type": file_type, "src_abs_path": str(abs_path), "original_rel_path": rel_path_str})
        if not is_sample_valid: return False, [], local_invalid_log, subject_id
        return True, files_to_copy_details, local_invalid_log, subject_id

    def _process_extracted_original_chunk_samples(self, samples_data_list, data_base_path, original_chunk_name): # Identical
        chunk_local_log = []; num_added_train, num_added_test, num_skipped, num_invalid = 0,0,0,0
        for sample_meta in tqdm(samples_data_list, desc=f"Filtering samples from {original_chunk_name}", unit="sample", leave=False):
            is_valid, files_to_copy_from_temp_extract, validation_logs, subject_id = self._validate_sample_and_get_files(sample_meta, data_base_path, original_chunk_name)
            chunk_local_log.extend(validation_logs)
            if not is_valid: num_invalid += 1; continue
            target_pending_list, target_sample_cache_base_dir, count_var = (None, None, None)
            if subject_id in self.train_subject_ids: target_pending_list, target_sample_cache_base_dir, count_var = self.pending_train_samples, self.pending_train_cache_dir, "train"
            elif subject_id in self.test_subject_ids: target_pending_list, target_sample_cache_base_dir, count_var = self.pending_test_samples, self.pending_test_cache_dir, "test"
            else: chunk_local_log.append(f"INFO (S:{subject_id}, {original_chunk_name}): Not in train/test. Skipping."); num_skipped += 1; continue
            sample_cache_uuid = str(uuid.uuid4()); sample_specific_cache_dir = target_sample_cache_base_dir / sample_cache_uuid
            sample_specific_cache_dir.mkdir(parents=True); staged_files_info_list = []; all_files_for_sample_cached = True
            for file_detail_from_temp in files_to_copy_from_temp_extract:
                src_abs_path_in_temp = Path(file_detail_from_temp["src_abs_path"]); dest_filename_in_cache = src_abs_path_in_temp.name
                dest_abs_path_in_cache = sample_specific_cache_dir / dest_filename_in_cache
                try:
                    shutil.copy2(src_abs_path_in_temp, dest_abs_path_in_cache)
                    staged_files_info_list.append({"type": file_detail_from_temp["type"], "staged_abs_path": str(dest_abs_path_in_cache), "original_rel_path_basename": dest_filename_in_cache})
                except Exception as e: chunk_local_log.append(f"ERROR caching {src_abs_path_in_temp} to {dest_abs_path_in_cache}: {e}"); all_files_for_sample_cached = False; break
            if all_files_for_sample_cached:
                target_pending_list.append({"original_metadata": sample_meta, "subject_id": subject_id, "original_chunk_name": original_chunk_name, "sample_cache_dir_abs_path": str(sample_specific_cache_dir), "staged_files_info": staged_files_info_list})
                if count_var == "train": num_added_train +=1 
                else: num_added_test +=1
            else: 
                if sample_specific_cache_dir.exists(): shutil.rmtree(sample_specific_cache_dir)
                num_invalid +=1; chunk_local_log.append(f"Failed to cache all files for sample (S:{subject_id}, from {original_chunk_name}). Discarded.")
        summary = f"Filtered {original_chunk_name}: Train+{num_added_train}, Test+{num_added_test}, Invalid:{num_invalid}, Skipped:{num_skipped}."; print(summary); chunk_local_log.insert(0, summary)
        return chunk_local_log

    def _repack_and_transfer_pending_samples(self, force_pack_incomplete=False): # Identical
        self.next_train_chunk_idx_to_create = self._process_and_pack_pending_list(
            self.pending_train_samples, "train", self.train_staging_build_dir, self.next_train_chunk_idx_to_create, force_pack_incomplete)
        self.next_test_chunk_idx_to_create = self._process_and_pack_pending_list(
            self.pending_test_samples, "test", self.test_staging_build_dir, self.next_test_chunk_idx_to_create, force_pack_incomplete)

    def _process_and_pack_pending_list(self, pending_samples_list, chunk_type_name, # Identical
                                       local_staging_build_root, current_chunk_start_idx, force_pack_incomplete):
        next_chunk_idx_val = current_chunk_start_idx
        while True:
            can_pack_this_chunk = (len(pending_samples_list) >= self.new_chunk_size) or (force_pack_incomplete and len(pending_samples_list) > 0)
            if not can_pack_this_chunk: break
            num_samples_for_this_new_chunk = min(len(pending_samples_list), self.new_chunk_size)
            samples_to_pack_into_new_chunk = pending_samples_list[:num_samples_for_this_new_chunk]
            new_chunk_numeric_id_str = f"{next_chunk_idx_val:06d}"; archive_base_name = f"{chunk_type_name}_dataset_chunk_{new_chunk_numeric_id_str}"
            tar_ext = f".tar.{self.output_compression}" if self.output_compression else ".tar"; new_tar_filename_on_remote = f"{archive_base_name}{tar_ext}"
            current_new_chunk_content_build_dir = local_staging_build_root / f"build_content_{new_chunk_numeric_id_str}"
            local_temp_tar_archive_path = local_staging_build_root / new_tar_filename_on_remote
            if current_new_chunk_content_build_dir.exists(): shutil.rmtree(current_new_chunk_content_build_dir)
            current_new_chunk_data_subdir = current_new_chunk_content_build_dir / "data"; current_new_chunk_data_subdir.mkdir(parents=True)
            repacked_metadata_for_new_chunk_pkl = []; successfully_processed_samples_for_this_chunk = []; all_samples_in_new_chunk_ok = True
            desc_repack = f"Populating local new {chunk_type_name} chunk {new_chunk_numeric_id_str}"
            for sample_j, original_sample_detail in enumerate(tqdm(samples_to_pack_into_new_chunk, desc=desc_repack, unit="sample", leave=False)):
                new_sample_id_within_this_chunk_str = f"{sample_j:06d}"; new_sample_files_destination_dir = current_new_chunk_data_subdir / new_sample_id_within_this_chunk_str
                new_sample_files_destination_dir.mkdir(); new_sample_metadata_entry = original_sample_detail["original_metadata"].copy()
                new_sample_metadata_entry.update({"new_chunk_id": new_chunk_numeric_id_str, "sample_id_in_new_chunk": new_sample_id_within_this_chunk_str, "original_subject_id_preserved": original_sample_detail["subject_id"], "source_original_chunk": original_sample_detail["original_chunk_name"]})
                sample_files_copied_ok = True
                for staged_file_info in original_sample_detail["staged_files_info"]:
                    src_abs_path_in_cache = Path(staged_file_info["staged_abs_path"]); new_relative_path_for_pkl = Path("data") / new_sample_id_within_this_chunk_str / staged_file_info["original_rel_path_basename"]
                    dest_abs_path_for_file_in_staging_build = new_sample_files_destination_dir / staged_file_info["original_rel_path_basename"]
                    try: 
                        shutil.copy2(src_abs_path_in_cache, dest_abs_path_for_file_in_staging_build)
                        file_type_key = staged_file_info["type"]
                        if file_type_key == "image": new_sample_metadata_entry["image_file"] = str(new_relative_path_for_pkl)
                        elif file_type_key == "depth": new_sample_metadata_entry["depth_file"] = str(new_relative_path_for_pkl)
                        elif file_type_key == "metadata_json": new_sample_metadata_entry["metadata_file"] = str(new_relative_path_for_pkl)
                    except Exception as e: self.invalid_data_log_entries.append(f"Error copying {src_abs_path_in_cache.name} for {new_tar_filename_on_remote}: {e}"); sample_files_copied_ok = False; all_samples_in_new_chunk_ok = False; break 
                if sample_files_copied_ok: repacked_metadata_for_new_chunk_pkl.append(new_sample_metadata_entry); successfully_processed_samples_for_this_chunk.append(original_sample_detail) 
                else: break
            if not all_samples_in_new_chunk_ok: self.invalid_data_log_entries.append(f"Failed local assembly for {new_tar_filename_on_remote}. Aborting this chunk build."); break
            new_samples_pkl_path_in_staging = current_new_chunk_content_build_dir / "samples.pkl"
            with open(new_samples_pkl_path_in_staging, "wb") as f: pickle.dump(repacked_metadata_for_new_chunk_pkl, f)
            archive_internal_root_dir_name = f"chunk_{new_chunk_numeric_id_str}"; tar_creation_ok = False
            try:
                with tarfile.open(local_temp_tar_archive_path, f"w:{self.output_compression}") as tar:
                    tar.add(new_samples_pkl_path_in_staging, arcname=Path(archive_internal_root_dir_name) / "samples.pkl"); tar.add(current_new_chunk_data_subdir, arcname=Path(archive_internal_root_dir_name) / "data")
                tar_creation_ok = True; self.invalid_data_log_entries.append(f"Local tar {local_temp_tar_archive_path.name} created successfully.")
            except Exception as e: self.invalid_data_log_entries.append(f"Error creating local tar {local_temp_tar_archive_path.name}: {e}")
            if not tar_creation_ok: break
            rclone_upload_success = self._rclone_copyto_for_final_chunk_upload(local_temp_tar_archive_path, chunk_type_name, new_tar_filename_on_remote)
            if rclone_upload_success:
                for packed_sample_detail in successfully_processed_samples_for_this_chunk:
                    cache_dir_to_clean = Path(packed_sample_detail["sample_cache_dir_abs_path"]);
                    if cache_dir_to_clean.exists(): shutil.rmtree(cache_dir_to_clean)
                del pending_samples_list[:num_samples_for_this_new_chunk]; next_chunk_idx_val += 1
            else: self.invalid_data_log_entries.append(f"Rclone upload FAILED for {local_temp_tar_archive_path.name}. Preserved in staging. Halting packing for '{chunk_type_name}'."); break 
            if current_new_chunk_content_build_dir.exists(): shutil.rmtree(current_new_chunk_content_build_dir)
            if local_temp_tar_archive_path.exists(): local_temp_tar_archive_path.unlink()
            if not force_pack_incomplete and len(pending_samples_list) < self.new_chunk_size: break
        return next_chunk_idx_val

    def run(self): # Identical
        self.invalid_data_log_entries.append(f"Run instance method started. Rclone Input: {self.rclone_input_prefix}")
        self._write_log_file()
        original_remote_chunk_paths = self._list_remote_rclone_input_chunks()
        if original_remote_chunk_paths is None: self.invalid_data_log_entries.append("Halting due to failure in listing remote input chunks."); self._write_log_file(); return
        if not original_remote_chunk_paths: self.invalid_data_log_entries.append(f"No original chunk files found at {self.rclone_input_prefix}. Nothing to process."); self._write_log_file(); return
        self.invalid_data_log_entries.append(f"Found {len(original_remote_chunk_paths)} remote original chunks. Will start/resume from index {self.original_chunk_process_start_idx}.")
        print(f"Found {len(original_remote_chunk_paths)} remote original chunks. Will start/resume from index {self.original_chunk_process_start_idx}.")
        all_original_chunks_processed_successfully = True
        for original_chunk_idx in range(self.original_chunk_process_start_idx, len(original_remote_chunk_paths)):
            remote_original_chunk_full_path = original_remote_chunk_paths[original_chunk_idx]; original_chunk_filename = remote_original_chunk_full_path.split('/')[-1]
            log_entry = f"\nProcessing remote original chunk {original_chunk_idx + 1}/{len(original_remote_chunk_paths)}: {original_chunk_filename} (from {remote_original_chunk_full_path})"
            print(log_entry); self.invalid_data_log_entries.append(log_entry)
            self._save_resume_state(current_original_chunk_idx_processed_or_next_to_process=original_chunk_idx)
            extracted_content_root_path, copy_extract_logs = self._copy_and_extract_single_original_chunk(remote_original_chunk_full_path, self.temp_extraction_dir)
            self.invalid_data_log_entries.extend(copy_extract_logs)
            if not extracted_content_root_path: self.invalid_data_log_entries.append(f"Fatal error downloading/extracting {original_chunk_filename}. Halting further original chunk processing."); all_original_chunks_processed_successfully = False; break 
            samples_data_list, data_files_base_path, load_meta_logs = self._load_metadata_from_extracted_chunk(extracted_content_root_path)
            self.invalid_data_log_entries.extend(load_meta_logs)
            if samples_data_list: filter_logs = self._process_extracted_original_chunk_samples(samples_data_list, data_files_base_path, original_chunk_filename); self.invalid_data_log_entries.extend(filter_logs)
            else: self.invalid_data_log_entries.append(f"No samples data loaded from {original_chunk_filename}.")
            if extracted_content_root_path.exists(): shutil.rmtree(extracted_content_root_path)
            self._save_resume_state(current_original_chunk_idx_processed_or_next_to_process=original_chunk_idx + 1)
            self.invalid_data_log_entries.append(f"Checking pending: Train ({len(self.pending_train_samples)}), Test ({len(self.pending_test_samples)})")
            print(f"Checking pending: Train ({len(self.pending_train_samples)}), Test ({len(self.pending_test_samples)})")
            self._repack_and_transfer_pending_samples(force_pack_incomplete=False); self._write_log_file()
        if all_original_chunks_processed_successfully:
            self.invalid_data_log_entries.append("\nAll specified original input chunks processed (or resumed)."); print("\nAll specified original input chunks processed (or resumed).")
            self.invalid_data_log_entries.append(f"Final check for remaining samples: Train ({len(self.pending_train_samples)}), Test ({len(self.pending_test_samples)})")
            print(f"Final check for remaining samples: Train ({len(self.pending_train_samples)}), Test ({len(self.pending_test_samples)})")
            self._repack_and_transfer_pending_samples(force_pack_incomplete=True)
            if not self.pending_train_samples and not self.pending_test_samples: self._cleanup_resume_state_and_cache(); self.invalid_data_log_entries.append("Run completed successfully. All pending samples packed. Resume state cleaned.")
            else: self.invalid_data_log_entries.append("Run completed original chunks, but some samples remain pending (likely due to upload error). Resume state preserved."); print("WARNING: Some samples remain pending, likely due to upload errors. Resume state preserved.")
        else: self.invalid_data_log_entries.append("Run halted due to critical error during original chunk processing. Resume state preserved."); print("Run halted. Resume state preserved. Check logs.")
        self._write_log_file(); print(f"\nFilter and re-chunking process finished. Log: {self.log_file}")

    def _write_log_file(self): # Identical
        try:
            with open(self.log_file, "w") as f:
                for log_entry in self.invalid_data_log_entries: f.write(log_entry + "\n")
        except Exception as e: print(f"Error writing issues log to {self.log_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset chunks from rclone, re-chunk, and upload to rclone remote, with resumability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rclone-input-prefix", required=True, help="Rclone remote path for INPUT original .tar chunks (e.g., myremote:bucket/input_chunks/).")
    parser.add_argument("--rclone-destination-prefix", required=True, help="Rclone remote path for OUTPUT new chunks (e.g., myremote:bucket/output_dataset/). Train/test subdirs created here.")
    parser.add_argument("--temp-extraction-dir", required=True, help="LOCAL TEMP dir for downloading and extracting ONE original chunk (e.g., /mnt/ramdisk or /dev/shm/extract_tmp).")
    parser.add_argument("--staging-dir", required=True, help="LOCAL STAGING dir for pending sample cache, resume state, logs, & temp new chunks before upload.")
    
    parser.add_argument("--train-subjects", required=True, help="Comma-sep subject IDs for training.")
    parser.add_argument("--test-subjects", required=True, help="Comma-sep subject IDs for testing.")
    parser.add_argument("--subject-id-key", default="subject_id", help="Key in sample metadata for subject ID.")
    
    parser.add_argument("--new-chunk-size", type=int, default=1000, help="Samples per new repacked chunk.")
    parser.add_argument("--output-compression", default="gz", choices=["gz", "bz2", "xz", "tar"], help="Compression for output .tar.")
    parser.add_argument("--rclone-common-flags", type=str, default="", help="Optional space-separated COMMON rclone flags for all operations (e.g., '--config my.conf'). Quote if it contains spaces. Specific operations like copyto might add their own progress flags.")
    
    args = parser.parse_args()

    current_user_login = "AliEmreSenel" 
    current_timestamp_utc_str = "2025-05-26 19:17:04" # This will be the timestamp from when you run it, or you can pass a fixed one

    train_ids = [s.strip() for s in args.train_subjects.split(',') if s.strip()]
    test_ids = [s.strip() for s in args.test_subjects.split(',') if s.strip()]

    if not train_ids or not test_ids: print("Error: Train/Test subjects required."); return
    if set(train_ids) & set(test_ids): print("Error: Subject IDs overlap."); return

    rclone_common_flags_list = args.rclone_common_flags.split() if args.rclone_common_flags else None

    Path(args.temp_extraction_dir).mkdir(parents=True, exist_ok=True)
    Path(args.staging_dir).mkdir(parents=True, exist_ok=True)
    
    filter_rechunker = None
    try:
        filter_rechunker = DatasetFiltererAndRechunker(
            rclone_input_prefix=args.rclone_input_prefix, 
            rclone_destination_prefix=args.rclone_destination_prefix,
            temp_extraction_dir_path=args.temp_extraction_dir,
            staging_dir_path=args.staging_dir,
            train_subject_ids=train_ids, test_subject_ids=test_ids,
            current_user=current_user_login, current_timestamp_utc=current_timestamp_utc_str, # Pass current user/time
            subject_id_key=args.subject_id_key,
            new_chunk_size=args.new_chunk_size,
            output_compression=args.output_compression,
            rclone_common_flags=rclone_common_flags_list
        )
        filter_rechunker.run()
    except Exception as e:
        print(f"FATAL ERROR in main process: {e}")
        import traceback
        traceback.print_exc()
        if filter_rechunker and hasattr(filter_rechunker, 'invalid_data_log_entries'):
            filter_rechunker.invalid_data_log_entries.append(f"\n--- Main Process CRASH: {time.asctime()} ---") # Use current time for crash
            filter_rechunker.invalid_data_log_entries.append(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
            filter_rechunker._write_log_file()
            print(f"Attempted to write crash details to log: {filter_rechunker.log_file}")
    finally:
        if filter_rechunker and hasattr(filter_rechunker, 'invalid_data_log_entries') and filter_rechunker.invalid_data_log_entries:
            print(f"Final attempt to ensure logs are written to {filter_rechunker.log_file}")
            filter_rechunker._write_log_file()

if __name__ == "__main__":
    main()
