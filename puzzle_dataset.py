import os
import json
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata
from utils.data_processing import process_vision_batch

from argdantic import ArgParser
from pydantic import BaseModel

class LazySafetensorsLoader:
    """
    Lazy loader for sharded safetensors files.
    Mimics a numpy array or tensor for read-only access.
    """
    def __init__(self, file_paths: List[Path], field_name: str):
        from safetensors import safe_open
        self.file_paths = sorted([str(p) for p in file_paths])
        self.field_name = field_name
        self.files = []
        self.file_ranges = [] # (start, end)
        self.total_length = 0
        self.shapes = []
        
        # Initialize files and index
        for fp in self.file_paths:
            try:
                # We keep files open for performance. 
                # Note: This might consume file descriptors.
                f = safe_open(fp, framework="np", device="cpu")
                
                # Check if field exists
                keys = f.keys()
                if field_name not in keys:
                    # If field is missing, we assume it's missing in this chunk.
                    # For consistency, we should probably error or skip?
                    # But if we skip, indices will be misaligned with other fields.
                    # We assume consistent schema.
                    print(f"[WARN] Field '{field_name}' missing in {fp}")
                    continue
                
                slice_info = f.get_slice(field_name)
                shape = slice_info.get_shape()
                length = shape[0]
                
                self.files.append(f)
                self.shapes.append(shape)
                self.file_ranges.append((self.total_length, self.total_length + length))
                self.total_length += length
            except Exception as e:
                print(f"[WARN] Failed to load {fp}: {e}")
                
    def __len__(self):
        return self.total_length
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_length)
            if step != 1:
                raise NotImplementedError("Slicing with step != 1 not supported")
            
            result = []
            current = start
            while current < stop:
                # Find file
                for i, (s, e) in enumerate(self.file_ranges):
                    if s <= current < e:
                        f = self.files[i]
                        local_start = current - s
                        local_end = min(stop, e) - s
                        slice_obj = f.get_slice(self.field_name)
                        result.append(slice_obj[local_start:local_end])
                        current += (local_end - local_start)
                        break
                else:
                    break # Should not happen
            
            if not result:
                return np.array([])
            return np.concatenate(result, axis=0)
            
        elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.numpy()
            if isinstance(idx, list):
                idx = np.array(idx)
                
            out_arrays = []
            for i in idx:
                found = False
                for f_idx, (s, e) in enumerate(self.file_ranges):
                    if s <= i < e:
                        f = self.files[f_idx]
                        local_i = i - s
                        slice_obj = f.get_slice(self.field_name)
                        out_arrays.append(slice_obj[local_i:local_i+1])
                        found = True
                        break
                if not found:
                    raise IndexError(f"Index {i} out of bounds")
            
            if not out_arrays:
                return np.array([])
            return np.concatenate(out_arrays, axis=0)
            
        else:
            # Single index
            for f_idx, (s, e) in enumerate(self.file_ranges):
                if s <= idx < e:
                    f = self.files[f_idx]
                    local_i = idx - s
                    slice_obj = f.get_slice(self.field_name)
                    return slice_obj[local_i]
            raise IndexError("Index out of bounds")


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        if total_puzzles > 0:
            mean_puzzle_examples = mean_puzzle_examples / total_puzzles
        else:
            mean_puzzle_examples = 0

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        # Check for NEW streaming format (shard_*.pt + shard_index.json)
        shard_index_path = os.path.join(dataset_path, "shard_index.json")
        if os.path.exists(shard_index_path):
            with open(shard_index_path, 'r') as f:
                shard_info = json.load(f)
            
            # Check if train/dataset.json exists for metadata
            train_metadata_path = os.path.join(dataset_path, self.split, "dataset.json")
            if os.path.exists(train_metadata_path):
                with open(train_metadata_path, 'r') as f:
                    return PuzzleDatasetMetadata(**json.load(f))
            
            # Fallback: create metadata from shard info
            return PuzzleDatasetMetadata(
                seq_len=512,
                vocab_size=2048 + 4,
                pad_id=0,
                ignore_label_id=-100,
                blank_identifier_id=0,
                num_puzzle_identifiers=shard_info['total_samples'],
                total_groups=shard_info['total_samples'],
                mean_puzzle_examples=1.0,
                total_puzzles=shard_info['total_samples'],
                sets=[self.split]
            )
        
        # Check if OLD streaming format (consolidated_*.pt or .safetensors files)
        stream_dir = os.path.join(dataset_path, "stream_checkpoints")
        if os.path.exists(stream_dir):
            # Check for safetensors first (preferred)
            safetensors_files = sorted(Path(stream_dir).glob("consolidated_*.safetensors"))
            if safetensors_files:
                from safetensors import safe_open
                # Load metadata from first chunk
                with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
                    # Assuming 'sketches' key exists
                    first_shape = f.get_tensor("sketches").shape
                
                # Calculate total samples efficiently
                num_samples = 0
                for sf in safetensors_files:
                    with safe_open(sf, framework="pt", device="cpu") as f:
                        # Read shape of 'sketches' tensor without loading data
                        slice_info = f.get_slice("sketches")
                        num_samples += slice_info.get_shape()[0]

                return PuzzleDatasetMetadata(
                    seq_len=first_shape[1],
                    vocab_size=0,
                    pad_id=0,
                    ignore_label_id=-100,
                    blank_identifier_id=0,
                    num_puzzle_identifiers=num_samples,
                    total_groups=1,
                    mean_puzzle_examples=1.0,
                    total_puzzles=num_samples,
                    sets=["default"]
                )

            consolidated_files = sorted(Path(stream_dir).glob("consolidated_*.pt"))
            if consolidated_files:
                # Load streaming format metadata from first chunk
                chunk = torch.load(consolidated_files[0], map_location='cpu')
                num_samples = sum(torch.load(f, map_location='cpu')['sketches'].shape[0] 
                                 for f in consolidated_files)
                
                # Create metadata for streaming format
                return PuzzleDatasetMetadata(
                    seq_len=chunk['sketches'].shape[1],  # Assuming capsule format
                    vocab_size=0,  # Not applicable for vision
                    pad_id=0,
                    ignore_label_id=-100,
                    blank_identifier_id=0,
                    num_puzzle_identifiers=num_samples,
                    total_groups=1,
                    mean_puzzle_examples=1.0,
                    total_puzzles=num_samples,
                    sets=["default"]
                )
        
        # Standard format
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                
                # Check for NEW shard format (shard_*.pt files)
                shard_index_path = os.path.join(dataset_path, "shard_index.json")
                if os.path.exists(shard_index_path):
                    with open(shard_index_path, 'r') as f:
                        shard_info = json.load(f)
                    
                    # Load all shards and extract samples
                    all_samples = []
                    for shard_file in shard_info['shard_files']:
                        # weights_only=False needed for custom DataSample objects
                        shard = torch.load(shard_file, map_location='cpu', weights_only=False)
                        all_samples.extend(shard['samples'])
                    
                    # Convert samples to input/label format
                    # For now, store raw samples - will be processed during training
                    num_samples = len(all_samples)
                    
                    # Create placeholder arrays (actual data loaded on-demand)
                    # Store samples directly for streaming access
                    self._data[set_name_] = {
                        "raw_samples": all_samples,  # Raw DataSample objects
                        "num_samples": num_samples,
                        "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                        "puzzle_indices": np.arange(num_samples + 1, dtype=np.int32),
                        "group_indices": np.arange(num_samples + 1, dtype=np.int32)
                    }
                    continue
                
                # Check if OLD streaming format
                stream_dir = os.path.join(dataset_path, "stream_checkpoints")
                if os.path.exists(stream_dir):
                    # Check for safetensors first (preferred)
                    safetensors_files = sorted(Path(stream_dir).glob("consolidated_*.safetensors"))
                    if safetensors_files:
                        # Lazy load using custom loader
                        # Check if 'sketches' exists in first file to confirm schema
                        from safetensors import safe_open
                        has_sketches = False
                        has_checksums = False
                        try:
                            with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
                                keys = f.keys()
                                if 'sketches' in keys:
                                    has_sketches = True
                                if 'checksums' in keys:
                                    has_checksums = True
                        except Exception as e:
                            print(f"[ERROR] Failed to inspect {safetensors_files[0]}: {e}")
                        
                        if has_sketches:
                            inputs = LazySafetensorsLoader(safetensors_files, "sketches")
                            num_samples = len(inputs)
                            
                            labels = None
                            if has_checksums:
                                labels = LazySafetensorsLoader(safetensors_files, "checksums")
                            
                            # group_indices are boundary markers: [0, 1, 2, ..., N]
                            # Each sample is its own group (1 sample per puzzle)
                            group_indices = np.arange(num_samples + 1, dtype=np.int32)
                            puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
                            
                            # Load children from metadata if available
                            all_children = []
                            for sf in safetensors_files:
                                meta_file = str(sf).replace('.safetensors', '_meta.json')
                                if os.path.exists(meta_file):
                                    try:
                                        with open(meta_file, 'r') as f:
                                            meta = json.load(f)
                                            if 'children' in meta:
                                                all_children.extend(meta['children'])
                                    except Exception as e:
                                        print(f"[WARN] Failed to load metadata {meta_file}: {e}")

                            children = None
                            if all_children:
                                try:
                                    children = np.array(all_children)
                                    print(f"[INFO] Loaded {len(children)} children records")
                                except Exception as e:
                                    print(f"[ERROR] Failed to convert children to numpy: {e}")

                            self._data[set_name_] = {
                                "inputs": inputs,
                                "labels": labels,
                                "children": children,
                                "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                                "puzzle_indices": puzzle_indices,
                                "group_indices": group_indices
                            }
                            continue

                    consolidated_files = sorted(Path(stream_dir).glob("consolidated_*.pt"))
                    if consolidated_files:
                        # Load and concatenate consolidated chunks
                        all_inputs = []
                        all_labels = []
                        
                        for chunk_file in consolidated_files:
                            chunk = torch.load(chunk_file, map_location='cpu')
                            if 'sketches' in chunk:
                                all_inputs.append(chunk['sketches'].numpy())
                            if 'checksums' in chunk:
                                all_labels.append(chunk['checksums'].numpy())
                        
                        inputs = np.concatenate(all_inputs, axis=0)
                        labels = np.concatenate(all_labels, axis=0)
                        num_samples = inputs.shape[0]
                        
                        # group_indices are boundary markers: [0, 1, 2, ..., N]
                        # Each sample is its own group (1 sample per puzzle)
                        # puzzle_indices also boundaries: [0, 1, 2, ..., N]
                        group_indices = np.arange(num_samples + 1, dtype=np.int32)
                        puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
                        
                        self._data[set_name_] = {
                            "inputs": inputs,
                            "labels": labels,
                            "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                            "puzzle_indices": puzzle_indices,
                            "group_indices": group_indices
                        }
                        continue
                
                # Check for raw_samples.pt format (vision-unified pipeline)
                raw_samples_path = os.path.join(dataset_path, self.split, 'raw_samples.pt')
                if os.path.exists(raw_samples_path):
                    # Load raw samples (images + metadata)
                    # weights_only=False needed for custom DataSample class
                    raw_samples = torch.load(raw_samples_path, map_location='cpu', weights_only=False)
                    num_samples = len(raw_samples)
                    
                    # Convert to expected format
                    # Each sample is a dict with 'input_image', 'output_image', 'task_id'
                    # Store as list of dicts - will be processed on-the-fly during training
                    group_indices = np.arange(num_samples + 1, dtype=np.int32)
                    puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
                    
                    self._data[set_name_] = {
                        "raw_samples": raw_samples,  # List of dicts
                        "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                        "puzzle_indices": puzzle_indices,
                        "group_indices": group_indices
                    }
                    continue
                
                # Standard format with .npy files
                field_mmap_modes = {
                    "inputs": "r",
                    "labels": "r",
                    "puzzle_identifiers": None,
                    "puzzle_indices": None,
                    "group_indices": None
                }
                
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}
        
        return batch

    def _package_batch(self, dataset, sample_indices, puzzle_identifier_indices):
        if "raw_samples" in dataset:
            batch_samples = [dataset["raw_samples"][i] for i in sample_indices]
            batch = {
                "raw_samples": batch_samples,
                "puzzle_identifiers": torch.from_numpy(dataset["puzzle_identifiers"][puzzle_identifier_indices].astype(np.int32))
            }
            # Process vision batch in worker process (parallel)
            batch = process_vision_batch(batch)
            # Remove raw_samples to save IPC bandwidth
            if 'raw_samples' in batch:
                del batch['raw_samples']
            return batch
        else:
            batch_data = {
                "inputs": dataset["inputs"][sample_indices],
                "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_identifier_indices]
            }
            if dataset["labels"] is not None:
                batch_data["labels"] = dataset["labels"][sample_indices]
            if dataset.get("children") is not None:
                batch_data["children"] = dataset["children"][sample_indices]
            
            return self._collate_batch(batch_data)

    def _iter_train(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        rng = np.random.default_rng(self.config.seed + worker_id)
        
        while True:
            self._iters += 1
            set_names = list(self._data.keys())
            rng.shuffle(set_names)
            
            for set_name in set_names:
                dataset = self._data[set_name]
                group_indices = dataset["group_indices"]
                total_groups = len(group_indices) - 1
                
                # Shard groups
                group_order = np.arange(total_groups, dtype=np.int32)
                rng.shuffle(group_order)
                group_order = group_order[worker_id::num_workers]
                
                start_index = 0
                while start_index < len(group_order):
                    start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                        rng, group_order, dataset["puzzle_indices"], dataset["group_indices"],
                        start_index, self.local_batch_size
                    )
                    
                    if len(batch_indices) == 0:
                        break
                        
                    batch = self._package_batch(dataset, batch_indices, batch_puzzle_indices)
                    yield set_name, batch, self.config.global_batch_size

    def _iter_test(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        for set_name, dataset in self._data.items():
            # Iterate over all samples
            num_samples = dataset["puzzle_identifiers"].shape[0]
            indices = np.arange(num_samples)
            
            # Shard indices
            my_indices = indices[worker_id::num_workers]
            
            for i in range(0, len(my_indices), self.local_batch_size):
                batch_indices = my_indices[i : i + self.local_batch_size]
                # For test, puzzle_identifiers align with samples
                batch = self._package_batch(dataset, batch_indices, batch_indices)
                yield set_name, batch, self.config.global_batch_size

                
    def __iter__(self):
        worker_info = get_worker_info()
        # assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


class CapsuleDataLoaderWrapper:
    """
    Wraps a standard DataLoader to yield batches in the format expected by the training loop.
    Used for multimodal capsule datasets.
    """
    def __init__(self, raw_loader, global_batch_size):
        self.raw_loader = raw_loader
        self.global_batch_size = global_batch_size
    
    def __iter__(self):
        for batch_data in self.raw_loader:
            # batch_data is tuple of (sketches,) or (sketches, checksums) or (sketches, checksums, children)
            # Sketches are already [B, 12, 512] from encoder.sketch_projection
            if len(batch_data) == 3:
                sketches = batch_data[0]  # [B, 12, 512]
                batch = {
                    'inputs': sketches,
                    'checksums': batch_data[1],
                    'children': batch_data[2],
                    'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                }
            elif len(batch_data) == 2:
                sketches = batch_data[0]  # [B, 12, 512]
                batch = {
                    'inputs': sketches,
                    'checksums': batch_data[1],
                    'children': None,
                    'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                }
            else:
                sketches = batch_data[0]  # [B, 12, 512]
                batch = {
                    'inputs': sketches,
                    'checksums': None,
                    'children': None,
                    'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                }
            
            # Yield in expected format: (set_name, batch, global_batch_size)
            yield 'capsule', batch, self.global_batch_size
    
    def __len__(self):
        return len(self.raw_loader)

