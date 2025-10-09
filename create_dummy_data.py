import numpy as np
import os

data_dir = "data/arc-aug-1000/train"
os.makedirs(data_dir, exist_ok=True)

# Based on dummy dataset.json
seq_len = 10
num_examples = 16  # Match global_batch_size for simplicity
num_puzzles = 1
num_groups = 1

# Create dummy data
inputs = np.zeros((num_examples, seq_len), dtype=np.int32)
labels = np.zeros((num_examples, seq_len), dtype=np.int32)
puzzle_identifiers = np.zeros((num_puzzles,), dtype=np.int32)
puzzle_indices = np.array([0, num_examples], dtype=np.int64)
group_indices = np.array([0, num_puzzles], dtype=np.int64)

# Save files for the 'train' set
np.save(os.path.join(data_dir, "train__inputs.npy"), inputs)
np.save(os.path.join(data_dir, "train__labels.npy"), labels)
np.save(os.path.join(data_dir, "train__puzzle_identifiers.npy"), puzzle_identifiers)
np.save(os.path.join(data_dir, "train__puzzle_indices.npy"), puzzle_indices)
np.save(os.path.join(data_dir, "train__group_indices.npy"), group_indices)

print("Dummy .npy files created successfully.")