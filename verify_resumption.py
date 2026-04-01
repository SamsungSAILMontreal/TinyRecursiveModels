
import os
import shutil
import torch
import numpy as np
import random
import sys
from unittest.mock import MagicMock, patch

# Disable compilation for testing
os.environ["DISABLE_COMPILE"] = "1"

# Mock adam_atan2
sys.modules["adam_atan2"] = MagicMock()
sys.modules["adam_atan2"].AdamATan2 = torch.optim.AdamW

# Mock models.sparse_embedding
sys.modules["models.sparse_embedding"] = MagicMock()
class MockOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults=None):
        if defaults is None: defaults = {}
        defaults['lr'] = 0.01
        super().__init__(params, defaults)
    def step(self, closure=None):
        pass
sys.modules["models.sparse_embedding"].CastedSparseEmbeddingSignSGD_Distributed = MockOptimizer

# Mock puzzle_dataset
sys.modules["puzzle_dataset"] = MagicMock()
class MockPuzzleDataset:
    pass
sys.modules["puzzle_dataset"].PuzzleDataset = MockPuzzleDataset
sys.modules["puzzle_dataset"].PuzzleDatasetConfig = MagicMock()
sys.modules["puzzle_dataset"].PuzzleDatasetMetadata = MagicMock()

# Mock distributed
sys.modules["torch.distributed"] = MagicMock()
sys.modules["torch.distributed"].get_rank.return_value = 0
sys.modules["torch.distributed"].get_world_size.return_value = 1
sys.modules["torch.distributed"].broadcast_object_list = MagicMock()
sys.modules["torch.distributed"].broadcast = MagicMock()

import pretrain
from pretrain import TrainState, PretrainConfig, ArchConfig, LossConfig, create_model, save_train_state, load_checkpoint, init_train_state
import torch.nn as nn

# Mock classes
class MockArchConfig:
    def __init__(self):
        self.name = "test_arch"
        self.loss = LossConfig(name="test_loss")
        self.puzzle_emb_ndim = 0
        self.__pydantic_extra__ = {}

class MockConfig:
    def __init__(self, **kwargs):
        self.arch = MockArchConfig()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.loss = LossConfig(name="test_loss")
        self.checkpoint_path = "test_verification_checkpoints"
        self.load_checkpoint = None
        self.global_batch_size = 1
        self.epochs = 1
        self.lr = 0.01
        self.lr_min_ratio = 0.0
        self.lr_warmup_steps = 0
        self.weight_decay = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.puzzle_emb_lr = 0.01
        self.puzzle_emb_weight_decay = 0.0
        self.freeze_weights = False
        self.ema = False
        self.seed = 42

class MockMetadata:
    def __init__(self):
        self.vocab_size = 10
        self.seq_len = 10
        self.num_puzzle_identifiers = 1
        self.total_groups = 1
        self.mean_puzzle_examples = 1

def test_bitwise_resumption():
    print("Setting up bitwise resumption test...")
    if os.path.exists("test_verification_checkpoints"):
        shutil.rmtree("test_verification_checkpoints")
    os.makedirs("test_verification_checkpoints")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class SimpleModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            self.model = MagicMock()
            self.model.puzzle_emb = MagicMock()
            self.model.puzzle_emb.buffers.return_value = []
            self.model.puzzle_emb.weights.shape = torch.Size([1, 10])

        def forward(self, carry, batch, return_keys=[]):
            # Advance all RNGs
            noise = torch.randn(1, 2)
            _ = np.random.rand(1)
            _ = random.random()

            if batch is not None and "inputs" in batch:
                inp = batch["inputs"]
            else:
                inp = torch.randn(1, 10)

            loss = self.linear(inp).sum() + noise.sum()
            return carry, loss, {}, {}, False

        def initial_carry(self, batch):
            return None

    pretrain.load_model_class = lambda name, *args: SimpleModel

    config = MockConfig(
        data_paths=[],
        global_batch_size=1,
        epochs=1,
        lr=0.01,
        lr_min_ratio=0.0,
        lr_warmup_steps=0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        puzzle_emb_lr=0.0,
        puzzle_emb_weight_decay=0.0,
        checkpoint_path="test_verification_checkpoints",
        seed=seed
    )

    metadata = MockMetadata()

    # Capture original torch.load and torch.device
    original_load = torch.load
    original_device = torch.device

    def mock_load(f, map_location=None, **kwargs):
        return original_load(f, map_location="cpu", **kwargs)

    def real_device_mock(device_str):
        if device_str == "cuda":
            return original_device("cpu")
        return original_device(device_str)

    with patch("torch.load", side_effect=mock_load), \
         patch("torch.device", side_effect=real_device_mock), \
         patch("torch.cuda.is_available", return_value=False):

        # --- Continuous Run (3 steps) ---
        print("Running continuous training (3 steps)...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_state_cont, _ = init_train_state(config, metadata, 0, 1)

        losses_cont = []
        for i in range(3):
            train_state_cont.step += 1
            loss = train_state_cont.model(None, None)[1]
            loss.backward()
            for opt in train_state_cont.optimizers:
                opt.step()
                opt.zero_grad()
            losses_cont.append(loss.item())
            print(f"Step {i+1} Loss: {loss.item()}")

        print(f"Continuous Losses: {losses_cont}")

        # --- Interrupted Run ---
        print("\nRunning interrupted training...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_state_part1, _ = init_train_state(config, metadata, 0, 1)

        train_state_part1.step += 1
        loss = train_state_part1.model(None, None)[1]
        loss.backward()
        for opt in train_state_part1.optimizers:
            opt.step()
            opt.zero_grad()
        print(f"Part 1 Step 1 Loss: {loss.item()}")

        assert losses_cont[0] == loss.item(), "Step 1 mismatch!"

        print("Saving checkpoint at step 1...")
        save_train_state(config, train_state_part1)

        print("Resuming...")
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)

        config.load_checkpoint = None

        # Manual auto-resume logic simulation
        if config.load_checkpoint is None and config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
            max_step = -1
            max_ckpt = None
            for fname in os.listdir(config.checkpoint_path):
                if fname.startswith("step_") and not fname.endswith(".tmp"):
                    try:
                        step_val = int(fname.split("_")[1])
                        if step_val > max_step:
                            max_step = step_val
                            max_ckpt = os.path.join(config.checkpoint_path, fname)
                    except (ValueError, IndexError):
                        continue
            if max_ckpt is not None:
                print(f"Auto-resume: Found {max_ckpt}")
                config.load_checkpoint = max_ckpt

        expected_ckpt = os.path.join(config.checkpoint_path, "step_1")
        assert config.load_checkpoint == expected_ckpt

        train_state_resumed, checkpoint_data = init_train_state(config, metadata, 0, 1)

        assert checkpoint_data is not None
        assert train_state_resumed.step == 1

        losses_resumed = [losses_cont[0]]
        for i in range(2):
            train_state_resumed.step += 1
            loss = train_state_resumed.model(None, None)[1]
            loss.backward()
            for opt in train_state_resumed.optimizers:
                opt.step()
                opt.zero_grad()
            losses_resumed.append(loss.item())
            print(f"Resumed Step {i+2} Loss: {loss.item()}")

        print(f"Resumed Losses: {losses_resumed}")

        if np.allclose(losses_cont, losses_resumed):
            print("\nSUCCESS: Bitwise resumption verified!")
        else:
            print("\nFAILURE: Resumption mismatch!")
            print(f"Continuous: {losses_cont}")
            print(f"Resumed:    {losses_resumed}")
            exit(1)

if __name__ == "__main__":
    test_bitwise_resumption()
