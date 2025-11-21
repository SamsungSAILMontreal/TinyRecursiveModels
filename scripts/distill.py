import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import sys
import os
import copy
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from scripts.load_model import ModelLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.checkpointing import AsyncCheckpointManager

logger = logging.getLogger(__name__)

class HyperDistillationTrainer:
    """
    Hyper-Distillation Trainer (Default):
    - Self-Distillation (Teacher = Snapshot of Student)
    - Deep Thinking Boost (Teacher runs with 2x H_cycles)
    - Q-Value Distillation (Student mimics Teacher's reasoning policy)
    - Cyclic Learning Rate (Forget-Relearn)
    """
    def __init__(self, cfg: DictConfig):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        
        # Hyper-Distillation Parameters (Defaults)
        self.teacher_cycles_factor = getattr(cfg, 'teacher_cycles_factor', 2.0)
        self.snapshot_interval = getattr(cfg, 'snapshot_interval', 1000)
        self.q_loss_weight = getattr(cfg, 'q_loss_weight', 1.0)
        self.temperature = getattr(cfg.distillation, 'temperature', 2.0)
        self.alpha = getattr(cfg.distillation, 'alpha', 0.5)
        
        print(f"🚀 Hyper-Distillation Initialized:")
        print(f"   Teacher Boost: {self.teacher_cycles_factor}x cycles")
        print(f"   Snapshot Interval: {self.snapshot_interval} steps")
        print(f"   Q-Value Loss Weight: {self.q_loss_weight}")

        # Initialize Student
        arch_config = OmegaConf.to_container(cfg.arch, resolve=True)
        
        # Inject defaults if missing
        if 'seq_len' not in arch_config: arch_config['seq_len'] = 1024
        if 'vocab_size' not in arch_config: arch_config['vocab_size'] = 4096
        if 'num_puzzle_identifiers' not in arch_config: arch_config['num_puzzle_identifiers'] = 0
        if 'batch_size' not in arch_config: arch_config['batch_size'] = cfg.global_batch_size if cfg else 1
        if 'input_vocab_size' not in arch_config: arch_config['input_vocab_size'] = arch_config['vocab_size']

        self.student = TinyRecursiveReasoningModel_ACTV1(arch_config).to(self.device)
        
        # Initialize Teacher (Snapshot of Student)
        self._update_teacher_snapshot()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=cfg.lr if cfg else 1e-4,
            weight_decay=cfg.weight_decay if cfg else 0.01
        )
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Checkpointing
        self.checkpoint_manager = AsyncCheckpointManager(
            checkpoint_dir="checkpoints/hyper_distilled_student",
            max_keep=3,
            use_safetensors=True
        )
        
        self.global_step = 0

    def _update_teacher_snapshot(self):
        """Update teacher weights with a copy of the student."""
        print(f"📸 Updating Teacher Snapshot (Step {getattr(self, 'global_step', 0)})...")
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, batch):
        # Handle different batch structures
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch.get('target_ids', input_ids).to(self.device)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            target_ids = batch[1].to(self.device) if len(batch) > 1 else input_ids.to(self.device)
        else:
            input_ids = batch.to(self.device)
            target_ids = input_ids

        # Reconstruct batch dict for TRM forward
        batch_dict = {"inputs": input_ids}
        if isinstance(batch, dict):
            batch_dict.update(batch)
            
        # 1. Periodic Snapshot Update
        if self.global_step > 0 and self.global_step % self.snapshot_interval == 0:
            self._update_teacher_snapshot()

        # 2. Teacher Forward (Deep Thinking Boost)
        with torch.no_grad():
            # Calculate boosted cycles
            base_cycles = self.student.config.H_cycles
            boosted_cycles = int(base_cycles * self.teacher_cycles_factor)
            
            # Initialize teacher carry
            teacher_carry = self.teacher.initial_carry(batch_dict)
            
            # Forward with boosted cycles
            teacher_out_carry, teacher_outputs = self.teacher(
                teacher_carry, batch_dict, 
                global_step=self.global_step,
                h_cycles_override=boosted_cycles
            )
            
            teacher_logits = teacher_outputs['logits']
            
            # Get Teacher Q-values
            t_q_halt = teacher_outputs.get('q_halt_logits')
            t_q_cont = teacher_outputs.get('q_continue_logits')
            t_q_exp = teacher_outputs.get('q_expand_logits')

        # 3. Student Forward (Standard)
        student_carry = self.student.initial_carry(batch_dict)
        student_out_carry, student_outputs = self.student(student_carry, batch_dict, global_step=self.global_step)
        student_logits = student_outputs['logits']
        
        s_q_halt = student_outputs.get('q_halt_logits')
        s_q_cont = student_outputs.get('q_continue_logits')
        s_q_exp = student_outputs.get('q_expand_logits')

        # 4. Loss Calculation
        
        # A. Task Loss (Hard Labels)
        # Using target_ids for task loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        task_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # B. Distillation Loss (KL Divergence on Logits)
        soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distill_loss = self.kl_div_loss(soft_prob, soft_targets) * (self.temperature ** 2)

        # C. Q-Value Distillation (MSE on Policy)
        q_loss = torch.tensor(0.0, device=self.device)
        if t_q_halt is not None and s_q_halt is not None:
            q_loss += F.mse_loss(s_q_halt, t_q_halt)
            q_loss += F.mse_loss(s_q_cont, t_q_cont)
            if t_q_exp is not None and s_q_exp is not None:
                q_loss += F.mse_loss(s_q_exp, t_q_exp)
        
        # Total Loss
        loss = (self.alpha * task_loss) + ((1 - self.alpha) * distill_loss) + (self.q_loss_weight * q_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), distill_loss.item(), task_loss.item(), q_loss.item()

    def train_loop(self, dataloader: DataLoader, num_epochs: int = 1):
        self.student.train()
        self.global_step = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                loss, d_loss, t_loss, q_loss = self.train_step(batch)
                total_loss += loss
                self.global_step += 1
                
                if i % 10 == 0:
                    print(f"Epoch {epoch} | Step {i} | Loss: {loss:.4f} (Distill: {d_loss:.4f}, Task: {t_loss:.4f}, Q: {q_loss:.4f})")
                
                if self.global_step % 1000 == 0:
                     self.checkpoint_manager.save_async(self.global_step, self.student, self.optimizer)

            print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")
            
            self.checkpoint_manager.save_async(self.global_step, self.student, self.optimizer)
        
        self.checkpoint_manager.shutdown()

@hydra.main(config_path="../config", config_name="distill", version_base="1.2")
def main(cfg: DictConfig):
    print("="*70)
    print("  Phase 3: Hyper-Distillation (Self-Distillation + Deep Thinking)")
    print("="*70)
    
    # Force self-distillation settings
    if not hasattr(cfg, 'teacher_cycles_factor'):
        OmegaConf.set_struct(cfg, False)
        cfg.teacher_cycles_factor = 2.0
        cfg.snapshot_interval = 1000
        cfg.q_loss_weight = 1.0
        OmegaConf.set_struct(cfg, True)

    print("Loading Dataset...")
    dataset_paths = cfg.data_paths
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=dataset_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1, 
        rank=0,
        num_replicas=1
    ), split='train')
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.global_batch_size, 
        shuffle=False, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    
    trainer = HyperDistillationTrainer(cfg)
    trainer.train_loop(dataloader, num_epochs=1)

if __name__ == "__main__":
    main()
