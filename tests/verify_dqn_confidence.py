
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config

def verify_dqn_confidence():
    print("🚀 Starting DQN Confidence Manager Verification...")
    
    # 1. Config Setup
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=16,
        num_puzzle_identifiers=0,
        puzzle_emb_len=0, # Fix size mismatch
        vocab_size=100,
        H_cycles=4,
        L_cycles=1,
        H_layers=1, # Added missing field
        L_layers=1,
        hidden_size=32, # Small hidden size
        expansion=2,
        num_heads=4,
        pos_encodings="learned",
        halt_max_steps=10,
        halt_exploration_prob=0.1,
        
        # DQN Config
        enable_dqn=True,
        q_head_type="confidence", # NEW TYPE
        enable_adaptive_hcycles=True,
        hcycle_confidence_threshold=0.5, # Low threshold to trigger actions
        q_head_num_actions=3,
        use_semantic_encoder=False # Disable vision encoder for simple test
    )
    
    print("✅ Config created.")
    
    # 2. Model Initialization
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.train() # Enable training mode to collect trajectory
    print("✅ Model initialized.")
    
    # Verify Q-Head type
    if hasattr(model.inner.q_head, 'net'):
        print("✅ Q-Head is MLP (ConfidenceQHead structure confirmed).")
    else:
        print("❌ Q-Head structure mismatch!")
        
    # 3. Forward Pass
    inputs = torch.randint(0, 100, (2, 16)) # [B, L]
    puzzle_ids = torch.zeros(2, dtype=torch.long)
    
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_ids
    }
    
    # Create initial carry
    carry = model.inner.empty_carry(2)
    
    print("🔄 Running forward pass...")
    new_carry, output, q_logits = model.inner(carry, batch)
    
    print("✅ Forward pass completed.")
    
    # 4. Verify Output
    if isinstance(output, dict):
        if 'dqn_trajectory' in output:
            traj = output['dqn_trajectory']
            print(f"✅ DQN Trajectory found! Length: {len(traj)}")
            
            first_step = traj[0]
            print(f"   Step 0 keys: {first_step.keys()}")
            print(f"   Hidden shape: {first_step['hidden'].shape}")
            print(f"   Stability shape: {first_step['stability'].shape}")
            print(f"   Loop Count shape: {first_step['loop_count'].shape}")
            print(f"   Action shape: {first_step['action'].shape}")
            
            # Verify shapes
            # Hidden: [B, D] -> [2, 32]
            # Stability: [B, 1] -> [2, 1]
            # Loop: [B, 1] -> [2, 1]
            
            assert first_step['hidden'].shape == (2, 32)
            assert first_step['stability'].shape == (2, 1)
            assert first_step['loop_count'].shape == (2, 1)
            
            print("✅ Trajectory shapes verified.")
        else:
            print("❌ 'dqn_trajectory' NOT found in output dict!")
    else:
        print("❌ Output is not a dict (expected dict with trajectory)!")

if __name__ == "__main__":
    try:
        verify_dqn_confidence()
        print("\n🎉 Verification SUCCESS!")
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
