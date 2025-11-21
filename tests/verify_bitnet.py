import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bitnet import BitNetTransformer

def test_bitnet_training():
    print("Testing BitNet Training (Simulated Quantization)...")
    
    # Parameters
    vocab_size = 100
    d_model = 32
    nhead = 4
    num_layers = 2
    seq_len = 10
    batch_size = 2
    
    # Instantiate model
    model = BitNetTransformer(vocab_size, d_model, nhead, num_layers)
    print("Model instantiated.")
    
    # Dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("Running forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Loss
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
    print(f"Loss: {loss.item()}")
    
    # Backward pass
    print("Running backward pass...")
    loss.backward()
    
    # Check gradients
    print("Checking gradients...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            # print(f"Gradient for {name}: {param.grad.norm()}")
            if param.grad.norm() == 0:
                print(f"WARNING: Zero gradient for {name}")
        else:
            print(f"WARNING: No gradient for {name}")
            
    if has_grad:
        print("PASS: Gradients flowed through the network (STE working).")
    else:
        print("FAIL: No gradients found.")

if __name__ == "__main__":
    test_bitnet_training()
