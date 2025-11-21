"""
Train Noise2Noise Denoiser for Text Rendered Images

Trains a lightweight U-Net to denoise text images using pairs of noisy variants
(different font sizes, positions) without requiring clean ground truth.

Usage:
    python scripts/train_noise2noise.py --samples 10000 --epochs 50 --output models/checkpoints/n2n_denoiser.pt
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')

from models.noise2noise_denoiser import Noise2NoiseDenoiser, NoisyVariantGenerator
from models.text_renderer import TextRenderer


class TextRenderDataset(Dataset):
    """Dataset of text samples for Noise2Noise training."""
    
    def __init__(self, texts, renderer, variant_generator):
        self.texts = texts
        self.renderer = renderer
        self.variant_gen = variant_generator
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Generate two noisy variants
        img1, img2 = self.variant_gen.generate_variant(text, variant_type='mixed')
        
        # Convert to tensors [C, H, W] in range [0, 1]
        img1 = torch.from_numpy(img1).float() / 255.0
        img2 = torch.from_numpy(img2).float() / 255.0
        
        # HWC -> CHW
        img1 = img1.permute(2, 0, 1)
        img2 = img2.permute(2, 0, 1)
        
        return img1, img2


def generate_training_texts(num_samples=10000):
    """Generate diverse text samples for training."""
    import random
    
    texts = []
    
    # Code snippets
    code_templates = [
        "def {func}():\n    return {val}",
        "class {cls}:\n    def __init__(self):\n        pass",
        "for i in range({n}):\n    print(i)",
        "if {var} > 0:\n    {action}",
        "import {module}\nfrom {pkg} import {item}",
    ]
    
    # Natural language
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "Neural networks learn hierarchical representations.",
        "Optimization algorithms minimize loss functions.",
        "Transformers use attention mechanisms for sequence modeling.",
        "Convolutional layers extract spatial features from images.",
        "Recurrent networks process sequential data effectively.",
        "Gradient descent updates model parameters iteratively.",
    ]
    
    # Math expressions
    math_templates = [
        "f(x) = {a}x^2 + {b}x + {c}",
        "y = {m}x + {b}",
        "∫{a} to {b} f(x)dx",
        "lim(x→{val}) f(x)",
        "∑_{{i=1}}^{{n}} x_i",
    ]
    
    for _ in range(num_samples):
        choice = random.choice(['code', 'text', 'math'])
        
        if choice == 'code':
            template = random.choice(code_templates)
            text = template.format(
                func=random.choice(['hello', 'process', 'compute', 'validate']),
                cls=random.choice(['Model', 'Handler', 'Parser', 'Manager']),
                val=random.randint(0, 100),
                n=random.randint(5, 20),
                var=random.choice(['x', 'y', 'count', 'value']),
                action=random.choice(['continue', 'break', 'return', 'pass']),
                module=random.choice(['torch', 'numpy', 'pandas', 'sklearn']),
                pkg=random.choice(['torch.nn', 'numpy.random', 'os.path']),
                item=random.choice(['Module', 'randn', 'join', 'split'])
            )
        elif choice == 'text':
            text = random.choice(sentences)
        else:  # math
            template = random.choice(math_templates)
            text = template.format(
                a=random.randint(1, 5),
                b=random.randint(1, 10),
                c=random.randint(-10, 10),
                m=random.randint(1, 5),
                val=random.randint(0, 10),
                n=random.randint(5, 20)
            )
        
        texts.append(text)
    
    return texts


def train_noise2noise(
    num_samples=10000,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    output_path='models/checkpoints/n2n_denoiser.pt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train Noise2Noise denoiser.
    
    Args:
        num_samples: Number of text samples to generate
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        output_path: Path to save trained model
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print(" NOISE2NOISE DENOISER TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Samples: {num_samples:,}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("="*70 + "\n")
    
    # Generate training texts
    print(" Generating training texts...")
    texts = generate_training_texts(num_samples)
    print(f"✓ Generated {len(texts):,} text samples\n")
    
    # Initialize renderer and variant generator
    print(" Initializing renderers...")
    renderer = TextRenderer(width=224, height=224, font_size=12)
    variant_gen = NoisyVariantGenerator(renderer)
    print("✓ Renderers initialized\n")
    
    # Create dataset and dataloader
    print(" Creating dataset...")
    dataset = TextRenderDataset(texts, renderer, variant_gen)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    print(f"✓ Dataset ready: {len(dataset):,} samples\n")
    
    # Initialize model
    print(" Initializing Noise2Noise model...")
    denoiser = Noise2NoiseDenoiser(device=device)
    optimizer = optim.Adam(denoiser.model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    params = sum(p.numel() for p in denoiser.model.parameters())
    print(f"✓ Model initialized: {params:,} parameters\n")
    
    # Training loop
    print(" Starting training...\n")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        denoiser.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for noisy1, noisy2 in pbar:
            loss = denoiser.train_step(noisy1, noisy2, optimizer)
            epoch_loss += loss
            
            pbar.set_postfix({'loss': f'{loss:.6f}'})
        
        # Average loss
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            denoiser.save(str(output_path))
            print(f"   💾 Best model saved (loss: {best_loss:.6f})\n")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train Noise2Noise denoiser for text images")
    parser.add_argument(
        '--samples',
        type=int,
        default=10000,
        help='Number of text samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/checkpoints/n2n_denoiser.pt',
        help='Output path for trained model (default: models/checkpoints/n2n_denoiser.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda if available)'
    )
    
    args = parser.parse_args()
    
    train_noise2noise(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
