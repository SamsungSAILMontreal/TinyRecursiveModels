"""
Model Benchmark Tester for TRM Models

Measures:
- Inference latency (ms)
- Throughput (tokens/sec)
- Memory usage (RAM)
- Model size
- Perplexity (optional, if test data provided)

Based on:
- https://www.evidentlyai.com/llm-guide/llm-benchmarks
- PyTorch benchmarking best practices
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import psutil
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_model import ModelLoader
from dataset.common import get_tokenizer


class ModelBenchmark:
    """Comprehensive model benchmarking suite."""
    
    def __init__(self, model, tokenizer, config: Dict, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.results = {}
        
    def measure_model_size(self) -> Dict:
        """Measure model size and parameter count."""
        print("\n📊 Measuring model size...")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size in bytes (assuming fp32)
        size_bytes = total_params * 4  # 4 bytes per fp32 parameter
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_mb / 1024
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'size_gb': round(size_gb, 2),
        }
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {size_gb:.2f} GB ({size_mb:.2f} MB)")
        
        return results
    
    def measure_inference_latency(self, sequence_lengths: List[int] = [128, 256, 512],
                                   num_iterations: int = 100) -> Dict:
        """
        Measure inference latency for different sequence lengths.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            num_iterations: Number of iterations per test
            
        Returns:
            results: Latency statistics
        """
        print(f"\n⚡ Measuring inference latency ({num_iterations} iterations)...")
        
        results = {}
        
        for seq_len in sequence_lengths:
            print(f"   Testing sequence length: {seq_len}")
            
            # Create dummy input
            batch = {
                'inputs': torch.randint(0, self.config['vocab_size'], (1, seq_len), 
                                       dtype=torch.long, device=self.device),
                'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device),
                'labels': torch.full((1, seq_len), -100, dtype=torch.long, device=self.device)
            }
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    carry = self.model.initial_carry(batch)
                    _ = self.model(carry, batch)
            
            # Synchronize if using CUDA
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            latencies = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    
                    carry = self.model.initial_carry(batch)
                    _ = self.model(carry, batch)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)  # Convert to ms
            
            results[f'seq_len_{seq_len}'] = {
                'mean_ms': round(sum(latencies) / len(latencies), 2),
                'min_ms': round(min(latencies), 2),
                'max_ms': round(max(latencies), 2),
                'p50_ms': round(sorted(latencies)[len(latencies)//2], 2),
                'p95_ms': round(sorted(latencies)[int(len(latencies)*0.95)], 2),
                'p99_ms': round(sorted(latencies)[int(len(latencies)*0.99)], 2),
            }
            
            print(f"      Mean: {results[f'seq_len_{seq_len}']['mean_ms']:.2f} ms")
            print(f"      P50:  {results[f'seq_len_{seq_len}']['p50_ms']:.2f} ms")
            print(f"      P95:  {results[f'seq_len_{seq_len}']['p95_ms']:.2f} ms")
        
        return results
    
    def measure_throughput(self, sequence_length: int = 256, 
                          num_tokens_to_generate: int = 100) -> Dict:
        """
        Measure text generation throughput.
        
        Args:
            sequence_length: Initial sequence length
            num_tokens_to_generate: Number of tokens to generate
            
        Returns:
            results: Throughput statistics
        """
        print(f"\n🚀 Measuring generation throughput ({num_tokens_to_generate} tokens)...")
        
        # Create initial input
        batch = {
            'inputs': torch.randint(0, self.config['vocab_size'], (1, sequence_length),
                                   dtype=torch.long, device=self.device),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device),
            'labels': torch.full((1, sequence_length), -100, dtype=torch.long, device=self.device)
        }
        
        # Generate tokens
        start_time = time.perf_counter()
        
        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            
            for _ in range(num_tokens_to_generate):
                # Forward pass
                carry, outputs = self.model(carry, batch)
                
                # Sample next token (greedy)
                next_token = outputs['logits'][0, -1, :].argmax().item()
                
                # Update batch
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                batch['inputs'] = torch.cat([batch['inputs'], next_token_tensor], dim=1)
                
                # Truncate if needed
                if batch['inputs'].shape[1] > self.config['seq_len']:
                    batch['inputs'] = batch['inputs'][:, -self.config['seq_len']:]
                
                batch['labels'] = torch.full_like(batch['inputs'], -100)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        tokens_per_sec = num_tokens_to_generate / total_time
        ms_per_token = (total_time / num_tokens_to_generate) * 1000
        
        results = {
            'tokens_generated': num_tokens_to_generate,
            'total_time_sec': round(total_time, 2),
            'tokens_per_sec': round(tokens_per_sec, 2),
            'ms_per_token': round(ms_per_token, 2),
        }
        
        print(f"   Tokens generated: {num_tokens_to_generate}")
        print(f"   Total time: {total_time:.2f} sec")
        print(f"   Throughput: {tokens_per_sec:.2f} tokens/sec")
        print(f"   Latency: {ms_per_token:.2f} ms/token")
        
        return results
    
    def measure_memory_usage(self, sequence_length: int = 512) -> Dict:
        """
        Measure memory usage during inference.
        
        Args:
            sequence_length: Sequence length to test
            
        Returns:
            results: Memory usage statistics
        """
        print(f"\n💾 Measuring memory usage...")
        
        # Get process
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_mb = process.memory_info().rss / (1024 ** 2)
        
        # Create batch
        batch = {
            'inputs': torch.randint(0, self.config['vocab_size'], (1, sequence_length),
                                   dtype=torch.long, device=self.device),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device),
            'labels': torch.full((1, sequence_length), -100, dtype=torch.long, device=self.device)
        }
        
        # Run inference
        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            _ = self.model(carry, batch)
        
        # Measure peak memory
        peak_mb = process.memory_info().rss / (1024 ** 2)
        
        # CUDA memory (if applicable)
        cuda_memory = {}
        if self.device == 'cuda' and torch.cuda.is_available():
            cuda_memory = {
                'allocated_mb': round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
                'reserved_mb': round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
                'max_allocated_mb': round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            }
        
        results = {
            'baseline_ram_mb': round(baseline_mb, 2),
            'peak_ram_mb': round(peak_mb, 2),
            'inference_ram_mb': round(peak_mb - baseline_mb, 2),
        }
        
        if cuda_memory:
            results['cuda'] = cuda_memory
        
        print(f"   Baseline RAM: {baseline_mb:.2f} MB")
        print(f"   Peak RAM: {peak_mb:.2f} MB")
        print(f"   Inference RAM: {peak_mb - baseline_mb:.2f} MB")
        
        if cuda_memory:
            print(f"   CUDA allocated: {cuda_memory['allocated_mb']:.2f} MB")
            print(f"   CUDA reserved: {cuda_memory['reserved_mb']:.2f} MB")
        
        return results
    
    def compute_perplexity(self, test_data_path: Optional[str] = None, 
                          max_samples: int = 100) -> Optional[Dict]:
        """
        Compute perplexity on test data (if provided).
        
        Args:
            test_data_path: Path to test data (text file or dataset)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            results: Perplexity statistics or None
        """
        if test_data_path is None:
            print("\n⚠️  Skipping perplexity (no test data provided)")
            return None
        
        print(f"\n📈 Computing perplexity on test data...")
        print(f"   Test data: {test_data_path}")
        
        # Load test data
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"   ❌ Error loading test data: {e}")
            return None
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Compute loss on chunks
        chunk_size = self.config['seq_len']
        total_loss = 0.0
        num_chunks = 0
        
        with torch.no_grad():
            for i in range(0, min(len(tokens) - chunk_size, max_samples * chunk_size), chunk_size):
                chunk = tokens[i:i+chunk_size]
                
                if len(chunk) < chunk_size:
                    continue
                
                # Create batch
                inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
                labels = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)
                
                batch = {
                    'inputs': inputs,
                    'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device),
                    'labels': labels
                }
                
                # Forward pass
                carry = self.model.initial_carry(batch)
                carry, outputs = self.model(carry, batch)
                
                # Compute loss
                logits = outputs['logits']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='mean'
                )
                
                total_loss += loss.item()
                num_chunks += 1
                
                if num_chunks >= max_samples:
                    break
        
        # Compute perplexity
        avg_loss = total_loss / num_chunks if num_chunks > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results = {
            'perplexity': round(perplexity, 2),
            'avg_loss': round(avg_loss, 4),
            'num_chunks_evaluated': num_chunks,
        }
        
        print(f"   Perplexity: {perplexity:.2f}")
        print(f"   Avg loss: {avg_loss:.4f}")
        print(f"   Chunks evaluated: {num_chunks}")
        
        return results
    
    def run_full_benchmark(self, test_data_path: Optional[str] = None) -> Dict:
        """Run complete benchmark suite."""
        print("\n" + "="*70)
        print("  🔬 TRM Model Benchmark Suite")
        print("="*70)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.config.get('name', 'TRM'),
            'device': self.device,
            'config': {
                'hidden_size': self.config.get('hidden_size'),
                'vocab_size': self.config.get('vocab_size'),
                'seq_len': self.config.get('seq_len'),
                'H_cycles': self.config.get('H_cycles'),
                'L_layers': self.config.get('L_layers'),
            }
        }
        
        # Run benchmarks
        self.results['model_size'] = self.measure_model_size()
        self.results['inference_latency'] = self.measure_inference_latency()
        self.results['throughput'] = self.measure_throughput()
        self.results['memory_usage'] = self.measure_memory_usage()
        self.results['perplexity'] = self.compute_perplexity(test_data_path)
        
        print("\n" + "="*70)
        print("  ✅ Benchmark Complete!")
        print("="*70)
        
        return self.results
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("  📊 BENCHMARK SUMMARY")
        print("="*70)
        
        # Model size
        print("\n🔹 Model Size:")
        print(f"   Parameters: {self.results['model_size']['total_parameters']:,}")
        print(f"   Size: {self.results['model_size']['size_gb']:.2f} GB")
        
        # Latency
        print("\n🔹 Inference Latency (seq_len=512):")
        lat = self.results['inference_latency'].get('seq_len_512', {})
        if lat:
            print(f"   Mean: {lat['mean_ms']:.2f} ms")
            print(f"   P95:  {lat['p95_ms']:.2f} ms")
        
        # Throughput
        print("\n🔹 Generation Throughput:")
        thr = self.results['throughput']
        print(f"   {thr['tokens_per_sec']:.2f} tokens/sec")
        print(f"   {thr['ms_per_token']:.2f} ms/token")
        
        # Memory
        print("\n🔹 Memory Usage:")
        mem = self.results['memory_usage']
        print(f"   Peak RAM: {mem['peak_ram_mb']:.2f} MB")
        print(f"   Inference: {mem['inference_ram_mb']:.2f} MB")
        
        # Perplexity
        if self.results['perplexity']:
            print("\n🔹 Perplexity:")
            print(f"   {self.results['perplexity']['perplexity']:.2f}")
        
        print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark TRM model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark.py --model model_cpu_bnn.pt
  
  # Benchmark with test data (compute perplexity)
  python benchmark.py --model checkpoints/text-trm-10m/latest.pt --test-data test.txt
  
  # Full benchmark with results saved
  python benchmark.py --model model_cpu_bnn.pt --output results.json
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data for perplexity computation')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output path for results JSON')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name (default: gpt2)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for latency benchmark (default: 100)')
    
    args = parser.parse_args()
    
    # Load model
    print("🔄 Loading model...")
    loader = ModelLoader(args.model, args.device)
    model = loader.load_model(quantized='bnn' in args.model.lower() or 'quantized' in args.model.lower())
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer: {args.tokenizer}")
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Create benchmark
    benchmark = ModelBenchmark(model, tokenizer, loader.config, args.device)
    
    # Run benchmark
    benchmark.run_full_benchmark(test_data_path=args.test_data)
    
    # Save and print results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
