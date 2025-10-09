# Snapdragon NPU Deployment Guide

This document outlines the complete plan and scripts for deploying TinyRecursiveInference models to Qualcomm Snapdragon NPU for inference-only execution.

## Overview

The deployment process consists of three main stages:
1. Export trained PyTorch model to ONNX format
2. Convert ONNX to Qualcomm DLC format with quantization
3. Create inference runtime for Snapdragon NPU

## Prerequisites

### Software Requirements
- Qualcomm Neural Processing SDK (SNPE) v2.x+
- ONNX Runtime
- PyTorch (matching training version)
- Python 3.8+

### Hardware Requirements
- Development machine with trained model checkpoints
- Snapdragon device with NPU support (8 Gen 2+, X Elite, etc.)

---

## Stage 1: ONNX Export

### Script: `export_to_onnx.py`

```python
#!/usr/bin/env python3
"""
Export trained TinyRecursiveInference model to ONNX format.

Usage:
    python export_to_onnx.py \
        --checkpoint checkpoints/trm/my_model/checkpoint_best.pt \
        --output models_onnx/trm_inference.onnx \
        --H_cycles 3 \
        --L_cycles 4 \
        --unroll
"""

import argparse
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import json

# Import model loading utilities
from pretrain import load_model_class, create_model
from omegaconf import OmegaConf


class ONNXInferenceWrapper(torch.nn.Module):
    """
    Wrapper that unrolls recursive reasoning for ONNX export.
    ONNX doesn't handle dynamic control flow well, so we unroll the loops.
    """

    def __init__(self, base_model, H_cycles, L_cycles, max_seq_len=2048):
        super().__init__()
        self.base_model = base_model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.max_seq_len = max_seq_len

    def forward(self, inputs, labels, puzzle_identifiers):
        """
        Forward pass with unrolled recursion.

        Args:
            inputs: [batch_size, seq_len] token IDs
            labels: [batch_size, seq_len] target token IDs
            puzzle_identifiers: [batch_size] puzzle IDs for embeddings

        Returns:
            logits: [batch_size, seq_len, vocab_size] output predictions
        """
        batch_size = inputs.shape[0]

        # Create initial batch dict
        batch = {
            'inputs': inputs,
            'labels': labels,
            'puzzle_identifiers': puzzle_identifiers,
        }

        # Initialize carry state
        carry = self.base_model.initial_carry(batch)

        # Unroll H and L cycles explicitly
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                # Run one step of reasoning
                carry = self.base_model.forward(carry, batch)

        # Extract final logits from carry
        logits = carry['logits']  # [batch_size, seq_len, vocab_size]

        return logits


def load_checkpoint_for_export(checkpoint_path, device='cpu'):
    """
    Load trained checkpoint and prepare for export.

    Returns:
        model: Unwrapped base model (no ACT wrapper, no compile)
        config: Model configuration dict
        metadata: Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = OmegaConf.to_container(checkpoint['config'], resolve=True)
    else:
        raise ValueError("Checkpoint missing config. Cannot reconstruct model.")

    # Force inference settings
    config['arch']['training_mode'] = False
    config['compile'] = False  # Don't compile for ONNX export

    # Load model class
    model_class = load_model_class(config['arch']['model_name'])

    # Create base model without wrappers
    base_model = model_class(config['arch'])

    # Load state dict (handle EMA if present)
    if 'ema_state_dict' in checkpoint and checkpoint.get('used_ema', False):
        print("Using EMA weights for export")
        state_dict = checkpoint['ema_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']

    # Remove wrapper prefixes if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix from DDP
        if key.startswith('module.'):
            key = key[7:]
        # Remove '_orig_mod.' prefix from torch.compile
        if key.startswith('_orig_mod.'):
            key = key[10:]
        cleaned_state_dict[key] = value

    base_model.load_state_dict(cleaned_state_dict, strict=False)
    base_model.eval()
    base_model.to(device)

    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'vocab_size': config.get('vocab_size', 512),
        'max_seq_len': config.get('max_seq_len', 2048),
        'H_cycles': config['arch'].get('H_cycles', 3),
        'L_cycles': config['arch'].get('L_cycles', 4),
    }

    return base_model, config, metadata


def export_to_onnx(
    checkpoint_path,
    output_path,
    H_cycles=None,
    L_cycles=None,
    batch_size=1,
    seq_len=2048,
    dynamic_axes=True,
    opset_version=17,
):
    """
    Export model to ONNX format.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_path: Where to save ONNX model
        H_cycles: Override H_cycles from checkpoint
        L_cycles: Override L_cycles from checkpoint
        batch_size: Batch size for dummy input (use 1 for dynamic)
        seq_len: Sequence length for dummy input
        dynamic_axes: Enable dynamic batch/sequence dimensions
        opset_version: ONNX opset version
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    base_model, config, metadata = load_checkpoint_for_export(checkpoint_path)

    # Use override cycles if provided
    H_cycles = H_cycles or metadata['H_cycles']
    L_cycles = L_cycles or metadata['L_cycles']
    vocab_size = metadata['vocab_size']

    print(f"Wrapping model for ONNX export (H={H_cycles}, L={L_cycles})")
    onnx_model = ONNXInferenceWrapper(
        base_model=base_model,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        max_seq_len=seq_len,
    )
    onnx_model.eval()

    # Create dummy inputs
    dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_puzzle_ids = torch.randint(0, 100, (batch_size,), dtype=torch.long)

    # Define dynamic axes for variable batch/sequence
    if dynamic_axes:
        dynamic_axes_dict = {
            'inputs': {0: 'batch_size', 1: 'seq_len'},
            'labels': {0: 'batch_size', 1: 'seq_len'},
            'puzzle_identifiers': {0: 'batch_size'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
        }
    else:
        dynamic_axes_dict = {}

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        onnx_model,
        (dummy_inputs, dummy_labels, dummy_puzzle_ids),
        output_path,
        input_names=['inputs', 'labels', 'puzzle_identifiers'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    # Save metadata alongside ONNX model
    metadata_path = Path(output_path).with_suffix('.json')
    export_metadata = {
        'checkpoint_source': str(checkpoint_path),
        'H_cycles': H_cycles,
        'L_cycles': L_cycles,
        'vocab_size': vocab_size,
        'max_seq_len': seq_len,
        'opset_version': opset_version,
        'config': config,
    }

    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)

    print(f"✓ ONNX export complete: {output_path}")
    print(f"✓ Metadata saved: {metadata_path}")

    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnx
    onnx_model_check = onnx.load(output_path)
    onnx.checker.check_model(onnx_model_check)
    print("✓ ONNX model is valid")

    return output_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description='Export TRM model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for ONNX model')
    parser.add_argument('--H_cycles', type=int, default=None,
                        help='Override H_cycles')
    parser.add_argument('--L_cycles', type=int, default=None,
                        help='Override L_cycles')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Sequence length for export')
    parser.add_argument('--static_shapes', action='store_true',
                        help='Disable dynamic axes (use static shapes)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dynamic_axes=not args.static_shapes,
        opset_version=args.opset,
    )


if __name__ == '__main__':
    main()
```

---

## Stage 2: ONNX to Qualcomm DLC Conversion

### Script: `convert_to_dlc.sh`

```bash
#!/bin/bash
#
# Convert ONNX model to Qualcomm DLC format with quantization.
#
# Prerequisites:
#   - Qualcomm SNPE SDK installed and sourced
#   - SNPE Python environment activated
#
# Usage:
#   ./convert_to_dlc.sh models_onnx/trm_inference.onnx models_dlc/trm_int8.dlc

set -e

ONNX_MODEL=$1
OUTPUT_DLC=$2
QUANTIZATION=${3:-"int8"}  # Options: fp32, fp16, int8

if [ -z "$ONNX_MODEL" ] || [ -z "$OUTPUT_DLC" ]; then
    echo "Usage: $0 <onnx_model> <output_dlc> [quantization=int8]"
    exit 1
fi

# Check SNPE is available
if ! command -v snpe-onnx-to-dlc &> /dev/null; then
    echo "ERROR: SNPE tools not found. Please source SNPE SDK environment:"
    echo "  source /path/to/snpe-x.y.z/bin/envsetup.sh"
    exit 1
fi

echo "Converting ONNX to DLC..."
echo "  Input:  $ONNX_MODEL"
echo "  Output: $OUTPUT_DLC"
echo "  Quantization: $QUANTIZATION"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_DLC")"

# Step 1: Convert ONNX to DLC (floating point)
echo ""
echo "Step 1: ONNX → DLC (FP32)..."
snpe-onnx-to-dlc \
    --input_network "$ONNX_MODEL" \
    --output_path "${OUTPUT_DLC%.dlc}_fp32.dlc" \
    --input_dim inputs "1,2048" \
    --input_dim labels "1,2048" \
    --input_dim puzzle_identifiers "1"

echo "✓ FP32 DLC created"

# Step 2: Generate quantization calibration data if doing INT8
if [ "$QUANTIZATION" == "int8" ]; then
    echo ""
    echo "Step 2: Generating quantization calibration data..."

    # Create calibration script
    cat > /tmp/generate_calibration_data.py << 'EOF'
import numpy as np
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else 'calibration_data'
num_samples = 100
vocab_size = 512
seq_len = 2048

Path(output_dir).mkdir(exist_ok=True)

print(f"Generating {num_samples} calibration samples...")

for i in range(num_samples):
    # Generate random inputs (in practice, use real validation data)
    inputs = np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int64)
    labels = np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int64)
    puzzle_ids = np.random.randint(0, 100, (1,), dtype=np.int64)

    # Save as raw binary files
    inputs.tofile(f"{output_dir}/inputs_{i}.raw")
    labels.tofile(f"{output_dir}/labels_{i}.raw")
    puzzle_ids.tofile(f"{output_dir}/puzzle_identifiers_{i}.raw")

print(f"✓ Calibration data saved to {output_dir}/")

# Create input list file for SNPE
with open(f"{output_dir}/input_list.txt", 'w') as f:
    for i in range(num_samples):
        f.write(f"inputs_{i}.raw\n")
EOF

    python3 /tmp/generate_calibration_data.py calibration_data

    echo ""
    echo "Step 3: Quantizing to INT8..."
    snpe-dlc-quantize \
        --input_dlc "${OUTPUT_DLC%.dlc}_fp32.dlc" \
        --output_dlc "$OUTPUT_DLC" \
        --input_list calibration_data/input_list.txt \
        --use_enhanced_quantizer \
        --optimizations cle \
        --axis_quant

    echo "✓ INT8 quantized DLC created"

elif [ "$QUANTIZATION" == "fp16" ]; then
    echo ""
    echo "Step 2: Quantizing to FP16..."
    snpe-dlc-quantize \
        --input_dlc "${OUTPUT_DLC%.dlc}_fp32.dlc" \
        --output_dlc "$OUTPUT_DLC" \
        --float_fallback \
        --use_enhanced_quantizer

    echo "✓ FP16 DLC created"

else
    # Just copy FP32 as final output
    cp "${OUTPUT_DLC%.dlc}_fp32.dlc" "$OUTPUT_DLC"
    echo "✓ FP32 DLC ready"
fi

# Step 3: Verify DLC
echo ""
echo "Step 4: Verifying DLC..."
snpe-dlc-info --input_dlc "$OUTPUT_DLC"

echo ""
echo "════════════════════════════════════════"
echo "✓ Conversion complete!"
echo "  DLC model: $OUTPUT_DLC"
echo "  Quantization: $QUANTIZATION"
echo "════════════════════════════════════════"
```

### Alternative: Python Conversion Script

```python
#!/usr/bin/env python3
"""
convert_to_dlc.py - Python wrapper for ONNX→DLC conversion

Usage:
    python convert_to_dlc.py \
        --onnx models_onnx/trm_inference.onnx \
        --output models_dlc/trm_int8.dlc \
        --quantization int8
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def check_snpe_available():
    """Check if SNPE tools are in PATH"""
    try:
        subprocess.run(['snpe-onnx-to-dlc', '--help'],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_calibration_data(output_dir, num_samples=100, vocab_size=512, seq_len=2048):
    """Generate dummy calibration data for quantization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating {num_samples} calibration samples...")

    input_list = []
    for i in range(num_samples):
        # In production, use real validation data
        inputs = np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int64)
        labels = np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int64)
        puzzle_ids = np.random.randint(0, 100, (1,), dtype=np.int64)

        inputs.tofile(output_dir / f"inputs_{i}.raw")
        labels.tofile(output_dir / f"labels_{i}.raw")
        puzzle_ids.tofile(output_dir / f"puzzle_identifiers_{i}.raw")

        input_list.append(f"inputs_{i}.raw")

    # Create input list for SNPE
    with open(output_dir / "input_list.txt", 'w') as f:
        f.write('\n'.join(input_list))

    print(f"✓ Calibration data: {output_dir}/")
    return output_dir / "input_list.txt"


def convert_onnx_to_dlc(onnx_path, dlc_output, quantization='int8'):
    """
    Convert ONNX model to Qualcomm DLC format

    Args:
        onnx_path: Path to ONNX model
        dlc_output: Path for output DLC
        quantization: 'fp32', 'fp16', or 'int8'
    """
    if not check_snpe_available():
        print("ERROR: SNPE tools not found. Please source SNPE environment:")
        print("  source /path/to/snpe-x.y.z/bin/envsetup.sh")
        sys.exit(1)

    dlc_output = Path(dlc_output)
    dlc_output.parent.mkdir(exist_ok=True, parents=True)

    # Step 1: ONNX → DLC (FP32)
    print("\nStep 1: Converting ONNX to FP32 DLC...")
    fp32_dlc = dlc_output.with_name(dlc_output.stem + '_fp32.dlc')

    cmd = [
        'snpe-onnx-to-dlc',
        '--input_network', str(onnx_path),
        '--output_path', str(fp32_dlc),
        '--input_dim', 'inputs', '1,2048',
        '--input_dim', 'labels', '1,2048',
        '--input_dim', 'puzzle_identifiers', '1',
    ]

    subprocess.run(cmd, check=True)
    print(f"✓ FP32 DLC created: {fp32_dlc}")

    # Step 2: Quantization
    if quantization == 'int8':
        print("\nStep 2: Generating calibration data...")
        calib_dir = Path('calibration_data')
        input_list = generate_calibration_data(calib_dir)

        print("\nStep 3: Quantizing to INT8...")
        cmd = [
            'snpe-dlc-quantize',
            '--input_dlc', str(fp32_dlc),
            '--output_dlc', str(dlc_output),
            '--input_list', str(input_list),
            '--use_enhanced_quantizer',
            '--optimizations', 'cle',
            '--axis_quant',
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ INT8 DLC created: {dlc_output}")

    elif quantization == 'fp16':
        print("\nStep 2: Quantizing to FP16...")
        cmd = [
            'snpe-dlc-quantize',
            '--input_dlc', str(fp32_dlc),
            '--output_dlc', str(dlc_output),
            '--float_fallback',
            '--use_enhanced_quantizer',
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ FP16 DLC created: {dlc_output}")

    else:  # fp32
        fp32_dlc.rename(dlc_output)
        print(f"✓ FP32 DLC ready: {dlc_output}")

    # Verify
    print("\nVerifying DLC...")
    subprocess.run(['snpe-dlc-info', '--input_dlc', str(dlc_output)])

    print("\n" + "="*50)
    print(f"✓ Conversion complete: {dlc_output}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to Qualcomm DLC')
    parser.add_argument('--onnx', required=True, help='Input ONNX model')
    parser.add_argument('--output', required=True, help='Output DLC path')
    parser.add_argument('--quantization', choices=['fp32', 'fp16', 'int8'],
                        default='int8', help='Quantization mode')

    args = parser.parse_args()

    convert_onnx_to_dlc(args.onnx, args.output, args.quantization)


if __name__ == '__main__':
    main()
```

---

## Stage 3: Snapdragon Inference Runtime

### Script: `snpe_inference.py`

```python
#!/usr/bin/env python3
"""
Snapdragon NPU inference runtime for TinyRecursiveInference.

Runs DLC models on Qualcomm devices using SNPE runtime.

Usage:
    # On Snapdragon device or emulator:
    python snpe_inference.py \
        --dlc models_dlc/trm_int8.dlc \
        --input puzzle_input.npy \
        --output predictions.npy \
        --runtime dsp  # Options: cpu, gpu, dsp (NPU)
"""

import argparse
import numpy as np
import json
from pathlib import Path
import time

# SNPE Python API
try:
    import snpe
    from snpe import snpe_utils
    SNPE_AVAILABLE = True
except ImportError:
    SNPE_AVAILABLE = False
    print("WARNING: SNPE not available. Install SNPE Python package.")


class SNPEInferenceEngine:
    """
    Inference engine for SNPE models on Snapdragon NPU.
    """

    def __init__(self, dlc_path, runtime='dsp', performance_profile='high_performance'):
        """
        Initialize SNPE inference engine.

        Args:
            dlc_path: Path to DLC model file
            runtime: 'cpu', 'gpu', 'dsp' (NPU), or 'aip' (AI accelerator)
            performance_profile: 'default', 'high_performance', 'power_saver',
                                'balanced', 'sustained_high_performance'
        """
        if not SNPE_AVAILABLE:
            raise RuntimeError("SNPE not available. Cannot initialize engine.")

        self.dlc_path = Path(dlc_path)
        self.runtime = runtime
        self.performance_profile = performance_profile
        self.model = None

        # Load metadata if available
        metadata_path = self.dlc_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self._load_model()

    def _load_model(self):
        """Load DLC model into SNPE runtime"""
        print(f"Loading DLC model: {self.dlc_path}")
        print(f"  Runtime: {self.runtime}")
        print(f"  Performance: {self.performance_profile}")

        # Map runtime strings to SNPE enums
        runtime_map = {
            'cpu': snpe.SNPE_RUNTIME_CPU,
            'gpu': snpe.SNPE_RUNTIME_GPU,
            'dsp': snpe.SNPE_RUNTIME_DSP,
            'aip': snpe.SNPE_RUNTIME_AIP_FIXED_TF,
        }

        # Configure SNPE
        runtime_config = snpe.SNPEConfig()
        runtime_config.runtime = runtime_map.get(self.runtime, snpe.SNPE_RUNTIME_DSP)
        runtime_config.performance_profile = self.performance_profile
        runtime_config.enable_init_caching = True

        # Load model
        self.model = snpe.SNPE(str(self.dlc_path), runtime_config)

        # Get input/output tensor names
        self.input_names = self.model.get_input_names()
        self.output_names = self.model.get_output_names()

        print(f"✓ Model loaded")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")

    def preprocess_puzzle(self, puzzle_data):
        """
        Preprocess puzzle data for model input.

        Args:
            puzzle_data: Dict with 'inputs', 'labels', 'puzzle_id'

        Returns:
            input_tensors: Dict of numpy arrays ready for SNPE
        """
        inputs = puzzle_data['inputs']
        labels = puzzle_data.get('labels', np.zeros_like(inputs))
        puzzle_id = puzzle_data.get('puzzle_id', 0)

        # Ensure correct shapes
        if inputs.ndim == 1:
            inputs = inputs[np.newaxis, :]  # Add batch dim
        if labels.ndim == 1:
            labels = labels[np.newaxis, :]
        if isinstance(puzzle_id, int):
            puzzle_id = np.array([puzzle_id], dtype=np.int64)

        return {
            'inputs': inputs.astype(np.int64),
            'labels': labels.astype(np.int64),
            'puzzle_identifiers': puzzle_id.astype(np.int64),
        }

    def run_inference(self, input_tensors, measure_time=False):
        """
        Run inference on Snapdragon NPU.

        Args:
            input_tensors: Dict of input arrays
            measure_time: Whether to measure inference time

        Returns:
            output_tensors: Dict of output arrays
            timing_info: Optional timing dict if measure_time=True
        """
        if measure_time:
            start_time = time.perf_counter()

        # Run inference
        output_tensors = self.model.execute(input_tensors)

        if measure_time:
            inference_time = time.perf_counter() - start_time
            timing_info = {
                'inference_time_ms': inference_time * 1000,
                'throughput_samples_per_sec': 1.0 / inference_time,
            }
        else:
            timing_info = None

        return output_tensors, timing_info

    def postprocess_outputs(self, output_tensors):
        """
        Convert model outputs to predictions.

        Args:
            output_tensors: Dict of output arrays from SNPE

        Returns:
            predictions: [batch_size, seq_len] predicted token IDs
            logits: [batch_size, seq_len, vocab_size] raw logits
        """
        logits = output_tensors['logits']  # [batch, seq_len, vocab]
        predictions = np.argmax(logits, axis=-1)  # [batch, seq_len]

        return predictions, logits

    def predict(self, puzzle_data, measure_time=False):
        """
        End-to-end prediction on puzzle.

        Args:
            puzzle_data: Dict with puzzle inputs
            measure_time: Whether to measure timing

        Returns:
            predictions: Predicted token IDs
            timing_info: Optional timing dict
        """
        # Preprocess
        input_tensors = self.preprocess_puzzle(puzzle_data)

        # Inference
        output_tensors, timing_info = self.run_inference(
            input_tensors, measure_time=measure_time
        )

        # Postprocess
        predictions, logits = self.postprocess_outputs(output_tensors)

        if measure_time:
            return predictions, timing_info
        else:
            return predictions

    def benchmark(self, puzzle_data, num_iterations=100):
        """
        Benchmark inference performance.

        Args:
            puzzle_data: Sample puzzle for benchmarking
            num_iterations: Number of inference runs

        Returns:
            stats: Dict with performance statistics
        """
        print(f"Benchmarking {num_iterations} iterations...")

        input_tensors = self.preprocess_puzzle(puzzle_data)
        times = []

        # Warmup
        for _ in range(10):
            self.model.execute(input_tensors)

        # Benchmark
        for i in range(num_iterations):
            start = time.perf_counter()
            self.model.execute(input_tensors)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        times = np.array(times)
        stats = {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_samples_per_sec': 1000.0 / np.mean(times),
        }

        return stats


def load_puzzle_from_file(input_path):
    """Load puzzle data from numpy file or JSON"""
    input_path = Path(input_path)

    if input_path.suffix == '.npy':
        # Raw numpy array
        inputs = np.load(input_path)
        return {
            'inputs': inputs,
            'labels': np.zeros_like(inputs),
            'puzzle_id': 0,
        }

    elif input_path.suffix == '.json':
        # JSON format with metadata
        with open(input_path) as f:
            data = json.load(f)
        return {
            'inputs': np.array(data['inputs'], dtype=np.int64),
            'labels': np.array(data.get('labels', []), dtype=np.int64),
            'puzzle_id': data.get('puzzle_id', 0),
        }

    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")


def save_predictions(predictions, output_path, metadata=None):
    """Save predictions to file"""
    output_path = Path(output_path)

    if output_path.suffix == '.npy':
        np.save(output_path, predictions)

    elif output_path.suffix == '.json':
        output_data = {
            'predictions': predictions.tolist(),
        }
        if metadata:
            output_data['metadata'] = metadata

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    print(f"✓ Predictions saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SNPE inference for TinyRecursiveInference')
    parser.add_argument('--dlc', required=True, help='Path to DLC model')
    parser.add_argument('--input', required=True, help='Input puzzle (.npy or .json)')
    parser.add_argument('--output', required=True, help='Output predictions path')
    parser.add_argument('--runtime', choices=['cpu', 'gpu', 'dsp', 'aip'],
                        default='dsp', help='SNPE runtime backend')
    parser.add_argument('--performance', default='high_performance',
                        help='Performance profile')
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Run benchmark with N iterations')

    args = parser.parse_args()

    # Initialize engine
    print("Initializing SNPE inference engine...")
    engine = SNPEInferenceEngine(
        dlc_path=args.dlc,
        runtime=args.runtime,
        performance_profile=args.performance,
    )

    # Load input
    print(f"\nLoading input: {args.input}")
    puzzle_data = load_puzzle_from_file(args.input)
    print(f"  Input shape: {puzzle_data['inputs'].shape}")

    # Run inference
    if args.benchmark > 0:
        print(f"\nRunning benchmark...")
        stats = engine.benchmark(puzzle_data, num_iterations=args.benchmark)
        print("\nBenchmark Results:")
        print(f"  Mean:   {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  Std:    {stats['std_ms']:.2f} ms")
        print(f"  P95:    {stats['p95_ms']:.2f} ms")
        print(f"  P99:    {stats['p99_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")

    print(f"\nRunning inference...")
    predictions, timing = engine.predict(puzzle_data, measure_time=True)

    print(f"✓ Inference complete")
    print(f"  Time: {timing['inference_time_ms']:.2f} ms")
    print(f"  Output shape: {predictions.shape}")

    # Save outputs
    save_predictions(
        predictions,
        args.output,
        metadata={
            'runtime': args.runtime,
            'inference_time_ms': timing['inference_time_ms'],
        }
    )


if __name__ == '__main__':
    main()
```

---

## Stage 4: Android Integration

### Script: `android_app/SnapdragonInference.java`

```java
/**
 * Android wrapper for SNPE inference.
 *
 * Build requirements:
 *   - SNPE Android SDK
 *   - NDK r21+
 *   - Gradle 7.0+
 */

package com.example.tinyrecursive;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class SnapdragonInference {

    private NeuralNetwork network;
    private Set<String> inputNames;
    private Set<String> outputNames;
    private String runtime;

    /**
     * Initialize SNPE model.
     *
     * @param dlcPath Path to DLC model file
     * @param runtime Runtime: "DSP", "GPU", "CPU"
     */
    public SnapdragonInference(String dlcPath, String runtime) throws IOException {
        this.runtime = runtime;

        // Build SNPE config
        SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(
            new File(dlcPath)
        );

        // Set runtime
        NeuralNetwork.Runtime runtimeEnum;
        switch (runtime.toUpperCase()) {
            case "DSP":
                runtimeEnum = NeuralNetwork.Runtime.DSP;
                break;
            case "GPU":
                runtimeEnum = NeuralNetwork.Runtime.GPU;
                break;
            default:
                runtimeEnum = NeuralNetwork.Runtime.CPU;
        }
        builder.setRuntimeOrder(runtimeEnum);

        // Performance settings
        builder.setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE);
        builder.setInitCacheMode(true);

        // Build network
        network = builder.build();

        // Get tensor names
        inputNames = network.getInputTensorsNames();
        outputNames = network.getOutputTensorsNames();

        System.out.println("✓ SNPE model loaded");
        System.out.println("  Inputs: " + inputNames);
        System.out.println("  Outputs: " + outputNames);
    }

    /**
     * Run inference on puzzle.
     *
     * @param inputs Input token IDs [seq_len]
     * @param labels Label token IDs [seq_len]
     * @param puzzleId Puzzle identifier
     * @return predictions Predicted token IDs [seq_len]
     */
    public int[] predict(int[] inputs, int[] labels, int puzzleId) {
        // Create input tensors
        Map<String, FloatTensor> inputTensors = new HashMap<>();

        // Convert int arrays to float tensors (SNPE limitation)
        float[] inputsFloat = new float[inputs.length];
        float[] labelsFloat = new float[labels.length];
        for (int i = 0; i < inputs.length; i++) {
            inputsFloat[i] = (float) inputs[i];
            labelsFloat[i] = (float) labels[i];
        }

        // Create tensors
        long[] shape1D = new long[]{1, inputs.length};
        FloatTensor inputTensor = network.createFloatTensor(shape1D);
        FloatTensor labelTensor = network.createFloatTensor(shape1D);
        FloatTensor puzzleTensor = network.createFloatTensor(new long[]{1, 1});

        inputTensor.write(inputsFloat, 0, inputs.length);
        labelTensor.write(labelsFloat, 0, labels.length);
        puzzleTensor.write(new float[]{(float) puzzleId}, 0, 1);

        inputTensors.put("inputs", inputTensor);
        inputTensors.put("labels", labelTensor);
        inputTensors.put("puzzle_identifiers", puzzleTensor);

        // Execute
        Map<String, FloatTensor> outputTensors = network.execute(inputTensors);

        // Extract logits
        FloatTensor logitsTensor = outputTensors.get("logits");
        float[] logitsFlat = new float[logitsTensor.getSize()];
        logitsTensor.read(logitsFlat, 0, logitsFlat.length);

        // Get predictions (argmax over vocab dimension)
        int seqLen = inputs.length;
        int vocabSize = logitsFlat.length / seqLen;
        int[] predictions = new int[seqLen];

        for (int i = 0; i < seqLen; i++) {
            int maxIdx = 0;
            float maxVal = logitsFlat[i * vocabSize];
            for (int j = 1; j < vocabSize; j++) {
                float val = logitsFlat[i * vocabSize + j];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            predictions[i] = maxIdx;
        }

        // Cleanup
        inputTensor.release();
        labelTensor.release();
        puzzleTensor.release();
        logitsTensor.release();

        return predictions;
    }

    /**
     * Benchmark inference performance.
     *
     * @param inputs Sample input for benchmarking
     * @param labels Sample labels
     * @param puzzleId Sample puzzle ID
     * @param iterations Number of benchmark iterations
     * @return Average inference time in milliseconds
     */
    public double benchmark(int[] inputs, int[] labels, int puzzleId, int iterations) {
        // Warmup
        for (int i = 0; i < 10; i++) {
            predict(inputs, labels, puzzleId);
        }

        // Benchmark
        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            predict(inputs, labels, puzzleId);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / iterations;
        return avgTimeMs;
    }

    /**
     * Release SNPE resources.
     */
    public void close() {
        if (network != null) {
            network.release();
            network = null;
        }
    }
}
```

---

## Deployment Checklist

### Development Phase
- [ ] Train TRM model on GPU cluster
- [ ] Export checkpoint to ONNX format
- [ ] Verify ONNX model correctness
- [ ] Test ONNX inference on CPU

### Conversion Phase
- [ ] Install Qualcomm SNPE SDK
- [ ] Generate calibration dataset (use real validation data)
- [ ] Convert ONNX → DLC (FP32)
- [ ] Quantize to INT8 with calibration
- [ ] Verify DLC model integrity

### Testing Phase
- [ ] Set up Snapdragon development device
- [ ] Install SNPE runtime on device
- [ ] Run inference on DLC model
- [ ] Benchmark performance (latency, throughput)
- [ ] Compare accuracy vs. PyTorch baseline

### Optimization Phase
- [ ] Profile model on NPU (identify bottlenecks)
- [ ] Optimize recursive loops (consider unrolling further)
- [ ] Tune quantization (try mixed precision)
- [ ] Cache embeddings for repeated puzzles
- [ ] Batch inference if possible

### Production Phase
- [ ] Package DLC model in Android APK
- [ ] Implement Java/Kotlin inference wrapper
- [ ] Add error handling and fallbacks
- [ ] Test on multiple Snapdragon SoCs
- [ ] Monitor real-world performance

---

## Performance Expectations

### Latency Targets (per inference)
- **FP32 on CPU**: ~500-1000ms
- **INT8 on NPU**: ~50-150ms (10x faster)
- **Goal**: <100ms for real-time applications

### Model Size
- **PyTorch checkpoint**: ~30MB
- **ONNX FP32**: ~25MB
- **DLC INT8**: ~7-10MB (optimized for mobile)

### Accuracy Impact
- **FP32**: 100% baseline
- **FP16**: ~99.5% of baseline
- **INT8**: ~97-99% of baseline (depends on calibration quality)

---

## Troubleshomarks

### Issue: ONNX export fails with dynamic control flow
**Solution**: Use `--unroll` flag to explicitly unroll H and L cycles

### Issue: DLC conversion fails with unsupported ops
**Solution**: Check SNPE version supports ONNX opset. Try older opset (15, 16)

### Issue: INT8 quantization degrades accuracy significantly
**Solution**:
- Use real validation data for calibration (not random)
- Try per-channel quantization
- Use mixed precision (keep critical layers in FP16)

### Issue: Inference slower on NPU than expected
**Solution**:
- Check runtime selection (DSP vs AIP)
- Profile with `snpe-diagview`
- Reduce H_cycles/L_cycles for mobile deployment
- Use batch inference where possible

### Issue: Large memory usage on device
**Solution**:
- Enable init caching
- Reuse tensors instead of reallocating
- Clear intermediate activations

---

## Next Steps

1. **Export first model**: Run `export_to_onnx.py` on best checkpoint
2. **Test locally**: Verify ONNX inference matches PyTorch
3. **Set up SNPE**: Install SDK and convert to DLC
4. **Benchmark**: Measure performance on target device
5. **Iterate**: Optimize based on profiling results

## References

- Qualcomm SNPE Documentation: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
- ONNX Export Guide: https://pytorch.org/docs/stable/onnx.html
- INT8 Quantization Best Practices: https://arxiv.org/abs/2004.09602
