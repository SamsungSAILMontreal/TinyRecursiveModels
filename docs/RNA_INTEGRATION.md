# RNA 3D Structure Prediction with TinyRecursiveModels

This integration enables the TinyRecursiveModels (TRM) framework to predict 3D RNA structures using the [jrc-rna-structure-pipeline](https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline).

## Overview

The RNA structure prediction task involves:
- **Input**: RNA sequences (composed of A, C, G, U nucleotides)
- **Output**: 5 different 3D conformations for each sequence
- **Target**: C1' atom coordinates (x, y, z) for each nucleotide

This follows the Stanford RNA 3D Folding competition format.

## Installation

### Prerequisites

1. Install the base TinyRecursiveModels requirements:
```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
```

2. Install RNA-specific dependencies:
```bash
pip install pandas biopython requests
```

3. Clone and set up the RNA structure pipeline:
```bash
git clone https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline.git
cd jrc-rna-structure-pipeline
curl -fsSL https://pixi.sh/install.sh | bash  # Install pixi
pixi install
cd ..
```

## Quick Start

### 1. Prepare the Dataset

Download RNA structure data from the competition or use the pipeline to generate it:

```bash
# Using pre-downloaded competition data
python dataset/build_rna_dataset.py \
  --sequences path/to/train_sequences.csv \
  --labels path/to/train_labels.csv \
  --output-dir data/rna-structure \
  --temporal-cutoff 2025-05-29
```

This will create:
- `data/rna-structure/train_sequences.json` - Training sequences
- `data/rna-structure/train_labels.json` - Training labels
- `data/rna-structure/val_sequences.json` - Validation sequences
- `data/rna-structure/val_labels.json` - Validation labels

### 2. Train the Model

Train a TRM model for RNA structure prediction:

```bash
python pretrain_rna.py \
  --data-dir data/rna-structure \
  --output-dir checkpoints/rna \
  --batch-size 16 \
  --epochs 100 \
  --lr 1e-4 \
  --num-structures 5
```

**Expected training time**: Varies by dataset size
- Small dataset (100 sequences): ~1-2 hours on a single GPU
- Medium dataset (1000 sequences): ~10-20 hours on a single GPU
- Large dataset (10000+ sequences): ~2-3 days on multiple GPUs

### 3. Generate Predictions

Create predictions for test sequences:

```bash
python predict_rna.py \
  --model-path checkpoints/rna/best_model.pth \
  --sequences data/rna-structure/test_sequences.json \
  --output submission.csv \
  --num-structures 5
```

This generates a `submission.csv` file in Kaggle competition format with 5 predicted structures per sequence.

## Dataset Format

### Input Sequences CSV

```csv
target_id,sequence,temporal_cutoff,description,stoichiometry,all_sequences,ligand_ids,ligand_SMILES
1A1T_A,CGCGAAUUAGCG,2000-01-01,Example RNA structure,A:1,>A\nCGCGAAUUAGCG,,
```

### Labels CSV

```csv
ID,resname,resid,x_1,y_1,z_1,chain,copy
1A1T_A_1,C,1,-10.123,5.456,3.789,A,1
1A1T_A_2,G,2,-9.234,6.123,4.567,A,1
```

### Submission Format

The model generates predictions in the required format:

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
1A1T_A_1,C,1,-10.123,5.456,3.789,-10.125,5.458,3.791,...
```

Each row contains:
- `ID`: target_id + residue number (1-based)
- `resname`: Nucleotide type (A/C/G/U)
- `resid`: Residue number (1-based)
- `x_i, y_i, z_i`: Coordinates for structure i (i=1 to 5)

## Model Architecture

The RNA structure prediction model uses TRM's recursive reasoning approach:

```
Input Sequence (ACGU)
    ↓
Nucleotide Embedding + Position Embedding
    ↓
Recursive Reasoning (H_cycles × L_cycles)
    ↓ (Transformer Encoder Layers)
    ↓
5 Structure Prediction Heads
    ↓
Output: 5 × (x, y, z) coordinates per residue
```

**Key features**:
- **Recursive refinement**: Multiple reasoning cycles improve predictions
- **Multi-conformation**: Generates 5 different valid structures
- **Attention mechanism**: Captures long-range dependencies in RNA structure
- **Coordinate clipping**: Ensures coordinates stay within valid range (-999.999 to 9999.999)

## Advanced Usage

### Using the RNA Structure Pipeline

To generate your own dataset from PDB:

```bash
cd jrc-rna-structure-pipeline
pixi shell

# Run the full pipeline for a date range
cd /path/to/output
bash /path/to/jrc-rna-structure-pipeline/workflows/kaggle2026_1978-01-01_2025-12-17.sh
```

This will:
1. Fetch RNA structures from PDB
2. Extract metadata and filter by quality
3. Cluster sequences to remove redundancy
4. Calculate structural metrics
5. Generate train/test splits
6. Create Kaggle-formatted outputs

### Custom Model Configuration

Modify the model architecture in `pretrain_rna.py`:

```python
model = RNAStructureModel(
    vocab_size=4,          # A, C, G, U
    embed_dim=256,         # Embedding dimension
    hidden_dim=512,        # Hidden layer size
    num_structures=5,      # Number of conformations
    max_length=500,        # Maximum sequence length
    H_cycles=3,           # High-level reasoning cycles
    L_cycles=6,           # Low-level reasoning cycles
    L_layers=2,           # Number of transformer layers
)
```

### Multi-GPU Training

For larger datasets, use distributed training:

```bash
torchrun --nproc-per-node 4 pretrain_rna.py \
  --data-dir data/rna-structure \
  --output-dir checkpoints/rna \
  --batch-size 64 \
  --epochs 200
```

## Evaluation Metrics

The model is evaluated using Mean Squared Error (MSE) between predicted and actual C1' atom coordinates:

```
MSE = Σ(predicted_coords - actual_coords)² / num_residues
```

Lower MSE indicates better prediction accuracy.

## File Structure

```
TinyRecursiveModels/
├── dataset/
│   └── build_rna_dataset.py      # Dataset preprocessing
├── pretrain_rna.py                # Training script
├── predict_rna.py                 # Prediction script
├── docs/
│   └── RNA_INTEGRATION.md         # This file
├── data/
│   └── rna-structure/            # Processed datasets
└── checkpoints/
    └── rna/                      # Model checkpoints
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size or sequence length:
```bash
python pretrain_rna.py --batch-size 8 --max-length 300
```

### Slow Training

Use multiple GPUs or reduce model size:
```bash
# Smaller model
python pretrain_rna.py --embed-dim 128 --hidden-dim 256
```

### Poor Predictions

Try:
1. Increase number of training epochs
2. Use more recursive reasoning cycles (H_cycles, L_cycles)
3. Add more training data
4. Tune learning rate

## References

1. [TinyRecursiveModels Paper](https://arxiv.org/abs/2510.04871)
2. [RNA Structure Pipeline](https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline)
3. [Stanford RNA 3D Folding Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)

## Citation

If you use this integration, please cite:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```
