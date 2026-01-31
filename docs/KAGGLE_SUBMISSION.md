# Kaggle Competition Submission Guide

## Stanford RNA 3D Folding Part 2

This guide explains how to submit predictions to the [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) competition using TinyRecursiveModels.

## Competition Overview

- **Task**: Predict 3D structures of RNA molecules
- **Input**: RNA sequences (ACGU nucleotides)
- **Output**: 5 different 3D conformations per sequence
- **Format**: C1' atom coordinates (x, y, z) for each residue
- **Metric**: TM-score (Template Modeling score)
- **Runtime**: ≤8 hours (CPU or GPU)

## Quick Start

### Option 1: Kaggle Notebook (Recommended)

1. **Upload the notebook to Kaggle**:
   - Go to [Kaggle Notebooks](https://www.kaggle.com/code)
   - Click "New Notebook" → "Import Notebook"
   - Upload `kaggle_submission_notebook.ipynb`

2. **Add model weights as a dataset**:
   - Upload your trained model (`rna_model.pth`) as a Kaggle dataset
   - In the notebook, click "Add data" → Select your model dataset
   - Update the checkpoint path in the notebook

3. **Add competition data**:
   - The competition data is automatically available at:
     `/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv`

4. **Run and submit**:
   - Click "Run All"
   - Once complete, the notebook outputs `submission.csv`
   - Click "Submit to Competition"

### Option 2: Python Script

For local testing or custom workflows:

```bash
# Ensure test_sequences.csv is in the current directory
python kaggle_submission.py
```

This will generate `submission.csv` in the current directory.

## File Structure

```
TinyRecursiveModels/
├── kaggle_submission.py           # Standalone submission script
├── kaggle_submission_notebook.ipynb  # Jupyter notebook for Kaggle
└── docs/
    └── KAGGLE_SUBMISSION.md       # This file
```

## Submission Format

The output `submission.csv` must have this exact format:

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,-7.301,9.023,8.932,...
R1107_2,G,2,-8.02,11.014,14.606,-7.953,10.02,12.127,...
```

**Columns**:
- `ID`: target_id + "_" + residue_number (1-based indexing)
- `resname`: Nucleotide (A, C, G, or U)
- `resid`: Residue number (1-based)
- `x_1, y_1, z_1` through `x_5, y_5, z_5`: Coordinates for 5 structures

**Constraints**:
- Exactly 5 structures per sequence (required)
- Coordinates clipped to [-999.999, 9999.999] (PDB format limitation)
- One row per residue in each sequence

## Model Training

Before submitting, you need a trained model:

### 1. Prepare Training Data

```bash
# Download competition training data
# Available at: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data

# Process the data
python dataset/build_rna_dataset.py \
  --sequences train_sequences.csv \
  --labels train_labels.csv \
  --output-dir data/rna-competition \
  --temporal-cutoff 2025-05-29
```

### 2. Train the Model

```bash
# Train with default settings
python pretrain_rna.py \
  --data-dir data/rna-competition \
  --output-dir checkpoints/rna-competition \
  --epochs 100 \
  --batch-size 16 \
  --num-structures 5

# For faster training with GPU
python pretrain_rna.py \
  --data-dir data/rna-competition \
  --output-dir checkpoints/rna-competition \
  --epochs 100 \
  --batch-size 32 \
  --device cuda
```

### 3. Upload Model to Kaggle

1. Create a new dataset on Kaggle:
   - Go to "Your Work" → "Datasets" → "New Dataset"
   - Upload `checkpoints/rna-competition/best_model.pth`
   - Name it `rna-structure-weights`

2. Make it public or add it to your notebook

## Competition Requirements Checklist

- [x] **Input**: Reads `test_sequences.csv`
- [x] **Output**: Produces `submission.csv`
- [x] **Format**: ID, resname, resid, + 15 coordinates (5 structures × 3 dims)
- [x] **Structures**: Exactly 5 predictions per sequence
- [x] **Coordinates**: C1' atoms only
- [x] **Clipping**: Coordinates in [-999.999, 9999.999]
- [x] **Runtime**: Completes within 8 hours
- [ ] **Validation**: Test on sample submission
- [ ] **Weights**: Upload trained model to Kaggle

## Testing Locally

Before submitting to Kaggle, test locally:

```bash
# Create a sample test_sequences.csv
cat > test_sequences.csv << EOF
target_id,sequence,temporal_cutoff,description
TEST_001,ACGU,2026-01-01,Test sequence
EOF

# Run submission script
python kaggle_submission.py

# Verify output
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'Expected: 18 columns (ID, resname, resid + 15 coords)')
print(df.head())
"
```

## Evaluation Metric

The competition uses **TM-score** (Template Modeling score):

- **Range**: 0.0 to 1.0 (higher is better)
- **Algorithm**: US-align for structure alignment
- **Scoring**: Best of 5 predictions per sequence
- **Final Score**: Average TM-score across all test sequences

### TM-score Formula

```
TM-score = (1/Lref) * Σ[1 / (1 + (di/d0)²)]
```

Where:
- `Lref`: Number of residues in reference structure
- `di`: Distance between aligned residue pairs (Angstroms)
- `d0`: Scaling factor (depends on sequence length)

## Tips for Better Performance

### 1. Model Architecture
- Increase recursive cycles for better reasoning
- Use larger embeddings for complex structures
- Add more transformer layers

```python
model = RNAStructureModel(
    embed_dim=512,      # Larger embeddings
    hidden_dim=1024,    # More capacity
    H_cycles=5,         # More reasoning cycles
    L_cycles=8,
    L_layers=4,         # Deeper network
)
```

### 2. Training Strategy
- Train for more epochs (200-500)
- Use learning rate scheduling
- Implement early stopping based on validation TM-score
- Add data augmentation (rotations, translations)

### 3. Ensemble Methods
- Train multiple models with different seeds
- Average predictions from ensemble
- Use different architectures

### 4. Post-processing
- Apply physical constraints (bond lengths, angles)
- Refine structures with energy minimization
- Use template-based refinement when templates available

## Troubleshooting

### Runtime Exceeds 8 Hours

**Solutions**:
- Reduce batch size
- Decrease max_length
- Use fewer recursive cycles
- Simplify model architecture

```python
# Fast inference configuration
model = RNAStructureModel(
    embed_dim=128,
    hidden_dim=256,
    H_cycles=2,
    L_cycles=4,
    L_layers=1,
)
batch_size = 64  # Larger batches for speed
```

### Out of Memory

**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Process sequences in smaller chunks

```python
batch_size = 8  # Smaller batches
max_length = 300  # Shorter sequences
```

### Poor TM-scores

**Solutions**:
- Train longer
- Use more training data
- Increase model capacity
- Try different hyperparameters
- Use MSA (Multiple Sequence Alignment) data if available

## Competition Rules

- **Internet**: Disabled during submission
- **External Data**: Allowed if freely and publicly available
- **Pretrained Models**: Allowed
- **Team Mergers**: Until March 18, 2026
- **Final Submission**: March 25, 2026

## Resources

- **Competition Page**: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
- **TRM Paper**: https://arxiv.org/abs/2510.04871
- **Documentation**: See `docs/RNA_INTEGRATION.md`
- **Examples**: See `docs/RNA_EXAMPLE.md`

## Citation

If you use this code in your submission, please cite:

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

## Support

- **Issues**: [GitHub Issues](https://github.com/drqsatoshi/TinyRecursiveModels/issues)
- **Discussion**: [Kaggle Discussion](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/discussion)

Good luck with the competition! 🧬🏆
