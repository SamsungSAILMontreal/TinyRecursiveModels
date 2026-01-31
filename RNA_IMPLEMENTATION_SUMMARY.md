# RNA 3D Structure Prediction - Implementation Summary

## Overview

This implementation adds comprehensive RNA 3D structure prediction capabilities to TinyRecursiveModels, with full support for the **Stanford RNA 3D Folding Part 2** Kaggle competition.

## Competition Details

- **Competition**: [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- **Task**: Predict 3D structures of RNA molecules
- **Output**: 5 different conformations per sequence
- **Metric**: TM-score (Template Modeling score)
- **Deadline**: March 25, 2026

## Implementation Status

### ✅ Core Features Complete

1. **Dataset Processing**
   - CSV input parser for competition format
   - Nucleotide encoding (A, C, G, U → 0, 1, 2, 3)
   - Temporal data splitting
   - Multi-structure label handling

2. **Model Architecture**
   - TRM-based recursive reasoning
   - 5 independent prediction heads
   - Transformer encoder with position embeddings
   - Configurable recursive cycles (H_cycles, L_cycles)

3. **Training Pipeline**
   - Multi-GPU support
   - MSE loss on 3D coordinates
   - Proper padding handling (index 4)
   - Checkpoint management

4. **Prediction Pipeline**
   - Batch inference
   - Coordinate clipping to PDB format limits
   - Competition CSV output format

5. **Kaggle Competition Support** ⭐
   - Standalone submission script
   - Jupyter notebook for Kaggle platform
   - Reads `test_sequences.csv`
   - Outputs `submission.csv`
   - Runtime optimized for <8 hours

## File Structure

```
TinyRecursiveModels/
├── kaggle_submission.py              # Kaggle submission script
├── kaggle_submission_notebook.ipynb  # Kaggle notebook
├── dataset/
│   └── build_rna_dataset.py         # Dataset builder
├── pretrain_rna.py                   # Training script
├── predict_rna.py                    # Prediction script
├── utils/
│   └── rna_utils.py                 # RNA utilities
├── workflows/
│   └── rna_structure_prediction.sh  # Automation script
├── tests/
│   ├── test_rna_integration.py      # Integration tests
│   └── test_kaggle_submission.py    # Kaggle tests
└── docs/
    ├── KAGGLE_SUBMISSION.md         # Competition guide
    ├── RNA_INTEGRATION.md           # Full docs
    ├── RNA_EXAMPLE.md               # Tutorial
    └── RNA_README.md                # Quick ref
```

## Testing Status

**All 15 tests passing:**

### RNA Integration Tests (6/6) ✓
- ✓ FASTA parsing
- ✓ RNA sequence validation
- ✓ RMSD calculation
- ✓ Coordinate clipping
- ✓ Stoichiometry merging
- ✓ Dataset building workflow

### Kaggle Submission Tests (9/9) ✓
- ✓ Sequence encoding
- ✓ Model creation
- ✓ Forward pass
- ✓ Prediction generation
- ✓ Submission file format
- ✓ Column validation
- ✓ Coordinate clipping
- ✓ Row count verification
- ✓ Sample data validation

## Usage

### For Kaggle Competition

**Option 1: Jupyter Notebook (Recommended)**
1. Upload `kaggle_submission_notebook.ipynb` to Kaggle
2. Add pretrained model as dataset
3. Run notebook
4. Submit `submission.csv`

**Option 2: Python Script**
```bash
python kaggle_submission.py
```

### For Local Development

**Train a model:**
```bash
python pretrain_rna.py \
  --data-dir data/rna-structure \
  --epochs 100 \
  --num-structures 5
```

**Generate predictions:**
```bash
python predict_rna.py \
  --model-path checkpoints/rna/best_model.pth \
  --sequences test_sequences.json \
  --output submission.csv
```

**Automated workflow:**
```bash
bash workflows/rna_structure_prediction.sh \
  --sequences train_sequences.csv \
  --labels train_labels.csv \
  --test-sequences test_sequences.csv
```

## Technical Details

### Model Architecture

```python
RNAStructureModel(
    vocab_size=4,        # A, C, G, U
    embed_dim=256,       # Embedding dimension
    hidden_dim=512,      # Hidden layer size
    num_structures=5,    # Number of conformations
    max_length=500,      # Max sequence length
    H_cycles=3,          # High-level reasoning
    L_cycles=6,          # Low-level reasoning
    L_layers=2,          # Transformer layers
)
```

**Parameters**: ~7M (configurable)

### Submission Format

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,-7.301,9.023,8.932,...
```

- **Columns**: 18 (ID, resname, resid + 15 coordinates)
- **Rows**: One per residue in test sequences
- **Coordinates**: Clipped to [-999.999, 9999.999]

### Evaluation Metric

**TM-score** (Template Modeling score):
- Range: 0.0 to 1.0 (higher is better)
- Algorithm: US-align for structure alignment
- Scoring: Best of 5 predictions per sequence
- Final: Average TM-score across all sequences

## Performance Optimization

### For Faster Inference

```python
# Reduce model complexity
model = RNAStructureModel(
    embed_dim=128,
    hidden_dim=256,
    H_cycles=2,
    L_cycles=4,
    L_layers=1,
)

# Increase batch size
batch_size = 64
```

### For Better Accuracy

```python
# Increase model capacity
model = RNAStructureModel(
    embed_dim=512,
    hidden_dim=1024,
    H_cycles=5,
    L_cycles=8,
    L_layers=4,
)

# Train longer
epochs = 200
```

## Documentation

1. **Kaggle Submission**: `docs/KAGGLE_SUBMISSION.md`
   - Competition requirements
   - Submission workflow
   - Troubleshooting guide

2. **Technical Integration**: `docs/RNA_INTEGRATION.md`
   - Architecture details
   - API reference
   - Advanced usage

3. **Tutorial**: `docs/RNA_EXAMPLE.md`
   - Step-by-step example
   - Sample data
   - Expected outputs

4. **Quick Reference**: `docs/RNA_README.md`
   - Common commands
   - File descriptions
   - Quick tips

## Next Steps

### For Competition Participants

1. **Train a model**:
   ```bash
   python pretrain_rna.py --data-dir data/rna-competition --epochs 100
   ```

2. **Upload to Kaggle**:
   - Model weights as dataset
   - Submission notebook

3. **Submit**:
   - Run notebook on Kaggle
   - Submit `submission.csv`

### For Improvement

- Add MSA (Multiple Sequence Alignment) features
- Implement ensemble predictions
- Add physics-based constraints
- Use template-based refinement
- Hyperparameter tuning

## Requirements

**Python Packages:**
- torch >= 2.0
- pandas >= 1.5
- numpy >= 1.24
- tqdm >= 4.65

**Optional:**
- CUDA for GPU acceleration
- Bioinformatics tools from jrc-rna-structure-pipeline

## Citation

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

- **GitHub Issues**: [drqsatoshi/TinyRecursiveModels](https://github.com/drqsatoshi/TinyRecursiveModels/issues)
- **Kaggle Discussion**: [Competition Forum](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/discussion)

---

**Status**: ✅ Ready for Competition
**Last Updated**: January 31, 2026
**Version**: 1.0.0
