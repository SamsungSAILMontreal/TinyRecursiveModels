# RNA 3D Structure Prediction - Quick Reference

This is a quick reference guide for using TinyRecursiveModels for RNA 3D structure prediction.

## Installation

```bash
# Install base requirements
pip install -r requirements.txt

# Clone RNA pipeline
git clone https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline.git
```

## Quick Start (3 Steps)

### 1. Prepare Dataset

```bash
python dataset/build_rna_dataset.py \
  --sequences train_sequences.csv \
  --labels train_labels.csv \
  --output-dir data/rna-structure
```

### 2. Train Model

```bash
python pretrain_rna.py \
  --data-dir data/rna-structure \
  --epochs 100
```

### 3. Generate Predictions

```bash
python predict_rna.py \
  --model-path checkpoints/rna/best_model.pth \
  --sequences test_sequences.json \
  --output submission.csv
```

## Automated Workflow

```bash
bash workflows/rna_structure_prediction.sh \
  --sequences train_sequences.csv \
  --labels train_labels.csv \
  --test-sequences test_sequences.csv
```

## Files

| File | Description |
|------|-------------|
| `dataset/build_rna_dataset.py` | Dataset preprocessing |
| `pretrain_rna.py` | Training script |
| `predict_rna.py` | Prediction script |
| `utils/rna_utils.py` | RNA utility functions |
| `workflows/rna_structure_prediction.sh` | Automated workflow |
| `docs/RNA_INTEGRATION.md` | Full documentation |
| `docs/RNA_EXAMPLE.md` | Step-by-step example |
| `tests/test_rna_integration.py` | Integration tests |

## Documentation

- **Full Guide**: [docs/RNA_INTEGRATION.md](docs/RNA_INTEGRATION.md)
- **Example Walkthrough**: [docs/RNA_EXAMPLE.md](docs/RNA_EXAMPLE.md)
- **Main README**: [README.md](../README.md)

## Dataset Format

### Input (CSV)
```csv
target_id,sequence,temporal_cutoff,description,stoichiometry,all_sequences,ligand_ids,ligand_SMILES
1A1T_A,CGCGAAUUAGCG,2000-01-01,Example,A:1,>A\nCGCGAAUUAGCG,,
```

### Output (CSV)
```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
1A1T_A_1,C,1,-10.1,5.2,3.8,-10.2,5.3,3.9,...
```

## Key Features

✓ **Multi-conformation prediction** - Generates 5 structures per sequence  
✓ **Recursive reasoning** - TRM's recursive approach for complex folding  
✓ **Competition-ready** - Kaggle format output  
✓ **Scalable** - Multi-GPU support  

## Common Commands

```bash
# Check dataset statistics
python -c "from utils.rna_utils import get_sequence_statistics; \
           import json; \
           seqs = json.load(open('data/rna-structure/train_sequences.json')); \
           stats = get_sequence_statistics([s['sequence'] for s in seqs]); \
           print(stats)"

# Validate RNA sequence
python -c "from utils.rna_utils import validate_rna_sequence; \
           print(validate_rna_sequence('ACGU'))"

# Parse FASTA
python -c "from utils.rna_utils import parse_fasta; \
           print(parse_fasta('>A\nACGU\n>B\nGCAU'))"

# Run tests
python tests/test_rna_integration.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` or `--max-length` |
| Slow training | Use multiple GPUs with `torchrun` |
| Poor accuracy | Increase `--epochs` or model size |
| Missing dependencies | Run `pip install pandas numpy biopython` |

## Support

- Issues: [GitHub Issues](https://github.com/drqsatoshi/TinyRecursiveModels/issues)
- Pipeline: [jrc-rna-structure-pipeline](https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline)
- TRM Paper: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
