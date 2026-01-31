# RNA Structure Prediction Example

This example demonstrates the complete workflow for RNA 3D structure prediction using TinyRecursiveModels.

## Prerequisites

Make sure you have:
1. TinyRecursiveModels installed
2. RNA pipeline dependencies installed
3. Sample RNA data (or use the provided example)

## Example Dataset

For this example, we'll create a minimal synthetic dataset to demonstrate the workflow.

### Step 1: Create Sample Data

Create a sample sequences CSV file (`example_sequences.csv`):

```csv
target_id,sequence,temporal_cutoff,description,stoichiometry,all_sequences,ligand_ids,ligand_SMILES
TEST_RNA_001,CGCGAAUUAGCG,2025-01-01,Example hairpin structure,A:1,>A\nCGCGAAUUAGCG,,
TEST_RNA_002,GGACUUCGGUCC,2025-02-01,Example stem-loop,A:1,>A\nGGACUUCGGUCC,,
TEST_RNA_003,AUCGAUCGAUCG,2025-03-01,Example repeating pattern,A:1,>A\nAUCGAUCGAUCG,,
```

Create sample labels CSV file (`example_labels.csv`):

```csv
ID,resname,resid,x_1,y_1,z_1,chain,copy
TEST_RNA_001_1,C,1,-10.5,5.2,3.1,A,1
TEST_RNA_001_2,G,2,-9.8,6.1,4.2,A,1
TEST_RNA_001_3,C,3,-8.5,7.3,5.1,A,1
TEST_RNA_001_4,G,4,-7.2,8.1,6.3,A,1
TEST_RNA_001_5,A,5,-6.1,9.2,7.2,A,1
TEST_RNA_001_6,A,6,-5.3,10.1,8.1,A,1
TEST_RNA_001_7,U,7,-4.2,11.3,9.2,A,1
TEST_RNA_001_8,U,8,-3.1,12.1,10.1,A,1
TEST_RNA_001_9,A,9,-2.3,13.2,11.3,A,1
TEST_RNA_001_10,G,10,-1.2,14.1,12.2,A,1
TEST_RNA_001_11,C,11,-0.5,15.3,13.1,A,1
TEST_RNA_001_12,G,12,0.2,16.1,14.2,A,1
TEST_RNA_002_1,G,1,-11.2,4.8,2.9,A,1
TEST_RNA_002_2,G,2,-10.5,5.7,3.8,A,1
TEST_RNA_002_3,A,3,-9.3,6.9,4.7,A,1
TEST_RNA_002_4,C,4,-8.0,7.7,5.9,A,1
TEST_RNA_002_5,U,5,-6.9,8.8,6.8,A,1
TEST_RNA_002_6,U,6,-6.1,9.7,7.7,A,1
TEST_RNA_002_7,C,7,-5.0,10.9,8.8,A,1
TEST_RNA_002_8,G,8,-3.9,11.7,9.7,A,1
TEST_RNA_002_9,G,9,-3.1,12.8,10.9,A,1
TEST_RNA_002_10,U,10,-2.0,13.7,11.8,A,1
TEST_RNA_002_11,C,11,-1.3,14.9,12.7,A,1
TEST_RNA_002_12,C,12,-0.2,15.7,13.8,A,1
```

### Step 2: Process the Dataset

```bash
python dataset/build_rna_dataset.py \
  --sequences example_sequences.csv \
  --labels example_labels.csv \
  --output-dir data/rna-example \
  --max-length 100 \
  --temporal-cutoff 2025-02-15
```

This will create:
- `data/rna-example/train_sequences.json`
- `data/rna-example/train_labels.json`
- `data/rna-example/val_sequences.json`
- `data/rna-example/val_labels.json`

**Expected output:**
```
Loading RNA data...
Loaded 3 sequences and 2 label sets
Splitting dataset...
Train: 2 sequences, 2 labels
Val: 1 sequences, 0 labels
Saved 2 sequences and 2 labels to data/rna-example/train_*.json
Saved 1 sequences and 0 labels to data/rna-example/val_*.json

Dataset saved to data/rna-example
```

### Step 3: Train the Model

For this example, we'll use a small number of epochs to demonstrate the process:

```bash
python pretrain_rna.py \
  --data-dir data/rna-example \
  --output-dir checkpoints/rna-example \
  --batch-size 2 \
  --epochs 10 \
  --lr 1e-3 \
  --num-structures 5 \
  --max-length 100
```

**Expected output:**
```
Loading datasets...
Train dataset: 2 samples
Val dataset: 1 samples
Creating model...
Model parameters: X,XXX,XXX

Epoch 1/10
Training: 100%|████████| 1/1 [00:XX<00:00]
Train loss: X.XXXXXX
Validating: 100%|████████| 1/1 [00:XX<00:00]
Val loss: X.XXXXXX
Saved best model (val_loss: X.XXXXXX)
...
```

### Step 4: Generate Predictions

Create a test sequences file or use the validation set:

```bash
python predict_rna.py \
  --model-path checkpoints/rna-example/best_model.pth \
  --sequences data/rna-example/val_sequences.json \
  --output example_submission.csv \
  --batch-size 2 \
  --num-structures 5
```

**Expected output:**
```
Loading model...
Loaded model from epoch X
Validation loss: X.XXXXXX
Generating predictions...
Predicting: 100%|████████| 1/1 [00:XX<00:00]
Generated predictions for 1 sequences
Saving submission...
Saved submission to example_submission.csv
Total residues: 12

Prediction complete!
Submission saved to example_submission.csv
```

### Step 5: Inspect Results

The submission file (`example_submission.csv`) will contain:

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
TEST_RNA_003_1,A,1,X.XXX,Y.YYY,Z.ZZZ,X.XXX,Y.YYY,Z.ZZZ,...
TEST_RNA_003_2,U,2,X.XXX,Y.YYY,Z.ZZZ,X.XXX,Y.YYY,Z.ZZZ,...
...
```

Each row contains:
- 1 residue ID
- 1 residue name (nucleotide)
- 1 residue number
- 15 coordinate values (5 structures × 3 coordinates)

## Using the Automated Workflow

Alternatively, use the workflow script to run all steps automatically:

```bash
bash workflows/rna_structure_prediction.sh \
  --sequences example_sequences.csv \
  --labels example_labels.csv \
  --test-sequences example_sequences.csv \
  --output example_submission.csv \
  --epochs 10 \
  --batch-size 2
```

## Using Real Competition Data

To use real competition data from Kaggle:

1. Download the data from the [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) competition

2. Process and train:

```bash
# Process dataset
python dataset/build_rna_dataset.py \
  --sequences kaggle/train_sequences.csv \
  --labels kaggle/train_labels.csv \
  --output-dir data/rna-kaggle \
  --temporal-cutoff 2025-05-29

# Train model (this will take several hours/days depending on data size)
python pretrain_rna.py \
  --data-dir data/rna-kaggle \
  --output-dir checkpoints/rna-kaggle \
  --batch-size 16 \
  --epochs 100 \
  --num-structures 5

# Generate predictions for test set
python predict_rna.py \
  --model-path checkpoints/rna-kaggle/best_model.pth \
  --sequences kaggle/test_sequences.csv \
  --output kaggle_submission.csv
```

## Tips for Better Results

### 1. Data Augmentation
- Use multiple conformations if available in the training data
- Consider synthetic augmentation (rotation, translation)

### 2. Model Tuning
```bash
# Increase model capacity
python pretrain_rna.py \
  --data-dir data/rna-kaggle \
  --embed-dim 512 \
  --hidden-dim 1024 \
  --epochs 200
```

### 3. Multi-GPU Training
```bash
# Use 4 GPUs for faster training
torchrun --nproc-per-node 4 pretrain_rna.py \
  --data-dir data/rna-kaggle \
  --batch-size 64 \
  --epochs 200
```

### 4. Learning Rate Scheduling
Modify `pretrain_rna.py` to add a learning rate scheduler for better convergence.

## Troubleshooting

### Memory Issues
If you run out of memory:
```bash
python pretrain_rna.py --batch-size 4 --max-length 200
```

### Slow Training
Use a smaller model or fewer epochs:
```bash
python pretrain_rna.py --embed-dim 128 --hidden-dim 256 --epochs 50
```

### Poor Predictions
1. Train for more epochs
2. Increase model capacity
3. Use more training data
4. Check data quality

## Next Steps

1. **Experiment with hyperparameters**: Try different model sizes, learning rates, and cycle counts
2. **Add features**: Incorporate MSA (Multiple Sequence Alignment) data if available
3. **Ensemble predictions**: Combine predictions from multiple models
4. **Post-processing**: Apply physical constraints to improve structure validity

## References

- [RNA Integration Guide](RNA_INTEGRATION.md)
- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [RNA Pipeline Repository](https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline)
