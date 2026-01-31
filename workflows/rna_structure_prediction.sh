#!/bin/bash
#
# RNA 3D Structure Prediction Workflow
#
# This script automates the entire workflow for RNA structure prediction:
# 1. Setup environment
# 2. Process dataset
# 3. Train model
# 4. Generate predictions
#
# Usage:
#   bash workflows/rna_structure_prediction.sh [--help] [--skip-setup] [--skip-train]

set -e  # Exit on error

# Default configuration
DATA_DIR="data/rna-structure"
CHECKPOINT_DIR="checkpoints/rna"
SEQUENCES_FILE=""
LABELS_FILE=""
TEST_SEQUENCES_FILE=""
OUTPUT_FILE="submission.csv"
BATCH_SIZE=16
EPOCHS=100
NUM_STRUCTURES=5
MAX_LENGTH=500
TEMPORAL_CUTOFF="2025-05-29"

# Flags
SKIP_SETUP=false
SKIP_TRAIN=false
SKIP_PREDICT=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
RNA 3D Structure Prediction Workflow

Usage: $0 [OPTIONS]

Options:
    --sequences FILE         Path to training sequences CSV file (required)
    --labels FILE           Path to training labels CSV file (required)
    --test-sequences FILE   Path to test sequences CSV file for predictions
    --output FILE           Output submission file (default: submission.csv)
    --data-dir DIR          Directory for processed data (default: data/rna-structure)
    --checkpoint-dir DIR    Directory for model checkpoints (default: checkpoints/rna)
    --batch-size N          Batch size for training (default: 16)
    --epochs N              Number of training epochs (default: 100)
    --num-structures N      Number of structures to predict (default: 5)
    --max-length N          Maximum sequence length (default: 500)
    --temporal-cutoff DATE  Date for train/val split (default: 2025-05-29)
    
    --skip-setup           Skip environment setup
    --skip-train           Skip model training (use existing checkpoint)
    --skip-predict         Skip prediction generation
    --dry-run              Show what would be done without executing
    
    --help                 Show this help message

Examples:
    # Full workflow with custom data
    $0 --sequences train_sequences.csv --labels train_labels.csv \\
       --test-sequences test_sequences.csv

    # Quick prediction with existing model
    $0 --sequences test_sequences.json --skip-setup --skip-train

    # Train only
    $0 --sequences train_sequences.csv --labels train_labels.csv --skip-predict

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            SEQUENCES_FILE="$2"
            shift 2
            ;;
        --labels)
            LABELS_FILE="$2"
            shift 2
            ;;
        --test-sequences)
            TEST_SEQUENCES_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --num-structures)
            NUM_STRUCTURES="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --temporal-cutoff)
            TEMPORAL_CUTOFF="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-predict)
            SKIP_PREDICT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validation
if [ -z "$SEQUENCES_FILE" ]; then
    log_error "Sequences file is required. Use --sequences to specify."
    print_usage
    exit 1
fi

# Main workflow
log_info "========================================="
log_info "RNA 3D Structure Prediction Workflow"
log_info "========================================="
log_info "Configuration:"
log_info "  Sequences: $SEQUENCES_FILE"
log_info "  Labels: $LABELS_FILE"
log_info "  Test sequences: $TEST_SEQUENCES_FILE"
log_info "  Data directory: $DATA_DIR"
log_info "  Checkpoint directory: $CHECKPOINT_DIR"
log_info "  Output file: $OUTPUT_FILE"
log_info "========================================="

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No commands will be executed"
fi

# Step 1: Environment setup
if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 1: Setting up environment"
    
    if [ "$DRY_RUN" = false ]; then
        # Check if RNA pipeline is cloned
        if [ ! -d "jrc-rna-structure-pipeline" ]; then
            log_info "Cloning RNA structure pipeline..."
            git clone https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline.git
        else
            log_info "RNA structure pipeline already cloned"
        fi
        
        # Check Python packages
        log_info "Checking Python dependencies..."
        python -c "import pandas, numpy, torch" 2>/dev/null || {
            log_warning "Some dependencies missing. Installing..."
            pip install pandas numpy torch
        }
    else
        log_info "[DRY RUN] Would clone jrc-rna-structure-pipeline and install dependencies"
    fi
    
    log_success "Environment setup complete"
else
    log_info "Skipping environment setup"
fi

# Step 2: Process dataset
if [ ! -z "$LABELS_FILE" ]; then
    log_info "Step 2: Processing dataset"
    
    CMD="python dataset/build_rna_dataset.py \
        --sequences $SEQUENCES_FILE \
        --labels $LABELS_FILE \
        --output-dir $DATA_DIR \
        --max-length $MAX_LENGTH \
        --temporal-cutoff $TEMPORAL_CUTOFF"
    
    if [ "$DRY_RUN" = false ]; then
        eval $CMD
        log_success "Dataset processing complete"
    else
        log_info "[DRY RUN] Would execute: $CMD"
    fi
else
    log_info "Step 2: No labels file provided, skipping dataset processing"
fi

# Step 3: Train model
if [ "$SKIP_TRAIN" = false ]; then
    log_info "Step 3: Training model"
    
    CMD="python pretrain_rna.py \
        --data-dir $DATA_DIR \
        --output-dir $CHECKPOINT_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --num-structures $NUM_STRUCTURES \
        --max-length $MAX_LENGTH"
    
    if [ "$DRY_RUN" = false ]; then
        eval $CMD
        log_success "Model training complete"
    else
        log_info "[DRY RUN] Would execute: $CMD"
    fi
else
    log_info "Skipping model training"
fi

# Step 4: Generate predictions
if [ "$SKIP_PREDICT" = false ]; then
    log_info "Step 4: Generating predictions"
    
    # Determine sequences file for prediction
    if [ ! -z "$TEST_SEQUENCES_FILE" ]; then
        PRED_SEQUENCES="$TEST_SEQUENCES_FILE"
    else
        PRED_SEQUENCES="$DATA_DIR/val_sequences.json"
    fi
    
    # Check if model checkpoint exists
    if [ ! -f "$CHECKPOINT_DIR/best_model.pth" ]; then
        log_error "Model checkpoint not found: $CHECKPOINT_DIR/best_model.pth"
        log_error "Train a model first or specify --skip-predict"
        exit 1
    fi
    
    CMD="python predict_rna.py \
        --model-path $CHECKPOINT_DIR/best_model.pth \
        --sequences $PRED_SEQUENCES \
        --output $OUTPUT_FILE \
        --batch-size $BATCH_SIZE \
        --num-structures $NUM_STRUCTURES \
        --max-length $MAX_LENGTH"
    
    if [ "$DRY_RUN" = false ]; then
        eval $CMD
        log_success "Prediction generation complete"
        log_success "Submission saved to: $OUTPUT_FILE"
    else
        log_info "[DRY RUN] Would execute: $CMD"
    fi
else
    log_info "Skipping prediction generation"
fi

log_success "========================================="
log_success "Workflow complete!"
log_success "========================================="

if [ "$DRY_RUN" = false ]; then
    log_info "Next steps:"
    log_info "  1. Review predictions in: $OUTPUT_FILE"
    log_info "  2. Submit to competition or evaluate locally"
    log_info "  3. Fine-tune model if needed"
fi
