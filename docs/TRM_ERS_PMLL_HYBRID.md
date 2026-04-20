# Hybrid TRM with ERS, PMLL, and Topic Integrator

This hybrid model combines the Tiny Recursive Model (TRM) with advanced memory management and recursive reasoning techniques from Dr. Josef Kurk Edwards' research (drqsatoshi.com).

## Overview

The hybrid model integrates:

1. **TRM (Tiny Recursive Model)**: Efficient recursive reasoning with minimal parameters
2. **ERS (Enhanced Reconsideration System)**: Persistent memory management with temporal decay and self-correction
3. **PMLL (Persistent Memory Logic Loops)**: Multi-pass validation and tensor routing through lattice structures
4. **Topic Integrator**: Knowledge graph integration for topic-based memory organization

## Architecture Components

### Enhanced Reconsideration System (ERS)

ERS provides stateful memory management through:

- **Persistent Memory Blocks**: Store past representations with confidence scores and timestamps
- **Temporal Decay**: Older memories naturally lose confidence over time
- **Consensus Strengthening**: Related memories reinforce each other
- **Contradiction Detection**: Conflicting memories penalize each other's confidence
- **Deferred Reconsideration Queue**: Prioritized queue for multi-pass memory validation

### PMLL (Persistent Memory Logic Loops)

PMLL enhances recursive reasoning with:

- **Lattice-based Tensor Routing**: Dynamic routing network for processing memory
- **Multi-petal Attention**: Multiple attention heads for embedding refinement
- **Commitment Scoring**: Evaluates confidence in memory commitments
- **Multi-pass Validation**: Iterative reconsideration within recursive cycles

### Topic Integrator

The Topic Integrator provides:

- **Topic Embedding Space**: Learned embeddings for different topic domains
- **Topic Assignment**: Automatic routing of information to relevant topics
- **Knowledge Graph Integration**: Connects memories through semantic relationships
- **Topic Fusion**: Combines topic context with current hidden states

## Usage

### Basic Configuration

```bash
run_name="pretrain_ers_pmll_sudoku"
python pretrain.py \
arch=trm_ers_pmll \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

### Configuration Parameters

#### ERS Configuration
- `ers_enabled: True` - Enable Enhanced Reconsideration System
- `ers_memory_size: 128` - Maximum number of memory blocks
- `ers_temporal_decay_rate: 0.95` - Memory confidence decay rate
- `ers_consensus_threshold: 0.7` - Threshold for consensus strengthening
- `ers_contradiction_threshold: 0.3` - Threshold for contradiction detection

#### PMLL Configuration
- `pmll_enabled: True` - Enable Persistent Memory Logic Loops
- `pmll_reconsideration_steps: 3` - Number of multi-pass validation loops
- `pmll_commitment_threshold: 0.8` - Threshold for memory commitment
- `pmll_lattice_dim: 64` - Dimension of lattice routing network

#### Topic Integrator Configuration
- `topic_integrator_enabled: True` - Enable Topic Integrator
- `topic_integrator_max_topics: 16` - Maximum number of topics to track

## Model Behavior

### Training

During training, the model:

1. Processes input through standard TRM embedding layers
2. Applies ERS to reconsider and update persistent memory
3. Integrates topic context from Topic Integrator
4. Executes PMLL-enhanced recursive loops with multi-pass validation
5. Applies PMLL lattice refinement for tensor routing
6. Updates memory blocks with temporal decay and consensus

### Inference

During inference, the model:

1. Maintains persistent memory across sequences
2. Applies temporal decay to existing memories
3. Detects contradictions and updates confidence scores
4. Routes information through PMLL lattice
5. Integrates topic context for knowledge-grounded predictions

## Performance Characteristics

- **Parameter Count**: ~13M parameters (similar to base TRM)
- **Memory Overhead**: Additional memory for ERS blocks (~128 blocks × embedding dimension)
- **Computational Cost**: Increased by PMLL reconsideration steps (default: 3× per H-cycle)
- **Benefits**: 
  - Improved long-term consistency through persistent memory
  - Better handling of contradictory information
  - Topic-aware reasoning for complex tasks

## Comparison with Base TRM

| Feature | Base TRM | Hybrid TRM-ERS-PMLL |
|---------|----------|---------------------|
| Recursive Reasoning | ✓ | ✓ |
| Persistent Memory | ✗ | ✓ (ERS) |
| Temporal Decay | ✗ | ✓ (ERS) |
| Contradiction Detection | ✗ | ✓ (ERS) |
| Multi-pass Validation | ✗ | ✓ (PMLL) |
| Topic Integration | ✗ | ✓ (Topic Integrator) |
| Parameter Efficiency | ✓✓ | ✓ |

## Research Background

This implementation is based on research from:

- **TRM**: "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau
- **ERS/PMLL**: Research by Dr. Josef Kurk Edwards (drQedwards) on Enhanced Reconsideration Systems and Persistent Memory Logic Loops
- **RTM**: "The Recursive Transformer Model: Architecture, Theory, and Implementation with Persistent Memory Logic Loops"

## References

- TRM Paper: https://arxiv.org/abs/2510.04871
- ERS Repository: https://github.com/drQedwards/ERS
- RTM Repository: https://github.com/drQedwards/RTM
- ERS White Paper: https://d197for5662m48.cloudfront.net/documents/publicationstatus/275810/preprint_pdf/ba3c7814470845671dced1012f7830ea.pdf

## Citation

If you use this hybrid model in your research, please cite both the TRM paper and the ERS/RTM work:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
    title={Less is More: Recursive Reasoning with Tiny Networks}, 
    author={Alexia Jolicoeur-Martineau},
    year={2025},
    eprint={2510.04871},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2510.04871}, 
}

@misc{edwards2025recursive,
    title={The Recursive Transformer Model: Architecture, Theory, and Implementation with Persistent Memory Logic Loops},
    author={Josef Kurk Edwards and Sarah Chen and Michael Rodriguez},
    year={2025}
}
```
