# Model Variations

This document summarizes the differences between the base `trm.py` model and its variations.

## `trm.py` (Base Model)

The base Tiny Recursion Model (TRM) is a recursive reasoning model that uses a tiny network to recursively improve its predicted answer. The core of the model is the `TinyRecursiveReasoningModel_ACTV1_Inner` class, which implements the recursive reasoning process.

The model has two latent states:

*   `z_H`: The "high-level" latent state, which is updated once per "H-cycle".
*   `z_L`: The "low-level" latent state, which is updated multiple times per "H-cycle" (specifically, `L_cycles` times).

The recursive process can be summarized as follows:

1.  The model takes as input the embedded question, the current answer, and the latent states `z_H` and `z_L`.
2.  The model updates `z_L` for `L_cycles` iterations, using `z_H` and the input embeddings as input.
3.  The model updates `z_H` using the final `z_L` as input.
4.  The model generates a new answer based on the updated `z_H`.
5.  This process is repeated for `H_cycles` iterations.

The model also uses an Adaptive Computation Time (ACT) mechanism to learn how many steps to run the recursive process for.

## `trm_hier6.py`

This variation of the model introduces a more complex hierarchical structure. Instead of a single `z_L` latent state, it uses six of them (`z_L1` through `z_L6`).

The recursive process is modified as follows:

1.  The model takes as input the embedded question, the current answer, and the latent states `z_H` and `z_L1` through `z_L6`.
2.  The model updates each of the `z_L` latent states in turn, using the sum of all `z_L` states and the `z_H` state as input.
3.  The model updates `z_H` using the sum of all `z_L` states as input.
4.  The model generates a new answer based on the updated `z_H`.
5.  This process is repeated for `H_cycles` iterations.

This variation of the model is more complex than the base model, but it has the potential to learn more complex reasoning processes.

## `trm_singlez.py`

This variation of the model simplifies the base model by using only a single latent state, `z_L`. This eliminates the distinction between "high-level" and "low-level" latent states.

The recursive process is modified as follows:

1.  The model takes as input the embedded question, the current answer, and the latent state `z_L`.
2.  The model updates `z_L` for `L_cycles` iterations, using the input embeddings as input.
3.  The model generates a new answer based on the updated `z_L`.
4.  This process is repeated for `H_cycles` iterations.

This variation of the model is simpler than the base model, which may make it faster to train and less prone to overfitting.
