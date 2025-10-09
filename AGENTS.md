# Multi-Machine Deployment and Scaling Guide

This guide provides an educational overview of how to deploy this project across multiple machines, how to scale the model, and how to estimate the required GPU resources.

## 1. Conceptual Overview: Distributed Training

This project uses **Distributed Data-Parallel (DDP)** training, which is a common strategy for training large models. Here’s how it works:

*   **Data Parallelism:** The training data is split into smaller chunks, and each chunk is processed by a different GPU.
*   **Model Replication:** A complete copy of the model is loaded onto each GPU.
*   **Gradient Synchronization:** During the backward pass, each GPU computes the gradients for its chunk of data. These gradients are then averaged across all GPUs using a process called **All-Reduce**. This ensures that all models are updated with the same gradients, keeping them synchronized.
*   **`torchrun`:** This is a utility provided by PyTorch that simplifies the process of launching distributed training jobs. It automatically sets up the necessary environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, etc.) that are used by `torch.distributed` to coordinate the processes.

## 2. Multi-Machine Deployment Guide

The `README.md` provides commands for running on a single machine with multiple GPUs. To run on multiple machines, you need to extend this to a multi-node setup.

### Prerequisites:

*   **Networking:** Your machines must be able to communicate with each other over the network. They should be on the same subnet, or you need to configure your firewall to allow traffic on a specific port.
*   **Shared Storage (Optional but Recommended):** For simplicity, it's best to have a shared network file system (like NFS) where your dataset and code are stored. This ensures that all nodes have access to the same files. If you don't have a shared file system, you'll need to copy the code and data to each machine.
*   **SSH Access:** You'll need to be able to SSH into each machine to launch the training script.

### Launching the Training Job:

You will use `torchrun` on each machine. You need to designate one machine as the "master" node, which will be responsible for coordinating the other nodes.

Let's say you have two machines, `machine-1` (master) and `machine-2`.

**On `machine-1` (Master Node):**

```bash
torchrun --nproc-per-node=4 \
         --nnodes=2 \
         --node-rank=0 \
         --rdzv-id=123 \
         --rdzv-backend=c10d \
         --rdzv-endpoint="machine-1:29500" \
         pretrain.py \
         arch=trm \
         data_paths="[data/arc1concept-aug-1000]" \
         arch.L_layers=2 \
         arch.H_cycles=3 arch.L_cycles=4 \
         +run_name="my-distributed-run" ema=True
```

**On `machine-2` (Worker Node):**

```bash
torchrun --nproc-per-node=4 \
         --nnodes=2 \
         --node-rank=1 \
         --rdzv-id=123 \
         --rdzv-backend=c10d \
         --rdzv-endpoint="machine-1:29500" \
         pretrain.py \
         arch=trm \
         data_paths="[data/arc1concept-aug-1000]" \
         arch.L_layers=2 \
         arch.H_cycles=3 arch.L_cycles=4 \
         +run_name="my-distributed-run" ema=True
```

### Explanation of `torchrun` Parameters:

*   `--nproc-per-node`: The number of GPUs to use on each machine.
*   `--nnodes`: The total number of machines (nodes) participating in the training.
*   `--node-rank`: A unique ID for each machine, from 0 to `nnodes - 1`. The master node should always have `rank=0`.
*   `--rdzv-id`: A unique ID for the job. This should be the same on all nodes.
*   `--rdzv-backend`: The rendezvous backend. `c10d` is the standard TCP-based backend.
*   `--rdzv-endpoint`: The address and port of the master node. All nodes will connect to this address to coordinate.

## 3. Model Scaling and GPU Requirements

The size of the model and the batch size are the primary factors that determine the GPU memory you'll need.

### Key Model Parameters (`config/arch/trm.yaml`):

*   `hidden_size`: The embedding dimension. This has a quadratic impact on the memory used by the attention mechanism.
*   `num_heads`: The number of attention heads. More heads increase parallelism but also memory usage.
*   `L_layers`: The number of "low-level" transformer layers. This is a linear scaling factor.
*   `H_cycles`, `L_cycles`: The number of recursive cycles. These don't significantly change the model's memory footprint, but they increase the computational cost (and thus the training time).

### Rules of Thumb for GPU Memory:

*   **Baseline:** The `README.md` mentions that the ARC-AGI-1 experiment runs on 4 H100 GPUs. The H100 has 80GB of VRAM. This gives you a starting point.
*   **Doubling `hidden_size`:** If you double the `hidden_size`, you will roughly quadruple the memory required for the attention mechanism. The memory for the feed-forward layers will double. Overall, expect the memory usage to increase by a factor of 2.5-3x.
*   **Doubling `L_layers`:** If you double the number of layers, you will roughly double the memory required for the model's parameters and activations.
*   **Batch Size:** The `global_batch_size` in `config/cfg_pretrain.yaml` is the total batch size across all GPUs. The per-GPU batch size is `global_batch_size / WORLD_SIZE`, where `WORLD_SIZE` is the total number of GPUs (`nproc-per-node * nnodes`). If you run out of memory, the first thing to try is reducing the `global_batch_size`.
*   **Mixed-Precision Training:** The `forward_dtype: bfloat16` in `config/arch/trm.yaml` indicates that the model uses 16-bit precision for the forward pass. This significantly reduces memory usage compared to 32-bit floating-point (`float32`).

### Estimating Your Needs:

1.  **Start Small:** Begin with the default configuration on a single machine.
2.  **Monitor GPU Usage:** Use `nvidia-smi` to monitor the VRAM usage.
3.  **Extrapolate:**
    *   If you want to train a larger model, make one change at a time (e.g., increase `hidden_size`) and observe the new memory usage.
    *   If a single GPU is not enough, you'll need to add more GPUs. The total memory required will be the memory per GPU multiplied by the number of GPUs.

## 4. Experimentation and Cost Management

Running experiments on multiple machines can be expensive. Here are some best practices:

*   **Use `wandb`:** The project is already integrated with Weights & Biases (`wandb`). Use it to log your experiments. This will help you keep track of your results and compare different runs.
*   **Start with Small-Scale Experiments:** Before launching a full multi-day training run, test your setup with a small dataset or for a small number of epochs. This will help you catch any bugs or configuration issues early.
*   **Use Checkpointing:** The `checkpoint_every_eval` option in `config/cfg_pretrain.yaml` is very useful. It saves the model periodically, so you can resume training if it gets interrupted.
*   **Automate Your Workflow:** For launching jobs on multiple machines, consider using a tool like `tmux` or `screen` to manage your SSH sessions, or even a more advanced cluster management tool if you have one available.

By following this guide, you should be well-equipped to deploy this project on your machines and start experimenting with different model sizes and configurations. Good luck with your educational project!