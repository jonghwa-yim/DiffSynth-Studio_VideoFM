#!/bin/bash
#SBATCH --job-name=wan2.2-t2v-lora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/wan2.2_lora_%j.out
#SBATCH --error=logs/wan2.2_lora_%j.err
#SBATCH --partition=main

###############################################################################
# Multi-node LoRA Training for Wan2.2-T2V-A14B with DeepSpeed Zero-2
###############################################################################
#
# This script uses accelerate + DeepSpeed with proper multi-node SLURM support.
#
# To change the number of nodes, update:
#   1. This file: --nodes=N
#   2. accelerate_config_14B_lora.yaml:
#      - num_machines: N
#      - num_processes: N * 8
#
# Usage:
#   sbatch examples/wanvideo/model_training/t_lora/slurm_launch.sh
#
###############################################################################

# Create logs directory if it doesn't exist
mkdir -p logs

# Number of GPUs per node
GPUS_PER_NODE=8
NNODES=$SLURM_JOB_NUM_NODES

# Set master node address for multi-node communication
# These MUST be exported for DeepSpeed to pick up
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Configure model cache directory
export DIFFSYNTH_MODEL_CACHE="./pretrained_models"

# Print job information
echo "=============================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "World size: $WORLD_SIZE"
echo "Model cache: $DIFFSYNTH_MODEL_CACHE"
echo "=============================================="

# Verify models exist
if [ ! -d "$DIFFSYNTH_MODEL_CACHE/Wan-AI/Wan2.2-T2V-A14B" ]; then
    echo "ERROR: Models not found"
    exit 1
fi

# Launch training on each node
# srun will set SLURM_NODEID (0, 1, 2, ...) for each node
srun --export=ALL bash examples/wanvideo/model_training/t_lora/Wan2.2-T2V-A14B.sh

echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
