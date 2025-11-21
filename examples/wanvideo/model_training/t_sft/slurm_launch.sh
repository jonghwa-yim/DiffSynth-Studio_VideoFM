#!/bin/bash
#SBATCH --job-name=wan2.2-t2v-14b-tsft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --output=logs/wan2.2_tsft_%j.out
#SBATCH --error=logs/wan2.2_tsft_%j.err

# Set master node address for multi-node communication
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Master node: $MASTER_ADDR"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch training on all nodes
srun bash examples/wanvideo/model_training/t_sft/Wan2.2-T2V-A14B_ft.sh

