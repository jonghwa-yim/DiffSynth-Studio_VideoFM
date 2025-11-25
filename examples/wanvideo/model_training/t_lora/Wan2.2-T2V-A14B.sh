#!/bin/bash
#SBATCH --job-name=wan2.2-t2v-lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/wan2.2_lora_%j.out
#SBATCH --error=logs/wan2.2_lora_%j.err
#SBATCH --partition=main

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Configure model cache directory
# Option 1: Use environment variable (recommended)
export DIFFSYNTH_MODEL_CACHE="./pretrained_models"
echo "Model cache directory: $DIFFSYNTH_MODEL_CACHE"

# Option 2: Use existing models from another location (e.g., shared storage)
# export DIFFSYNTH_MODEL_CACHE="/scratch/shared/pretrained_models"
# mkdir -p "$DIFFSYNTH_MODEL_CACHE"

# Stage 1: High noise LoRA training [timesteps 875-1000]
accelerate launch \
  --mixed_precision bf16 \
  --num_processes 8 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/jonghwa/data/training_datasets/videoufo/ \
  --dataset_metadata_path /home/jonghwa/data/training_datasets/videoufo/metadata.csv \
  --max_pixels 921600 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./checkpoints/Wan2.2-T2V-A14B_high_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [875, 1000]

# Stage 2: Low noise LoRA training [timesteps 0-875)
accelerate launch \
  --mixed_precision bf16 \
  --num_processes 8 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/jonghwa/data/training_datasets/videoufo/ \
  --dataset_metadata_path /home/jonghwa/data/training_datasets/videoufo/metadata.csv \
  --max_pixels 921600 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./checkpoints/Wan2.2-T2V-A14B_low_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.417
# boundary corresponds to timesteps [0, 875)

# Print job completion information
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"