#!/bin/bash

# Get node rank from SLURM
NODE_RANK=${SLURM_NODEID:-0}
NNODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=8

# Master address should be set by slurm_launch.sh
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Configure model cache directory
export DIFFSYNTH_MODEL_CACHE="${DIFFSYNTH_MODEL_CACHE:-./pretrained_models}"

# Print node info
echo "=============================================="
echo "Node: $(hostname)"
echo "Node rank: $NODE_RANK / $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Model cache: $DIFFSYNTH_MODEL_CACHE"
echo "=============================================="

###############################################################################
# Stage 1: High noise LoRA training [timesteps 875-1000]
###############################################################################
echo ""
echo "Starting Stage 1: High noise LoRA training..."

accelerate launch \
  --config_file examples/wanvideo/model_training/t_lora/accelerate_config_14B_lora_offload.yaml \
  --num_machines $NNODES \
  --num_processes $((NNODES * GPUS_PER_NODE)) \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --rdzv_backend static \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/jonghwa/data/training_datasets/openvid/ \
  --dataset_metadata_path /home/jonghwa/data/training_datasets/openvid/metadata.csv \
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
  --min_timestep_boundary 0 \
  --use_gradient_checkpointing_offload

echo "Stage 1 completed at: $(date)"

###############################################################################
# Stage 2: Low noise LoRA training [timesteps 0-875)
###############################################################################
echo ""
echo "Starting Stage 2: Low noise LoRA training..."

accelerate launch \
  --config_file examples/wanvideo/model_training/t_lora/accelerate_config_14B_lora_offload.yaml \
  --num_machines $NNODES \
  --num_processes $((NNODES * GPUS_PER_NODE)) \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --rdzv_backend static \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/jonghwa/data/training_datasets/openvid/ \
  --dataset_metadata_path /home/jonghwa/data/training_datasets/openvid/metadata.csv \
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
  --min_timestep_boundary 0.417 \
  --use_gradient_checkpointing_offload

echo "Stage 2 completed at: $(date)"
echo "All training completed!"
