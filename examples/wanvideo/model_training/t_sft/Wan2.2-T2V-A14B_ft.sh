#!/bin/bash

# Set master address and port for multi-node training (SLURM will set these)
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Stage 1: High noise training [timesteps 875-1000]
accelerate launch \
  --config_file examples/wanvideo/model_training/t_sft/accelerate_config_14B_ft.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank ${SLURM_NODEID:-0} \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/custom_video_dataset \
  --dataset_metadata_path data/custom_video_dataset/metadata.csv \
  --max_pixels 921600 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 2e-6 \
  --num_epochs 1 \
  --warmup_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-T2V-A14B_high_noise_t_sft" \
  --trainable_models "dit" \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [875, 1000]

# Stage 2: Low noise training [timesteps 0-875)
accelerate launch \
  --config_file examples/wanvideo/model_training/t_sft/accelerate_config_14B_ft.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank ${SLURM_NODEID:-0} \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/custom_video_dataset \
  --dataset_metadata_path data/custom_video_dataset/metadata.csv \
  --max_pixels 921600 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 2e-6 \
  --num_epochs 1 \
  --warmup_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-T2V-A14B_low_noise_t_sft" \
  --trainable_models "dit" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.417
# boundary corresponds to timesteps [0, 875)