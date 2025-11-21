# WAN 2.2 Multi-Node Fine-Tuning Configuration

Fine-tuning WAN 2.2 T2V-A14B checkpoint on 10B high-quality custom clips using 2 nodes × 8 H100 GPUs.

## Configuration Overview

### `accelerate_config_14B_ft.yaml`
- **Multi-node**: 2 nodes, 16 processes
- **DeepSpeed ZeRO Stage 2**: Good balance on 80GB H100 GPU
- **No CPU offloading**: Full H100 utilization
- **Gradient accumulation**: 4 steps (effective batch size = 64)
- **Mixed precision**: BF16

### `Wan2.2-T2V-A14B_ft.sh`
Two-stage fine-tuning on pre-trained Wan2.2 checkpoint:
1. **High noise** (timesteps 875-1000): Refine coarse structure
2. **Low noise** (timesteps 0-875): Enhance fine details

## Usage

### 1. Prepare Dataset
Create `data/custom_video_dataset/metadata.csv`:
```csv
video,prompt
videos/001.mp4,description of video 1
videos/002.mp4,description of video 2
```

### 2. Launch Training
```bash
sbatch examples/wanvideo/model_training/t_sft/slurm_launch.sh
```

### 3. Monitor Progress
```bash
tail -f logs/wan2.2_tsft_<JOB_ID>.out
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 2e-6 | Conservative for fine-tuning pre-trained checkpoint |
| Warmup steps | 500 | Smooth transition from pretrained weights |
| Num epochs | 1 | Sufficient for 10B high-quality clips |
| Dataset repeat | 1 | No repetition needed with large dataset |
| Gradient accumulation | 2 | Effective batch = 32 per stage |
| ZeRO stage | 2 | Optimal for 14B model using H100 |

## Output
Models saved to:
- `./models/train/Wan2.2-T2V-A14B_high_noise_t_sft/`
- `./models/train/Wan2.2-T2V-A14B_low_noise_t_sft/`

## Fine-Tuning vs Training from Scratch

This config is optimized for **fine-tuning** a pre-trained Wan2.2 checkpoint:
- **Lower learning rate (2e-6)**: Prevents catastrophic forgetting of pre-trained knowledge
- **Warmup steps (500)**: Gradual adaptation to new data distribution
- **Smaller gradient accumulation (2)**: Faster iteration during fine-tuning

If training from scratch, increase learning rate to 5e-6 and gradient accumulation to 4.

