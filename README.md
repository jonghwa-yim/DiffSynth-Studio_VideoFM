# DiffSynth-Studio VideoFM

Video generation framework based on DiffSynth-Studio, focused on Wan Video series and related models.

## Quick Start

```bash
# Install
git clone <repo>
cd DiffSynth-Studio_VideoFM
pip install -e .

# Generate video
python examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py
```

## Code Structure

```
DiffSynth-Studio_VideoFM/
├── diffsynth/                    # Core library
│   ├── models/                   # Model implementations
│   │   ├── wan_video_dit.py      # DiT backbone (T2V, I2V)
│   │   ├── wan_video_dit_s2v.py  # S2V DiT variant
│   │   ├── wan_video_vae.py      # VAE 2.1/2.2
│   │   ├── wan_video_text_encoder.py  # T5 encoder
│   │   └── wan_video_*.py        # Additional modules
│   │
│   ├── pipelines/                # Inference pipelines
│   │   ├── wan_video_new.py      # Main unified pipeline
│   │   └── wan_video.py          # Legacy pipeline
│   │
│   ├── trainers/                 # Training infrastructure
│   │   ├── unified_dataset.py    # Data loading
│   │   └── utils.py              # Training utilities
│   │
│   ├── distributed/              # Distributed inference
│   │   └── xdit_context_parallel.py  # xDiT USP
│   │
│   ├── vram_management/          # Memory optimization
│   │   ├── layers.py             # Layer-by-layer offload
│   │   └── gradient_checkpointing.py
│   │
│   ├── extensions/               # Enhancement tools
│   │   ├── RIFE/                 # Frame interpolation
│   │   ├── FastBlend/            # Video deflickering
│   │   └── ImageQualityMetric/   # Quality assessment
│   │
│   └── schedulers/               # Noise schedulers
│       └── flow_match.py         # Flow matching
│
├── examples/wanvideo/            # Usage examples
│   ├── model_inference/          # Inference scripts
│   ├── model_training/           # Training scripts
│   │   ├── train.py              # Main training script
│   │   ├── t_lora/               # LoRA training configs
│   │   ├── t_sft/                # Full fine-tuning configs
│   │   ├── lora/                 # Legacy LoRA scripts
│   │   └── full/                 # Legacy full training
│   └── acceleration/             # Acceleration configs
│
├── reference/Wan2.2/             # Original Wan2.2 (read-only)
│   └── wan/modules/animate/preprocess/  # Preprocessing reference
│
└── pretrained_models/            # Model checkpoints
```

## Supported Models

| Model | Type | VRAM | Training |
|-------|------|------|----------|
| Wan2.2-T2V-A14B | Text-to-Video | 24GB+ | ✅ LoRA, ✅ Full |
| Wan2.2-I2V-A14B | Image-to-Video | 24GB+ | ✅ LoRA, ✅ Full |
| Wan2.2-TI2V-5B | Text+Image-to-Video | 16GB+ | ✅ LoRA, ✅ Full |
| Wan2.2-S2V-14B | Speech-to-Video | 24GB+ | ✅ LoRA, ✅ Full |
| Wan2.2-Animate-14B | Character Animation | 24GB+ | ✅ LoRA, ✅ Full |
| Wan2.1-T2V-1.3B | Text-to-Video | 8GB+ | ✅ LoRA, ✅ Full |

Full model list: [examples/wanvideo/README.md](examples/wanvideo/README.md)

## Usage

### Inference

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", 
                   origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", 
                   origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", 
                   origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
pipe.enable_vram_management()

video = pipe(prompt="A cat playing in the garden", seed=42, tiled=True)
save_video(video, "output.mp4", fps=15)
```

### LoRA Training (Single GPU)

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/videos \
  --dataset_metadata_path /path/to/metadata.csv \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:..." \
  --lora_base_model dit \
  --lora_rank 32 \
  --output_path ./checkpoints/my_lora
```

### Multi-Node Training (SLURM)

```bash
# Edit node count in slurm_launch.sh and accelerate_config.yaml
sbatch examples/wanvideo/model_training/t_lora/slurm_launch.sh
```

**Config files**:
- `t_lora/accelerate_config_14B_lora.yaml` - DeepSpeed Zero-2, 2 nodes × 8 GPUs
- `t_sft/accelerate_config_14B_ft.yaml` - Full fine-tuning config

## Training Infrastructure

| Component | Implementation |
|-----------|----------------|
| Distributed Backend | DeepSpeed Zero-2 |
| Orchestration | HuggingFace Accelerate |
| Job Scheduler | SLURM |
| Gradient Checkpointing | ✅ Supported |
| Mixed Precision | BF16 |

### Scaling Configuration

Edit `accelerate_config_*.yaml`:
```yaml
num_machines: 4        # Number of nodes
num_processes: 32      # Total GPUs (nodes × 8)
deepspeed_config:
  zero_stage: 2        # Optimizer + gradient sharding
```

Edit `slurm_launch.sh`:
```bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
```

## Memory Optimization

| Technique | Effect | Usage |
|-----------|--------|-------|
| Layer-by-layer offload | 40%+ VRAM reduction | `pipe.enable_vram_management()` |
| Gradient checkpointing | 30%+ VRAM reduction | `--use_gradient_checkpointing` |
| CPU offload | Additional reduction | `--use_gradient_checkpointing_offload` |
| Tiled VAE | Large resolution | `tiled=True` in pipeline |

## Reference Code

Original Wan2.2 repository is in `reference/Wan2.2/` for:
- **Animate preprocessing**: Pose extraction, SAM-2 segmentation
- **S2V audio processing**: Speech feature extraction
- **FSDP reference**: Alternative distributed training approach

See [reference/Wan2.2/INTEGRATION_PLAN.md](reference/Wan2.2/INTEGRATION_PLAN.md) for details.

## Key Differences from Original Wan2.2

| Aspect | DiffSynth-Studio | Original Wan2.2 |
|--------|------------------|-----------------|
| Training | ✅ LoRA + Full | ❌ Inference only |
| Distributed | DeepSpeed Zero-2 | FSDP |
| VRAM Mgmt | Advanced offload | Basic |
| Attention | Flash Attn 2/3 + SageAttn | Flash Attn only |

## License

Apache 2.0. See [LICENSE](LICENSE).

