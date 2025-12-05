# Wan2.2 Code Comparison: Original vs DiffSynth-Studio

## Summary

**DiffSynth-Studio is the primary codebase for all Wan video generation work.**

The original Wan2.2 repository is kept as reference for:
1. Animate preprocessing scripts (pose extraction, SAM-2 segmentation)
2. Alternative distributed strategies (FSDP vs DeepSpeed)

## Feature Comparison

| Feature | DiffSynth-Studio | Original Wan2.2 |
|---------|------------------|-----------------|
| **Multi-GPU Training** | ✅ DeepSpeed Zero-2 | ✅ FSDP |
| **Multi-Node Training** | ✅ SLURM + accelerate | ✅ torchrun |
| **LoRA Training** | ✅ Full support | ❌ Not included |
| **Full Fine-tuning** | ✅ Full support | ❌ Not included |
| **Inference** | ✅ Complete | ✅ Complete |
| **VRAM Management** | ✅ Layer-by-layer offload | ⚠️ Basic offload |
| **FP8 Quantization** | ✅ Supported | ❌ No |
| **Attention Backends** | ✅ Flash Attn 2/3 + SageAttn | ⚠️ Flash Attn only |
| **TeaCache Acceleration** | ✅ Supported | ❌ No |
| **Animate Preprocessing** | ❌ Not included | ✅ Full pipeline |
| **S2V Audio Processing** | ⚠️ Partial | ✅ Full support |

## Distributed Training

### DiffSynth-Studio Approach
- **Backend**: DeepSpeed Zero-2 via `accelerate`
- **Sharding**: Optimizer states + gradients
- **Config**: `accelerate_config_*.yaml`
- **Launch**: SLURM `srun` + `accelerate launch`

### Original Wan2.2 Approach  
- **Backend**: PyTorch FSDP
- **Sharding**: Parameters + gradients + optimizer
- **Config**: Code-based in `wan/distributed/fsdp.py`
- **Launch**: `torchrun`

Both achieve the same goal with different implementations. DeepSpeed Zero-2 is simpler to configure.

## Model Architecture

Both implement identical DiT architecture with same layer structure:
- Patch embedding (Conv3D)
- Transformer blocks with self-attention + cross-attention + FFN
- RoPE 3D positional encoding
- T5-based text encoder
- VAE 2.1/2.2 for latent encoding

## What to Use from Original Wan2.2

### 1. Animate Preprocessing (Priority: High)
Location: `reference/Wan2.2/wan/modules/animate/preprocess/`

| Script | Purpose |
|--------|---------|
| `process_pipepline.py` | Main orchestration |
| `pose2d.py` | ViTPose-based 2D pose extraction |
| `sam_utils.py` | SAM-2 human segmentation |
| `retarget_pose.py` | Pose retargeting between characters |

**Dependencies**: SAM-2, decord, onnxruntime, FLUX-Kontext (optional)

### 2. S2V Audio Utils (Priority: Medium)
Location: `reference/Wan2.2/wan/modules/s2v/`
- Audio encoding and processing utilities
- Whisper integration for speech

### 3. FSDP Reference (Priority: Low)
Location: `reference/Wan2.2/wan/distributed/`
- Only needed if DeepSpeed Zero-2 is insufficient
- Reference for PyTorch native distributed training

## Directory Mapping

| DiffSynth-Studio | Original Wan2.2 | Notes |
|------------------|-----------------|-------|
| `diffsynth/models/wan_video_dit.py` | `wan/modules/model.py` | Equivalent |
| `diffsynth/models/wan_video_vae.py` | `wan/modules/vae2_1.py`, `vae2_2.py` | Unified |
| `diffsynth/models/wan_video_text_encoder.py` | `wan/modules/t5.py` | Simplified |
| `diffsynth/distributed/xdit_context_parallel.py` | `wan/distributed/sequence_parallel.py` | Different impl |
| `diffsynth/pipelines/wan_video_new.py` | `wan/text2video.py`, etc. | Unified pipeline |
| ❌ Missing | `wan/modules/animate/preprocess/` | To port |

## Conclusion

- **Use DiffSynth-Studio** for all training and inference
- **Reference Original Wan2.2** only for Animate preprocessing scripts
- **No need** to port FSDP (DeepSpeed Zero-2 works well)
