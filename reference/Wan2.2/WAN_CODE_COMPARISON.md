# Wan2.2 Code Comparison: Original vs DiffSynth-Studio

## Executive Summary

**Recommendation: You DO NOT need to pull the original Wan2.2 model code.**

DiffSynth-Studio has a **complete and enhanced** implementation of Wan models with additional features. The original Wan2.2 repository is functionally equivalent for the core models but uses a different API design.

## Detailed Comparison

### 1. Repository Architecture

| Aspect | Original Wan2.2 | DiffSynth-Studio |
|--------|-----------------|------------------|
| **Focus** | Wan-specific models only | Multi-model framework (Wan, HunyuanVideo, SVD, etc.) |
| **Structure** | Clean separation: `wan/text2video.py`, `wan/image2video.py`, etc. | Unified pipeline: `diffsynth/pipelines/wan_video_new.py` |
| **API Style** | Class-based (WanT2V, WanI2V, WanAnimate, WanS2V) | Unified WanVideoPipeline with from_pretrained() |
| **Entry Point** | `generate.py` script | Python API + Examples |

### 2. Model Implementation Comparison

#### Core Model Files

**Original Wan2.2:**
- `wan/modules/model.py` (~546 lines) - DiT model
- `wan/modules/vae2_1.py` (~663 lines) - VAE 2.1
- `wan/modules/vae2_2.py` (~1051 lines) - VAE 2.2
- `wan/modules/t5.py` (~513 lines) - T5 text encoder

**DiffSynth-Studio:**
- `diffsynth/models/wan_video_dit.py` (~773 lines) - DiT model
- `diffsynth/models/wan_video_vae.py` (~1382 lines) - VAE (both 2.1 & 2.2)
- `diffsynth/models/wan_video_text_encoder.py` (~269 lines) - T5 encoder
- **Additional:** `wan_video_dit_s2v.py`, `wan_video_image_encoder.py`, `wan_video_vace.py`, `wan_video_motion_controller.py`, `wan_video_animate_adapter.py`, `wan_video_mot.py`, `wan_video_camera_controller.py`

**Analysis:** DiffSynth-Studio has **more comprehensive** model implementations with additional specialized modules.

### 3. Supported Models

Both repositories support the same Wan model variants:
- ✅ Wan2.2-T2V-A14B (Text-to-Video)
- ✅ Wan2.2-I2V-A14B (Image-to-Video)
- ✅ Wan2.2-TI2V-5B (Text+Image-to-Video)
- ✅ Wan2.2-S2V-14B (Speech-to-Video)
- ✅ Wan2.2-Animate-14B (Character Animation)
- ✅ Wan2.1 series (all variants)
- ✅ Control models (VACE, Camera, Motion)

### 4. Feature Comparison

| Feature | Original Wan2.2 | DiffSynth-Studio | Winner |
|---------|-----------------|------------------|--------|
| **Core Inference** | ✅ Yes | ✅ Yes | Tie |
| **Multi-GPU (FSDP)** | ✅ Yes | ❌ No | Original |
| **Sequence Parallel (USP)** | ✅ Yes (custom) | ✅ Yes (xDiT) | Tie (different implementations) |
| **VRAM Management** | ⚠️ Basic offloading | ✅ Advanced layer-by-layer offload | **DiffSynth** |
| **FP8 Quantization** | ❌ No | ✅ Yes | **DiffSynth** |
| **LoRA Training** | ❌ Not in repo | ✅ Full support | **DiffSynth** |
| **Full Fine-tuning** | ❌ Not in repo | ✅ Full support | **DiffSynth** |
| **TeaCache** | ❌ No | ✅ Yes | **DiffSynth** |
| **ExVideo Integration** | ❌ No | ✅ Yes | **DiffSynth** |
| **FastBlend/RIFE** | ❌ No | ✅ Yes | **DiffSynth** |
| **Image Quality Metrics** | ❌ No | ✅ Yes | **DiffSynth** |
| **Multi-model Support** | ❌ Wan only | ✅ Wan, Hunyuan, etc. | **DiffSynth** |

### 5. Distributed Training/Inference

**Original Wan2.2:**
- Full FSDP (Fully Sharded Data Parallel) support for T5 and DiT
- Custom Sequence Parallel implementation (Ulysses attention)
- Designed for multi-GPU production environments

**DiffSynth-Studio:**
- xDiT-based Unified Sequence Parallel (USP)
- Layer-by-layer VRAM offloading for consumer GPUs
- More suitable for research and single-GPU scenarios

### 6. API Comparison

**Original Wan2.2 Usage:**
```python
from wan import WanT2V
from wan.configs import wan_t2v_A14B

model = WanT2V(
    config=wan_t2v_A14B,
    checkpoint_dir="path/to/checkpoints",
    device_id=0
)
video = model.generate(
    prompt="A cat playing",
    num_frames=81,
    height=480,
    width=832
)
```

**DiffSynth-Studio Usage:**
```python
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
video = pipe(prompt="A cat playing", seed=0, tiled=True)
```

**Analysis:** DiffSynth-Studio uses a more modular, flexible API with automatic model downloading from ModelScope/HuggingFace.

### 7. Dependencies Comparison

**Original Wan2.2 Specific Dependencies:**
- SAM-2 (for Animate preprocessing)
- openai-whisper (for S2V)
- CosyVoice dependencies (for S2V text-to-speech)
- decord (for video processing)

**DiffSynth-Studio Additional Dependencies:**
- xDiT (for distributed inference)
- Various quality metric models
- RIFE interpolation
- FastBlend deflickering

## Key Differences in Implementation

### 1. Attention Mechanism
Both use Flash Attention but implement it differently:
- **Original:** Direct flash_attn integration with window attention support
- **DiffSynth:** Supports Flash Attention 2, 3, and SageAttention with compatibility mode

### 2. RoPE (Rotary Position Embedding)
- **Original:** Implements RoPE directly in attention layers
- **DiffSynth:** Separate RoPE implementation with precomputation support

### 3. VAE Architecture
- **Original:** Separate files for VAE 2.1 and 2.2
- **DiffSynth:** Unified VAE implementation supporting both versions with WanVideoVAE38 class

## What You're Missing (if anything)

### From Original Wan2.2:
1. **FSDP Support** - If you need to train/run on multiple GPUs with FSDP sharding
2. **Command-line Interface** - The `generate.py` script provides a convenient CLI
3. **Official preprocessing scripts** - For Animate task (pose extraction, face detection, etc.)

### Already in DiffSynth-Studio:
1. ✅ All core model implementations
2. ✅ Inference pipelines for all Wan variants
3. ✅ Training support (LoRA and full fine-tuning)
4. ✅ Low-VRAM optimizations
5. ✅ Advanced features (TeaCache, ExVideo, etc.)

## Recommendations

### ✅ Stick with DiffSynth-Studio if you want:
- ✅ Low VRAM usage (consumer GPUs like RTX 4090, 3090)
- ✅ LoRA training and fine-tuning
- ✅ Integration with other models (HunyuanVideo, etc.)
- ✅ Research features (TeaCache, ExVideo, quality metrics)
- ✅ Unified API across multiple model families
- ✅ Active development and community support

### ⚠️ Consider Original Wan2.2 if you need:
- Multi-GPU production deployment with FSDP
- Official preprocessing pipelines for Animate
- Command-line interface for simple usage
- Minimal dependencies (Wan-only focus)

### 🔄 Hybrid Approach (Optional):
You could:
1. Keep DiffSynth-Studio as your main codebase
2. Reference the original Wan2.2 for:
   - FSDP implementation patterns
   - Preprocessing scripts for Animate task
   - CLI design inspiration

## Conclusion

**You have everything you need in DiffSynth-Studio.** The implementation is complete, well-maintained, and enhanced with additional features that make it superior for research and development on consumer hardware.

The original Wan2.2 repository is designed more for production deployment in enterprise environments with multi-GPU clusters, while DiffSynth-Studio is optimized for research, experimentation, and accessible hardware.

## Action Items

✅ **No action needed** - Your current codebase is complete

⭐ **Optional enhancements** (if desired):
1. Add CLI wrapper similar to `generate.py` for convenience
2. Port FSDP support for multi-GPU training (if needed)
3. Integrate official Animate preprocessing scripts (if using Animate heavily)

---
*Generated: November 19, 2024*
*Based on comparison of Wan-Video/Wan2.2 (latest) and DiffSynth-Studio (current)*

