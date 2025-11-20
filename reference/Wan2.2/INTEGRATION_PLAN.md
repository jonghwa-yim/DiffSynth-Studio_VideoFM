# Integration Plan: Original Wan2.2 Code + DiffSynth-Studio

## Updated Requirements

Based on your needs for:
1. ✅ Multi-GPU cluster optimization for massive training
2. ✅ Preprocessing scripts for Animate task development

**You SHOULD keep the original Wan2.2 code as a reference.**

## Recommended Directory Structure

### Option 1: Side-by-side (RECOMMENDED)
```
~/workspace/
├── DiffSynth-Studio_VideoFM/          # Your main working repository
│   ├── diffsynth/
│   ├── examples/
│   ├── README.md
│   └── WAN_CODE_COMPARISON.md
│
└── reference_wan2.2/                   # Original Wan2.2 for reference
    ├── wan/
    │   ├── distributed/
    │   │   ├── fsdp.py                # ⭐ FSDP implementation
    │   │   ├── sequence_parallel.py   # ⭐ Sequence parallel
    │   │   └── ulysses.py             # ⭐ Ulysses attention
    │   ├── modules/
    │   │   └── animate/
    │   │       └── preprocess/        # ⭐ Preprocessing scripts
    │   ├── text2video.py
    │   ├── image2video.py
    │   └── animate.py
    ├── generate.py
    └── requirements_animate.txt
```

### Option 2: Inside DiffSynth-Studio (Alternative)
```
DiffSynth-Studio_VideoFM/
├── diffsynth/                          # Your main code
├── examples/
├── reference/                          # Reference code (gitignored)
│   └── wan2.2/                        # Original Wan2.2
│       ├── wan/
│       ├── generate.py
│       └── README.md
├── .gitignore                         # Add reference/ to gitignore
└── README.md
```

## What to Extract from Original Wan2.2

### 1. Multi-GPU Training (FSDP)

**Key Files to Reference:**
```
wan/distributed/
├── fsdp.py              # ⭐ Fully Sharded Data Parallel implementation
├── sequence_parallel.py # ⭐ Sequence parallel for long videos
├── ulysses.py           # ⭐ Ulysses attention for distributed
└── util.py              # Distributed utilities
```

**Key Features:**
- FSDP sharding for T5 and DiT models
- Gradient checkpointing strategies
- Mixed precision training optimizations
- Memory-efficient parameter management

**Integration Strategy:**
1. Study `wan/distributed/fsdp.py` for sharding patterns
2. Adapt to DiffSynth-Studio's model architecture
3. Test with your training pipeline in `examples/wanvideo/model_training/`

### 2. Animate Preprocessing Scripts

**Key Files to Extract:**
```
wan/modules/animate/preprocess/
├── __init__.py
├── preprocess_data.py       # ⭐ Main preprocessing pipeline
├── process_pipeline.py      # ⭐ End-to-end processing
├── pose2d.py                # ⭐ 2D pose extraction
├── pose2d_utils.py          # Pose utilities
├── human_visualization.py   # Visualization tools
├── retarget_pose.py         # Pose retargeting
├── sam_utils.py             # ⭐ SAM-2 integration for segmentation
├── video_predictor.py       # Video prediction
└── utils.py                 # General utilities
```

**Dependencies (from requirements_animate.txt):**
```bash
# Add to your requirements if using preprocessing
SAM-2  # For human segmentation
openai-whisper  # For audio processing (if needed for S2V)
decord  # For video I/O
onnxruntime  # For pose detection
pandas
matplotlib
```

**What These Scripts Do:**
1. **Pose Extraction** - Extract 2D/3D human pose from videos
2. **Face Detection** - Detect and track facial landmarks
3. **Segmentation** - Use SAM-2 to segment humans from background
4. **Retargeting** - Transfer motion from one character to another
5. **Visualization** - Preview pose/face overlays on videos

### 3. Distributed Training Patterns

**From `wan/text2video.py` (WanT2V class):**

```python
# Key patterns to adapt:

# 1. FSDP Model Initialization
from .distributed.fsdp import shard_model
shard_fn = partial(shard_model, device_id=device_id)
self.text_encoder = T5EncoderModel(
    ...
    shard_fn=shard_fn if t5_fsdp else None
)

# 2. Sequence Parallel Integration
from .distributed.sequence_parallel import sp_dit_forward, sp_attn_forward
# Replace forward methods for distributed inference

# 3. Memory-efficient Loading
if init_on_cpu:
    # Load model on CPU first
    model = WanModel(..., device=torch.device('cpu'))
    # Then shard and move to GPU
```

## Step-by-Step Integration Plan

### Phase 1: Clone and Organize (Now)

```bash
# Navigate to your workspace
cd ~/workspace

# Clone original Wan2.2 as reference
git clone https://github.com/Wan-Video/Wan2.2.git reference_wan2.2

# Mark it as read-only reference (optional)
cd reference_wan2.2
echo "# This is a READ-ONLY reference repository" > REFERENCE_ONLY.md
```

### Phase 2: Extract Preprocessing Scripts (Week 1-2)

**Create preprocessing module in DiffSynth-Studio:**
```
DiffSynth-Studio_VideoFM/
└── diffsynth/
    └── processors/
        └── animate/                    # New module
            ├── __init__.py
            ├── pose_extraction.py      # Adapted from original
            ├── face_detection.py       # Adapted from original
            ├── segmentation.py         # SAM-2 integration
            └── preprocessing_pipeline.py
```

**Action Items:**
1. Copy preprocessing scripts from `reference_wan2.2/wan/modules/animate/preprocess/`
2. Adapt to DiffSynth-Studio's structure
3. Update dependencies in `requirements.txt`
4. Test with sample videos
5. Document usage in `examples/wanvideo/README.md`

### Phase 3: Implement FSDP Support (Week 3-4)

**Create distributed training module:**
```
DiffSynth-Studio_VideoFM/
└── diffsynth/
    └── distributed/
        ├── __init__.py
        ├── xdit_context_parallel.py   # Existing
        ├── fsdp_wrapper.py            # New: Adapted from original
        └── multi_gpu_trainer.py       # New: Training utilities
```

**Action Items:**
1. Study `reference_wan2.2/wan/distributed/fsdp.py`
2. Create FSDP wrapper for DiffSynth-Studio models
3. Integrate with existing training scripts in `examples/wanvideo/model_training/`
4. Test on multi-GPU setup
5. Benchmark against single-GPU training

### Phase 4: Training Pipeline Enhancement (Week 5-6)

**Enhance existing training scripts:**
```
examples/wanvideo/model_training/
├── full/
│   ├── train_fsdp.sh              # New: Multi-GPU training
│   └── train_single_gpu.sh        # Existing
└── configs/
    └── fsdp_config.py             # New: FSDP configuration
```

## Code Snippets to Adapt

### 1. FSDP Sharding Pattern (from wan/distributed/fsdp.py)

```python
# Original Wan2.2 pattern:
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def shard_model(model, device_id):
    """Shard model using FSDP"""
    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={
            WanAttentionBlock,  # Your attention block class
        }
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=device_id,
        mixed_precision=...,
    )
    return model

# Adapt to DiffSynth-Studio:
# - Use your model classes from diffsynth/models/wan_video_dit.py
# - Integrate with your training loop
# - Add to diffsynth/distributed/fsdp_wrapper.py
```

### 2. Preprocessing Pipeline (from wan/modules/animate/preprocess/)

```python
# Original pattern for pose extraction:
from .pose2d import extract_pose
from .sam_utils import segment_human
from .video_predictor import predict_video

def preprocess_animate_video(video_path, output_dir):
    """Extract pose, face, and segmentation"""
    # 1. Extract pose keypoints
    pose_data = extract_pose(video_path)
    
    # 2. Segment human using SAM-2
    mask = segment_human(video_path)
    
    # 3. Extract face landmarks
    face_data = extract_face(video_path)
    
    # 4. Visualize and save
    save_preprocessing_results(
        pose_data, mask, face_data, output_dir
    )
    
    return pose_data, mask, face_data

# Adapt to DiffSynth-Studio:
# - Create diffsynth/processors/animate/preprocessing_pipeline.py
# - Use VideoData class for I/O consistency
# - Add CLI script in examples/wanvideo/preprocessing/
```

## Gitignore Setup

Add to `.gitignore`:
```gitignore
# Reference code (keep out of version control)
reference/
reference_*/

# Preprocessing outputs
data/preprocessed/
*.pose.pkl
*.face.pkl
*.mask.mp4
```

## Documentation to Create

### 1. Multi-GPU Training Guide
**Location:** `examples/wanvideo/MULTI_GPU_TRAINING.md`
- Setup instructions
- FSDP configuration
- Performance benchmarks
- Troubleshooting

### 2. Preprocessing Guide
**Location:** `examples/wanvideo/ANIMATE_PREPROCESSING.md`
- Installation of preprocessing dependencies
- Usage examples
- Input/output formats
- Visualization tools

## Quick Reference Commands

### Clone Original Wan2.2
```bash
cd ~/workspace
git clone https://github.com/Wan-Video/Wan2.2.git reference_wan2.2
```

### Extract Specific Files
```bash
# Copy preprocessing scripts
cp -r reference_wan2.2/wan/modules/animate/preprocess/ \
      DiffSynth-Studio_VideoFM/diffsynth/processors/animate/

# Copy distributed utilities
cp reference_wan2.2/wan/distributed/fsdp.py \
   DiffSynth-Studio_VideoFM/diffsynth/distributed/fsdp_wrapper.py
```

### Install Additional Dependencies
```bash
cd DiffSynth-Studio_VideoFM

# For preprocessing
pip install -e git+https://github.com/facebookresearch/sam2.git@main#egg=SAM-2
pip install onnxruntime decord

# For distributed training
pip install torch.distributed
```

## Timeline & Priorities

### Immediate (This Week)
- [x] Clone original Wan2.2 to `~/workspace/reference_wan2.2/`
- [ ] Review FSDP implementation patterns
- [ ] Review preprocessing scripts structure

### Short-term (Next 2 Weeks)
- [ ] Adapt preprocessing scripts to DiffSynth-Studio
- [ ] Test preprocessing on sample videos
- [ ] Document preprocessing pipeline

### Medium-term (Next Month)
- [ ] Implement FSDP wrapper for multi-GPU training
- [ ] Test on 2-4 GPU cluster
- [ ] Benchmark training speed improvements
- [ ] Create training guides

### Long-term (Next Quarter)
- [ ] Optimize preprocessing for production
- [ ] Scale to 8+ GPU training
- [ ] Integrate preprocessing into training pipeline
- [ ] Publish training results

## Key Differences to Handle

### 1. Model Loading
- **Original:** Direct checkpoint loading with paths
- **DiffSynth:** ModelConfig + from_pretrained() from ModelScope/HF
- **Solution:** Create hybrid loader that supports both methods

### 2. Data Format
- **Original:** Uses custom data structures
- **DiffSynth:** Uses VideoData class
- **Solution:** Write adapters between formats

### 3. Training Loop
- **Original:** Custom training loop in each task file
- **DiffSynth:** Unified training via trainers/
- **Solution:** Integrate FSDP into existing unified_dataset.py

## Best Practices

### DO ✅
- Keep original Wan2.2 code **read-only** as reference
- Adapt code to DiffSynth-Studio's architecture (don't copy-paste)
- Document all changes and adaptations
- Write tests for new features
- Version control your adaptations

### DON'T ❌
- Don't modify the reference repository
- Don't copy entire modules without understanding
- Don't create conflicting implementations
- Don't skip testing on small scale first

## Support Resources

### Original Wan2.2 Documentation
- GitHub: https://github.com/Wan-Video/Wan2.2
- Paper: https://arxiv.org/abs/2503.20314
- User Guide: https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y

### DiffSynth-Studio Documentation
- Your README: `README.md`
- Wan Examples: `examples/wanvideo/README.md`
- Training Examples: `examples/wanvideo/model_training/`

---
**Next Steps:** Clone the original Wan2.2 repository and review the two key areas (FSDP + preprocessing).

