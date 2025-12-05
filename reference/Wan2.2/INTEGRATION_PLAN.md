# Integration Plan: Original Wan2.2 Reference

## Current Status

**DiffSynth-Studio is fully functional** for:
- ✅ Multi-GPU training (DeepSpeed Zero-2)
- ✅ Multi-node training (SLURM + accelerate)
- ✅ LoRA and full fine-tuning
- ✅ Inference with VRAM management

**Original Wan2.2 provides reference for**:
- ⭐ Animate preprocessing scripts
- ⭐ S2V audio processing utilities
- 📚 FSDP implementation (alternative approach)

## What to Extract

### Priority 1: Animate Preprocessing

**Source**: `reference/Wan2.2/wan/modules/animate/preprocess/`

**Target**: Create `diffsynth/processors/animate/`

| File | Purpose | Action |
|------|---------|--------|
| `process_pipepline.py` | Main pipeline | Adapt |
| `pose2d.py` | ViTPose extraction | Copy + adapt |
| `sam_utils.py` | SAM-2 segmentation | Copy + adapt |
| `retarget_pose.py` | Pose retargeting | Copy + adapt |
| `human_visualization.py` | Pose visualization | Copy |

**Required Dependencies**:
```bash
pip install sam2 decord onnxruntime moviepy
# Optional for retargeting:
pip install diffusers[flux]  # FLUX-Kontext
```

### Priority 2: S2V Audio Utils (If Using S2V)

**Source**: `reference/Wan2.2/wan/modules/s2v/`

Useful for Speech-to-Video preprocessing:
- `audio_encoder.py` - Audio feature extraction
- `audio_utils.py` - Audio processing utilities

## Integration Steps

### Step 1: Create Animate Processor Module

```
diffsynth/
└── processors/
    └── animate/
        ├── __init__.py
        ├── pose_extraction.py    # From pose2d.py
        ├── segmentation.py       # From sam_utils.py
        ├── retarget.py           # From retarget_pose.py
        └── pipeline.py           # From process_pipepline.py
```

### Step 2: Add CLI Script

```
examples/wanvideo/preprocessing/
├── preprocess_animate.py
└── README.md
```

### Step 3: Update Dependencies

Add to `requirements.txt`:
```
sam2>=1.0
decord>=0.6.0
onnxruntime>=1.16.0
```

## Directory Structure

```
~/workspace/DiffSynth-Studio_VideoFM/
├── diffsynth/                    # Main codebase
│   ├── models/                   # Model implementations
│   ├── pipelines/                # Inference pipelines
│   ├── trainers/                 # Training infrastructure
│   ├── distributed/              # xDiT context parallel
│   └── processors/               # Post-processing tools
│       └── animate/              # [TO CREATE] From Wan2.2
│
├── examples/wanvideo/
│   ├── model_inference/          # Inference scripts
│   ├── model_training/
│   │   ├── t_lora/               # LoRA training (multi-node ready)
│   │   └── t_sft/                # Full fine-tuning (multi-node ready)
│   └── preprocessing/            # [TO CREATE] Animate prep
│
└── reference/Wan2.2/             # READ-ONLY reference
    └── wan/
        ├── modules/animate/preprocess/  # ⭐ Source for preprocessing
        ├── modules/s2v/                 # Audio utilities
        └── distributed/                 # FSDP reference
```

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| Done | Multi-GPU/node training | ✅ Complete |
| Done | LoRA + Full fine-tuning | ✅ Complete |
| Done | Inference pipeline | ✅ Complete |
| Todo | Port Animate preprocessing | 🔲 Not started |
| Todo | S2V audio utilities | 🔲 Not started |

## Notes

- **Do NOT modify** files in `reference/Wan2.2/`
- Keep it as read-only reference
- Adapt code to DiffSynth-Studio's architecture when porting
- Test preprocessing with sample videos before production use
