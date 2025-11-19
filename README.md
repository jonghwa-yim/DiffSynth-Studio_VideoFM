# DiffSynth-Studio (Video Generation Focus)

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/) 

## Introduction

Welcome to the magic world of Diffusion models! DiffSynth-Studio is an open-source Diffusion model engine developed and maintained by [ModelScope](https://www.modelscope.cn/) team. We aim to foster technical innovation through framework development, bring together the power of the open-source community, and explore the limits of generative models!

**This fork focuses exclusively on video generation models, particularly the Wan Video series and related technologies.**

DiffSynth currently includes two open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Focused on aggressive technical exploration, for academia, providing support for more cutting-edge model capabilities.
* [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine): Focused on stable model deployment, for industry, offering higher computing performance and more stable features.

## Installation

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

<details>
<summary>Other installation methods</summary>

Install from PyPI (version updates may be delayed; for latest features, install from source)

```
pip install diffsynth
```

If you meet problems during installation, they might be caused by upstream dependencies. Please check the docs of these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

</details>

## Video Generation Framework

DiffSynth-Studio redesigns the inference and training pipelines for video diffusion models, enabling efficient memory management and flexible model training.

### Wan Video Series

Detail page: [./examples/wanvideo/](./examples/wanvideo/)

https://github.com/user-attachments/assets/1d66ae74-3b02-40a9-acc3-ea95fc039314

<details>

<summary>Quick Start</summary>

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

video = pipe(
    prompt="A documentary photography style scene: a lively puppy rapidly running on green grass. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. Sunlight shines on its body, making the fur appear soft and shiny. The background is an open field with occasional wildflowers, and faint blue sky and clouds in the distance. Strong sense of perspective captures the motion of the puppy and the vitality of the surrounding grass. Mid-shot side-moving view.",
    negative_prompt="Bright colors, overexposed, static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)
```

</details>

<details>

<summary>Model Overview</summary>

| Model ID | Extra Parameters | Inference | Full Training | Validate After Full Training | LoRA Training | Validate After LoRA Training |
|-|-|-|-|-|-|-|
|[Wan-AI/Wan2.2-Animate-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)|`input_image`, `animate_pose_video`, `animate_face_video`, `animate_inpaint_video`, `animate_mask_video`|[code](./examples/wanvideo/model_inference/Wan2.2-Animate-14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-Animate-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-Animate-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-14B.py)|
|[Wan-AI/Wan2.2-S2V-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)|`input_image`, `input_audio`, `audio_sample_rate`, `s2v_pose_video`|[code](./examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-S2V-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-S2V-14B.py)|
|[Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.2-I2V-A14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-I2V-A14B.py)|
|[Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)||[code](./examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-T2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-T2V-A14B.py)|
|[Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-TI2V-5B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py)|
|[Wan-AI/Wan2.2-VACE-Fun-A14B](https://www.modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-VACE-Fun-A14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-VACE-Fun-A14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-VACE-Fun-A14B.py)|
|[PAI/Wan2.2-Fun-A14B-InP](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.2-Fun-A14B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-InP.py)|
|[PAI/Wan2.2-Fun-A14B-Control](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control)|`control_video`, `reference_image`|[code](./examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control.py)|
|[PAI/Wan2.2-Fun-A14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera)|`control_camera_video`, `input_image`|[code](./examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control-Camera.py)|
|[Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)||[code](./examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-T2V-1.3B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B.py)|
|[Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)||[code](./examples/wanvideo/model_inference/Wan2.1-T2V-14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-T2V-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-T2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-14B.py)|
|[Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-I2V-14B-480P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-I2V-14B-480P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-480P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-480P.py)|
|[Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-720P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-720P.py)|
|[Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-FLF2V-14B-720P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-FLF2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-FLF2V-14B-720P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-FLF2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py)|
|[PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)|`control_video`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-14B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-InP.py)|
|[PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)|`control_video`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-14B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)|`control_video`, `reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control)|`control_video`, `reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera)|`control_camera_video`, `input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)|`control_camera_video`, `input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|
|[iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-1.3B-Preview.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B-Preview.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B-Preview.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-Preview.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B-Preview.py)|
|[Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py)|
|[Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-14B.py)|
|[DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)|`motion_bucket_id`|[code](./examples/wanvideo/model_inference/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py)|
|[krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video)||[code](./examples/wanvideo/model_inference/krea-realtime-video.py)|[code](./examples/wanvideo/model_training/full/krea-realtime-video.sh)|[code](./examples/wanvideo/model_training/validate_full/krea-realtime-video.py)|[code](./examples/wanvideo/model_training/lora/krea-realtime-video.sh)|[code](./examples/wanvideo/model_training/validate_lora/krea-realtime-video.py)|
|[meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video)|`longcat_video`|[code](./examples/wanvideo/model_inference/LongCat-Video.py)|[code](./examples/wanvideo/model_training/full/LongCat-Video.sh)|[code](./examples/wanvideo/model_training/validate_full/LongCat-Video.py)|[code](./examples/wanvideo/model_training/lora/LongCat-Video.sh)|[code](./examples/wanvideo/model_training/validate_lora/LongCat-Video.py)|
|[ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B)|`vap_video`, `vap_prompt`|[code](./examples/wanvideo/model_inference/Video-As-Prompt-Wan2.1-14B.py)|[code](./examples/wanvideo/model_training/full/Video-As-Prompt-Wan2.1-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Video-As-Prompt-Wan2.1-14B.py)|[code](./examples/wanvideo/model_training/lora/Video-As-Prompt-Wan2.1-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Video-As-Prompt-Wan2.1-14B.py)|

</details>

### Other Video Generation Models

<details>
<summary>HunyuanVideo</summary>

Detail page: [./examples/HunyuanVideo/](./examples/HunyuanVideo/)

https://github.com/user-attachments/assets/48dd24bb-0cc6-40d2-88c3-10feed3267e9  

HunyuanVideo is a state-of-the-art video generation model featuring:
- Advanced 3D RoPE (Rotary Position Embeddings) for video
- Dynamic resolution bucketing for flexible aspect ratios
- Dual text encoder architecture (CLIP + LLM)
- Advanced VRAM management (6GB-80GB configurations)

</details>

## Video Enhancement Tools

<details>
<summary>Frame Interpolation & Deflickering</summary>

We provide powerful video post-processing tools:

- **RIFE**: State-of-the-art frame interpolation to increase FPS
- **FastBlend**: Video deflickering algorithm for temporal consistency

These tools can be applied to any video generation output to improve quality.

Detail page: [FastBlend Paper](https://arxiv.org/abs/2311.09265)

</details>

<details>
<summary>Image Quality Assessment Models</summary>

We have integrated a series of image quality assessment models. These models can be used for evaluating video generation models, alignment training, and similar tasks.

Detail page: [./examples/image_quality_metric/](./examples/image_quality_metric/)

* [ImageReward](https://github.com/THUDM/ImageReward)
* [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* [PickScore](https://github.com/yuvalkirstain/pickscore)
* [CLIP](https://github.com/openai/CLIP)
* [HPSv2](https://github.com/tgxs002/HPSv2)
* [HPSv2.1](https://github.com/tgxs002/HPSv2)
* [MPS](https://github.com/Kwai-Kolors/MPS)

</details>

## Research Contributions

<details>
<summary>ExVideo: Extended Training for Video Generation Models</summary>

- Project Page: [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- Paper: [ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- Code Example: [./examples/ExVideo/](./examples/ExVideo/)
- Model: [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

ExVideo is a post-tuning technique for extending video length (e.g., 25→128 frames) using parameter-efficient LoRA training. This technique can be applied to Wan models to generate longer videos.

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>

## Inference Acceleration

<details>
<summary>TeaCache</summary>

TeaCache is a powerful acceleration technique for video generation that can significantly speed up inference without quality loss.

Detail page: [./examples/TeaCache/](./examples/TeaCache/)

Compatible with:
- Wan Video models
- HunyuanVideo models

</details>

## Update History (Video Models)

- **November 4, 2025**: We support [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) model, which is trained on Wan 2.1 and enables motion generation conditioned on reference videos.

- **October 30, 2025**: We support [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) model, which enables text-to-video, image-to-video, and video continuation capabilities. This model adopts Wan's framework for both inference and training in this project.

- **October 27, 2025**: We support [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) model, further expanding Wan's ecosystem.

- **August 28, 2025**: We support Wan2.2-S2V, an audio-driven cinematic video generation model open-sourced by Alibaba. See [./examples/wanvideo/](./examples/wanvideo/).

- **July 28, 2025**: With the open-sourcing of Wan 2.2, we immediately provided comprehensive support, including low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, full training. See [./examples/wanvideo/](./examples/wanvideo/).

- **March 13, 2025**: We support HunyuanVideo-I2V, the image-to-video generation version of HunyuanVideo open-sourced by Tencent. Please refer to [./examples/HunyuanVideo/](./examples/HunyuanVideo/) for more details.

- **February 25, 2025**: We support Wan-Video, a collection of SOTA video synthesis models open-sourced by Alibaba. See [./examples/wanvideo/](./examples/wanvideo/).

- **December 19, 2024**: We implement advanced VRAM management for HunyuanVideo, making it possible to generate videos at a resolution of 129x720x1280 using 24GB of VRAM, or at 129x512x384 resolution with just 6GB of VRAM. Please refer to [./examples/HunyuanVideo/](./examples/HunyuanVideo/) for more details.

- **October 8, 2024**: We release the extended LoRA based on CogVideoX-5B and ExVideo. You can download this model from [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) or [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1).

- **June 21, 2024**: We propose ExVideo, a post-tuning technique aimed at enhancing the capability of video generation models. We have extended Stable Video Diffusion to achieve the generation of long videos up to 128 frames.
  - [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - Source code is released in this repo. See [`examples/ExVideo`](./examples/ExVideo/).
  - Models are released on [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) and [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1).
  - Technical report is released on [arXiv](https://arxiv.org/abs/2406.14130).

- **Nov 15, 2023**: We propose FastBlend, a powerful video deflickering algorithm.
  - The sd-webui extension is released on [GitHub](https://github.com/Artiprocher/sd-webui-fastblend).
  - Demo videos are shown on Bilibili, including three tasks.
    - [Video deflickering](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [Video interpolation](https://www.bilibili.com/video/BV1Lw411m71p)
    - [Image-driven video rendering](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - The technical report is released on [arXiv](https://arxiv.org/abs/2311.09265).

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{diffsynth2024,
  title={DiffSynth-Studio: Video Generation Framework},
  author={ModelScope Team},
  journal={GitHub repository},
  year={2024},
  url={https://github.com/modelscope/DiffSynth-Studio}
}
```
