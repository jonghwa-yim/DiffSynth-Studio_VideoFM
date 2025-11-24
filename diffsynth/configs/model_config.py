from typing_extensions import Literal, TypeAlias

from ..models.svd_image_encoder import SVDImageEncoder
from ..models.svd_unet import SVDUNet
from ..models.svd_vae_decoder import SVDVAEDecoder
from ..models.svd_vae_encoder import SVDVAEEncoder

from ..models.hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from ..models.hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder

from ..extensions.RIFE import IFNet

from ..models.hunyuan_video_dit import HunyuanVideoDiT

from ..models.wan_video_dit import WanModel
from ..models.wan_video_dit_s2v import WanS2VModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel

from ..models.longcat_video_dit import LongCatVideoTransformer3DModel

model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "2a07abce74b4bdc696b76254ab474da6", ["svd_image_encoder", "svd_unet", "svd_vae_decoder", "svd_vae_encoder"], [SVDImageEncoder, SVDUNet, SVDVAEDecoder, SVDVAEEncoder], "civitai"),
    (None, "9b9313d104ac4df27991352fec013fd4", ["rife"], [IFNet], "civitai"),
    (None, "aeb82dce778a03dcb4d726cb03f3c43f", ["hunyuan_video_vae_decoder", "hunyuan_video_vae_encoder"], [HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder], "diffusers"),
    (None, "b9588f02e78f5ccafc9d7c0294e46308", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    (None, "84ef4bd4757f60e906b54aa6a7815dc6", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5ec04e02b42d2580483ad69f4e76346a", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5f90e66a0672219f12d9a626c8c21f61", ["wan_video_dit", "wan_video_vap"], [WanModel,MotWanModel], "diffusers"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    (None, "8b27900f680d7251ce44e2dc8ae1ffef", ["wan_video_dit"], [LongCatVideoTransformer3DModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
    (None, "31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),
]
huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("CogVideoXTransformer3DModel", "diffsynth.models.cog_dit", "cog_dit", "CogDiT"),
    ("LlamaForCausalLM", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoLLMEncoder"),
    ("LlavaForConditionalGeneration", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoMLLMEncoder"),
]
patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
    ("9a4ab6869ac9b7d6e31f9854e397c867", ["svd_unet"], [SVDUNet], {"add_positional_conv": 128}),
]

preset_models_on_huggingface = {
    # Stable Video Diffusion
    "stable-video-diffusion-img2vid-xt": [
        ("stabilityai/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
    # RIFE
    "RIFE": [
        ("AlexWortega/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # CogVideo
    "CogVideoX-5B": [
        ("THUDM/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
    ],
}

preset_models_on_modelscope = {
    # Stable Video Diffusion
    "stable-video-diffusion-img2vid-xt": [
        ("AI-ModelScope/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    # ExVideo
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-CogVideoX-LoRA-129f-v1": [
        ("ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1", "ExVideo-CogVideoX-LoRA-129f-v1.safetensors", "models/lora"),
    ],
    # RIFE
    "RIFE": [
        ("AI-ModelScope/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # CogVideo
    "CogVideoX-5B": {
        "file_list": [
            ("ZhipuAI/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
        ],
        "load_path": [
            "models/CogVideo/CogVideoX-5b/text_encoder",
            "models/CogVideo/CogVideoX-5b/transformer",
            "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
        ],
    },
    # HunyuanVideo
    "HunyuanVideo":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideoI2V":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideoI2V/text_encoder"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00001-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00002-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00003-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00004-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "config.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model.safetensors.index.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/vae/pytorch_model.pt", "models/HunyuanVideoI2V/vae"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideoI2V/transformers")
        ],
        "load_path": [
            "models/HunyuanVideoI2V/text_encoder/model.safetensors",
            "models/HunyuanVideoI2V/text_encoder_2",
            "models/HunyuanVideoI2V/vae/pytorch_model.pt",
            "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideo-fp8":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("DiffSynth-Studio/HunyuanVideo-safetensors", "model.fp8.safetensors", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/model.fp8.safetensors"
        ],
    },
}

Preset_model_id: TypeAlias = Literal[
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1",
    "ExVideo-CogVideoX-LoRA-129f-v1",
    "RIFE",
    "CogVideoX-5B",
    "HunyuanVideo",
    "HunyuanVideo-fp8",
    "HunyuanVideoI2V",
]
