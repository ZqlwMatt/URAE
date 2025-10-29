import os
import torch
from huggingface_hub import hf_hub_download
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import random

bfl_repo="black-forest-labs/FLUX.1-dev"
device = torch.device('cuda')
dtype = torch.bfloat16
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype)
pipe.scheduler.config.use_dynamic_shifting = False
pipe.scheduler.config.time_shift = 10
pipe = pipe.to(device)

if not os.path.exists('ckpt/urae_2k_adapter.safetensors'):
    hf_hub_download(repo_id="Huage001/URAE", filename='urae_2k_adapter.safetensors', local_dir='ckpt', local_dir_use_symlinks=False)
pipe.load_lora_weights("ckpt/urae_2k_adapter.safetensors")

prompts = [
    # "A serene woman in a flowing azure dress, gracefully perched on a sunlit cliff overlooking a tranquil sea, her hair gently tousled by the breeze. The scene is infused with a sense of peace, evoking a dreamlike atmosphere, reminiscent of Impressionist paintings.",
    "Photo of a lowpoly fantasy house from warcraft game, lawn.",
    "A photo of a demon knight, flame in eyes, warcraft style.",
    "Photo of link in the legend of zelda, photo-realistic, unreal 5.",
]
height = 2048
width = 2048

for i, prompt in enumerate(prompts):
    seed = random.randint(0, 10000)
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(seed),
        ntk_factor=10,
        proportional_attention=True
    ).images[0]

    # Save image
    os.makedirs('output', exist_ok=True)
    image.save(f'output/urae_2k_adapter_dev_{i}_{seed}.png')
