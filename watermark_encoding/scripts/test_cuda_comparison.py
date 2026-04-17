import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_INFERENCE_STEPS = 30
SEED = 42

LORA_PATHS = [
    "watermark_encoding/models/watermark_s1_alpha1.0_rank4_noxattn/watermark_s1_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s2_alpha1.0_rank4_noxattn/watermark_s2_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s3_alpha1.0_rank4_noxattn/watermark_s3_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s4_alpha1.0_rank4_noxattn/watermark_s4_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s5_alpha1.0_rank4_noxattn/watermark_s5_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s6_alpha1.0_rank4_noxattn/watermark_s6_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s7_alpha1.0_rank4_noxattn/watermark_s7_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s8_alpha1.0_rank4_noxattn/watermark_s8_alpha1.0_rank4_noxattn_last.safetensors",
]

prompts = [
    "a photo of a mountain landscape",
    "a portrait of a person outdoors",
    "a dog running in a park",
]

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

# generate baselines first
for i, prompt in enumerate(prompts):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]
    image.save(f"watermark_encoding/data/cuda_baseline_p0{i}.png")
    print(f"Saved cuda_baseline_p0{i}.png")

# load loras and generate at scale 0.3
adapter_names = [f"s{i+1}" for i in range(8)]
for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)

pipe.set_adapters(adapter_names, adapter_weights=[-0.3] * 8)

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]
    image.save(f"watermark_encoding/data/cuda_scale03_neg_p0{i}.png")
    print(f"Saved cuda_scale03_neg_p0{i}.png")

print("Done.")
