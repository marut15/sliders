import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "mps"
DTYPE = torch.float16
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

prompt = "a photo of a mountain landscape"

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

adapter_names = [f"s{i+1}" for i in range(8)]
for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)

# test all-negative direction at three scales
for scale in [0.5, 0.3, 0.2]:
    alphas = [-scale] * 8
    pipe.set_adapters(adapter_names, adapter_weights=alphas)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]
    image.save(f"watermark_encoding/data/test_scale_{str(scale).replace('.','')}_neg.png")
    print(f"Saved scale {scale} negative")

print("Done.")
