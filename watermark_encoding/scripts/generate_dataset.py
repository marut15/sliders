import torch
import json
import os
from pathlib import Path
from diffusers import DiffusionPipeline
from PIL import Image

# --- CONFIG ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
DTYPE = torch.bfloat16
SCALE = 0.75  # slider strength — subtle enough to be invisible, strong enough to detect
NUM_INFERENCE_STEPS = 30
SEED = 42

OUTPUT_DIR = Path("watermark_encoding/data/images")
METADATA_PATH = Path("watermark_encoding/data/metadata.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

PROMPTS = [
    "a photo of a mountain landscape",
    "a portrait of a person outdoors",
    "a city street at noon",
    "a bowl of fruit on a wooden table",
    "a dog running in a park",
    "a beach at sunset",
    "a forest path in autumn",
    "a modern kitchen interior",
    "a cat sitting on a windowsill",
    "a snowy village at night",
]

# --- LOAD MODEL ---
print("Loading SDXL pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

# --- LOAD ALL LORAS ---
print("Loading LoRAs...")
adapter_names = [f"s{i+1}" for i in range(8)]
for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)
    print(f"Loaded slider {i+1}")

# --- GENERATE ---
metadata = []
total = 256 * len(PROMPTS)
count = 0

for id_int in range(256):
    bits = [int(b) for b in f"{id_int:08b}"]
    alphas = [SCALE if b == 1 else -SCALE for b in bits]

    pipe.set_adapters(adapter_names, adapter_weights=alphas)

    for prompt_idx, prompt in enumerate(PROMPTS):
        generator = torch.Generator(device=DEVICE).manual_seed(SEED + prompt_idx)
        image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]

        filename = f"id{id_int:03d}_p{prompt_idx:02d}.png"
        image.save(OUTPUT_DIR / filename)

        metadata.append({
            "file": filename,
            "id_int": id_int,
            "bits": bits,
            "prompt": prompt,
        })

        count += 1
        if count % 50 == 0:
            print(f"Progress: {count}/{total}")

# --- SAVE METADATA ---
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done. Generated {count} images.")
print(f"Metadata saved to {METADATA_PATH}")
