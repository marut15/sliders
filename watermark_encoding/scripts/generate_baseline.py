import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "mps"
DTYPE = torch.float16
NUM_INFERENCE_STEPS = 30
SEED = 42

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

prompts = [
    "a photo of a mountain landscape",
    "a portrait of a person outdoors",
    "a dog running in a park",
]

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]
    image.save(f"watermark_encoding/data/baseline_p0{i}.png")
    print(f"Saved baseline_p0{i}.png")

print("Done.")
