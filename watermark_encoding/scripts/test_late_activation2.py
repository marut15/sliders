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
    "a beach at sunset",
    "a forest path in autumn",
    "a modern kitchen interior",
    "a snowy village at night",
    "a city street at noon",
]

adapter_names = [f"s{i+1}" for i in range(8)]
SCALE = 0.3
ACTIVATE_AT_STEP = 25

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)

pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)

def make_callback(target_alphas, activate_at_step):
    def callback(pipe, step, timestep, kwargs):
        if step == activate_at_step:
            pipe.set_adapters(adapter_names, adapter_weights=target_alphas)
        return kwargs
    return callback

for i, prompt in enumerate(prompts):
    # save baseline
    pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0]
    image.save(f"watermark_encoding/data/test2_baseline_p{i:02d}.png")
    print(f"Saved test2_baseline_p{i:02d}.png")

    # save watermarked
    alphas = [-SCALE] * 8
    cb = make_callback(alphas, ACTIVATE_AT_STEP)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        callback_on_step_end=cb,
    ).images[0]
    image.save(f"watermark_encoding/data/test2_watermarked_p{i:02d}.png")
    print(f"Saved test2_watermarked_p{i:02d}.png")

print("Done.")
