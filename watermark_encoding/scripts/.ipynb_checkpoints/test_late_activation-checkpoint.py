import torch
from diffusers import DiffusionPipeline
from diffusers.callbacks import PipelineCallback

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

adapter_names = [f"s{i+1}" for i in range(8)]
SCALE = 0.3
ACTIVATE_AT_STEP = 25  # activate LoRAs only after step 20 out of 30

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)

# start with all scales at 0
pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)

def make_callback(target_alphas, activate_at_step):
    def callback(pipe, step, timestep, kwargs):
        if step == activate_at_step:
            pipe.set_adapters(adapter_names, adapter_weights=target_alphas)
        return kwargs
    return callback

for i, prompt in enumerate(prompts):
    alphas = [-SCALE] * 8  # all negative = ID 0
    cb = make_callback(alphas, ACTIVATE_AT_STEP)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
    image = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        callback_on_step_end=cb,
    ).images[0]
    image.save(f"watermark_encoding/data/test_late2_p0{i}.png")
    print(f"Saved test_late_p0{i}.png")

print("Done.")
