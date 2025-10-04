Runpod Serverless - Inpaint Anything (LaMa | Stable Diffusion)

Env vars
- LAMA_CONFIG=./lama/configs/prediction/default.yaml
- LAMA_CKPT=/workspace/pretrained_models/big-lama
- SD_MODEL_ID=stabilityai/stable-diffusion-2-inpainting

Payload (SD)
{
  "input": {
    "backend": "sd",
    "image_b64": "...",
    "mask_b64": "...",
    "prompt": "clean background, manga style",
    "negative_prompt": "blur, smear, artifacts",
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "seed": 42
  }
}

Payload (LaMa)
{
  "input": {
    "backend": "lama",
    "image_b64": "...",
    "mask_b64": "...",
    "invert_mask": false
  }
}

