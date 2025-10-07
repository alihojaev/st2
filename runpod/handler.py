import os
import base64
import io
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import runpod


_model = None
_device = "cuda"
_sd_pipe = None
_lama_inpaint_fn = None  # set lazily to avoid heavy imports at startup


def init_model():
    global _model, _device, _lama_inpaint_fn
    if _model is not None:
        return
    lama_config = os.environ.get("LAMA_CONFIG", "./lama/configs/prediction/default.yaml")
    lama_ckpt = os.environ.get("LAMA_CKPT", "./pretrained_models/big-lama")
    if not os.path.exists(lama_ckpt):
        raise RuntimeError("LAMA_CKPT directory not found. Set env LAMA_CKPT to LaMa checkpoint folder.")
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _device = "cpu"
    # Lazy import to prevent Hydra/PL from loading at startup unless needed
    from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
    _lama_inpaint_fn = inpaint_img_with_builded_lama
    _model = build_lama_model(lama_config, lama_ckpt, device=_device)


def init_sd():
    global _sd_pipe, _device
    if _sd_pipe is not None:
        return
    from diffusers import AutoPipelineForInpainting
    import torch
    model_id = os.environ.get("SD_MODEL_ID", "stabilityai/stable-diffusion-2-inpainting")
    torch_dtype = torch.float16 if (_device == "cuda" and torch.cuda.is_available()) else torch.float32
    _sd_pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch_dtype)
    if _device == "cuda" and torch.cuda.is_available():
        _sd_pipe = _sd_pipe.to("cuda")
    else:
        _sd_pipe = _sd_pipe.to("cpu")
    try:
        _sd_pipe.enable_attention_slicing("max")
    except Exception:
        pass


def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def _b64_to_image(b64: str) -> Image.Image:
    return _bytes_to_image(base64.b64decode(b64))


def _pil_to_np(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def _ensure_mask_2d(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.uint8)
    threshold = 127
    return (mask_arr > threshold).astype(np.uint8) * 255


def _image_to_b64(img_arr: np.ndarray, format_: str = "PNG") -> str:
    img = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format_)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # event expects: { "input": { "image_b64": str, "mask_b64": str, "invert_mask": bool, "backend": "lama"|"sd", "prompt": str, ... } }
    try:
        payload = event.get("input") or {}
        image_b64 = payload.get("image_b64")
        mask_b64 = payload.get("mask_b64")
        invert_mask = bool(payload.get("invert_mask", False))
        backend = str(payload.get("backend", "lama")).lower()
        prompt: Optional[str] = payload.get("prompt")
        negative_prompt: Optional[str] = payload.get("negative_prompt")
        guidance_scale: float = float(payload.get("guidance_scale", 7.5))
        num_inference_steps: int = int(payload.get("num_inference_steps", 30))
        seed = payload.get("seed")
        sd_max_side = payload.get("sd_max_side")
        if not image_b64 or not mask_b64:
            return {"error": "image_b64 and mask_b64 are required"}
        image_pil = _b64_to_image(image_b64)
        mask_pil = _b64_to_image(mask_b64)
        img_arr = _pil_to_np(image_pil)
        mask_arr = _pil_to_np(mask_pil)
        mask_arr = _ensure_mask_2d(mask_arr)
        if invert_mask:
            mask_arr = 255 - mask_arr
        if backend == "sd":
            init_sd()
            # Convert to PIL for diffusers
            image = Image.fromarray(img_arr.astype(np.uint8))
            mask = Image.fromarray(((mask_arr > 0).astype(np.uint8) * 255), mode="L")
            orig_w, orig_h = image.width, image.height

            # Compute working size for SD: optional downscale and enforce multiples of 8
            work_w, work_h = orig_w, orig_h
            try:
                if sd_max_side:
                    ratio = float(sd_max_side) / float(max(orig_w, orig_h))
                    ratio = ratio if ratio < 1.0 else 1.0
                    work_w = int(orig_w * ratio)
                    work_h = int(orig_h * ratio)
            except Exception:
                pass
            # Enforce divisibility by 8
            work_w = max(8, (work_w // 8) * 8)
            work_h = max(8, (work_h // 8) * 8)
            if (work_w, work_h) != (orig_w, orig_h):
                image = image.resize((work_w, work_h), Image.LANCZOS)
                mask = mask.resize((work_w, work_h), Image.NEAREST)
            import torch
            generator = torch.Generator(device=_device).manual_seed(int(seed)) if seed is not None else None
            if _device == "cuda" and torch.cuda.is_available():
                with torch.autocast("cuda"):
                    out = _sd_pipe(
                        prompt=prompt or "",
                        negative_prompt=negative_prompt,
                        image=image,
                        mask_image=mask,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=work_w,
                        height=work_h,
                    ).images[0]
            else:
                out = _sd_pipe(
                    prompt=prompt or "",
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    width=work_w,
                    height=work_h,
                ).images[0]
            out_arr = np.array(out)
            if out_arr.shape[0] != img_arr.shape[0] or out_arr.shape[1] != img_arr.shape[1]:
                out_arr = np.array(Image.fromarray(out_arr).resize((img_arr.shape[1], img_arr.shape[0]), Image.LANCZOS))
            return {"image_b64": _image_to_b64(out_arr)}
        else:
            # Lazy initialize LaMa only when requested
            init_model()
            result = _lama_inpaint_fn(_model, img_arr, mask_arr, device=_device)
            return {"image_b64": _image_to_b64(result)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


