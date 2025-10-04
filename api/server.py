import os
import base64
import io
import threading
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama


_model_lock = threading.Lock()
_model = None
_device = "cuda"
_sd_pipe = None


def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _load_model_once():
    global _model, _device
    with _model_lock:
        if _model is not None:
            return
        lama_config = _get_env("LAMA_CONFIG", "./lama/configs/prediction/default.yaml")
        lama_ckpt = _get_env("LAMA_CKPT", "./pretrained_models/big-lama")
        if not os.path.exists(lama_ckpt):
            # Fallback to OpenCV inpaint if checkpoints are not provided.
            _model = "cv_inpaint"
            _device = "cpu"
            return
        # Auto device
        try:
            import torch  # lazy import for startup
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
        _model = build_lama_model(lama_config, lama_ckpt, device=_device)


def _load_sd_pipe_once():
    global _sd_pipe, _device
    with _model_lock:
        if _sd_pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        model_id = _get_env("SD_MODEL_ID", "runwayml/stable-diffusion-inpainting")
        torch_dtype = torch.float16 if (_device == "cuda" and torch.cuda.is_available()) else torch.float32
        _sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        if _device == "cuda" and torch.cuda.is_available():
            _sd_pipe = _sd_pipe.to("cuda")
        else:
            _sd_pipe = _sd_pipe.to("cpu")
        try:
            _sd_pipe.enable_attention_slicing("max")
        except Exception:
            pass


def _pil_to_np(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def _ensure_mask_2d(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        # convert to grayscale by taking single channel if it is binary/monochrome
        mask_arr = mask_arr[:, :, 0]
    # binarize
    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.uint8)
    threshold = 127
    mask_arr = (mask_arr > threshold).astype(np.uint8) * 255
    return mask_arr


def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def _b64_to_image(b64: str) -> Image.Image:
    return _bytes_to_image(base64.b64decode(b64))


def _image_to_b64(img_arr: np.ndarray, format_: str = "PNG") -> str:
    img = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format_)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cv_inpaint(img_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    # OpenCV inpaint expects BGR image and mask with 0/255, non-zero = inpaint
    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.uint8)
    mask_bin = (mask_arr > 0).astype(np.uint8) * 255
    bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    # Radius=3 usually ok for text; method TELEA gives smooth results
    out_bgr = cv2.inpaint(bgr, mask_bin, 3, cv2.INPAINT_TELEA)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


def _sd_inpaint(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
) -> np.ndarray:
    image_pil = Image.fromarray(img_arr.astype(np.uint8))
    mask_bin = (mask_arr > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode="L")
    orig_w, orig_h = image_pil.width, image_pil.height

    generator = None
    try:
        import torch
        if seed is not None:
            generator = torch.Generator(device=_device).manual_seed(int(seed))
    except Exception:
        generator = None

    result = _sd_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        width=orig_w,
        height=orig_h,
    ).images[0]

    out = np.array(result)
    if out.shape[0] != img_arr.shape[0] or out.shape[1] != img_arr.shape[1]:
        out = np.array(Image.fromarray(out).resize((img_arr.shape[1], img_arr.shape[0]), Image.LANCZOS))
    return out


app = FastAPI(title="Inpaint Anything API", version="0.1.0")


@app.on_event("startup")
def _startup():
    _load_model_once()


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(None),
    mask: UploadFile = File(None),
    image_b64: Optional[str] = Form(None),
    mask_b64: Optional[str] = Form(None),
    invert_mask: bool = Form(False),
    backend: str = Form("lama"),  # lama | sd | cv
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(7.5),
    num_inference_steps: Optional[int] = Form(30),
    seed: Optional[int] = Form(None),
):
    try:
        _load_model_once()

        if image is None and image_b64 is None:
            return JSONResponse(status_code=400, content={"error": "Provide image (file) or image_b64."})
        if mask is None and mask_b64 is None:
            return JSONResponse(status_code=400, content={"error": "Provide mask (file) or mask_b64."})

        if image is not None:
            image_pil = _bytes_to_image(await image.read())
        else:
            image_pil = _b64_to_image(image_b64)

        if mask is not None:
            mask_pil = _bytes_to_image(await mask.read())
        else:
            mask_pil = _b64_to_image(mask_b64)

        img_arr = _pil_to_np(image_pil)
        mask_arr = _pil_to_np(mask_pil)
        mask_arr = _ensure_mask_2d(mask_arr)

        if invert_mask:
            mask_arr = 255 - mask_arr

        if img_arr.shape[0] != mask_arr.shape[0] or img_arr.shape[1] != mask_arr.shape[1]:
            return JSONResponse(status_code=400, content={"error": "Image and mask sizes must match."})

        backend = (backend or "lama").lower()
        if backend == "cv":
            result = _cv_inpaint(img_arr, mask_arr)
        elif backend == "sd":
            _load_sd_pipe_once()
            result = _sd_inpaint(
                img_arr,
                mask_arr,
                prompt or "",
                negative_prompt or None,
                guidance_scale or 7.5,
                num_inference_steps or 30,
                seed,
            )
        else:
            with _model_lock:
                if _model == "cv_inpaint":
                    result = _cv_inpaint(img_arr, mask_arr)
                else:
                    result = inpaint_img_with_builded_lama(
                        _model,
                        img_arr,
                        mask_arr,
                        device=_device,
                    )

        result_b64 = _image_to_b64(result)
        return {"image_b64": result_b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/healthz")
async def healthz():
    try:
        _load_model_once()
        return {"status": "ok", "device": _device, "sd_loaded": _sd_pipe is not None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


