"""
Feature 4 – Real text-to-image renderer for Neuroforge
This module forwards image generation requests to an external Cloudflare
Worker.  The worker URL and authentication key can be configured via
module-level constants.  The feature registers two endpoints when
enabled:

* ``POST /features/4/render`` – Render an image from a text prompt.
* ``GET /features/4/images`` – List previously rendered images in the
  save directory.

The save directory can be configured via ``routing.json`` under
``features.4.save_root`` or will default to ``<projects>/images`` if
present in the folders configuration.
"""

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse
import os
import json
import uuid
import logging
import time
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import List

# --- Captioning API Configuration ---
# This feature supports a Whisk‑like pipeline by performing an
# image→text→image round trip.  To enable more detailed captions, you can
# point CAPTION_API_URL at your own Cloudflare worker that runs
# LLaVA (with optional fallback to another model).  That worker will
# expect the uploaded image in a base64 JSON payload and an x-auth-key
# header for authentication.  See documentation for details.
#
# Example Cloudflare caption worker URL:
#   https://captionv1.<yourname>.workers.dev
# Example access key (should match the AUTH_KEY secret configured on the
# worker):
#   Tr1bXimgEnV1-sdbAI25
#
# If you leave CAPTION_API_URL pointing to the Hugging Face endpoint,
# BLIP will be used as the remote captioner.  However, if you deploy
# your own worker, update CAPTION_API_URL and CAPTION_API_KEY below.

#
# The caption worker and render worker can be consolidated under the same
# Cloudflare Worker.  By default this code uses the `imgenv1` worker on
# workers.dev for both captioning and rendering.  When you deploy your
# own Worker, ensure that `/caption` routes to the caption handler
# (image→text) and the root path handles rendering (text→image).
# Updated to use the new consolidated worker.  The prmptrndr worker handles
# both caption and render requests.  Ensure that your Cloudflare Worker
# is deployed at this URL and that the `/caption` subpath returns a
# caption JSON object.
CAPTION_API_URL = "https://prmptrndr.csirico9.workers.dev/caption"
CAPTION_API_KEY = "Tr1bXimgEnV1-sdbAI25"

# Endpoint for video generation using Stable Video Diffusion.  This
# route will be used by the UI to animate an uploaded image or a
# rendered result.  The same Cloudflare worker (prmptrndr) handles
# caption, render and video generation.  The `/video` path must be
# exposed on that worker.
VIDEO_API_URL = "https://prmptrndr.csirico9.workers.dev/video"
VIDEO_API_KEY = CAPTION_API_KEY

# Private cache for the local captioning model.  These globals are lazily
# initialised when the local captioner is first used.  They allow us to
# avoid re-loading the model on every request.
_BLIP_PROCESSOR = None
_BLIP_MODEL = None

def caption_image(image_bytes: bytes, instruction: str = "") -> str:
    """
    Generate a descriptive caption for the given image bytes using the
    configured captioning API.  This function sends the raw image to
    the configured endpoint and returns the generated caption.

    Returns
    -------
    str
        A caption describing the input image.

    Raises
    ------
    RuntimeError
        If the captioning API call fails or returns an unexpected
        payload.
    """
    # Prepare headers for the Cloudflare caption worker.  Use x-auth-key
    # rather than Authorization header so that your Cloudflare Worker can
    # authenticate the request.  Send the image as base64 in a JSON body.
    headers = {
        "x-auth-key": CAPTION_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8")
    }
    if instruction:
        payload["instruction"] = instruction
    # Attempt remote captioning first.  If the remote call fails (for example
    # due to a network issue or the API returns an error), fall back to a local
    # captioner using the transformers library.  The local captioner will
    # only be used if transformers is installed and a compatible model
    # is available.
    #
    # Debug logging: show the request details and response head for troubleshooting.
    try:
        # Log the outgoing caption request
        print("🧠 Sending to caption worker:", CAPTION_API_URL)
        # Note: do not log the entire payload (image) to avoid huge output
        try:
            resp = requests.post(
                CAPTION_API_URL, headers=headers, json=payload, timeout=90
            )
        except Exception as e:
            # Log the network error before rethrowing
            print("🧠 Caption request error:", e)
            raise
        # Log response status and first 300 characters of body for diagnostics
        print("🧠 Caption worker status:", resp.status_code)
        try:
            # If the response body is large, slice it to avoid flooding logs
            body_head = resp.text[:300]
        except Exception:
            body_head = '<non-text body>'
        print("🧠 Caption worker response (first 300 chars):", body_head)
        resp.raise_for_status()
        try:
            result = resp.json()
        except Exception:
            raise RuntimeError("Caption API returned non‑JSON response.")
        # Cloudflare models return either a list of predictions or a dict.
        # Keys may vary by model.  Accept the first non‑empty string among
        # generated_text, caption, description, or output.
        caption = ""
        if isinstance(result, list) and result:
            first_item = result[0]
            caption = (
                first_item.get("generated_text")
                or first_item.get("caption")
                or first_item.get("description")
                or first_item.get("output")
                or ""
            ).strip()
        elif isinstance(result, dict):
            caption = (
                result.get("generated_text")
                or result.get("caption")
                or result.get("description")
                or result.get("output")
                or ""
            ).strip()
        else:
            raise RuntimeError(f"Unexpected caption API response: {result}")
        if not caption:
            raise RuntimeError("Caption API returned an empty caption.")
        return caption
    except Exception as err:
        # Remote captioning failed; attempt local captioning as a fallback.
        try:
            return _caption_image_local(image_bytes)
        except ImportError as imp_err:
            raise RuntimeError(
                f"Captioning failed (remote error: {err}). Additionally, the local "
                "captioner could not be loaded because required libraries are missing: "
                f"{imp_err}. Please install `transformers` and its dependencies or "
                "provide a working CAPTION_API_URL and CAPTION_API_KEY."
            )
        except Exception as local_err:
            raise RuntimeError(
                f"Captioning failed (remote error: {err}, local error: {local_err})."
            )

# ---------------------------------------------------------------------------
# Local captioner implementation

def _caption_image_local(image_bytes: bytes) -> str:
    """
    Generate a caption locally using the BLIP model via the transformers library.

    This function will lazily load the BLIP processor and model on first use.
    It requires the `transformers` and `torch` libraries to be installed.  If
    these dependencies are not available, an ImportError will be raised.

    Parameters
    ----------
    image_bytes: bytes
        Raw image bytes to caption.

    Returns
    -------
    str
        A caption describing the input image.
    """
    global _BLIP_PROCESSOR, _BLIP_MODEL
    # Import inside the function to avoid ImportError at module import time.
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        # Propagate the ImportError to the caller so they can handle it.
        raise ImportError(
            "The transformers or torch packages are not installed; cannot run "
            "the local captioner." + str(e)
        )
    # Lazy loading of the model and processor
    if _BLIP_PROCESSOR is None or _BLIP_MODEL is None:
        # Load the BLIP model and processor from Hugging Face.  This may
        # download weights the first time it is called.
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        # Put the model in evaluation mode and move to CPU.
        _BLIP_MODEL.eval()
        _BLIP_MODEL.to("cpu")
    # Load image and generate caption
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = _BLIP_PROCESSOR(image, return_tensors="pt")
    with torch.no_grad():
        out = _BLIP_MODEL.generate(**inputs)
    caption = _BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True)
    return caption

# --- Cloudflare Worker Configuration ---
# Update these values to point at your image generation backend.
# If you consolidate your caption and render workers into a single
# Cloudflare worker (for example `imgenv1`), set WORKER_URL to the
# root of that worker.  Otherwise, if you use a custom domain, update
# WORKER_URL accordingly.  The AUTH_KEY must match the secret stored
# in your Cloudflare Worker.
WORKER_URL = "https://prmptrndr.csirico9.workers.dev"
AUTH_KEY = "Tr1bXimgEnV1-sdbAI25"  # same key stored in Cloudflare Secrets

# ---------------------------------------------------------------------------
# Worker mapping for different Stable Diffusion variants
#
# To support separate Cloudflare workers for base/lightning, img2img and
# inpainting models, define the appropriate base URL for each model type
# below.  When a client specifies a model identifier containing a
# particular keyword, the request will be forwarded to the corresponding
# worker.  For example, any model containing ``img2img`` will be sent
# to ``stbl-img2img`` and any containing ``inpaint`` will be sent to
# ``stbl-inptng``.  The base and lightning variants share the
# ``stbl-bselghtn`` worker.
WORKER_MAP = {
    "base": "https://stblbselghtn.csirico9.workers.dev",
    "lightning": "https://stblbselghtn.csirico9.workers.dev",
    "img2img": "https://stblimg2img.csirico9.workers.dev",
    "inpainting": "https://stblinptng.csirico9.workers.dev",
}

def get_worker_url_for_model(model: str) -> str:
    """
    Return the appropriate Cloudflare worker URL for the given model.

    The function examines the model identifier and selects the worker URL
    based on substrings within the model name.  If no match is found,
    the base worker is used by default.

    Parameters
    ----------
    model : str
        The model identifier provided by the client, e.g.
        "@cf/stabilityai/stable-diffusion-xl-img2img".

    Returns
    -------
    str
        The base URL of the Cloudflare worker to handle the request.
    """
    if not model:
        return WORKER_MAP["base"]
    m = model.lower()
    if "inpaint" in m:
        return WORKER_MAP["inpainting"]
    if "img2img" in m:
        return WORKER_MAP["img2img"]
    if "lightning" in m:
        return WORKER_MAP["lightning"]
    return WORKER_MAP["base"]


def _ensure_dir(path: str) -> None:
    """Ensure that the given directory exists, creating parents as needed."""
    os.makedirs(path, exist_ok=True)


def register(app, feature_cfg: dict, routing_path: str):
    """
    Register the text-to-image feature (slot 4) with the given FastAPI app.

    Parameters
    ----------
    app: FastAPI
        The FastAPI application instance.
    feature_cfg: dict
        Configuration for this feature from routing.json.
    routing_path: str
        Path to the routing.json configuration file.

    Returns
    -------
    dict
        Metadata describing the registered routes.
    """
    router = APIRouter(prefix="/features/4", tags=["feature-4-image-render"])

    # Load overall folder configuration to determine allowed save roots.
    try:
        with open(routing_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    allowed_roots = {
        "projects": cfg.get("folders", {}).get("projects"),
        "logs": cfg.get("folders", {}).get("logs"),
    }

    enabled = bool(feature_cfg.get("enabled", True))
    width = int(feature_cfg.get("width", 1024) or 1024)
    height = int(feature_cfg.get("height", 1024) or 1024)

    # Determine where to save images.  If not explicitly configured, use
    # <projects>/images as a default.  The backend will create this
    # directory on first run.
    save_root = feature_cfg.get("save_root")
    if not save_root:
        base = allowed_roots.get("projects")
        if base:
            save_root = os.path.join(base, "images")
    if enabled and save_root:
        _ensure_dir(save_root)

    # ------------------------------------------------------------------
    # Internal helper: send a prompt to the Cloudflare worker and save
    # the resulting image to disk.  Returns the file path and base64
    # encoded image data for inline display.
    def generate_image(
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        req_width: int = None,
        req_height: int = None,
        style: str = "",
        model: str = None,
        image_bytes: bytes = None,
    ) -> dict:
        """Generate an image from the provided prompt.

        If the configured Cloudflare worker is available, this function
        forwards the request to the external API and returns the
        resulting image.  If the worker call fails (due to a network
        error or non‑200 status code) the function falls back to
        generating a simple placeholder image locally that embeds the
        prompt text.  This ensures that clients always receive a
        response, even when the remote generator is unavailable.

        Additional parameters:
        - negative_prompt: Text describing what to avoid in the image.
        - seed: Integer for reproducible generation.
        - req_width, req_height: Override default dimensions.
        - style: Append a style description to the prompt.
        - model: Optional identifier of the diffusion model to use.  When
          provided, this value is included in the payload sent to the
          worker so that the backend can select a specific model.
        - image_bytes: Raw image data.  When provided and the model supports
          image-to-image generation (such as img2img or inpainting), the
          image will be encoded to base64 and sent to the Cloudflare
          worker under the ``image_base64`` key.
        """
        if not prompt:
            raise RuntimeError("Prompt cannot be empty.")

        # Apply style to prompt if provided
        if style:
            prompt += f", in the style of {style}"

        # Determine dimensions
        gen_width = req_width or width
        gen_height = req_height or height

        # Try remote rendering first
        try:
            headers = {
                "Content-Type": "application/json",
                "x-auth-key": AUTH_KEY,
            }
            payload = {"prompt": prompt}
            # Forward the model identifier if one is specified.  This allows
            # clients to select among multiple diffusion models exposed by the
            # Cloudflare workers.  The worker URL will be determined by the
            # model name using get_worker_url_for_model().
            if model:
                payload["model"] = model
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            if seed is not None:
                payload["seed"] = seed
            if gen_width:
                payload["width"] = gen_width
            if gen_height:
                payload["height"] = gen_height
            # If image_bytes are provided and the selected model is an
            # image-to-image variant (e.g., img2img or inpainting), include
            # the image data in the payload.  Cloudflare's img2img and
            # inpainting models require a base64-encoded image under
            # "image_base64".  We allow sending the image for any model if
            # supplied; models that do not support an image input will
            # simply ignore this field.
            if image_bytes is not None:
                try:
                    # Encode to base64
                    import base64 as _b64
                    b64_str = _b64.b64encode(image_bytes).decode("utf-8")
                    payload["image_base64"] = b64_str
                except Exception:
                    pass
            # Determine the worker to forward the request to based on the model
            worker_url = get_worker_url_for_model(model)
            # Debug logging: show details of the render call
            print("🎯 Rendering via:", worker_url)
            print("🎯 Render headers:", headers)
            # Log only the first 200 chars of the JSON payload to avoid large logs
            try:
                import json as _json
                payload_repr = _json.dumps(payload)[:200]
            except Exception:
                payload_repr = str(payload)[:200]
            print("🎯 Render payload:", payload_repr)
            r = requests.post(worker_url, json=payload, headers=headers, timeout=120)
            # Log the status and a snippet of the response body
            print("🎯 Render status:", r.status_code)
            try:
                body_head = r.text[:200]
            except Exception:
                body_head = '<non-text body>'
            print("🎯 Render body head:", body_head)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            # Remote worker failed; fall back to a local placeholder image.
            logging.warning(f"Remote image generation failed ({e}); falling back to local placeholder.")
            # Create a white canvas and draw the prompt text onto it
            from PIL import ImageDraw, ImageFont
            # Start with a reasonable canvas size; adjust based on text length
            canvas_width, canvas_height = gen_width, gen_height
            img = Image.new("RGB", (canvas_width, canvas_height), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            # Use a default font; if the system font is unavailable PIL will substitute
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except Exception:
                font = ImageFont.load_default()
            # Wrap text to fit within the image width
            max_width = canvas_width - 40
            words = prompt.split()
            lines: List[str] = []
            current_line = ""
            for word in words:
                if draw.textlength(f"{current_line} {word}", font=font) <= max_width:
                    current_line = f"{current_line} {word}".strip()
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            # Add negative prompt if provided
            if negative_prompt:
                lines.append("")
                lines.append("Avoid: " + negative_prompt)
            # Calculate vertical starting position
            text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines) + len(lines) * 10
            y_offset = (canvas_height - text_height) // 2
            for line in lines:
                w, h = draw.textbbox((0, 0), line, font=font)[2], draw.textbbox((0, 0), line, font=font)[3]
                x = (canvas_width - w) // 2
                draw.text((x, y_offset), line, fill=(50, 50, 50), font=font)
                y_offset += h + 10

        # Resize image if limits are defined and not already applied
        if width and height and (gen_width != width or gen_height != height):
            img.thumbnail((width, height))

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        if not save_root:
            raise RuntimeError("Save root not configured.")
        _ensure_dir(save_root)
        out_path = os.path.join(save_root, filename)
        img.save(out_path)

        # Encode image for inline display
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"path": out_path, "image_base64": b64_data}

    # ------------------------------------------------------------------
    # HTTP endpoints
    @router.post("/caption")
    async def caption_endpoint(request: Request):
        """
        Generate a caption for an uploaded image without rendering the final image.

        This endpoint accepts a JSON body with a single key:
        - image_base64: The base64‑encoded image data (JPEG or PNG).

        It returns a JSON response containing the generated caption.  If the
        captioning service fails, a 500 error is returned.
        """
        if not enabled:
            raise HTTPException(403, "Feature slot 4 disabled")
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")
        image_b64 = data.get("image_base64")
        instruction = data.get("instruction", "")
        if not isinstance(instruction, str):
            instruction = ""
        if not image_b64:
            raise HTTPException(400, "image_base64 is required")
        # Strip data URL prefix if present
        if isinstance(image_b64, str) and image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(400, "image_base64 is not valid base64")
        try:
            caption = caption_image(image_bytes, instruction.strip())
        except Exception as e:
            logging.error(f"Caption generation failed: {e}")
            raise HTTPException(500, f"Caption generation failed: {e}")
        return JSONResponse({"caption": caption})
    @router.post("/render")
    async def render_image(request: Request):
        """
        Generate an image using either a direct text prompt or an
        image→text→image pipeline.  By default the endpoint behaves
        like the original feature: it accepts a text prompt and
        forwards it to the configured backend (Cloudflare worker).

        To use the Whisk‑like pipeline, include ``method="caption"``
        in the JSON body and provide ``image_base64`` containing a
        base64‑encoded JPEG/PNG.  The server will generate a caption
        from the image using the configured caption API and then
        forward that caption as the prompt to the backend.  You can
        optionally supply a ``style`` string to influence the final
        render.

        JSON Body Parameters
        --------------------
        method : str, optional
            Either "prompt" (default) to render from text or
            "caption" to render from an uploaded image.
        prompt : str
            The text prompt to render (required for ``method='prompt'``).
        image_base64 : str
            Base64 encoded image data (required for ``method='caption'``).
        negative_prompt : str, optional
            Elements to avoid in the generated image.
        seed : int, optional
            Seed for reproducible generation.
        width, height : int, optional
            Override default dimensions.
        style : str, optional
            A style description appended to the prompt.
        model : str, optional
            Identifier of the Stable Diffusion variant to use (e.g.,
            "@cf/bytedance/stable-diffusion-xl-lightning").  If provided,
            this value is forwarded to the Cloudflare worker.
        """
        if not enabled:
            raise HTTPException(403, "Feature slot 4 disabled")
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")

        # Determine which pipeline to use.  Default to simple prompt
        # rendering for backward compatibility.
        method = data.get("method", "prompt").lower().strip()
        negative_prompt = data.get("negative_prompt", "").strip()
        seed = data.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                raise HTTPException(400, "seed must be an integer")
        style = data.get("style", "").strip()
        instruction = data.get("instruction", "")
        if not isinstance(instruction, str):
            instruction = ""
        instruction = instruction.strip()
        # Optional model identifier to select a specific diffusion backend
        model = data.get("model")
        if model is not None:
            model = str(model).strip() or None
        req_width = data.get("width")
        if req_width is not None:
            try:
                req_width = int(req_width)
            except ValueError:
                raise HTTPException(400, "width must be an integer")
        req_height = data.get("height")
        if req_height is not None:
            try:
                req_height = int(req_height)
            except ValueError:
                raise HTTPException(400, "height must be an integer")

        try:
            if method == "caption":
                # Expect a base64‑encoded image in the request body.
                image_b64 = data.get("image_base64")
                if not image_b64:
                    raise HTTPException(400, "image_base64 is required when method='caption'")
                try:
                    # Remove data URL prefix if present
                    if image_b64.startswith("data:image"):
                        image_b64 = image_b64.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_b64)
                except Exception:
                    raise HTTPException(400, "image_base64 is not valid base64")
                # Generate a caption using the configured API
                try:
                    caption = caption_image(image_bytes, instruction)
                except Exception as e:
                    logging.error(f"Caption generation failed: {e}")
                    raise HTTPException(500, f"Caption generation failed: {e}")
                # Use the caption as the prompt
                prompt = caption
                result = generate_image(
                    prompt,
                    negative_prompt,
                    seed,
                    req_width,
                    req_height,
                    style,
                    model,
                    image_bytes=image_bytes,
                )
                return JSONResponse({
                    "ok": True,
                    "saved_as": result["path"],
                    "image_base64": result["image_base64"],
                    "prompt": prompt,
                    "caption": caption,
                    "method": "caption"
                })
            else:
                # Standard prompt‑based rendering.  The client may optionally
                # provide image_base64 to perform img2img or inpainting.  If
                # the selected model requires an image and none is provided,
                # return a validation error.
                prompt = data.get("prompt", "").strip()
                if not prompt:
                    raise HTTPException(400, "No prompt provided")
                image_bytes_param = None
                image_b64_input = data.get("image_base64")
                if image_b64_input:
                    try:
                        # Remove data URL prefix if present
                        if isinstance(image_b64_input, str) and image_b64_input.startswith("data:image"):
                            image_b64_input = image_b64_input.split(",", 1)[1]
                        image_bytes_param = base64.b64decode(image_b64_input)
                    except Exception:
                        raise HTTPException(400, "image_base64 is not valid base64")
                # Validate that required image is present for img2img/inpainting models
                if model and (
                    "img2img" in model.lower() or "inpainting" in model.lower()
                ) and image_bytes_param is None:
                    raise HTTPException(
                        400,
                        "Selected diffusion model requires an image. Provide 'image_base64' or use the caption mode to upload an image.",
                    )
                result = generate_image(
                    prompt,
                    negative_prompt,
                    seed,
                    req_width,
                    req_height,
                    style,
                    model,
                    image_bytes=image_bytes_param,
                )
                return JSONResponse({
                    "ok": True,
                    "saved_as": result["path"],
                    "image_base64": result["image_base64"],
                    "prompt": prompt,
                    "method": "prompt",
                })
        except HTTPException:
            # Reraise HTTPExceptions without modification
            raise
        except Exception as e:
            logging.error(f"Image rendering failed: {e}")
            raise HTTPException(500, f"Image rendering failed: {e}")

    @router.post("/video")
    async def video_endpoint(request: Request):
        """
        Generate a short video from an image using the Stable Video
        Diffusion model via the configured Cloudflare Worker.  This
        endpoint accepts a JSON body with the following keys:

        - image_base64 (str): Required.  Base64‑encoded JPEG or PNG
          image to animate.
        - prompt (str): Optional.  A text prompt to guide the video.
          If omitted, the prompt defaults to an empty string and the
          model will simply animate the input image.
        - num_frames (int): Optional.  The number of frames to
          generate.  Allowed values depend on the model (e.g. 8, 14,
          20, etc.).  If unspecified, the model will choose a
          reasonable default.
        - fps (int): Optional.  Frames per second for the output
          video.  If unspecified, the model default is used.

        It returns a JSON object with ``video_base64`` containing
        the MP4 video data encoded as base64.  If the remote call
        fails or the worker returns an unexpected response, a 500
        error is raised.  A 400 error is returned if required
        fields are missing.
        """
        if not enabled:
            raise HTTPException(403, "Feature slot 4 disabled")
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")
        image_b64 = data.get("image_base64")
        if not image_b64:
            raise HTTPException(400, "image_base64 is required")
        # Remove data URL prefix if present
        if isinstance(image_b64, str) and image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
        try:
            # Validate base64
            base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(400, "image_base64 is not valid base64")
        prompt = data.get("prompt", "")
        # Optional integer parameters
        num_frames = data.get("num_frames")
        fps = data.get("fps")
        # Build payload for the Cloudflare worker
        payload = {
            "image_base64": image_b64,
            "prompt": prompt,
        }
        # Include optional fields if present and valid
        if num_frames is not None:
            try:
                payload["num_frames"] = int(num_frames)
            except ValueError:
                raise HTTPException(400, "num_frames must be an integer")
        if fps is not None:
            try:
                payload["fps"] = int(fps)
            except ValueError:
                raise HTTPException(400, "fps must be an integer")
        # Prepare headers for the Cloudflare video worker
        headers = {
            "x-auth-key": VIDEO_API_KEY,
            "Content-Type": "application/json",
        }
        # Forward the request to the video generation worker
        try:
            print("🎬 Sending to video worker:", VIDEO_API_URL)
            resp = requests.post(VIDEO_API_URL, headers=headers, json=payload, timeout=120)
        except Exception as e:
            print("🎬 Video request error:", e)
            raise HTTPException(500, f"Video generation request failed: {e}")
        print("🎬 Video worker status:", resp.status_code)
        try:
            body_head = resp.content[:200]
        except Exception:
            body_head = b'<non-binary response>'
        print("🎬 Video worker response (first 200 bytes):", body_head)
        if resp.status_code != 200:
            # Try to parse error message if JSON
            try:
                err_json = resp.json()
                detail = err_json.get("error") or err_json.get("detail") or resp.text
            except Exception:
                detail = resp.text
            raise HTTPException(500, f"Video generation failed: {detail}")
        # The worker returns raw MP4 bytes.  Encode to base64 for JSON transport.
        video_bytes = resp.content
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        return JSONResponse({
            "ok": True,
            "video_base64": video_b64,
            "prompt": prompt,
            "num_frames": payload.get("num_frames"),
            "fps": payload.get("fps"),
            "method": "video"
        })

    @router.get("/images")
    def list_images():
        """List saved image files for this feature."""
        if not enabled:
            raise HTTPException(403, "Feature slot 4 disabled")
        if not save_root or not os.path.isdir(save_root):
            return {"ok": True, "files": []}
        try:
            files = [
                f for f in os.listdir(save_root)
                if os.path.isfile(os.path.join(save_root, f))
            ]
            return {"ok": True, "files": sorted(files)}
        except Exception as e:
            logging.error(f"Could not list images: {e}")
            raise HTTPException(500, f"Could not list images: {e}")

    # Register the router and return route metadata
    app.include_router(router)
    return {"routes": ["/features/4/render", "/features/4/images", "/features/4/video"]}
