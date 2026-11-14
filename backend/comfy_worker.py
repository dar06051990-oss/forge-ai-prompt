"""Local Comfy-compatible CLIP generation worker. Runs with the current Forge Python.

The pinned Comfy backend sources are installed in this extension, not in
an external ComfyUI installation. stdout emits a marker-prefixed JSON result.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import time
from pathlib import Path

RESULT_MARKER = "__FORGE_AI_PROMPT_RESULT__:"


def result(obj, path=None):
    text = json.dumps(obj, ensure_ascii=False)
    if path:
        # A separate output channel keeps the user's prompt and full response
        # out of terminal logs, while normal progress remains live on stdout.
        target = Path(path)
        target.write_text(text, encoding="utf-8")
    else:
        # Used by legacy/probe calls (the parent captures this output).
        print(RESULT_MARKER + text, flush=True)


def main(req):
    started = time.perf_counter()
    print("[Forge AI Prompt] Initializing local generation components...", flush=True)
    core = Path(req["comfy_root"]).expanduser().resolve()
    if not (core / "comfy" / "sd.py").is_file():
        raise RuntimeError(f"Not a ComfyUI core directory (missing comfy/sd.py): {core}")
    sys.path.insert(0, str(core))
    # ComfyUI uses its installation root for tokenizer files and core resources.
    os.chdir(core)

    # The worker has its own sys.path. Never import comfy in the Forge UI process.
    import torch
    import comfy.sd
    import comfy.model_management
    import folder_paths
    if req.get("probe_only"):
        return {"ok": True, "runtime": f"torch {torch.__version__} | Comfy backend loaded via Forge Python"}

    enc_root = Path(req["encoder_root"]).expanduser().resolve()
    # .absolute() normalizes the requested filename without following the
    # model's own symlink. It must be *listed* inside the user-selected folder,
    # but a symlink to a shared model store is valid on Windows.
    enc_file = Path(os.path.abspath(os.path.expanduser(req["encoder_file"])))
    if not enc_file.is_file() or enc_root not in enc_file.parents:
        raise RuntimeError("The selected encoder is missing or its path is outside the configured folder")
    if enc_file.suffix.lower() != ".safetensors":
        raise RuntimeError("Only ComfyUI-compatible .safetensors text encoders are supported in this prototype")

    family = str(req.get("clip_type", "moss_image")).upper()
    if family not in {"MOSS_IMAGE", "MINIMAX", "QWEN_IMAGE", "FLUX3", "KREA2", "FIBO2", "JOYIMAGE", "IDEOGRAM4"}:
        raise RuntimeError(f"Unsupported encoder loader type: {family}")
    clip_type = getattr(comfy.sd.CLIPType, family, None)
    if clip_type is None:
        raise RuntimeError(f"Your ComfyUI core does not support loader type '{family.lower()}'. Please update it.")

    # Keep the specified model in its original folder. The Comfy model loader
    # accepts absolute paths; it does not require a ComfyUI server or graph.
    print(f"[Forge AI Prompt] Loading encoder: {enc_file.name} (loader: {family.lower()})", flush=True)
    model_started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(enc_file)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )
    if not hasattr(clip, "generate") or not hasattr(clip, "decode"):
        raise RuntimeError("The selected text encoder does not support text generation")
    print(f"[Forge AI Prompt] Encoder loaded in {time.perf_counter() - model_started:.1f} s.", flush=True)
    if torch.cuda.is_available():
        print(f"[Forge AI Prompt] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB (worker process).", flush=True)

    image_tensor = None
    image_info = ""
    if req.get("image_path"):
        import numpy as np
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(req["image_path"])).convert("RGB")
        img.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        image_tensor = torch.from_numpy(np.asarray(img).astype("float32") / 255.0)[None, ...]
        image_info = (f"Vision input passed to clip.tokenize: {img.width}x{img.height} RGB. "
                      "This confirms input delivery, not that the model interpreted it correctly.")
        print("[Forge AI Prompt worker] " + image_info, file=sys.stderr, flush=True)
    elif req.get("vision_mode"):
        raise RuntimeError("Vision mode selected but the worker received no image path")

    prompt = str(req.get("prompt", ""))
    system_prompt = str(req.get("system_prompt", ""))
    if not prompt and image_tensor is not None:
        prompt = "Describe this image as a complete image generation prompt."
    if not prompt:
        raise ValueError("Enter a prompt or upload an image")

    print("[Forge AI Prompt] Tokenizing request...", flush=True)
    tokens = clip.tokenize(
        prompt,
        image=image_tensor,
        skip_template=False,
        min_length=1,
        thinking=bool(req.get("thinking", False)),
        system_prompt=system_prompt,
    )
    do_sample = bool(req.get("sampling", True))
    kwargs = dict(
        do_sample=do_sample,
        max_length=int(req.get("max_tokens", 1024)),
        temperature=float(req.get("temperature", 0.7)),
        top_k=int(req.get("top_k", 64)),
        top_p=float(req.get("top_p", 0.95)),
        min_p=float(req.get("min_p", 0.05)),
        repetition_penalty=float(req.get("repetition_penalty", 1.05)),
        presence_penalty=0.0,
        seed=int(req.get("seed", 0)),
        mtp=False,  # Speculative decoding is optional; first version stays conservative.
    )
    print(f"[Forge AI Prompt] Generating text (maximum {kwargs['max_length']} tokens)...", flush=True)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        # The bundled Comfy implementation uses tqdm('Generating tokens').
        # Parent inherits stderr/stdout, so token progress is visible live.
        ids = clip.generate(tokens, **kwargs)
        elapsed = time.perf_counter() - generation_started
        num_tokens = len(ids)
        speed = num_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[Forge AI Prompt] Tokens generated: {num_tokens} in {elapsed:.1f} s ({speed:.2f} tokens/s).", flush=True)
        print("[Forge AI Prompt] Decoding generated text...", flush=True)
        decoded = clip.decode(ids)
    print(f"[Forge AI Prompt] Worker finished in {time.perf_counter() - started:.1f} s.", flush=True)
    thought, sep, text = decoded.partition("</think>")
    if sep:
        output = text.strip()
        thinking = thought.replace("<think>", "", 1).strip()
    elif decoded.lstrip().startswith("<think>"):
        output, thinking = "", decoded.replace("<think>", "", 1).strip()
    else:
        output, thinking = decoded.strip(), ""
    return {"ok": True, "text": output, "thinking": thinking, "image_info": image_info}


if __name__ == "__main__":
    request = None
    try:
        # Parent sends JSON with encoding="utf-8". On Windows, redirected
        # stdin may otherwise be decoded with the system ANSI code page
        # (e.g. cp1251), corrupting Cyrillic and other non-ASCII prompts.
        request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        answer = main(request)
    except Exception as exc:
        # With Status removed from the UI, all failures must be visible in Forge.
        print("[Forge AI Prompt] Generation failed:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=7)}
    result(answer, (request or {}).get("result_path"))
    if not answer.get("ok"):
        sys.exit(1)
