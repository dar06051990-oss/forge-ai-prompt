"""Local Comfy-compatible CLIP generation worker. Runs with the current Forge Python.

The pinned Comfy backend sources are installed in this extension, not in
an external ComfyUI installation. stdout emits a marker-prefixed JSON result.
"""
from __future__ import annotations

import gc
import logging
import json
import os
import sys
import traceback
import time
from pathlib import Path

RESULT_MARKER = "__FORGE_AI_PROMPT_RESULT__:"
_CACHE_CLIP = None
_CACHE_KEY = None
_ENGINE_ROOT = None
_ENGINE = None
_PERSISTENT = False
# FAP_DYNAMIC_V0218
_DYNAMIC = False
_GPU_FREE_BEFORE = None


def result(obj, path=None):
    text = json.dumps(obj, ensure_ascii=False)
    if path:
        # A separate output channel keeps the user's prompt and full response
        # out of terminal logs, while normal progress remains live on stdout.
        target = Path(path)
        pending = target.with_suffix(".tmp")
        pending.write_text(text, encoding="utf-8")
        os.replace(pending, target)
    else:
        # Used by legacy/probe calls (the parent captures this output).
        print(RESULT_MARKER + text, flush=True)


def _engine(core_dir):
    global _ENGINE, _ENGINE_ROOT, _DYNAMIC
    core = Path(core_dir).expanduser().resolve()
    if _ENGINE is not None:
        if core != _ENGINE_ROOT:
            raise RuntimeError("Cannot change Comfy core inside a running worker")
        return _ENGINE
    started = time.perf_counter()
    # FAP_QUIET_STARTUP_V2: Engine ready message below is sufficient.
    if not (core / "comfy" / "sd.py").is_file():
        raise RuntimeError(f"Not a ComfyUI core directory (missing comfy/sd.py): {core}")
    sys.path.insert(0, str(core))
    os.chdir(core)
    # ComfyUI's main.py normally performs this bootstrap. Our private
    # worker imports the core directly instead of starting its server.
    # Do not enable argument parsing: Forge and the worker have their own CLI.
    logging.basicConfig(level=logging.INFO, format="[Comfy backend] %(message)s")
    # FAP_QUIET_KITCHEN_V1: suppress verbose capability discovery only.
    class _QuietKitchenCapabilities(logging.Filter):
        def filter(self, record):
            if record.levelno >= logging.WARNING:
                return True
            # FAP_QUIET_STARTUP_V2: hide backend diagnostics during warm-up.
            # Warnings/errors pass the threshold above at every stage.
            if _ENGINE is None:
                return False
            message = record.getMessage()
            return not (message.startswith("Found comfy_kitchen backend ")
                        and "capabilities" in message)

    for handler in logging.getLogger().handlers:
        handler.addFilter(_QuietKitchenCapabilities())
    control = None
    try:
        from comfy.cli_args import args, enables_dynamic_vram
        if enables_dynamic_vram() and os.environ.get("FAP_DISABLE_DYNAMIC_VRAM") != "1":
            import comfy_aimdo.control as control
            reserve = getattr(args, "reserve_vram", None)
            control.init(simple_vram_headroom=None if reserve is None else int(reserve * 1024**3),
                         nvml_pressure=not getattr(args, "disable_nvml_pressure", False))
    except Exception as exc:
        control = None
        print(f"[Forge AI Prompt] Dynamic VRAM pre-init unavailable: {exc}", flush=True)
    import torch
    import comfy.model_management
    import comfy.model_patcher
    import comfy.memory_management
    if control is not None:
        try:
            mm = comfy.model_management
            if mm.is_nvidia() and mm.torch_version_numeric >= (2, 8):
                devices = [d.index for d in mm.get_all_torch_devices() if d.type == "cuda"]
                if devices:
                    headroom = int(getattr(args, "vram_headroom", 0) * 1024**3)
                    try:
                        initialized = control.init_devices((i, headroom) for i in devices)
                    except TypeError:
                        initialized = control.init_devices(iter(devices))
                    if initialized:
                        control.set_log_info()
                        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
                        comfy.memory_management.aimdo_enabled = True
                        _DYNAMIC = True
        except Exception as exc:
            print(f"[Forge AI Prompt] Dynamic VRAM initialization failed: {exc}", flush=True)
    print("[Forge AI Prompt] Dynamic VRAM: " + ("ENABLED (AIMDO)" if _DYNAMIC else "OFF (legacy loader)"), flush=True)
    import comfy.sd
    import folder_paths
    _ENGINE = (torch, comfy, folder_paths)
    _ENGINE_ROOT = core
    print(f"[Forge AI Prompt] Engine ready in {time.perf_counter() - started:.2f} s.", flush=True)
    return _ENGINE


def _release_gpu():
    """Release Comfy weights before returning the Forge queue lock."""
    if _ENGINE is None:
        return True
    torch, comfy, _ = _ENGINE
    mm = comfy.model_management
    started = time.perf_counter()
    # AIMDO uses allocations which are not all visible to PyTorch's allocator.
    # Release temporary cast buffers as well as cached GPU weights.
    if _DYNAMIC and hasattr(mm, "reset_cast_buffers"):
        mm.reset_cast_buffers()
    mm.unload_all_models()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[Forge AI Prompt] GPU handoff: {allocated:.0f} MiB allocated, {reserved:.0f} MiB reserved; {time.perf_counter() - started:.2f} s.", flush=True)
        # Do not keep the worker alive if substantial Torch GPU allocations
        # survive offload. CUDA context/driver memory is not measured here.
        safe = allocated <= 128 and reserved <= 256
        if _DYNAMIC and _GPU_FREE_BEFORE is not None:
            free_now, _ = torch.cuda.mem_get_info()
            deficit = max(0, _GPU_FREE_BEFORE - free_now) / 1024**2
            print(f"[Forge AI Prompt] CUDA free VRAM after handoff: {free_now / 1024**2:.0f} MiB; delta from request start: {deficit:.0f} MiB less free.", flush=True)
            # External GPU activity may also alter this measurement. Prefer
            # restarting the worker over leaving an uncertain allocation.
            safe = safe and deficit <= 512
        return safe
    return True


def main(req):
    global _CACHE_CLIP, _CACHE_KEY, _GPU_FREE_BEFORE
    started = time.perf_counter()
    torch, comfy, folder_paths = _engine(req["comfy_root"])
    if req.get("probe_only"):
        return {"ok": True, "runtime": f"torch {torch.__version__} | Comfy backend loaded via Forge Python"}
    _GPU_FREE_BEFORE = torch.cuda.mem_get_info()[0] if _DYNAMIC and torch.cuda.is_available() else None

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
    stat = enc_file.stat()
    key = (str(enc_file), family, stat.st_size, stat.st_mtime_ns)
    if _PERSISTENT and _CACHE_CLIP is not None and key == _CACHE_KEY:
        clip = _CACHE_CLIP
        print(f"[Forge AI Prompt] Encoder cache hit: {enc_file.name}", flush=True)
    else:
        if _CACHE_CLIP is not None:
            _CACHE_CLIP = None
            _CACHE_KEY = None
            _release_gpu()
        print(f"[Forge AI Prompt] Loading encoder: {enc_file.name} (loader: {family.lower()})", flush=True)
        model_started = time.perf_counter()
        clip = comfy.sd.load_clip(
            ckpt_paths=[str(enc_file)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        if not hasattr(clip, "generate") or not hasattr(clip, "decode"):
            raise RuntimeError("The selected text encoder does not support text generation")
        if _PERSISTENT:
            _CACHE_CLIP, _CACHE_KEY = clip, key
        print(f"[Forge AI Prompt] Encoder loaded in {time.perf_counter() - model_started:.1f} s.", flush=True)
    patcher = getattr(clip, "patcher", None)
    print(f"[Forge AI Prompt] Encoder patcher: {type(patcher).__name__}", flush=True)
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
    token_started = time.perf_counter()
    tokens = clip.tokenize(
        prompt,
        image=image_tensor,
        skip_template=False,
        min_length=1,
        thinking=bool(req.get("thinking", False)),
        system_prompt=system_prompt,
    )
    print(f"[Forge AI Prompt] Tokenization: {time.perf_counter() - token_started:.2f} s.", flush=True)
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
        # Time the load_model call already made by CLIP.generate; do not
        # load the model twice or alter token generation parameters.
        original_load = getattr(clip, "load_model", None)
        if callable(original_load):
            had_override = "load_model" in vars(clip)
            def timed_load(*args, **kwargs):
                t = time.perf_counter()
                try:
                    return original_load(*args, **kwargs)
                finally:
                    print(f"[Forge AI Prompt] CLIP.load_model: {time.perf_counter() - t:.2f} s (before decoding).", flush=True)
            clip.load_model = timed_load
            try:
                ids = clip.generate(tokens, **kwargs)
            finally:
                if had_override:
                    clip.load_model = original_load
                else:
                    del clip.load_model
        else:
            ids = clip.generate(tokens, **kwargs)
        elapsed = time.perf_counter() - generation_started
        num_tokens = len(ids)
        speed = num_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[Forge AI Prompt] Tokens generated: {num_tokens} in {elapsed:.1f} s ({speed:.2f} tokens/s).", flush=True)
        print("[Forge AI Prompt] Decoding generated text...", flush=True)
        decoded = clip.decode(ids)
    print(f"[Forge AI Prompt] Generation request finished in {time.perf_counter() - started:.1f} s.", flush=True)
    thought, sep, text = decoded.partition("</think>")
    if sep:
        output = text.strip()
        thinking = thought.replace("<think>", "", 1).strip()
    elif decoded.lstrip().startswith("<think>"):
        output, thinking = "", decoded.replace("<think>", "", 1).strip()
    else:
        output, thinking = decoded.strip(), ""
    return {"ok": True, "text": output, "thinking": thinking, "image_info": image_info}


def serve():
    global _PERSISTENT
    _PERSISTENT = True
    # Parent supplies the installed core as cwd. Warm up imports only; no
    # encoder weights are loaded until the first generation request.
    _engine(Path.cwd())
    for raw in sys.stdin.buffer:
        request = None
        try:
            request = json.loads(raw.decode("utf-8"))
            answer = main(request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=7)}
        try:
            reusable = _release_gpu()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            reusable = False
        if not reusable or not answer.get("ok"):
            print("[Forge AI Prompt] Worker restart required for safe GPU handoff.", flush=True)
            answer["restart_worker"] = True
        result(answer, (request or {}).get("result_path"))
        if answer.get("restart_worker"):
            return


if __name__ == "__main__" and "--serve" in sys.argv:
    sys.argv.remove("--serve")
    serve()
    sys.exit(0)


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
