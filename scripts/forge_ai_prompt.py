"""Forge AI Prompt: local text generation using Forge Python and a private installed engine."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
from modules import scripts

EXTENSION = Path(__file__).resolve().parents[1]
WORKER = EXTENSION / "backend" / "comfy_worker.py"
PRESETS = EXTENSION / "presets"
STATE = EXTENSION / ".forge_ai_prompt_settings.json"
# Installer backend is private to this extension; no ComfyUI installation required.
sys.path.insert(0, str(EXTENSION / "backend"))
import runtime_install
import fap_preset_store as preset_store
import fap_worker_client

LOCK = threading.Lock()
ENCODER_SUFFIXES = {".safetensors"}
LOADER_TYPES = ["moss_image", "minimax", "qwen_image", "flux3", "krea2", "fibo2", "joyimage", "ideogram4"]
RESULT_MARKER = "__FORGE_AI_PROMPT_RESULT__:"

IMAGE_INSTRUCTION = (
    "Examine the attached image. Describe only the objects, people, clothing, poses, "
    "background, colors, lighting and composition actually visible. "
    "Write a faithful English image-generation prompt; do not invent an elf, "
    "staff, castle or other details unless they are visible."
)
MODE_TO_PRESET = {"Enhance text": "Enhance prompt", "Describe image": "Image to prompt"}


def _default_model_folder():
    try:
        from modules import paths
        models = Path(paths.models_path)
    except Exception:
        models = Path.cwd() / "models"
    for name in ("text_encoders", "text_encoder", "clip"):
        candidate = models / name
        if candidate.is_dir():
            return str(candidate)
    return str(models / "text_encoders")


def _read_config():
    try:
        data = json.loads(STATE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return {}


def _save_config(encoder_root, encoder=None, clip_type=None):
    cfg = _read_config()
    cfg["encoder_root"] = encoder_root
    if encoder:
        cfg["encoder_name"] = encoder
    if clip_type:
        cfg["clip_type"] = clip_type
    # Legacy ComfyUI paths are no longer used. Remove them on save.
    cfg.pop("comfy_root", None)
    cfg.pop("python_exe", None)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE)
    print("[Forge AI Prompt] Encoder selection saved.", flush=True)


def _worker_env():
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(runtime_install.CORE) + (os.pathsep + py_path if py_path else "")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run_worker(request, timeout):
    return subprocess.run(
        [sys.executable, "-u", str(WORKER)],
        input=json.dumps(request, ensure_ascii=False),
        text=True, encoding="utf-8", errors="replace", capture_output=True,
        timeout=timeout, cwd=str(runtime_install.CORE), shell=False, env=_worker_env(),
    )



def _release_forge_models_for_text():
    """Free Forge's VRAM *in the main Forge process*, not in the worker.

    Do not call this on tab/preset switches: reloading large checkpoints would
    slow the user's normal workflow. Caller must hold Forge's queue_lock so
    native image generation cannot overlap this separate Qwen process.
    """
    from modules import sd_models

    free_before = None
    try:
        import torch
        if torch.cuda.is_available():
            free_before, _ = torch.cuda.mem_get_info()
    except Exception:
        pass

    print('[Forge AI Prompt] Unloading Forge image-generation weights before text generation...', flush=True)
    # This resets model_data.forge_hash and sd_model as well as offloading
    # the underlying model patchers. Native txt2img/img2img will therefore
    # reload the selected checkpoint on the next image generation.
    sd_models.unload_model_weights()

    free_after = None
    try:
        import torch
        if torch.cuda.is_available():
            free_after, total = torch.cuda.mem_get_info()
    except Exception:
        pass
    if free_after is not None:
        freed = (free_after - free_before) / (1024 ** 2) if free_before is not None else None
        amount = f' (freed approximately {freed:+.0f} MiB)' if freed is not None else ''
        print(f'[Forge AI Prompt] GPU memory available: {free_after / (1024 ** 2):,.0f} MiB{amount}', flush=True)
    print('[Forge AI Prompt] Forge model release complete. Sending request to Qwen worker.', flush=True)


def _run_worker_console(request, timeout):
    return fap_worker_client.run(WORKER, runtime_install.CORE, _worker_env(), request, timeout)


def _warmup_worker(*_args, **_kwargs):
    # Startup is best effort: imports happen in the child, not in Forge.
    # If the callback is unavailable, the first request starts the worker.
    try:
        if runtime_install.ready():
            fap_worker_client.warmup(WORKER, runtime_install.CORE, _worker_env())
    except Exception as exc:
        print(f"[Forge AI Prompt] Worker warm-up skipped: {exc}", flush=True)


try:
    from modules import script_callbacks
    script_callbacks.on_app_started(_warmup_worker)
except (ImportError, AttributeError):
    pass


def _decode_result(proc):
    line = next((line[len(RESULT_MARKER):] for line in reversed(proc.stdout.splitlines())
                 if line.startswith(RESULT_MARKER)), None)
    if line is None:
        return None
    return json.loads(line)


def _scan_models(folder):
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ENCODER_SUFFIXES:
            try:
                files.append(path.relative_to(root).as_posix())
            except ValueError:
                pass
        if len(files) >= 2000:
            break
    return sorted(files, key=str.casefold)


def _refresh_models(folder):
    choices = _scan_models(folder)
    print(f"[Forge AI Prompt] Model list refreshed: {len(choices)} encoder(s) in {folder}", flush=True)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _check_setup(encoder_root, encoder, clip_type):
    """Check the current Forge Python + local installed text backend."""
    messages = ["Forge Python: " + sys.executable, runtime_install.describe()]
    issues = []
    if not runtime_install.ready():
        issues.append("Generation components are missing. Restart Forge to run automatic installation; see the console log.")
    if not (encoder_root or "").strip():
        issues.append("Text encoder folder is empty.")
    else:
        root = Path(encoder_root.strip().strip('"')).expanduser()
        if not root.is_dir():
            issues.append(f"Text encoder folder not found: {root}")
        else:
            choices = _scan_models(str(root))
            messages.append(f"Found {len(choices)} encoders in {root}")
            if not encoder or encoder not in choices:
                issues.append("Select an encoder from the list; click Refresh models if needed.")
            else:
                messages.append(f"Selected encoder: {encoder} ({clip_type})")
    if not issues and runtime_install.ready():
        try:
            result = _run_worker({"comfy_root": str(runtime_install.CORE), "probe_only": True}, 90)
            obj = _decode_result(result)
            if not obj or not obj.get("ok"):
                issues.append(
                    "Engine import test failed: " +
                    ((obj or {}).get("error", "unknown error") + "\n" + (obj or {}).get("traceback", "")[-1400:]
                     if obj else (result.stderr or result.stdout)[-1800:])
                )
            else:
                messages.append("Engine import: OK")
                messages.append(obj.get("runtime", ""))
        except Exception as exc:
            issues.append(f"Cannot test local engine: {type(exc).__name__}: {exc}")
    if issues:
        return "SETUP NOT READY:\n- " + "\n- ".join(issues) + "\n\n" + "\n".join(messages)
    return "SETUP READY:\n" + "\n".join(messages) + "\nNow try Generate Text."


def _test_ui():
    print("[Forge AI Prompt] UI callback works.", flush=True)
    return "UI CALLBACK OK: buttons are working. Next click Check setup."


def _list_presets():
    return preset_store.list_names()


def _load_preset(name):
    return preset_store.load(name)


def _preferred_preset(mode):
    choices = _list_presets()
    suggested = MODE_TO_PRESET.get(mode, "Enhance prompt")
    return suggested if suggested in choices else (choices[0] if choices else None)


def _on_preset_change(name):
    return _load_preset(name)


def _refresh_presets(current):
    names = _list_presets()
    active = current if current in names else _preferred_preset("Enhance text")
    print(f"[Forge AI Prompt] Presets refreshed: {len(names)} available.", flush=True)
    return gr.update(choices=names, value=active), _load_preset(active)


def _editor_open(current):
    names = _list_presets()
    active = current if current in names else (names[0] if names else None)
    return (gr.update(visible=True), gr.update(choices=names, value=active),
            active or "", _load_preset(active), "Select a preset to edit, or click New.")


def _editor_select(name):
    return name or "", _load_preset(name), "", ""


def _editor_new():
    return gr.update(value=None), "", "", "New preset: enter a name and system instruction.", ""


def _editor_result(name, message):
    names = _list_presets()
    chosen = name if name in names else (_preferred_preset("Enhance text") or None)
    return (gr.update(choices=names, value=chosen), _load_preset(chosen),
            gr.update(choices=names, value=chosen), chosen or "",
            _load_preset(chosen), message, "")


def _editor_save(name, text, previous):
    try:
        saved = preset_store.save(name, text, previous_name=previous)
        return _editor_result(saved, f"Saved: {saved}")
    except (ValueError, OSError) as exc:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                f"Cannot save preset: {exc}", "")


def _editor_duplicate(selected):
    try:
        created = preset_store.duplicate(selected)
        return _editor_result(created, f"Created a copy: {created}")
    except (ValueError, OSError) as exc:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                f"Cannot duplicate: {exc}", "")


def _editor_delete(selected, armed):
    # A click on Delete removes the selected preset immediately.
    # Keep the state-output slot for compatibility with the existing Gradio UI;
    # _editor_result() clears it after the operation.
    if not selected:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                "Choose a preset first.", "")
    try:
        preset_store.delete(selected)
        return _editor_result(_preferred_preset("Enhance text"), f"Deleted: {selected}")
    except (ValueError, OSError) as exc:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                f"Cannot delete: {exc}", "")


# FAP_PRESERVE_IMAGE_SETTINGS_V1
def _on_mode_change(mode):
    image_mode = mode == "Describe image"
    return (
        "Describe image: the uploaded image is sent to Qwen3-VL. Optional extra text can be enabled below."
        if image_mode else "Enhance text: the reference image is ignored.",
        gr.update(visible=image_mode), gr.update(visible=image_mode),
    )


# FAP_CLIPBOARD_V1
_FAP_CLIPBOARD_JS = """async (_) => {
    try {
        if (!navigator.clipboard || typeof navigator.clipboard.read !== 'function') {
            return ['error:Clipboard image access is unavailable. Use localhost/HTTPS or Ctrl+V.'];
        }
        const items = await navigator.clipboard.read();
        let blob = null;
        const formats = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp'];
        for (const item of items) {
            const format = formats.find(type => item.types.includes(type));
            if (format) {
                blob = await item.getType(format);
                break;
            }
        }
        if (!blob) return ['error:No image found in the clipboard.'];
        if (blob.size > 24 * 1024 * 1024) return ['error:Clipboard image exceeds 24 MiB. Use the normal image upload.'];
        const data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Cannot read clipboard image'));
            reader.readAsDataURL(blob);
        });
        return [data];
    } catch (error) {
        return ['error:Clipboard access failed. Allow clipboard access in the browser or use Ctrl+V.'];
    }
}"""


def _fap_paste_clipboard(data):
    import base64
    import io
    from PIL import Image, ImageOps

    def unchanged(message):
        if hasattr(gr, "Warning"):
            gr.Warning(message)
        else:
            print("[Forge AI Prompt] " + message, flush=True)
        return gr.update(), gr.update(), gr.update()

    if not isinstance(data, str) or not data:
        return unchanged("No image received from the clipboard.")
    if data.startswith("error:"):
        return unchanged(data[6:])
    try:
        header, sep, encoded = data.partition(",")
        allowed = {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64",
                   "data:image/gif;base64", "data:image/bmp;base64"}
        if not sep or header.lower() not in allowed:
            return unchanged("Unsupported clipboard image format.")
        if len(encoded) > 32 * 1024 * 1024:
            return unchanged("Clipboard image exceeds 24 MiB.")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 24 * 1024 * 1024:
            return unchanged("Clipboard image exceeds 24 MiB.")
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 50_000_000:
                return unchanged("Clipboard image is too large. Resize it first.")
            image = ImageOps.exif_transpose(source).convert("RGB")
        # Preserve checkbox value and all other prompt/settings. Explicitly set
        # visibility for both components even if mode.change is not dispatched.
        return (gr.update(value=image, visible=True),
                gr.update(value="Describe image"), gr.update(visible=True))
    except Exception:
        return unchanged("Could not decode clipboard image. Use normal image upload or Ctrl+V.")


def _on_image_change(image, mode):
    if image is None:
        return (gr.update(), "Image removed. Choose Enhance text for text-only generation.")
    return (gr.update() if mode == "Describe image" else gr.update(value="Describe image"),
            f"Image uploaded ({image.width}x{image.height}). Current prompt and settings preserved.")


def _generate(encoder_root, encoder, clip_type, prompt, system_prompt,
              image, max_tokens, temperature, top_k, top_p, thinking,
              mode="Enhance text", include_text=False):
    """Generate text; user-visible diagnostics go to the Forge console."""
    print(f"[Forge AI Prompt] Generate clicked. Mode: {mode}", flush=True)
    tmp_name = ""
    try:
        if not runtime_install.ready():
            raise RuntimeError("Local generation components are missing. Restart Forge to run automatic installation; see the console log.")
        if mode == "Describe image" and image is None:
            raise ValueError("Describe image requires an uploaded image.")
        if mode == "Describe image":
            supplied = (prompt or "").strip()
            prompt = IMAGE_INSTRUCTION + ((" Additional instruction: " + supplied) if include_text and supplied else "")
            print(f"[Forge AI Prompt] Reference image: {image.width}x{image.height}", flush=True)
        else:
            image = None
        if not encoder:
            raise ValueError("No text encoder selected. Click Refresh if necessary.")
        if not (encoder_root or "").strip():
            raise ValueError("Text encoder folder is empty.")
        root = Path(encoder_root.strip().strip('"')).expanduser().resolve()
        available = _scan_models(str(root))
        if not isinstance(encoder, str) or encoder not in available:
            raise ValueError("The selected encoder is not in the current model list. Click Refresh.")
        target = root / encoder
        if not target.is_file() or target.suffix.lower() not in ENCODER_SUFFIXES:
            raise ValueError(f"Selected encoder cannot be accessed: {target}")
        payload = {
            "comfy_root": str(runtime_install.CORE),
            "encoder_root": str(root), "encoder_file": str(target),
            "clip_type": clip_type, "prompt": prompt, "system_prompt": system_prompt,
            "max_tokens": int(max_tokens), "temperature": temperature,
            "top_k": int(top_k), "top_p": top_p, "min_p": 0.05,
            "thinking": thinking, "sampling": True,
        }
        if image is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_name = f.name
            image.save(tmp_name)
            payload["image_path"] = tmp_name
            payload["vision_mode"] = True

        # Serialize both the Forge-side release and the Qwen subprocess with
        # native txt2img/img2img GPU jobs. Otherwise two processes would fight
        # over the same VRAM, or an image could start while text is generating.
        from modules.call_queue import queue_lock
        with LOCK:
            with queue_lock:
                _release_forge_models_for_text()
                print(
                    f"[Forge AI Prompt] Starting local generation: {encoder} | "
                    f"loader={clip_type} | max_tokens={int(max_tokens)}",
                    flush=True,
                )
                obj = _run_worker_console(payload, 600)

        if not obj.get("ok"):
            raise RuntimeError(obj.get("error", "Unknown model error") + "\n" + obj.get("traceback", "")[-2600:])
        output = obj.get("text", "")
        if output:
            print(f"[Forge AI Prompt] Completed successfully. Output: {len(output)} characters.", flush=True)
        else:
            print("[Forge AI Prompt] Completed but text is empty. Try increasing Max tokens or disabling Thinking.", flush=True)
        if obj.get("thinking"):
            print("[Forge AI Prompt] Thinking text was generated (not printed to the console).", flush=True)
        return output
    except subprocess.TimeoutExpired:
        print("[Forge AI Prompt] ERROR: Timeout after 600 seconds. Worker was stopped.", flush=True)
        return ""
    except Exception as exc:
        print(f"[Forge AI Prompt] ERROR: {type(exc).__name__}: {exc}", flush=True)
        return ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

class Script(scripts.Script):
    """Native Forge panel in both txt2img and img2img; never intercepts image sampling."""

    def title(self):
        return "AI Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        cfg = _read_config()
        folder = cfg.get("encoder_root", _default_model_folder())
        models = _scan_models(folder)
        selected = cfg.get("encoder_name")
        if selected not in models:
            selected = next((x for x in models if "qwen3-vl-4b-heretic" in x.casefold()), models[0] if models else None)
        clip_default = cfg.get("clip_type", "moss_image")
        if clip_default not in LOADER_TYPES:
            clip_default = "moss_image"
        presets = _list_presets()
        suffix = "i2i" if is_img2img else "t2i"
        parent_prompt = "img2img_prompt" if is_img2img else "txt2img_prompt"

        # As in Forge Neo's Tagger Prompt script: this accordion is a native
        # always-visible script control, and Forge places it with other scripts.
        # No separate top-level tab is registered.
        with gr.Accordion("AI Prompt", open=False, elem_id=f"fap_panel_{suffix}"):
            # Keep the generation mode separate so encoder and loader remain
            # side by side even in Forge's narrow extensions panel.
            with gr.Row():
                mode = gr.Radio(
                    choices=list(MODE_TO_PRESET), value="Enhance text",
                    label="Mode", scale=1, elem_id=f"fap_mode_{suffix}",
                )
            with gr.Row():
                model = gr.Dropdown(
                    choices=models, value=selected, label="Text encoder", scale=4,
                    elem_id=f"fap_model_{suffix}",
                )
                clip_type = gr.Dropdown(
                    label="CLIP loader type", choices=LOADER_TYPES,
                    value=clip_default, scale=2,
                    elem_id=f"fap_clip_type_{suffix}",
                )
                refresh = gr.Button("Refresh", scale=0, min_width=78)

            with gr.Row():
                preset = gr.Dropdown(
                    label="System preset", choices=presets,
                    value=_preferred_preset("Enhance text"),
                    scale=4, elem_id=f"fap_preset_{suffix}",
                )
                open_editor = gr.Button("Edit Presets", scale=1, min_width=110,
                                        elem_id=f"fap_preset_open_{suffix}")
                refresh_presets = gr.Button("Refresh", scale=0, min_width=76)

            # Presets store ONLY system-prompt text. Generation parameters
            # are shared sliders and never depend on preset selection.
            system_prompt = gr.State(value=_load_preset(_preferred_preset("Enhance text")))

            # Full-screen modal styles live in the Gradio page so they also
            # apply when the JS moves the editor outside nested Forge panels.
            # A viewport-level modal, with field-specific sizing. Gradio may
            # render one-line Textbox as <textarea>; NEVER style all textareas
            # as large editors (this was the v0.2.8 Name-field regression).
            gr.HTML("""<style>
                .fap-preset-backdrop {
                    position: fixed !important;
                    inset: 0 !important;
                    z-index: 2147483000 !important;
                    background: rgba(0, 0, 0, .75) !important;
                    backdrop-filter: blur(1.5px);
                }
                [id^="fap_preset_editor_"].fap-editor-portaled {
                    position: fixed !important;
                    /* Avoid transform: Gradio dropdown popups are positioned
                       using viewport coordinates and a transformed ancestor
                       shifts their menu away from the input control. */
                    inset: 0 !important;
                    margin: auto !important;
                    transform: none !important;
                    z-index: 2147483001 !important;
                    box-sizing: border-box !important;
                    width: min(780px, calc(100vw - 28px)) !important;
                    max-width: calc(100vw - 28px) !important;
                    /* Explicit intrinsic height plus inset/auto margins
                       centers the editor without creating a transformed
                       containing block for floating dropdowns. */
                    height: max-content !important;
                    max-height: min(82vh, 760px) !important;
                    max-height: min(82dvh, 760px) !important;
                    min-height: 0 !important;
                    overflow-x: hidden !important;
                    overflow-y: auto !important;
                    overscroll-behavior: contain;
                    margin: 0 !important;
                    padding: 18px !important;
                    border-radius: 12px !important;
                    border: 1px solid #435267 !important;
                    background: #171f2d !important;
                    color: var(--body-text-color, #f3f4f6) !important;
                    opacity: 1 !important;
                    box-shadow: 0 18px 64px rgba(0,0,0,.75) !important;
                }
                /* Name is one line, not a second large prompt editor. */
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_name_"] {
                    min-height: 0 !important;
                    height: auto !important;
                    flex: 0 0 auto !important;
                }
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_name_"] textarea,
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_name_"] input {
                    box-sizing: border-box !important;
                    min-height: 38px !important;
                    height: 38px !important;
                    max-height: 38px !important;
                    overflow-y: hidden !important;
                    resize: none !important;
                }
                /* Text, unlike Name, gets a usable scrollable editor. */
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_text_"] {
                    min-height: 0 !important;
                    height: auto !important;
                    flex: 0 0 auto !important;
                }
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_text_"] textarea {
                    box-sizing: border-box !important;
                    font-family: inherit;
                    min-height: 170px !important;
                    height: clamp(190px, 32vh, 325px) !important;
                    height: clamp(190px, 32dvh, 325px) !important;
                    max-height: 40vh !important;
                    overflow-y: auto !important;
                    resize: none !important;
                }
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_status_"] textarea,
                [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_status_"] input {
                    box-sizing: border-box !important;
                    min-height: 36px !important;
                    height: 36px !important;
                    max-height: 36px !important;
                    overflow-y: hidden !important;
                    resize: none !important;
                }
                [id^="fap_editor_actions_"] {
                    position: sticky !important;
                    bottom: -18px !important;
                    z-index: 3 !important;
                    background: #171f2d !important;
                    padding: 10px 0 4px !important;
                    margin-top: 6px !important;
                    gap: 8px !important;
                    flex-wrap: wrap !important;
                }
                [id^="fap_editor_actions_"] button {
                    min-width: 98px !important;
                    height: 40px !important;
                    min-height: 40px !important;
                    flex: 1 1 100px !important;
                }
                [id^="fap_preset_editor_"].fap-editor-closing { display:none !important; }
                @media (max-height: 650px) {
                    [id^="fap_preset_editor_"].fap-editor-portaled { padding: 10px !important; }
                    [id^="fap_preset_editor_"].fap-editor-portaled [id^="fap_editor_text_"] textarea {
                        height: 26vh !important;
                        min-height: 120px !important;
                    }
                    [id^="fap_editor_actions_"] { bottom: -10px !important; }
                }
            </style>""")

            with gr.Group(visible=False, elem_id=f"fap_preset_editor_{suffix}") as editor_box:
                gr.Markdown("### System prompt presets")
                editor_pick = gr.Dropdown(label="Preset", choices=presets,
                                          value=_preferred_preset("Enhance text"))
                editor_name = gr.Textbox(label="Name", lines=1, max_lines=1,
                                          elem_id=f"fap_editor_name_{suffix}")
                editor_text = gr.Textbox(label="Text", lines=12, max_lines=22,
                                         elem_id=f"fap_editor_text_{suffix}")
                editor_status = gr.Textbox(label="Preset status", lines=1,
                                           max_lines=1, interactive=False,
                                           elem_id=f"fap_editor_status_{suffix}")
                delete_armed = gr.State(value="")
                with gr.Row(elem_id=f"fap_editor_actions_{suffix}"):
                    editor_new = gr.Button("New")
                    editor_save = gr.Button("Save", variant="primary")
                    editor_duplicate = gr.Button("Duplicate")
                    editor_delete = gr.Button("Delete")
                    editor_close = gr.Button("Close", elem_id=f"fap_preset_close_{suffix}")

            with gr.Row():
                read_prompt = gr.Button("Read Forge prompt", variant="secondary")
                # Saves the model choice without touching the generation backend.
                save_model = gr.Button("Save model choice", variant="secondary")

            mode_info = gr.Textbox(
                label="Mode", value="Enhance text: the reference image is ignored.",
                visible=False, interactive=False,
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=250):
                    prompt = gr.Textbox(
                        label="Input prompt / instructions", lines=4,
                        show_copy_button=True,
                        placeholder="Enter text, or click Read Forge prompt...",
                        elem_id=f"fap_input_{suffix}",
                    )
                    ref_image = gr.Image(
                        label="Reference image (Qwen3-VL)", type="pil", height=170,
                        visible=False, elem_id=f"fap_image_{suffix}",
                    )
                    paste_clipboard = gr.Button(
                        "Paste from Clipboard", variant="secondary",
                        elem_id=f"fap_paste_clipboard_{suffix}",
                    )
                    clipboard_data = gr.Textbox(value="", visible=False)
                    include_text = gr.Checkbox(
                        label="Include text instructions with image", value=False,
                        visible=False,
                    )
                with gr.Column(scale=1, min_width=250):
                    output = gr.Textbox(
                        label="Generated prompt", lines=5, show_copy_button=True,
                        elem_id=f"fap_output_{suffix}",
                    )
                    with gr.Row():
                        run = gr.Button("Generate Text", variant="primary")
                        insert = gr.Button("Insert into Prompt", variant="secondary")

            with gr.Accordion("Generation settings", open=False):
                with gr.Row():
                    tokens = gr.Slider(
                        label="Max tokens", minimum=64, maximum=4096, step=64, value=1024,
                    )
                    temperature = gr.Slider(
                        label="Temperature", minimum=0.05, maximum=2.0, step=0.05, value=0.7,
                    )
                with gr.Row():
                    top_k = gr.Slider(label="Top K", minimum=0, maximum=128, step=1, value=64)
                    top_p = gr.Slider(label="Top P", minimum=0.05, maximum=1.0, step=0.01, value=0.95)
                    thinking = gr.Checkbox(label="Thinking", value=False)

            # The private runtime is managed by the extension; no maintenance
            # controls or internal paths are shown in the generation UI.
            # Keep the configured model folder as a hidden Gradio input so
            # the existing worker, model refresh and save callbacks work.
            encoder_root = gr.State(value=folder)

            refresh.click(_refresh_models, inputs=[encoder_root], outputs=[model])
            save_model.click(
                _save_config, inputs=[encoder_root, model, clip_type], outputs=[],
            )
            preset.change(_on_preset_change, inputs=[preset], outputs=[system_prompt], queue=False)
            refresh_presets.click(
                _refresh_presets, inputs=[preset], outputs=[preset, system_prompt], queue=False,
            )
            open_editor.click(
                _editor_open, inputs=[preset],
                outputs=[editor_box, editor_pick, editor_name, editor_text, editor_status], queue=False,
            )
            editor_pick.change(
                _editor_select, inputs=[editor_pick],
                outputs=[editor_name, editor_text, editor_status, delete_armed], queue=False,
            )
            editor_new.click(
                _editor_new, inputs=[],
                outputs=[editor_pick, editor_name, editor_text, editor_status, delete_armed], queue=False,
            )
            editor_outputs = [preset, system_prompt, editor_pick, editor_name,
                              editor_text, editor_status, delete_armed]
            editor_save.click(_editor_save,
                              inputs=[editor_name, editor_text, editor_pick],
                              outputs=editor_outputs, queue=False)
            editor_duplicate.click(_editor_duplicate, inputs=[editor_pick],
                                   outputs=editor_outputs, queue=False)
            editor_delete.click(_editor_delete, inputs=[editor_pick, delete_armed],
                                outputs=editor_outputs, queue=False)
            editor_close.click(lambda: gr.update(visible=False), inputs=[],
                               outputs=[editor_box], queue=False)
            mode.change(
                _on_mode_change, inputs=[mode],
                outputs=[mode_info, ref_image, include_text],
                queue=False,
            )
            ref_image.change(
                _on_image_change, inputs=[ref_image, mode],
                outputs=[mode, mode_info],
                queue=False,
            )
            paste_clipboard.click(
                _fap_paste_clipboard, inputs=[clipboard_data],
                outputs=[ref_image, mode, include_text],
                _js=_FAP_CLIPBOARD_JS, queue=False,
            )
            run.click(
                _generate,
                inputs=[encoder_root, model, clip_type, prompt, system_prompt,
                        ref_image, tokens, temperature, top_k, top_p, thinking,
                        mode, include_text],
                outputs=[output], show_progress="full",
            )

            # Uses the same Gradio-compatible _js callback approach as Tagger
            # Prompt. The main Forge prompt and this tool's input are distinct.
            # Read it only on explicit click; never overwrite user text silently.
            read_prompt.click(
                fn=None, inputs=[], outputs=[prompt],
                _js=f"""() => {{
                    const root = (typeof gradioApp === 'function') ? gradioApp() : document;
                    const el = root.querySelector('#{parent_prompt} textarea');
                    return el ? el.value : '';
                }}""",
            )
            insert.click(
                fn=None, inputs=[output], outputs=[],
                _js=f"""(value) => {{
                    const root = (typeof gradioApp === 'function') ? gradioApp() : document;
                    const target = root.querySelector('#{parent_prompt} textarea');
                    if (!target) {{ console.warn('[Forge AI Prompt] Forge prompt field was not found.'); return []; }}
                    const text = (value ?? '').toString();
                    if (!text.trim()) {{ console.warn('[Forge AI Prompt] Nothing to insert.'); return []; }}
                    target.value = text;
                    if (typeof updateInput === 'function') {{
                        try {{ updateInput(target); }} catch (e) {{
                            target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }} else {{
                        target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                    target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    // FAP_PROMPT_SCROLL_V1
                    try {{ target.focus({{ preventScroll: true }}); }} catch (e) {{ target.focus(); }}
                    requestAnimationFrame(() => {{
                        if (target.isConnected) {{
                            target.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
                        }}
                    }});
                    console.info('[Forge AI Prompt] Inserted into {"img2img" if is_img2img else "txt2img"} prompt.');
                    return [];
                }}""",
            )

        # The extension is a utility, not a sampling script. Do not add inputs
        # to the native image-generation pipeline.
        return []
