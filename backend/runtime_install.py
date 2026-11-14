"""Install a pinned Comfy text-generation backend inside this Forge extension.

This is NOT a separate ComfyUI application; only Python source is extracted.
It uses sys.executable, torch and CUDA from the hosting Forge instance.
The installer NEVER installs/upgrades torch, torchvision, xformers, triton,
SageAttention or other GPU acceleration packages.
"""
from __future__ import annotations

import importlib.util
from importlib import metadata as package_metadata
import io
import json
import os
import shutil
import platform
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

EXTENSION = Path(__file__).resolve().parents[1]
RUNTIME_PARENT = EXTENSION / ".runtime"
CORE = RUNTIME_PARENT / "ComfyUI"
COMFY_COMMIT = "dc9ffe6adde25671bf535e99556ca9f8085c7d0f"
SOURCE_URL = f"https://codeload.github.com/Comfy-Org/ComfyUI/zip/{COMFY_COMMIT}"
FALLBACK_URL = f"https://github.com/Comfy-Org/ComfyUI/archive/{COMFY_COMMIT}.zip"
LOCK = threading.Lock()
STARTUP_STAMP = RUNTIME_PARENT / "startup_verified.json"

# Import name => pip distribution. Deliberately no torch, torchvision, CUDA etc.
# These are installed only when the Forge interpreter does not already have them.
SAFE_IMPORTS = {
    "comfy_kitchen": "comfy-kitchen",
    "comfy_aimdo": "comfy-aimdo==0.6.0",
    "comfy_angle": "comfy-angle",
    "safetensors": "safetensors",
    "einops": "einops",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece",
    "omegaconf": "omegaconf",
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface-hub",
    "accelerate": "accelerate",
    "kornia": "kornia",
    "spandrel": "spandrel",
    "torchsde": "torchsde",
    "av": "av",
    "scipy": "scipy",
    "psutil": "psutil",
    "pydantic": "pydantic",
    "PIL": "Pillow",
    "numpy": "numpy",
    "blake3": "blake3",
    "tqdm": "tqdm",
    "simpleeval": "simpleeval",
    "filelock": "filelock",
    "requests": "requests",
}

# Extract the backend code needed for comfy.sd and clip generation. No
# standalone UI/server bootstrap, model weights, custom nodes or Python executables.
FOLDERS = {"comfy", "comfy_extras", "comfy_api", "comfy_execution", "comfy_utils", "utils", "blueprints"}
# Root-level helpers are imported from inside comfy/ during import comfy.sd.
# Keep all Comfy backend root helpers but not the standalone server entrypoints.
ROOT_FILES = {
    "folder_paths.py", "comfyui_version.py", "LICENSE", "LICENSE.txt", "requirements.txt",
    "node_helpers.py", "latent_preview.py", "cuda_malloc.py", "protocol.py",
    "hook_breaker_ac10a0.py", "execution.py", "nodes.py",
}
# Block runtime/UI/data folders ONLY at the root of the archive. In
# particular, comfy/ldm/models is Python source code, NOT downloadable weights.
ROOT_BLOCKED = {"models", "input", "output", "temp", "user", ".github", "tests", "test", ".venv", "venv"}
ANY_DEPTH_BLOCKED = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".github"}
# These exist in the pinned source tree and are imported by comfy.sd. Older
# installer builds could miss them while still passing the previous ready().
REQUIRED_SOURCE = (
    "comfy/sd.py",
    "folder_paths.py",
    "comfy/ldm/models/autoencoder.py",
    # v0.2.3 excluded root-level Python helpers, breaking comfy.hooks.
    "node_helpers.py",
    "cuda_malloc.py",
    "latent_preview.py",
    "protocol.py",
    "hook_breaker_ac10a0.py",
    "execution.py",
    "nodes.py",
)
MAX_ZIP_DOWNLOAD = 170 * 1024 * 1024
MAX_UNCOMPRESSED = 260 * 1024 * 1024


def missing_source() -> list[str]:
    """Detect missing local source modules from earlier incomplete installers."""
    return [part for part in REQUIRED_SOURCE if not (CORE / Path(part)).is_file()]


def ready() -> bool:
    return not missing_source()


def describe() -> str:
    if not ready():
        missing = ", ".join(missing_source())
        if CORE.is_dir():
            return "Local runtime is incomplete. Restart Forge to restore missing files.\nMissing: " + missing
        return "Local generation components are not installed. Restart Forge to install them automatically."
    manifest = CORE / ".forge_ai_prompt_manifest.json"
    try:
        version = json.loads(manifest.read_text(encoding="utf-8")).get("commit", "unknown")
    except Exception:
        version = "unknown"
    return f"Local engine installed: {CORE}\nComfy source revision: {version}\nPython: {sys.executable}"


def _download() -> bytes:
    last = None
    for url in (SOURCE_URL, FALLBACK_URL):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Forge-AI-Prompt/0.2.4"})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_ZIP_DOWNLOAD:
                    raise RuntimeError("Comfy source archive is unexpectedly large")
                buf = io.BytesIO()
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    if buf.tell() + len(block) > MAX_ZIP_DOWNLOAD:
                        raise RuntimeError("Comfy source download exceeded the safety limit")
                    buf.write(block)
                return buf.getvalue()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not download the official Comfy source archive: {last}")


def _extract_archive(raw: bytes, target: Path) -> int:
    written = 0
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        if len(infos) > 40000:
            raise RuntimeError("Unexpectedly large number of files in Comfy source archive")
        for info in infos:
            if info.is_dir():
                continue
            # Reject paths that Windows could reinterpret as absolute or traversing.
            if "\\" in info.filename or ":" in info.filename:
                continue
            pp = PurePosixPath(info.filename)
            parts = pp.parts
            # Zip entries are ComfyUI-<sha>/comfy/..., not system paths.
            if len(parts) < 2 or not parts[0].lower().startswith("comfyui-") \
                    or any(p in {"", ".", ".."} for p in parts):
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.is_absolute() or any(p in ANY_DEPTH_BLOCKED for p in relative.parts) \
                    or relative.parts[0] in ROOT_BLOCKED:
                continue
            if relative.parts[0] not in FOLDERS and str(relative) not in ROOT_FILES:
                continue
            # Ignore symlinks, preserving only regular source files.
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                continue
            total_size += info.file_size
            if total_size > MAX_UNCOMPRESSED:
                raise RuntimeError("Comfy source archive exceeded maximum uncompressed size")
            dest = target.joinpath(*relative.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, 256 * 1024)
            written += 1
    missing = [part for part in REQUIRED_SOURCE if not (target / Path(part)).is_file()]
    if missing:
        raise RuntimeError("Downloaded source is incomplete (missing " + ", ".join(missing) + ")")
    return written


def _find_missing():
    """Check names and the pinned AIMDO ABI under the Forge interpreter."""
    result = []
    for mod, package in SAFE_IMPORTS.items():
        try:
            installed = importlib.util.find_spec(mod) is not None
        except (ModuleNotFoundError, ValueError):
            installed = False
        if mod == "comfy_aimdo" and installed:
            # A stale version can have a package but not host_buffer or
            # other modules required by the pinned Comfy core snapshot.
            try:
                installed = package_metadata.version("comfy-aimdo") == "0.6.0"
            except package_metadata.PackageNotFoundError:
                installed = False
        if not installed:
            result.append((mod, package))
    return result


def _install_packages(missing, log):
    """Install only absent packages; never resolve/replace Forge's CUDA stack."""
    if not missing:
        log.append("All checked Python libraries are already present in Forge.")
        return
    names = [package for _, package in missing]
    log.append("Missing packages: " + ", ".join(names))
    uv = shutil.which("uv")
    runners = []
    if uv:
        runners.append(("uv", [uv, "pip", "install", "--python", sys.executable,
                                 "--no-deps", "--only-binary", ":all:", *names]))
    runners.append(("pip", [sys.executable, "-m", "pip", "install", "--no-input",
                            "--disable-pip-version-check", "--no-deps",
                            "--only-binary=:all:", *names]))
    last_failure = None
    for name, command in runners:
        log.append(f"Trying {name} in the current Forge interpreter (no dependencies, binary wheels only)...")
        try:
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_failure = f"{name}: {type(exc).__name__}: {exc}"
            log.append(last_failure)
            continue
        log.append(proc.stdout[-4000:].strip() or f"{name}: no console output")
        if proc.returncode == 0:
            log.append(f"{name} completed successfully.")
            importlib.invalidate_caches()
            return
        last_failure = f"{name} exited with code {proc.returncode}"
    raise RuntimeError("Could not install missing runtime packages without modifying Forge Torch/CUDA. "
                       + str(last_failure))


def _pip_missing(log):
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Forge Python has no PyTorch. Will not install or replace PyTorch.")
    missing = _find_missing()
    # Install the failing core package first: a less relevant package being
    # unavailable must not prevent repairing the reported comfy_aimdo error.
    essential = [(mod, package) for mod, package in missing if mod in {"comfy_aimdo"}]
    other = [(mod, package) for mod, package in missing if mod != "comfy_aimdo"]
    if essential:
        _install_packages(essential, log)
    _install_packages(other, log)


def _probe(log):
    """Verify the SAME worker and Python environment that Generate Text uses."""
    if not ready():
        raise RuntimeError("Local runtime source is incomplete: " + ", ".join(missing_source()))
    worker = EXTENSION / "backend" / "comfy_worker.py"
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(CORE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        proc = subprocess.run([sys.executable, "-u", str(worker)],
                              input=json.dumps({"comfy_root": str(CORE), "probe_only": True}),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              encoding="utf-8", errors="replace", timeout=120,
                              cwd=str(CORE), shell=False, env=env)
    except Exception as exc:
        raise RuntimeError(f"Could not run backend import test: {type(exc).__name__}: {exc}") from exc
    prefix = "__FORGE_AI_PROMPT_RESULT__:"
    result_line = next((line[len(prefix):] for line in reversed(proc.stdout.splitlines())
                        if line.startswith(prefix)), None)
    try:
        answer = json.loads(result_line) if result_line else None
    except ValueError:
        answer = None
    if not answer or not answer.get("ok"):
        message = (answer or {}).get("error") or (proc.stderr or proc.stdout)[-1400:] or "No worker output"
        raise RuntimeError(f"Runtime import test failed: {message}"
                           "\nRestart Forge. If installation still fails, send the complete console error text.")
    log.append("Engine import: OK — " + answer.get("runtime", "worker probe succeeded"))


def _ensure_source(log):
    """Repair incomplete source from the pinned archive, without touching Forge.

    Existing complete local source is kept. Recovery replaces only our own
    .runtime/ComfyUI folder, through a staging directory and rollback backup.
    """
    if ready():
        log.append("Local Comfy components are complete; no source download needed.")
        return
    missing = missing_source()
    if CORE.is_dir():
        log.append("Incomplete source detected: " + ", ".join(missing))
        log.append("Downloading pinned source to repair missing files (not reinstalling Forge).")
    else:
        log.append("Local Comfy components are not installed; downloading pinned source.")
    raw = _download()
    log.append(f"Downloaded {len(raw) / 1048576:.1f} MiB from the official ComfyUI repository.")
    RUNTIME_PARENT.mkdir(parents=True, exist_ok=True)
    backup = RUNTIME_PARENT / "ComfyUI.previous"
    with tempfile.TemporaryDirectory(prefix="forge_ai_prompt_", dir=RUNTIME_PARENT) as temp:
        scratch = Path(temp) / "ComfyUI"
        scratch.mkdir()
        files = _extract_archive(raw, scratch)
        (scratch / ".forge_ai_prompt_manifest.json").write_text(
            json.dumps({"commit": COMFY_COMMIT, "source": "Comfy-Org/ComfyUI", "extractor": 3}, indent=2),
            encoding="utf-8",
        )
        log.append(f"Extracted {files} source files and verified required modules.")
        # Rename is local to .runtime. On failure, restore the previous folder.
        if backup.exists():
            shutil.rmtree(backup)
        if CORE.exists():
            CORE.rename(backup)
        try:
            scratch.rename(CORE)
        except Exception:
            if backup.exists() and not CORE.exists():
                backup.rename(CORE)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    log.append("Local source repaired: " + str(CORE))


def repair() -> str:
    """Restore missing source and Python libraries without replacing Torch/CUDA."""
    if not LOCK.acquire(blocking=False):
        return "A runtime installation or repair is already running."
    log = []
    try:
        log.append("Repairing local generation components using Forge Python: " + sys.executable)
        _ensure_source(log)
        _pip_missing(log)
        _probe(log)
        log.append("REPAIR COMPLETED. Try Generate Text again.")
        return "\n".join(log)
    except Exception as exc:
        log.append(f"ERROR: {type(exc).__name__}: {exc}")
        log.append("Torch/CUDA, xFormers and SageAttention were not deliberately changed.")
        return "\n".join(log)
    finally:
        LOCK.release()


def install() -> str:
    if not LOCK.acquire(blocking=False):
        return "An installation or repair is already running."
    log = []
    try:
        log.append("Installing/verifying local generation components using Forge Python: " + sys.executable)
        _ensure_source(log)
        _pip_missing(log)
        _probe(log)
        log.append("INSTALLATION VERIFIED. You can now Generate Text.")
        return "\n".join(log)
    except Exception as exc:
        log.append(f"ERROR: {type(exc).__name__}: {exc}")
        log.append("Torch/CUDA, xFormers and SageAttention were not deliberately changed.")
        return "\n".join(log)
    finally:
        LOCK.release()


def _startup_signature() -> dict:
    """Cheap, import-free fingerprint of the Forge venv and local sources.

    Recreating a venv at the same path changes pyvenv.cfg's timestamp, so the
    next Forge startup runs the worker import probe again even if the package
    versions look identical. Missing deps are checked *every* startup anyway.
    """
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    try:
        venv_mtime = cfg.stat().st_mtime_ns
    except OSError:
        venv_mtime = None
    manifest = CORE / ".forge_ai_prompt_manifest.json"
    try:
        manifest_revision = json.loads(manifest.read_text(encoding="utf-8")).get("commit")
    except (OSError, ValueError, AttributeError):
        manifest_revision = None
    versions = {}
    for package in ("torch", "comfy-aimdo", "comfy-kitchen", "transformers", "tokenizers"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "venv_config_mtime_ns": venv_mtime,
        "source_commit": manifest_revision,
        "required_source_revision": COMFY_COMMIT,
        "packages": versions,
    }


def _startup_stamp_matches(signature) -> bool:
    try:
        previous = json.loads(STARTUP_STAMP.read_text(encoding="utf-8"))
        return previous == signature
    except (OSError, ValueError):
        return False


def _write_startup_stamp(signature):
    RUNTIME_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                     prefix=".fap_verified_", suffix=".json", dir=RUNTIME_PARENT) as tmp:
        temp_path = Path(tmp.name)
        json.dump(signature, tmp, indent=2, ensure_ascii=False)
    try:
        os.replace(temp_path, STARTUP_STAMP)
    finally:
        temp_path.unlink(missing_ok=True)


def startup_install() -> str:
    """Idempotent installer for Forge launch_utils.run_extension_installer.

    Runs in the startup installer process, *before* Gradio starts. Never
    imports the Comfy package into Forge and never installs the GPU stack.
    """
    if not LOCK.acquire(blocking=False):
        return "ERROR: another AI Prompt runtime installation is already running."
    log = []
    try:
        # Forge installs its own requirements first. If the user has deleted
        # the *entire* venv and Forge hasn't restored Torch, this extension
        # cannot reconstruct Forge's environment on its behalf.
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "PyTorch is not installed in Forge's Python. Restore Forge's "
                "own venv/startup requirements first; AI Prompt will not install Torch."
            )
        source_missing = missing_source()
        package_missing = _find_missing()
        before = _startup_signature()
        clean = not source_missing and not package_missing
        if clean and _startup_stamp_matches(before):
            return "Runtime and Python dependencies already present; skipping installation."

        if source_missing:
            log.append("Missing local Comfy components: " + ", ".join(source_missing))
            _ensure_source(log)
        else:
            log.append("Local Comfy components: OK (no download).")

        if package_missing:
            log.append("Missing or incompatible Python libraries detected; restoring without Torch/CUDA changes.")
            _pip_missing(log)
        else:
            log.append("Python dependencies: OK (no pip/uv changes).")

        # Only perform the slow worker import check on the first installation,
        # a recreated venv, or when dependencies/source were repaired.
        _probe(log)
        _write_startup_stamp(_startup_signature())
        log.append("Auto-install verified. AI Prompt is ready.")
        return "\n".join(log)
    except Exception as exc:
        # A failed probe/repair must not be treated as successful on next boot.
        try:
            STARTUP_STAMP.unlink(missing_ok=True)
        except OSError:
            pass
        log.append(f"ERROR: {type(exc).__name__}: {exc}")
        log.append("AI Prompt installation failed; Forge's Torch/CUDA stack was not deliberately modified.")
        return "ERROR:\n" + "\n".join(log)
    finally:
        LOCK.release()
