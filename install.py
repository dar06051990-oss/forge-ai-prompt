"""Forge AI Prompt bootstrap, called by Forge's extension installer at startup.

Runs with the Python interpreter of the hosting Forge installation. Repairs
only this extension's local Comfy-derived sources and its missing dependencies.
Does not replace Torch, CUDA, xFormers, Triton or SageAttention.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


def main() -> int:
    extension = Path(__file__).resolve().parent
    sys.path.insert(0, str(extension / "backend"))

    try:
        import runtime_install
        message = runtime_install.startup_install()
        # FAP_QUIET_STARTUP_V2: suppress only the routine no-op result.
        if message == "Runtime and Python dependencies already present; skipping installation.":
            return 0
        for line in message.splitlines():
            print("[Forge AI Prompt: installer] " + line, flush=True)
        return 1 if message.startswith("ERROR:") else 0
    except Exception as exc:
        print(f"[Forge AI Prompt: installer] ERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("[Forge AI Prompt: installer] Existing Forge packages were not modified by this launcher.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
