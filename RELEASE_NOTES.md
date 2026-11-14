# Forge AI Prompt v0.2.16

First public release of **Forge AI Prompt**, a local AI prompt generation extension for **Forge Neo**.

### Highlights
- **Text → prompt:** Enhance short image-generation instructions.
- **Image → prompt:** Describe a reference image using a compatible Qwen3-VL text encoder.
- **Russian and English:** Fixed Windows UTF-8 input handling for Cyrillic prompts.
- **Native Forge UI:** Available in both txt2img and img2img, with copy/insert actions.
- **Preset manager:** Built-in and editable custom system prompts.
- **Live console progress:** Model loading, generation progress, elapsed time and tokens/s.
- **VRAM handoff:** Unloads Forge image-generation model weights before text generation when necessary.
- **Automatic setup:** `install.py` verifies and restores the local backend and missing dependencies on normal Forge startup.

### Requirements
Forge Neo on Windows with a working Python environment; a compatible generation-capable `.safetensors` Qwen3-VL encoder. Model weights are **not included**.

### Notes
The extension uses a private snapshot of ComfyUI components downloaded during installation; it does **not** install or run a separate ComfyUI server. The installer avoids replacing PyTorch and GPU accelerator packages. Only Windows/Forge Neo has been tested.

**License:** GNU GPL v3.0. See [LICENSE](LICENSE).

See [README.md](README.md) for full requirements, configuration, usage and troubleshooting.
