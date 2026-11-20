# Forge AI Prompt

**Local AI prompt generation for Forge Neo — directly inside txt2img and img2img.**

Forge AI Prompt turns short instructions into detailed image-generation prompts, or describes a reference image as a prompt. It runs locally using a compatible Qwen3-VL text encoder and a private, ComfyUI-based generation backend. No LM Studio, external API key, or separately running ComfyUI server is required.

> **Current version: v0.2.19** · Performance and usability update · Tested on Forge Neo for Windows with NVIDIA RTX 4070. Other Forge variants and hardware have not yet been verified.

## Features

- **Enhance text:** Expand short instructions into a detailed prompt.
- **Describe image:** Generate a prompt from an uploaded reference image; optionally include text instructions. Adding, replacing or removing the image preserves the current prompt, system preset and generation settings.
- **Clipboard images:** Use **Paste from Clipboard** to paste a reference image directly from the browser (clipboard permission and a secure context such as localhost/HTTPS required). Ctrl+V and normal image upload remain available.
- **Multilingual input:** English and Russian prompts are supported. v0.2.16 fixes UTF-8 input handling on Windows.
- **System prompt presets:** Choose from bundled presets or create, edit, duplicate and delete your own.
- **Native Forge UI:** One collapsible **AI Prompt** panel in both txt2img and img2img. Read the current Forge prompt, copy text, and insert generated text back into Forge. **Insert into Prompt** also focuses and smoothly scrolls to the main prompt field.
- **Generation controls:** Max tokens, Temperature, Top K, Top P and Thinking.
- **Persistent worker:** The private generation engine warms up once at Forge startup and caches the selected encoder across requests.
- **Dynamic VRAM (AIMDO):** Uses ComfyUI's dynamic model patcher when supported; automatically falls back to the legacy loader if initialization is unavailable.
- **Console progress:** Model loading, cache hits, token progress, elapsed time and tokens/s appear in the Forge console. Routine backend startup diagnostics and no-op installer messages are suppressed; warnings and errors remain visible.
- **VRAM handoff:** Releases Forge's currently loaded image-generation weights only when **Generate Text** is pressed. Forge reloads the selected checkpoint when image generation resumes.
- **Automatic setup:** Forge invokes `install.py` at startup to verify or restore missing local backend components and dependencies.

## Requirements

- A working installation of **Forge Neo** with its own Python environment and GPU dependencies already installed.
- An internet connection for the **first installation** or a repair of missing backend files/packages.
- A compatible **generation-capable** text encoder in `.safetensors` format. The configuration tested in development uses `qwen3-vl-4b-heretic.safetensors`. **Model weights are not included or downloaded automatically.**
- Sufficient GPU memory for the chosen text encoder. The tested setup uses an NVIDIA GPU with 12 GB VRAM; other configurations may behave differently.

## Installation

1. Close Forge Neo.
2. Open the `extensions` directory in your Forge installation.
3. Clone this repository as a folder named `forge-ai-prompt`:

   ```bash
   git clone https://github.com/dar06051990-oss/forge-ai-prompt.git
   ```

   Alternatively, download the source ZIP and extract its contents to `extensions/forge-ai-prompt/`.
4. Place your compatible `.safetensors` text encoder in Forge's model folder, usually `models/text_encoder/` or `models/text_encoders/`.
5. Start Forge Neo **normally**, without `--skip-install`. The extension's `install.py` checks the environment, downloads a pinned snapshot of the necessary ComfyUI Python components into `.runtime/ComfyUI` when needed, and installs missing packages into the **active Forge Python** environment.
6. In **txt2img** or **img2img**, expand **AI Prompt**, click **Refresh** if necessary, select the **Text encoder** and suitable **CLIP loader type**, and click **Save model choice**.
7. Select a preset, enter instructions or upload an image, then click **Generate Text**. You can copy the generated text or use **Insert into Prompt**.

> **Important:** The installer does not reinstall/upgrade PyTorch, torchvision, CUDA, xFormers, Triton or SageAttention. If Forge's entire `venv` was deleted, restore Forge's own environment using Forge's launcher first. Then the extension can reinstall its own missing dependencies.

## Usage

### Enhance text

Select **Enhance text**, then **Enhance prompt** (or a custom preset). Enter a short instruction such as:

`реалистичное фото, маленький рыжий котенок сидит на ветке дерева`

Click **Generate Text**. The resulting prompt appears in **Generated prompt** and can be sent to the native Forge prompt box.

### Describe image

Select **Describe image** to show the reference image control, then upload, paste with Ctrl+V, or use **Paste from Clipboard**. The clipboard button can also be used from Enhance text and switches the mode automatically. Choose **Image to prompt** or another system preset; switching mode or loading an image no longer changes your chosen preset or clears your text. Enable **Include text instructions with image** to combine the reference with your written instructions. Click **Generate Text**. The model may occasionally invent details; review its output before using it.

### Presets

The **Preset** dropdown selects the system instructions. Open **Edit presets** to create/edit/duplicate/delete presets. Bundled presets are in `presets/`; user-created presets are stored separately in `user_presets/` and are not intended to be overwritten during extension updates. Generation settings are shared, not stored per preset.

Bundled prompts include **Enhance prompt**, **Image to prompt**, and adapted presets for MOSS Image, Qwen-Image 2.1, FLUX.3 Klein and Krea 2.

### Performance and logging

The text generator runs in a persistent private Python subprocess using Forge's Python environment and local ComfyUI Python sources. It warms up once at Forge startup and caches one encoder and its tokenizer, with automatic reloading when the selected encoder or loader changes. On supported NVIDIA CUDA systems, AIMDO Dynamic VRAM and the dynamic model patcher avoid eagerly moving the entire encoder to the GPU; if initialization is unavailable, the legacy loader is retained. Set `FAP_DISABLE_DYNAMIC_VRAM=1` in the environment before starting Forge to test legacy loading. Logs and token progress remain in the **Forge console** (there is no Status textbox in the UI).

To make GPU memory available for text generation, the extension unloads Forge's image model when you press **Generate Text**. The worker releases GPU weights after each request while keeping its reusable model state; it is restarted when an error or unsafe VRAM handoff is detected. Your next image generation reloads the Forge image model. Switching tabs or changing presets **does not** unload the model.

In user-reported development tests on an RTX 4070, the initial text request took 35.0 seconds in v0.2.16 (303 tokens) and 12.83 seconds in v0.2.18 (342 tokens); after image generation, the persistent-worker-only build took 34.03 seconds and the Dynamic VRAM build took 11.87 seconds (303 and 350 tokens respectively). These are observed individual runs, not a controlled benchmark or a performance guarantee.

## Troubleshooting

- **No text encoder in the dropdown:** Put a compatible `.safetensors` model in Forge's default text encoder directory and click **Refresh**. The dropdown does not download models.
- **Missing dependencies or local Comfy sources:** Restart Forge normally and inspect console lines beginning `[Forge AI Prompt: installer]`. Make sure `--skip-install` is not enabled.
- **Deleted Forge `venv`:** First restore the main Forge environment. This extension cannot replace Forge's launcher or install its PyTorch stack.
- **Model loads but output is empty:** Check **Thinking**, **Max tokens**, selected model and **CLIP loader type**; review the console for errors.
- **Very slow text generation after image generation:** Check available VRAM and whether Forge model release is reported in the console.
- **Russian text generates unrelated results:** Update to **v0.2.16 or newer**; older versions could corrupt Cyrillic text when reading worker stdin on Windows.

## Files and local data

| Path | Purpose |
| --- | --- |
| `install.py` | Forge startup installer |
| `scripts/forge_ai_prompt.py` | Native Forge UI and model-memory handoff |
| `backend/comfy_worker.py` | Persistent local model-generation worker and Dynamic VRAM initialization |
| `backend/fap_worker_client.py` | Worker lifecycle, UTF-8 request protocol, and safe restart |
| `backend/runtime_install.py` | Downloads/verifies pinned Comfy Python sources and missing libraries |
| `backend/fap_preset_store.py` | Preset management |
| `javascript/fap_preset_modal.js` | Preset editor dialog |
| `presets/` | Bundled system prompts |
| `.runtime/` | Locally installed backend; **not** committed to Git |
| `user_presets/` | Personal presets; **not** committed to Git |
| `.forge_ai_prompt_settings.json` | Local encoder selection; **not** committed to Git |

## License

Forge AI Prompt is distributed under the **GNU General Public License version 3.0 (GPL-3.0)**. See [LICENSE](LICENSE). Third-party runtime components and model weights are governed by their respective licenses.

## Third-party software and attribution

- The runtime installer downloads selected components from the official [ComfyUI](https://github.com/Comfy-Org/ComfyUI) source archive, pinned to commit `dc9ffe6adde25671bf535e99556ca9f8085c7d0f`. Those components are **not included in this repository**. ComfyUI is licensed under **GPL-3.0**; its source archive contains the applicable license and notices.
- Some system prompt presets are **adaptations** of Comfy-Org workflow-template examples. They are identified as *adapted* in their names.
- Text encoder/model weights are distributed separately by their respective authors. Please check their model cards and licenses before downloading or redistributing them.

## Version history

See [CHANGELOG.md](CHANGELOG.md). The first public source release was **v0.2.16**. Version **v0.2.19** incorporates the tested v0.2.17/v0.2.18 speed patches and subsequent UI/console fixes; runtime pins are unchanged.

If you installed the experimental patches manually, back up your local extension before updating: its tracked files may be modified and `git pull` may refuse to overwrite them. Preserve `.runtime/`, `user_presets/` and `.forge_ai_prompt_settings.json`. The published source already contains the patches; do not reapply the old patch installers.
