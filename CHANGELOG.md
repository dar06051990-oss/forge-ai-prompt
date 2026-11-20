# Changelog

## v0.2.19 — Performance and usability update

- Integrated the tested v0.2.17 persistent worker: one-time engine warm-up, encoder/tokenizer cache, UTF-8 request protocol, and recovery on worker failure.
- Integrated v0.2.18 Dynamic VRAM (AIMDO) bootstrap from the pinned ComfyUI core; legacy loading remains available when initialization is unavailable.
- Release GPU weights after requests and verify the VRAM handoff, including dynamic allocations; retain encoder state between requests where safe.
- Added timing for tokenization, CLIP.load_model, token generation, GPU handoff and request round trip.
- Reduced noisy backend startup logs and suppressed the routine no-op installer result without suppressing warnings/errors or changing installation checks.
- Focus and smoothly scroll to the native Forge prompt on **Insert into Prompt** in txt2img and img2img.
- Preserve text instructions, selected system preset and generation settings on reference image upload/replacement/removal and mode changes.
- Added **Paste from Clipboard** with safe browser permission/error handling and image size limits.
- Kept pinned runtime, dependencies, generation controls and user presets unchanged.

The integrated changes were tested in the preceding local patches on Windows/Forge Neo with RTX 4070. No new model weights or dependencies are bundled.

## v0.2.16 — Initial public release

- Fixed Cyrillic and other non-ASCII input on Windows by decoding worker stdin explicitly as UTF-8.
- Added automatic runtime/dependency verification at Forge startup (`install.py`, introduced in v0.2.15).
- Unloads Forge image-generation weights when **Generate Text** starts, reducing VRAM contention (v0.2.14).
- Removed obsolete runtime/status controls from the UI; live model-loading and token progress remain visible in the Forge console.
- Added copy buttons to input and output fields.
- Added editable system prompt presets and the preset editor dialog.
- Kept the extension as a native always-visible Forge script in txt2img and img2img.
- Text enhancement and reference-image description supported with compatible Qwen3-VL generation-capable encoders.

This is the first public release. Versions prior to v0.2.16 were development/test builds.
