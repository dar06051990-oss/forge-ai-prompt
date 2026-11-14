# Changelog

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
