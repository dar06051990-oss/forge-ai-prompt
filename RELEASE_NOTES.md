# Forge AI Prompt v0.2.19

Performance and usability update for the local AI prompt generation extension for Forge Neo.

### Highlights

- Persistent private worker, model/tokenizer caching and one-time startup initialization.
- ComfyUI Dynamic VRAM (AIMDO) with legacy fallback and safe GPU memory handoff.
- Faster request preparation and detailed generation timing.
- Cleaner startup console without hiding warnings or errors.
- Automatic focus and scrolling after inserting text into the main Forge prompt.
- Reference images and text instructions can be combined without resetting the prompt or preset.
- New **Paste from Clipboard** button (browser permission and localhost/HTTPS required).

This version incorporates the tested v0.2.17 and v0.2.18 development patches and subsequent UI fixes. The pinned ComfyUI source and Python/GPU dependencies are unchanged. Model weights and local user data are not included.

### Updating

Close Forge first. If you installed development patches manually, back up the current extension before updating; local modifications may prevent a normal Git pull. Keep `.runtime/`, `user_presets/`, and `.forge_ai_prompt_settings.json`. Do not reapply the old patches after installing v0.2.19.

See [README.md](README.md) for requirements and usage. Licensed under [GNU GPL v3.0](LICENSE).
