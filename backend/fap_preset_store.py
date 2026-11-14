"""Forge AI Prompt system-prompt presets.

Bundled read-only defaults: <extension>/presets/*.txt
User edits, new presets and deleted-default markers: <extension>/user_presets/

Only prompt text is stored. Generation parameters are intentionally not saved here.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path

EXTENSION = Path(__file__).resolve().parents[1]
BUILTIN = EXTENSION / "presets"
USER = EXTENSION / "user_presets"
DELETED = USER / "_deleted.json"
_LOCK = threading.RLock()
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{x}" for x in range(1, 10)), *(f"LPT{x}" for x in range(1, 10))}


def validate_name(raw: str) -> str:
    name = str(raw or "").strip()
    if not name or len(name) > 96:
        raise ValueError("Preset name must contain 1-96 characters.")
    if _FORBIDDEN.search(name) or name in {".", ".."} or name.endswith((".", " ")):
        raise ValueError("Preset name contains characters not allowed in Windows filenames.")
    if name.split(".", 1)[0].upper() in _RESERVED:
        raise ValueError("Preset name is reserved by Windows.")
    if name.casefold() == "_deleted":
        raise ValueError("This preset name is reserved for internal settings.")
    return name


def _deleted_names() -> set[str]:
    try:
        data = json.loads(DELETED.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x).casefold() for x in data}
    except (OSError, ValueError, TypeError):
        pass
    return set()


def _atomic_write(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".fap_", suffix=".tmp", delete=False, newline="\n") as stream:
            tmp = Path(stream.name)
            stream.write(value)
        os.replace(str(tmp), str(path))
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)


def _filenames(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        return {}
    found = {}
    for item in folder.glob("*.txt"):
        if item.is_file() and not item.is_symlink():
            try:
                name = validate_name(item.stem)
            except ValueError:
                continue
            found.setdefault(name.casefold(), name)
    return found


def list_names() -> list[str]:
    with _LOCK:
        names = _filenames(BUILTIN)
        names.update(_filenames(USER))
        for name in _deleted_names():
            names.pop(name, None)
        return sorted(names.values(), key=str.casefold)


def load(name: str) -> str:
    with _LOCK:
        # Only allow names in the visible list. Never construct a filesystem
        # path from an unchecked component provided by the browser.
        choices = {s.casefold(): s for s in list_names()}
        actual = choices.get(str(name or "").casefold())
        if actual is None:
            return ""
        for path in (USER / (actual + ".txt"), BUILTIN / (actual + ".txt")):
            if path.is_file():
                return path.read_text(encoding="utf-8")
        return ""


def save(name: str, text: str, *, previous_name: str | None = None) -> str:
    name = validate_name(name)
    if not str(text or "").strip():
        raise ValueError("System prompt cannot be empty.")
    if len(text) > 150_000:
        raise ValueError("System prompt is too long (150,000 character limit).")
    with _LOCK:
        choices = {s.casefold(): s for s in list_names()}
        old = choices.get(str(previous_name or "").casefold())
        match = choices.get(name.casefold())
        if match and (not old or match.casefold() != old.casefold()):
            raise ValueError(f"Preset '{match}' already exists. Select it before editing.")
        if match:
            # Windows file systems ignore case; keep the existing spelling.
            name = match
        _atomic_write(USER / (name + ".txt"), text)
        deleted = _deleted_names()
        deleted.discard(name.casefold())
        if old and old.casefold() != name.casefold():
            # Renaming a built-in default hides the original, without deleting
            # the packaged copy. Renaming custom presets removes their old file.
            old_user = USER / (old + ".txt")
            if old_user.is_file():
                old_user.unlink()
            if (BUILTIN / (old + ".txt")).is_file():
                deleted.add(old.casefold())
        _atomic_write(DELETED, json.dumps(sorted(deleted), ensure_ascii=False, indent=2))
        return name


def duplicate(name: str) -> str:
    with _LOCK:
        actual = {s.casefold(): s for s in list_names()}.get(str(name or "").casefold())
        if not actual:
            raise ValueError("Select a preset to duplicate.")
        # Allow lengthy names while respecting the Windows filename limit.
        prefix = actual[:77]
        candidate = f"{prefix} - Copy"
        k = 2
        existing = {s.casefold() for s in list_names()}
        while candidate.casefold() in existing:
            candidate = f"{prefix} - Copy {k}"
            k += 1
        return save(candidate, load(actual))


def delete(name: str):
    with _LOCK:
        actual = {s.casefold(): s for s in list_names()}.get(str(name or "").casefold())
        if not actual:
            raise ValueError("Select a preset to delete.")
        user_file = USER / (actual + ".txt")
        if user_file.is_file():
            user_file.unlink()
        deleted = _deleted_names()
        deleted.add(actual.casefold())
        _atomic_write(DELETED, json.dumps(sorted(deleted), ensure_ascii=False, indent=2))
