"""Persistent private worker. No Comfy imports in Forge's UI process."""
from __future__ import annotations
import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_LOCK = threading.RLock()
_proc = None
_key = None


def _stop_locked():
    global _proc, _key
    p, _proc, _key = _proc, None, None
    if p is None:
        return
    try:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
    finally:
        if p.stdin is not None:
            try:
                p.stdin.close()
            except OSError:
                pass


def stop():
    with _LOCK:
        _stop_locked()


atexit.register(stop)


def _start_locked(worker, core, env):
    global _proc, _key
    key = (str(worker), str(core), sys.executable)
    if _proc is not None and _proc.poll() is None and _key == key:
        return _proc
    _stop_locked()
    print('[Forge AI Prompt] Starting persistent worker (engine warm-up)...', flush=True)
    _proc = subprocess.Popen(
        [sys.executable, '-u', str(worker), '--serve'],
        stdin=subprocess.PIPE, stdout=None, stderr=None,
        cwd=str(core), env=env, shell=False,
    )
    _key = key
    return _proc


def warmup(worker, core, env):
    with _LOCK:
        _start_locked(worker, core, env)


def run(worker, core, env, request, timeout):
    with _LOCK:
        with tempfile.TemporaryDirectory(prefix='forge_ai_prompt_result_') as temp_dir:
            result_file = Path(temp_dir) / 'result.json'
            payload = dict(request, result_path=str(result_file))
            wire = (json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8')
            try:
                p = _start_locked(worker, core, env)
                start = time.monotonic()
                p.stdin.write(wire)
                p.stdin.flush()
                while not result_file.is_file():
                    if p.poll() is not None:
                        # A result may have been atomically written just before exit.
                        if result_file.is_file():
                            break
                        raise RuntimeError(f'Text worker exited with code {p.returncode}. Check the console.')
                    if time.monotonic() - start >= timeout:
                        raise subprocess.TimeoutExpired(p.args, timeout)
                    time.sleep(0.05)
                obj = json.loads(result_file.read_text(encoding='utf-8'))
                if not obj.get('ok') or obj.get('restart_worker'):
                    _stop_locked()
                print(f'[Forge AI Prompt] Request round trip: {time.monotonic() - start:.2f} s.', flush=True)
                return obj
            except BaseException:
                _stop_locked()
                raise
