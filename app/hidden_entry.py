"""Single application entry point used by installed and portable launches.

Installed Studio uses its installer-provisioned private CPython environment.
Portable Studio packages an independent embedded CPython environment beside
this same application source. The native portable launcher is only a splash
host; it delegates to this entry point without a second app implementation.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

try:
    from gpu_runtime_paths import activate_optional_gpu_packages
except ImportError:
    from .gpu_runtime_paths import activate_optional_gpu_packages

activate_optional_gpu_packages()

def _open_log(environment_key: str, legacy_environment_key: str | None = None):
    target = os.environ.get(environment_key, "").strip()
    if not target and legacy_environment_key:
        target = os.environ.get(legacy_environment_key, "").strip()
    if not target:
        return None
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8", buffering=1, errors="replace")

def _log_uncaught_exception(exc_type, exc_value, exc_tb) -> None:
    stream = sys.stderr
    if stream is not None:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=stream)
        stream.flush()

def _create_native_splash_client():
    """Connect to the optional portable native splash; installed launch is identical without the host UI."""
    if not os.environ.get("POLYFITTY_STARTUP_READY_EVENT") or not os.environ.get("POLYFITTY_STARTUP_STATUS_FILE"):
        return None
    try:
        from startup_splash import StartupSplash
        client = StartupSplash()
        return client if client.enabled else None
    except Exception:
        return None

def main() -> int:
    stdout_log = _open_log("POLYFITTY_STARTUP_STDOUT", "POLYFIT_STARTUP_STDOUT")
    stderr_log = _open_log("POLYFITTY_STARTUP_STDERR", "POLYFIT_STARTUP_STDERR")
    if stdout_log is not None:
        sys.stdout = stdout_log
    if stderr_log is not None:
        sys.stderr = stderr_log
    sys.excepthook = _log_uncaught_exception

    try:
        from runtime_contract import verify_runtime_manifest
        verify_runtime_manifest(require_source_match=True)

        splash = _create_native_splash_client()
        if splash is not None:
            splash.update("Loading the embedded Polyfitty Studio environment...")

        from startup_gpu_preflight import _run_startup_gpu_preflight
        startup_gpu_probe = _run_startup_gpu_preflight(splash)

        from main import main as gui_main
        return int(gui_main(startup_splash=splash, startup_gpu_probe=startup_gpu_probe))
    except BaseException:
        if 'splash' in locals() and splash is not None:
            splash.fail("Application startup failed. Review the Polyfitty Studio startup logs.")
        if sys.stderr is not None:
            traceback.print_exc(file=sys.stderr)
            try:
                sys.stderr.flush()
            except Exception:
                pass
        raise

if __name__ == "__main__":
    raise SystemExit(main())
