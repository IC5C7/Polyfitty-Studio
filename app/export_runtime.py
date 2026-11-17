from __future__ import annotations

"""Runtime preflight helpers for reliable mesh export on the private Windows Python."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

def required_export_modules(
    suffix: str, bake_enabled: bool = False, conversion_from_baked: bool = False
) -> tuple[str, ...]:
    """Return Python modules required by the selected export path."""
    extension = str(suffix or "").lower()
    if extension == ".obj" and not conversion_from_baked:

        return ()
    modules = ["trimesh"]
    if conversion_from_baked or extension == ".fbx":
        modules.append("PIL")
    if extension == ".3mf":

        modules.append("lxml")
    if extension in {".glb", ".gltf"} and bake_enabled:
        modules.extend(("PIL", "scipy", "xatlas"))
    return tuple(modules)

def missing_export_modules(
    suffix: str, bake_enabled: bool = False, conversion_from_baked: bool = False
) -> tuple[str, ...]:
    """List missing imports before starting an export operation."""
    missing: list[str] = []
    for module_name in required_export_modules(suffix, bake_enabled, conversion_from_baked):
        try:
            found = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            found = None
        if found is None:
            missing.append(module_name)
    return tuple(missing)

def export_dependency_message(missing: tuple[str, ...] | list[str]) -> str:
    names = ", ".join(str(name) for name in missing)
    return (
        "Export components are missing from Polyfitty Studio's private Python runtime: "
        f"{names}.\n\nRun INSTALL.bat from the extracted Polyfitty Studio folder to restore "
        "the pinned export dependencies. The installer does not modify your system Python."
    )

def ensure_pip_available(
    python_executable: str | Path,
    bootstrap_script: str | Path,
    cache_dir: str | Path,
    *,
    platform: str | None = None,
    run: Callable = subprocess.run,
    which: Callable = shutil.which,
) -> tuple[bool, str]:
    """Check `python -m pip`; restore it with the app-local PS helper on Windows.

    The executable shim (Scripts/pip.exe) is intentionally not used as the
    health check: only importing pip through the exact app interpreter proves
    that `python -m pip` will work for installer/export maintenance.
    """
    executable = str(python_executable)
    helper = Path(bootstrap_script)
    cache = Path(cache_dir)
    probe = [executable, "-c", "import pip; print(pip.__version__)"]
    try:
        check = run(probe, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not check pip in the private Python runtime: {exc}"
    if check.returncode == 0:
        return True, (check.stdout or "pip is available").strip()

    current_platform = sys.platform if platform is None else platform
    if current_platform != "win32":
        return False, (
            "pip is missing from the private Python runtime. Automatic recovery is "
            "available through the Windows INSTALL.bat / START.bat scripts."
        )
    if not helper.is_file():
        return False, f"The private pip bootstrap helper is missing: {helper}"

    powershell = which("powershell.exe") or which("powershell") or "powershell.exe"
    command = [
        powershell,
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(helper),
        "-RuntimePath", str(Path(executable).resolve().parent),
        "-CachePath", str(cache.resolve()),
    ]
    try:
        repaired = run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not launch the private pip repair helper: {exc}"
    if repaired.returncode != 0:
        details = (repaired.stderr or repaired.stdout or "No diagnostic output was returned.").strip()
        return False, f"Private pip repair failed.\n{details}"

    try:
        verified = run(probe, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"pip repair ran, but the runtime could not be checked afterward: {exc}"
    if verified.returncode != 0:
        details = (verified.stderr or verified.stdout or "pip import failed.").strip()
        return False, f"pip repair completed but `python -m pip` is still unavailable.\n{details}"
    return True, (verified.stdout or "pip restored").strip()
