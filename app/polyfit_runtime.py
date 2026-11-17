from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

POLYFIT_WHEEL_NAME = "polyfit-1.6.0-cp312-cp312-win_amd64.whl"

class PolyFitBindingError(RuntimeError):
    pass

def _clear_polyfit_modules() -> None:
    for name in tuple(sys.modules):
        if (
            name == "polyfit" or name.startswith("polyfit.")
            or name == "PyPolyFit" or name.startswith("PyPolyFit.")
        ):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

def _import_binding():
    """Load the public ``polyfit`` package, with the native module as fallback.

    Upstream's executable Python examples import ``polyfit``. Some upstream
    distribution/build layouts also expose the native ``PyPolyFit`` module
    directly, so retain that as a compatibility fallback.
    """
    errors = []
    for module_name in ("polyfit", "PyPolyFit"):
        try:
            return importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            errors.append((module_name, exc))
    if errors:
        details = "; ".join(
            f"{name}: {type(error).__name__}: {error}" for name, error in errors
        )
        raise ImportError(f"Neither PolyFit module could be loaded ({details})") from errors[-1][1]
    raise ImportError("Neither polyfit nor PyPolyFit could be imported")

def _safe_find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        return None

def _runtime_diagnostic(root: Path, original_error: BaseException, repair_note: str) -> str:
    wheel = root / ".cache" / POLYFIT_WHEEL_NAME
    paths = "\n".join(f"  - {entry}" for entry in sys.path)
    spec = _safe_find_spec("polyfit")
    native_spec = _safe_find_spec("PyPolyFit")
    return (
        "Could not load the PolyFit Python binding.\n"
        f"Python executable: {sys.executable}\n"
        f"Python version: {sys.version.split()[0]}\n"
        f"Import preference: polyfit package, then PyPolyFit native module\n"
        f"polyfit package spec: {getattr(spec, 'origin', None) or 'not found'}\n"
        f"PyPolyFit native module spec: {getattr(native_spec, 'origin', None) or 'not found'}\n"
        f"Cached wheel: {wheel} ({'present' if wheel.is_file() else 'missing'})\n"
        f"Repair attempt: {repair_note or 'not attempted'}\n"
        f"Import error: {type(original_error).__name__}: {original_error}\n"
        f"Python search paths:\n{paths}\n\n"
        + (
            "The portable executable contains its Python and PolyFit runtime. Re-download the current "
            "Polyfitty Studio portable EXE if its embedded engine files are damaged."
            if getattr(sys, "frozen", False) else
            "Run INSTALL.bat from the extracted Polyfitty Studio folder to install or repair the private runtime."
        )
    )

def load_polyfit_binding(project_root: str | Path | None = None, allow_repair: bool = True):
    default_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1]
    )
    root = Path(project_root or default_root).resolve()
    if getattr(sys, "frozen", False):
        allow_repair = False
    initial_error = None
    repair_note = ""
    try:
        return _import_binding()
    except (ImportError, OSError) as exc:
        initial_error = exc

    wheel = root / ".cache" / POLYFIT_WHEEL_NAME
    if allow_repair and wheel.is_file():
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
            if completed.returncode == 0:
                _clear_polyfit_modules()
                try:
                    module = _import_binding()
                    return module
                except (ImportError, OSError) as retry_error:
                    initial_error = retry_error
                    repair_note = "cached wheel reinstalled, but import still failed"
            else:
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
                repair_note = f"pip repair failed with exit code {completed.returncode}: {output[-2400:]}"
        except Exception as repair_error:
            repair_note = f"cached-wheel repair could not run: {type(repair_error).__name__}: {repair_error}"
    elif allow_repair:
        repair_note = "no cached PolyFit wheel is available for automatic repair"

    if initial_error is None:
        initial_error = ImportError("PolyFit binding is unavailable")
    raise PolyFitBindingError(_runtime_diagnostic(root, initial_error, repair_note)) from initial_error
