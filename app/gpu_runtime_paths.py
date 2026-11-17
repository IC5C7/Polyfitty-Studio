from __future__ import annotations

"""Load GPU packages installed for this standalone portable runtime.

The portable distribution embeds its own CPython 3.12 interpreter and core
dependencies under ``.runtime/python312``. Vendor-specific wheels are installed
on first launch into the writable ``.cache/gpu_runtime/site-packages`` overlay
inside that portable folder. The installed product and portable product use the
same application source; only portable mode performs recipient-GPU setup on
first launch.
"""

import os
import sys
from pathlib import Path

def gpu_site_packages_path() -> Path:

    project_root = Path(__file__).resolve().parent.parent
    return (project_root / ".cache" / "gpu_runtime" / "site-packages").resolve()

def activate_optional_gpu_packages() -> Path | None:
    """Prepend this portable instance's GPU overlay when it exists.

    Keeping vendor wheels outside the embedded CPython core lets first-run
    setup choose packages for the recipient GPU without rebuilding the shipped
    application source. CPU-only startup remains fully supported when absent.
    """
    site_packages = gpu_site_packages_path()
    if not site_packages.is_dir():
        return None
    resolved = str(site_packages)
    normalized = os.path.normcase(resolved)
    if all(os.path.normcase(os.path.abspath(entry or os.getcwd())) != normalized for entry in sys.path):
        sys.path.insert(0, resolved)
    return site_packages
