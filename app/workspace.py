from __future__ import annotations

"""Per-input temporary workspace management."""

import hashlib
import os
import re
import tempfile
from pathlib import Path

def mesh_work_dir(input_path: str | os.PathLike, temp_root: str | os.PathLike | None = None) -> Path:
    """Return a stable per-input workspace under the OS temporary directory.

    Example on Windows:
        %TEMP%\\PolyfittyStudio\\geometry_0_002_reconstructed_work_a1b2c3d4

    The short path hash prevents collisions when different folders contain files
    with the same basename, while preserving a readable per-mesh folder name.
    """
    source = Path(input_path).expanduser()
    try:
        identity = str(source.resolve()).casefold()
    except OSError:
        identity = str(source.absolute()).casefold()
    digest = hashlib.sha1(identity.encode("utf-8", errors="replace")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-") or "mesh"
    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    folder_name = f"{stem}_reconstructed_work_{digest}"
    current = root / "PolyfittyStudio" / folder_name
    legacy = root / "PolyFitStudio" / folder_name

    if legacy.exists() and not current.exists():
        try:
            current.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(current)
            try:
                legacy.parent.rmdir()
            except OSError:
                pass
        except OSError:

            return legacy
    return current

def reconstructed_obj_path(
    input_path: str | os.PathLike,
    temp_root: str | os.PathLike | None = None,
) -> Path:
    """Return the generated reconstruction OBJ inside its per-input temp workspace."""
    source = Path(input_path).expanduser()
    return mesh_work_dir(source, temp_root=temp_root) / f"{source.stem}_reconstructed.obj"

def reconstructed_baked_glb_path(
    input_path: str | os.PathLike,
    temp_root: str | os.PathLike | None = None,
) -> Path:
    """Return the canonical baked reconstruction GLB inside its temp workspace."""
    source = Path(input_path).expanduser()
    return mesh_work_dir(source, temp_root=temp_root) / f"{source.stem}_reconstructed_baked.glb"
