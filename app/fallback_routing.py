from __future__ import annotations

from pathlib import Path

_TRIANGLE_MESH_SUFFIXES = {".obj", ".stl", ".off", ".glb", ".gltf", ".3mf"}

def _ply_contains_faces(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(1024 * 1024).split(b"end_header", 1)[0]
        if not header.startswith(b"ply"):
            return False
        for raw_line in header.splitlines():
            fields = raw_line.strip().split()
            if len(fields) == 3 and fields[0] == b"element" and fields[1] == b"face":
                return int(fields[2]) > 0
    except (OSError, ValueError):
        return False
    return False

def resolve_polyfit_timeout_fallback_source(
    original_input: str | Path,
    candidate_path: str | Path,
) -> tuple[Path, str]:
    """Prefer the original triangle mesh; candidate faces are only for point-set inputs."""
    original = Path(original_input)
    candidate = Path(candidate_path)
    suffix = original.suffix.lower()
    has_source_mesh = suffix in _TRIANGLE_MESH_SUFFIXES or (
        suffix == ".ply" and _ply_contains_faces(original)
    )
    if has_source_mesh:
        if not original.is_file():
            raise FileNotFoundError(f"The original triangle mesh is missing: {original}")
        return original, "original_input_mesh"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate, "polyfit_generated_candidate"
    raise FileNotFoundError(
        "No original triangle mesh or non-empty PolyFit candidate mesh is available for automatic fallback."
    )
