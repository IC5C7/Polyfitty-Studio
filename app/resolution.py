"""Output-resolution presets shared by the UI and reconstruction pipeline."""

RESOLUTION_PRESETS = {
    "draft": (100000, "100k samples · Fast preview"),
    "standard": (180000, "180k samples · Balanced"),
    "high": (300000, "300k samples · Detailed"),
    "ultra": (500000, "500k samples · High memory use"),
}

OUTPUT_FACE_BUDGETS = {
    "draft": 3000,
    "standard": 8000,
    "high": 12000,
    "ultra": 16000,
    "custom": 8000,
}

def get_resolution_preset(name: str) -> tuple[int, str]:
    """Return (maximum surface samples, UI hint) for a named preset."""
    key = str(name).strip().lower()
    if key not in RESOLUTION_PRESETS:
        raise ValueError(f"Unknown resolution preset: {name}")
    return RESOLUTION_PRESETS[key]

def get_output_face_budget(name: str, source_faces: int = 0) -> int:
    """Return the optional face budget used by the source-constrained planar fit helper."""
    key = str(name).strip().lower()
    budget = int(OUTPUT_FACE_BUDGETS.get(key, OUTPUT_FACE_BUDGETS["custom"]))
    source_count = int(source_faces or 0)
    if source_count > 0:
        budget = min(budget, source_count)
    return max(4, budget)
