"""Overall progress mapping for the integrated reconstruction + GLB pipeline."""

RECONSTRUCTION_WEIGHT = 70
BAKE_WEIGHT = 100 - RECONSTRUCTION_WEIGHT

def _clamp_percent(value: int | float) -> float:
    return max(0.0, min(100.0, float(value)))

def reconstruction_pipeline_percent(value: int | float) -> int:
    """Map reconstruction-local progress into the first 70% of the full pipeline."""
    return int(round(_clamp_percent(value) * RECONSTRUCTION_WEIGHT / 100.0))

def bake_pipeline_percent(value: int | float) -> int:
    """Map automatic bake/GLB validation progress into the final 30%."""
    return int(round(RECONSTRUCTION_WEIGHT + _clamp_percent(value) * BAKE_WEIGHT / 100.0))
