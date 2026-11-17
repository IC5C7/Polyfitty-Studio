from __future__ import annotations

def radial_grid_fade(distance: float, radius: float, fade_start_fraction: float = 0.58) -> float:
    """Return a smooth, circular fade from a clear grid center to a rounded edge."""
    radius_value = max(float(radius), 1.0e-12)
    distance_value = max(float(distance), 0.0)
    start = min(max(float(fade_start_fraction), 0.0), 0.999999)
    normalized = min(max(distance_value / radius_value, 0.0), 1.0)
    if normalized <= start:
        return 1.0

    fade_position = (normalized - start) / (1.0 - start)
    smoothstep = fade_position * fade_position * (3.0 - 2.0 * fade_position)
    return max(0.0, 1.0 - smoothstep)
