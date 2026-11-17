from __future__ import annotations

"""Shared installed/portable GPU capability probe and real-kernel preflight."""

def _run_startup_gpu_preflight(splash, gpu_probe=None, display_adapter_probe=None) -> dict | None:
    """Validate GPU runtimes prepared by the launch path while splash is visible.

    Installed mode uses the GPU backend provisioned by its setup manager.
    Portable mode first provisions recipient-specific user-space packages into
    its own embedded distribution, then reaches this same real-kernel self-test.
    Windows vendor display drivers are never silently replaced. CPU baking
    remains available on failure.
    """
    def update(message: str) -> None:
        if splash is not None:
            splash.update(message)
        else:
            print(f"[GPU startup] {message}", flush=True)

    try:
        update("Checking installed GPU acceleration runtimes...")
        if gpu_probe is None:
            from gpu_compute import probe_gpu_runtime as gpu_probe
        result = gpu_probe(progress_callback=lambda _percent, message: update(str(message)))
        if display_adapter_probe is None:
            from main import _discover_system_gpu_names as display_adapter_probe
        try:
            result["display_adapters"] = list(display_adapter_probe() or [])
        except Exception as exc:
            result["display_adapters"] = []
            result.setdefault("errors", []).append(
                f"Display adapter inventory unavailable ({type(exc).__name__}: {exc})"
            )
        devices = list(result.get("devices") or [])
        if result.get("compute_ready"):
            vendor = str(result.get("active_vendor") or "GPU")
            device = str(result.get("active_device") or "Unknown GPU")
            backend = str(result.get("active_backend") or "GPU")
            summary = f"{vendor} {device} · {backend} READY"
            update(summary + ". Starting Studio...")
        elif devices:
            summary = "GPU detected · CPU compatibility mode"
            update("GPU detected, but compute validation failed. Starting in CPU compatibility mode...")
        elif result.get("display_adapters"):
            summary = "Display GPU found · CPU compute mode"
            update("Display adapter found; no compatible compute runtime is ready. Starting in CPU mode...")
        else:
            summary = "No compute GPU · CPU mode"
            update("No supported GPU compute device detected. Starting in CPU mode...")
        result["startup_splash_summary"] = summary
        return result
    except Exception as exc:

        return {
            "devices": [],
            "errors": [f"Startup GPU preflight failed ({type(exc).__name__}: {exc})"],
            "display_adapters": [],
            "compute_ready": False,
            "activation_verified": False,
            "active_backend": "CPU",
            "active_device": "CPU",
            "active_vendor": "CPU",
            "active_memory_bytes": 0,
            "fallback_reason": f"Startup GPU preflight failed ({type(exc).__name__}: {exc})",
            "startup_splash_summary": "GPU check unavailable · CPU mode",
        }
