from __future__ import annotations

"""Optional GPU acceleration for baking and Li--Nan Stage-1 filtering.

NVIDIA GPUs can use CuPy/CUDA, supported Windows AMD GPUs can use the native
HIP/HIPRTC bridge (or HIP Python on other platforms), Intel GPUs can use
oneAPI/SYCL through dpnp, and OpenCL provides the cross-vendor fallback. GPU
adapters execute cage-ray/triangle tests, closest-point triangle distances and
Li--Nan bilateral-normal/vertex-update batches. Spatial candidate lookup stays
on SciPy cKDTree; UV/material sampling, topology traversal and GLB serialization
remain CPU-side so CPU and GPU runs share the same authoritative output path.
"""

from dataclasses import dataclass
import json
import time
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

try:
    from gpu_runtime_paths import activate_optional_gpu_packages
except ImportError:
    from .gpu_runtime_paths import activate_optional_gpu_packages

activate_optional_gpu_packages()

import numpy as np

ProgressCallback = Optional[Callable[[Optional[float], str], None]]

class GPUComputeRequiredError(RuntimeError):
    """Raised when strict GPU Only mode cannot continue on a usable GPU backend."""

_CUDA_DLL_DIRECTORY_HANDLES = []
_CUDA_DLL_PATHS_REGISTERED = False
_OPENCL_DLL_DIRECTORY_HANDLES = []
_OPENCL_DLL_PATHS_REGISTERED = False
_HIP_DLL_DIRECTORY_HANDLES = []
_HIP_DLL_PATHS_REGISTERED = False
_SYCL_DLL_DIRECTORY_HANDLES = []
_SYCL_DLL_PATHS_REGISTERED = False

def _prepare_hip_dll_search_paths() -> list[str]:
    """Expose bundled HIPRTC and driver-provided HIP runtime DLLs on Windows.

    The optional native bridge is built with the AMD HIP SDK and dynamically
    resolves the HIP runtime from the target PC's compatible AMD graphics
    driver. HIPRTC/COMGR and device bitcode may be bundled with the app.
    """
    global _HIP_DLL_PATHS_REGISTERED
    if _HIP_DLL_PATHS_REGISTERED or sys.platform != "win32":
        return []
    _HIP_DLL_PATHS_REGISTERED = True
    candidates: list[Path] = []
    roots = []
    for variable in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(variable, "").strip()
        if value:
            roots.append(Path(value))
    roots.extend((
        Path(sys.executable).resolve().parent / "amd_hip",
        Path(sys.executable).resolve().parent / "hip_runtime",
    ))
    for root in roots:
        candidates.extend((root / "bin", root))
    added: list[str] = []
    seen: set[str] = set()
    add_directory = getattr(os, "add_dll_directory", None)
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            resolved = str(candidate.resolve())
            key = os.path.normcase(resolved)
            if key in seen:
                continue
            seen.add(key)
            if add_directory is not None:
                _HIP_DLL_DIRECTORY_HANDLES.append(add_directory(resolved))
            if resolved not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
            added.append(resolved)
            if candidate.name.lower() == "bin" and candidate.parent.name.lower() == "amd_hip":
                os.environ["POLYFITTY_HIP_BUNDLE_DIR"] = resolved
                os.environ["HIP_PATH"] = str(candidate.parent.resolve())
            elif candidate.name.lower() == "bin" and candidate.parent.name.lower() in {"hip_runtime", "rocm"}:
                os.environ.setdefault("HIP_PATH", str(candidate.parent.resolve()))
        except OSError:
            continue
    return added

_WINDOWS_HIP_BRIDGE = None
_WINDOWS_HIP_BRIDGE_PATH: Path | None = None
_WINDOWS_HIP_BRIDGE_ERROR = ""

def _load_windows_hip_bridge():
    """Load the optional native MSVC HIP/HIPRTC bridge bundled by Windows builds."""
    global _WINDOWS_HIP_BRIDGE, _WINDOWS_HIP_BRIDGE_PATH, _WINDOWS_HIP_BRIDGE_ERROR
    if sys.platform != "win32":
        return None
    if _WINDOWS_HIP_BRIDGE is not None:
        return _WINDOWS_HIP_BRIDGE, _WINDOWS_HIP_BRIDGE_PATH
    if _WINDOWS_HIP_BRIDGE_ERROR:
        return None

    import ctypes

    _prepare_hip_dll_search_paths()
    roots: list[Path] = []
    roots.extend((Path(sys.executable).resolve().parent, Path(__file__).resolve().parent.parent))
    candidates: list[Path] = []
    explicit = os.environ.get("POLYFITTY_HIP_BRIDGE", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    for root in roots:
        candidates.extend((
            root / "amd_hip" / "bin" / "PolyfittyHipBridge.dll",
            root / "native" / "hip_bridge" / "build" / "PolyfittyHipBridge.dll",
        ))
    bridge_path = next((item.resolve() for item in candidates if item.is_file()), None)
    if bridge_path is None:
        return None

    bundle_bin = bridge_path.parent
    os.environ["POLYFITTY_HIP_BUNDLE_DIR"] = str(bundle_bin)
    os.environ["HIP_PATH"] = str(bundle_bin.parent)
    add_directory = getattr(os, "add_dll_directory", None)
    if add_directory is not None:
        try:
            _HIP_DLL_DIRECTORY_HANDLES.append(add_directory(str(bundle_bin)))
        except OSError:
            pass
    os.environ["PATH"] = str(bundle_bin) + os.pathsep + os.environ.get("PATH", "")

    try:
        bridge = ctypes.CDLL(str(bridge_path))
        char_ptr = ctypes.POINTER(ctypes.c_char)
        double_ptr = ctypes.POINTER(ctypes.c_double)
        bridge.pfhip_device_count.argtypes = [ctypes.POINTER(ctypes.c_int), char_ptr, ctypes.c_int]
        bridge.pfhip_device_count.restype = ctypes.c_int
        bridge.pfhip_device_info.argtypes = [
            ctypes.c_int, char_ptr, ctypes.c_int, char_ptr, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int), char_ptr, ctypes.c_int,
        ]
        bridge.pfhip_device_info.restype = ctypes.c_int
        bridge.pfhip_create_backend.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), char_ptr, ctypes.c_int]
        bridge.pfhip_create_backend.restype = ctypes.c_int
        bridge.pfhip_destroy_backend.argtypes = [ctypes.c_void_p]
        bridge.pfhip_destroy_backend.restype = None
        bridge.pfhip_intersect_rays.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            double_ptr, double_ptr, double_ptr,
            ctypes.c_double, ctypes.c_double,
            double_ptr, double_ptr, double_ptr, char_ptr, ctypes.c_int,
        ]
        bridge.pfhip_intersect_rays.restype = ctypes.c_int
        bridge.pfhip_closest_points.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, char_ptr, ctypes.c_int,
        ]
        bridge.pfhip_closest_points.restype = ctypes.c_int
        if hasattr(bridge, "pfhip_qem_candidate_batch"):
            bridge.pfhip_qem_candidate_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
                double_ptr, double_ptr, ctypes.POINTER(ctypes.c_ulonglong),
                double_ptr, double_ptr, char_ptr, ctypes.c_int,
            ]
            bridge.pfhip_qem_candidate_batch.restype = ctypes.c_int
        _WINDOWS_HIP_BRIDGE = bridge
        _WINDOWS_HIP_BRIDGE_PATH = bridge_path
        return bridge, bridge_path
    except Exception as exc:
        _WINDOWS_HIP_BRIDGE_ERROR = f"Could not load PolyfittyHipBridge.dll ({type(exc).__name__}: {exc})"
        return None

def _prepare_sycl_dll_search_paths() -> list[str]:
    """Keep bundled oneAPI/SYCL DLLs discoverable in Windows private-runtime launches."""
    global _SYCL_DLL_PATHS_REGISTERED
    if _SYCL_DLL_PATHS_REGISTERED or sys.platform != "win32":
        return []
    _SYCL_DLL_PATHS_REGISTERED = True
    roots: list[Path] = []
    roots.extend((
        Path(sys.executable).resolve().parent,
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "dpctl.libs",
        Path(sys.prefix) / "Lib" / "site-packages" / "dpnp.libs",
    ))

    for entry in sys.path:
        if not entry:
            continue
        site_root = Path(entry)
        if site_root.is_dir() and site_root.name.lower() == "site-packages":
            roots.extend((site_root / "dpctl.libs", site_root / "dpnp.libs"))
    added: list[str] = []
    seen: set[str] = set()
    add_directory = getattr(os, "add_dll_directory", None)
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in (root, *[item for item in root.rglob("*") if item.is_dir() and item.name.lower() in {"bin", "lib", ".libs"}]):
            try:
                resolved = str(candidate.resolve())
                key = os.path.normcase(resolved)
                if key in seen:
                    continue
                seen.add(key)
                if add_directory is not None:
                    _SYCL_DLL_DIRECTORY_HANDLES.append(add_directory(resolved))
                added.append(resolved)
            except OSError:
                continue
    return added

def _prepare_opencl_dll_search_paths() -> list[str]:
    """Keep the bundled OpenCL loader and vendor ICD loader visible in the shared Windows runtime."""
    global _OPENCL_DLL_PATHS_REGISTERED
    if _OPENCL_DLL_PATHS_REGISTERED or sys.platform != "win32":
        return []
    _OPENCL_DLL_PATHS_REGISTERED = True
    candidates: list[Path] = []
    candidates.extend((
        Path(sys.executable).resolve().parent,
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
    ))
    try:
        import importlib.util
        module_spec = importlib.util.find_spec("pyopencl")
        if module_spec is not None and module_spec.origin:
            package_dir = Path(module_spec.origin).resolve().parent
            candidates.extend((package_dir, package_dir / ".libs", package_dir.parent / "pyopencl.libs"))
    except (ImportError, ValueError, OSError):
        pass

    added: list[str] = []
    seen: set[str] = set()
    add_directory = getattr(os, "add_dll_directory", None)
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            resolved = str(candidate.resolve())
            key = os.path.normcase(resolved)
            if key in seen:
                continue
            seen.add(key)
            if add_directory is not None:
                _OPENCL_DLL_DIRECTORY_HANDLES.append(add_directory(resolved))
            added.append(resolved)
        except OSError:
            continue
    return added

def _prepare_cuda_dll_search_paths() -> list[str]:
    global _CUDA_DLL_PATHS_REGISTERED
    if _CUDA_DLL_PATHS_REGISTERED or sys.platform != "win32":
        return []
    _CUDA_DLL_PATHS_REGISTERED = True
    candidates = []
    roots = []
    site_roots = {
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.executable).resolve().parent / "Lib" / "site-packages",
    }
    site_roots.update(Path(entry) for entry in list(sys.path) if entry)
    bundled_cuda_runtime = None
    for site_root in site_roots:
        root = site_root / "nvidia"
        if root.is_dir():
            roots.append(root)
            runtime_root = root / "cuda_runtime"

            if (runtime_root / "bin").is_dir() and (runtime_root / "include").is_dir():
                if bundled_cuda_runtime is None:
                    bundled_cuda_runtime = runtime_root
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if bundled_cuda_runtime is not None:
        cuda_path = str(bundled_cuda_runtime.resolve())
        os.environ["CUDA_PATH"] = cuda_path
        os.environ["CUDA_HOME"] = cuda_path
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    for root in roots:
        candidates.extend((root / "cublas" / "bin", root / "cuda_runtime" / "bin", root / "cuda_nvrtc" / "bin"))
        try:
            candidates.extend(path for path in root.glob("*/bin") if path.is_dir())
        except OSError:
            continue
    added = []
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized = {os.path.normcase(os.path.abspath(item)) for item in path_entries if item}
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            continue
        if not candidate.is_dir():
            continue
        key = os.path.normcase(resolved)
        if key not in normalized:
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
            normalized.add(key)
        try:
            add_directory = getattr(os, "add_dll_directory", None)
            if add_directory is not None:
                _CUDA_DLL_DIRECTORY_HANDLES.append(add_directory(resolved))
        except OSError:
            continue
        added.append(resolved)
    return added

@dataclass(frozen=True)
class GPUDevice:
    backend: str
    index: int
    name: str
    vendor: str
    memory_bytes: int = 0
    native: object | None = None
    integrated: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} · {self.backend.upper()}"

@runtime_checkable
class GPUComputeBackend(Protocol):
    """Common contract implemented by every accelerated compute backend.

    All adapters consume/return NumPy host arrays, keeping geometry, baking,
    and worker code independent from the vendor runtime.
    """

    precision: str
    device_name: str

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit): ...

    def closest_points(self, points, candidate_triangles): ...

    def qem_candidate_batch(self, vertices, quadrics, edges): ...

GPU_BACKEND_PRIORITY = {"cuda": 0, "hip": 1, "sycl": 2, "opencl": 3}

def _safe_progress(callback: ProgressCallback, message: str) -> None:
    if callback is None:
        return
    try:
        callback(None, message)
    except Exception:
        pass

def _qem_candidate_batch_xp(
    xp,
    vertices,
    quadrics,
    edges,
    *,
    array_factory=None,
    to_host=None,
    dtype=None,
):
    """Vectorized float64 QEM segment placement for NumPy-style GPU arrays.

    The scalar operation order mirrors feature_decimation._optimal_edge_placement.
    A backend is accepted for deterministic QEM only when its result matches
    that CPU reference exactly on the first real batch.
    """
    compute_dtype = dtype or xp.float64
    make_array = array_factory or (lambda value, requested=None: xp.asarray(value, dtype=requested or compute_dtype))
    host_array = to_host or (lambda value: np.asarray(value))
    source_vertices = make_array(vertices, compute_dtype)
    source_quadrics = make_array(quadrics, compute_dtype)
    source_edges = make_array(edges, xp.int64)
    if int(source_edges.shape[0]) == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    first = source_edges[:, 0]
    second = source_edges[:, 1]
    p0 = source_vertices[first]
    p1 = source_vertices[second]
    direction = p1 - p0
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    x, y, z = p0[:, 0], p0[:, 1], p0[:, 2]
    length_sq = dx * dx + dy * dy + dz * dz
    quadric = source_quadrics[first] + source_quadrics[second]

    qh0_x = quadric[:, 0, 0] * x + quadric[:, 0, 1] * y + quadric[:, 0, 2] * z + quadric[:, 0, 3]
    qh0_y = quadric[:, 1, 0] * x + quadric[:, 1, 1] * y + quadric[:, 1, 2] * z + quadric[:, 1, 3]
    qh0_z = quadric[:, 2, 0] * x + quadric[:, 2, 1] * y + quadric[:, 2, 2] * z + quadric[:, 2, 3]
    numerator = dx * qh0_x + dy * qh0_y + dz * qh0_z
    qd_x = quadric[:, 0, 0] * dx + quadric[:, 0, 1] * dy + quadric[:, 0, 2] * dz
    qd_y = quadric[:, 1, 0] * dx + quadric[:, 1, 1] * dy + quadric[:, 1, 2] * dz
    qd_z = quadric[:, 2, 0] * dx + quadric[:, 2, 1] * dy + quadric[:, 2, 2] * dz
    denominator = dx * qd_x + dy * qd_y + dz * qd_z

    base_cost = (
        quadric[:, 0, 0] * x * x + quadric[:, 1, 1] * y * y + quadric[:, 2, 2] * z * z
        + 2.0 * quadric[:, 0, 1] * x * y
        + 2.0 * quadric[:, 0, 2] * x * z
        + 2.0 * quadric[:, 1, 2] * y * z
        + 2.0 * quadric[:, 0, 3] * x
        + 2.0 * quadric[:, 1, 3] * y
        + 2.0 * quadric[:, 2, 3] * z
        + quadric[:, 3, 3]
    )
    base_cost = xp.where(xp.isfinite(base_cost), xp.maximum(base_cost, 0.0), xp.inf)
    best_cost = base_cost.copy()
    best_t = xp.zeros_like(base_cost)
    for candidate_t in (0.5, 1.0):
        candidate_cost = (
            base_cost + 2.0 * numerator * candidate_t
            + denominator * candidate_t * candidate_t
        )
        candidate_cost = xp.where(
            xp.isfinite(candidate_cost), xp.maximum(candidate_cost, 0.0), xp.inf
        )
        better = candidate_cost < best_cost
        best_cost = xp.where(better, candidate_cost, best_cost)
        best_t = xp.where(better, candidate_t, best_t)

    valid_optimum = xp.isfinite(denominator) & (denominator > 1e-18)
    safe_denominator = xp.where(valid_optimum, denominator, 1.0)
    optimum_t = xp.clip(-numerator / safe_denominator, 0.0, 1.0)
    optimum_cost = (
        base_cost + 2.0 * numerator * optimum_t
        + denominator * optimum_t * optimum_t
    )
    finite_optimum = valid_optimum & xp.isfinite(optimum_cost)
    optimum_cost = xp.where(finite_optimum, xp.maximum(optimum_cost, 0.0), xp.inf)
    better_optimum = optimum_cost < best_cost
    best_cost = xp.where(better_optimum, optimum_cost, best_cost)
    best_t = xp.where(better_optimum, optimum_t, best_t)

    degenerate = ~xp.isfinite(length_sq) | (length_sq <= 1e-24)
    computed_positions = p0 + best_t[:, None] * direction
    midpoint_positions = 0.5 * (p0 + p1)
    computed_positions = xp.where(degenerate[:, None], midpoint_positions, computed_positions)
    mx, my, mz = midpoint_positions[:, 0], midpoint_positions[:, 1], midpoint_positions[:, 2]
    midpoint_cost = (
        quadric[:, 0, 0] * mx * mx + quadric[:, 1, 1] * my * my + quadric[:, 2, 2] * mz * mz
        + 2.0 * quadric[:, 0, 1] * mx * my
        + 2.0 * quadric[:, 0, 2] * mx * mz
        + 2.0 * quadric[:, 1, 2] * my * mz
        + 2.0 * quadric[:, 0, 3] * mx
        + 2.0 * quadric[:, 1, 3] * my
        + 2.0 * quadric[:, 2, 3] * mz
        + quadric[:, 3, 3]
    )
    midpoint_cost = xp.where(
        xp.isfinite(midpoint_cost), xp.maximum(midpoint_cost, 0.0), xp.inf
    )
    best_cost = xp.where(degenerate, midpoint_cost, best_cost)
    best_cost = xp.where(xp.isfinite(computed_positions).all(axis=1), best_cost, xp.inf)
    return (
        np.asarray(host_array(best_cost), dtype=np.float64).reshape(-1),
        np.asarray(host_array(computed_positions), dtype=np.float64).reshape(-1, 3),
    )

def _decode_name(value: object, fallback: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip() or fallback
        except Exception:
            return fallback
    return str(value or fallback).strip()

def _find_windows_cuda_driver_library() -> Path | None:
    """Return the NVIDIA kernel-driver DLL path when present on Windows.

    The installed and embedded-portable Python environments may each contain
    optional CUDA packages, while the recipient PC may have another GPU vendor.
    Avoid importing CuPy without the NVIDIA driver; OpenCL discovery can still
    select AMD/Intel devices without CUDA initialization.
    """
    if sys.platform != "win32":
        return None
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    candidates = (system_root / "System32" / "nvcuda.dll",)
    return next((candidate for candidate in candidates if candidate.is_file()), None)

def _discover_cuda(errors: list[str]) -> list[GPUDevice]:
    devices: list[GPUDevice] = []
    if sys.platform == "win32":
        driver_library = _find_windows_cuda_driver_library()
        if driver_library is None:
            errors.append(
                "NVIDIA CUDA driver library nvcuda.dll was not found in Windows System32; "
                "CUDA was skipped and OpenCL discovery will continue."
            )
            return devices
    try:
        _prepare_cuda_dll_search_paths()
        import cupy as cp
    except Exception as exc:
        errors.append(f"CUDA runtime unavailable ({type(exc).__name__}: {exc})")
        return devices
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            devices.append(GPUDevice(
                backend="cuda", index=index,
                name=_decode_name(props.get("name"), f"NVIDIA GPU {index}"),
                vendor="NVIDIA",
                memory_bytes=int(props.get("totalGlobalMem", 0) or 0),
                native=cp,
            ))
    except Exception as exc:
        errors.append(f"CUDA device scan failed ({type(exc).__name__}: {exc})")
    return devices

def _discover_opencl(errors: list[str]) -> list[GPUDevice]:
    """Enumerate vendor OpenCL ICDs and retain actionable loader diagnostics."""
    devices: list[GPUDevice] = []
    try:
        _prepare_opencl_dll_search_paths()
        import pyopencl as cl
    except Exception as exc:
        errors.append(
            "OpenCL runtime unavailable (PyOpenCL import failed: "
            f"{type(exc).__name__}: {exc})"
        )
        return devices

    try:
        platforms = list(cl.get_platforms())
    except Exception as exc:
        errors.append(
            "OpenCL platform enumeration failed; the PyOpenCL package is present, "
            "but no vendor ICD platform could be opened "
            f"({type(exc).__name__}: {exc})"
        )
        return devices

    if not platforms:
        errors.append(
            "OpenCL loader returned zero platforms. Install or repair the discrete/integrated "
            "GPU vendor driver with its OpenCL ICD component."
        )
        return devices

    platform_names = []
    initial_error_count = len(errors)
    for platform in platforms:
        platform_name = _decode_name(getattr(platform, "name", ""), "Unknown OpenCL platform")
        platform_vendor = _decode_name(getattr(platform, "vendor", ""), "Unknown vendor")
        platform_names.append(f"{platform_name} ({platform_vendor})")
        try:
            platform_devices = platform.get_devices(device_type=cl.device_type.GPU)
        except Exception as exc:
            errors.append(
                f"OpenCL GPU enumeration failed on {platform_name} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        for device in platform_devices:
            if bool(getattr(device, "available", True)) is False:
                continue
            if bool(getattr(device, "compiler_available", True)) is False:
                errors.append(
                    f"OpenCL GPU is visible but its driver has no compiler: "
                    f"{_decode_name(getattr(device, 'name', ''), 'Unknown GPU')}"
                )
                continue
            index = len(devices)
            devices.append(GPUDevice(
                backend="opencl", index=index,
                name=_decode_name(getattr(device, "name", ""), f"OpenCL GPU {index}"),
                vendor=_decode_name(getattr(device, "vendor", ""), platform_vendor),
                memory_bytes=int(getattr(device, "global_mem_size", 0) or 0),
                native=(cl, device),
                integrated=bool(getattr(device, "host_unified_memory", False)),
            ))

    if not devices and len(errors) == initial_error_count:
        errors.append(
            "OpenCL platforms were found, but none exposed an available GPU device: "
            + "; ".join(platform_names)
        )
    return devices

def _hip_result(call_result, success_value, label: str):
    """Unpack HIP Python's (error, *results) convention with useful errors."""
    if not isinstance(call_result, tuple):
        call_result = (call_result,)
    if not call_result:
        raise RuntimeError(f"{label} returned no status value")
    error = call_result[0]
    if error != success_value:
        raise RuntimeError(f"{label} failed: {error}")
    values = call_result[1:]
    if len(values) == 1:
        return values[0]
    return values

def _discover_hip(errors: list[str]) -> list[GPUDevice]:
    """Discover AMD HIP devices through the Windows bridge or HIP Python."""
    devices: list[GPUDevice] = []
    if sys.platform == "win32":
        bridge_info = _load_windows_hip_bridge()
        if bridge_info is not None:
            bridge, bridge_path = bridge_info
            import ctypes
            count = ctypes.c_int(0)
            error = ctypes.create_string_buffer(4096)
            result = bridge.pfhip_device_count(ctypes.byref(count), error, len(error))
            if result != 0:
                errors.append(
                    "AMD HIP bridge loaded, but the target AMD driver did not expose HIP devices: "
                    + error.value.decode("utf-8", errors="replace")
                )
                return devices
            for index in range(max(0, int(count.value))):
                name = ctypes.create_string_buffer(512)
                arch = ctypes.create_string_buffer(128)
                memory = ctypes.c_ulonglong(0)
                integrated = ctypes.c_int(0)
                error = ctypes.create_string_buffer(4096)
                result = bridge.pfhip_device_info(
                    index, name, len(name), arch, len(arch), ctypes.byref(memory),
                    ctypes.byref(integrated), error, len(error),
                )
                if result != 0:
                    errors.append(
                        f"AMD HIP device {index} could not be inspected: "
                        + error.value.decode("utf-8", errors="replace")
                    )
                    continue
                label = _decode_name(name.value, f"AMD HIP GPU {index}")
                architecture = _decode_name(arch.value, "")
                devices.append(GPUDevice(
                    backend="hip", index=index, name=label, vendor="AMD",
                    memory_bytes=int(memory.value),
                    native=(bridge, architecture, bridge_path),
                    integrated=bool(integrated.value),
                ))
            if not devices and count.value == 0:
                errors.append("AMD HIP runtime was loaded successfully, but it reported zero HIP devices.")
            return devices
        if _WINDOWS_HIP_BRIDGE_ERROR:
            errors.append(_WINDOWS_HIP_BRIDGE_ERROR)

    try:
        _prepare_hip_dll_search_paths()
        from hip import hip
    except Exception as exc:
        hip_requested = os.environ.get("POLYFITTY_ENABLE_HIP", "0") == "1"
        hip_root_present = any(os.environ.get(key, "").strip() for key in ("HIP_PATH", "ROCM_PATH"))
        if hip_requested or hip_root_present:
            errors.append(
                "ROCm/HIP runtime unavailable (native bridge/HIP Python binding could not be loaded: "
                f"{type(exc).__name__}: {exc}). On Windows, rebuild with the AMD HIP SDK option and Visual Studio C++ tools; otherwise OpenCL remains available."
            )
        return devices

    try:
        count = _hip_result(hip.hipGetDeviceCount(), hip.hipError_t.hipSuccess, "hipGetDeviceCount")
        for index in range(int(count)):
            properties = hip.hipDeviceProp_t()
            _hip_result(
                hip.hipGetDeviceProperties(properties, index),
                hip.hipError_t.hipSuccess, f"hipGetDeviceProperties({index})",
            )
            devices.append(GPUDevice(
                backend="hip", index=index,
                name=_decode_name(getattr(properties, "name", ""), f"AMD HIP GPU {index}"),
                vendor="AMD", memory_bytes=int(getattr(properties, "totalGlobalMem", 0) or 0),
                native=(hip, properties), integrated=bool(getattr(properties, "integrated", False)),
            ))
    except Exception as exc:
        errors.append(f"ROCm/HIP device enumeration failed ({type(exc).__name__}: {exc})")
    return devices

def _discover_sycl(errors: list[str]) -> list[GPUDevice]:
    """Discover SYCL GPU devices through Intel's dpctl runtime.

    dpctl supplies device discovery and queues; dpnp supplies the accelerated
    NumPy-style array operations used by this application's bake kernels.
    """
    devices: list[GPUDevice] = []
    try:
        _prepare_sycl_dll_search_paths()
        import dpctl
        import dpnp
    except Exception as exc:
        errors.append(
            "Intel oneAPI/SYCL runtime unavailable (dpctl/dpnp import failed: "
            f"{type(exc).__name__}: {exc}). Install or repair the Intel oneAPI Python packages for this Polyfitty Studio environment."
        )
        return devices

    try:
        sycl_devices = list(dpctl.get_devices(device_type="gpu"))
    except Exception as exc:
        errors.append(f"SYCL GPU enumeration failed ({type(exc).__name__}: {exc})")
        return devices

    if not sycl_devices:
        errors.append("SYCL runtime was loaded, but it exposed no GPU devices (Level Zero/OpenCL GPU plugin not available).")
        return devices

    for index, device in enumerate(sycl_devices):
        try:
            name = _decode_name(getattr(device, "name", ""), f"SYCL GPU {index}")
            vendor = _decode_name(getattr(device, "vendor", ""), "Unknown SYCL vendor")
            memory = int(getattr(device, "global_mem_size", 0) or 0)
            integrated = bool(getattr(device, "is_integrated", False))
            devices.append(GPUDevice(
                backend="sycl", index=index, name=name, vendor=vendor,
                memory_bytes=memory, native=device, integrated=integrated,
            ))
        except Exception as exc:
            errors.append(f"SYCL device {index} could not be inspected ({type(exc).__name__}: {exc})")
    return devices

def discover_gpu_devices(
    progress_callback: ProgressCallback = None,
) -> tuple[list[GPUDevice], list[str]]:
    """Discover CUDA, HIP/ROCm, SYCL/oneAPI, and OpenCL devices with status updates."""
    errors: list[str] = []
    devices: list[GPUDevice] = []
    discoverers = (
        (_discover_cuda, "NVIDIA CUDA"),
        (_discover_hip, "AMD HIP"),
        (_discover_sycl, "Intel oneAPI / SYCL"),
        (_discover_opencl, "cross-vendor OpenCL"),
    )
    for discoverer, label in discoverers:
        _safe_progress(progress_callback, f"Detecting {label} devices...")
        before_count = len(devices)
        try:
            devices.extend(discoverer(errors))
        except Exception as exc:
            errors.append(f"{discoverer.__name__} failed ({type(exc).__name__}: {exc})")
        detected_count = len(devices) - before_count
        if detected_count:
            _safe_progress(
                progress_callback,
                f"Detected {detected_count} {label} compute device(s).",
            )

    devices.sort(key=lambda item: (
        1 if item.integrated else 0,
        GPU_BACKEND_PRIORITY.get(item.backend.lower(), 99),
        -item.memory_bytes,
        item.index,
    ))
    return devices, errors

def intersect_rays_cpu(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    candidate_triangles: np.ndarray,
    epsilon: float,
    ray_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized CPU Moller-Trumbore intersection, matching the legacy path."""
    origins = np.asarray(ray_origins, dtype=np.float64)
    directions = np.asarray(ray_directions, dtype=np.float64)
    triangles = np.asarray(candidate_triangles, dtype=np.float64)
    v0 = triangles[:, :, 0, :]
    edge1 = triangles[:, :, 1, :] - v0
    edge2 = triangles[:, :, 2, :] - v0
    pvec = np.cross(directions[:, :, None, :], edge2[:, None, :, :])
    determinant = np.einsum("nci,nrci->nrc", edge1, pvec)
    valid = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
    tvec = origins[:, :, None, :] - v0[:, None, :, :]
    u = np.einsum("nrci,nrci->nrc", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1[:, None, :, :])
    v = np.einsum("nri,nrci->nrc", directions, qvec) * inverse
    ray_t = np.einsum("nci,nrci->nrc", edge2, qvec) * inverse
    valid &= (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)
    valid &= (ray_t >= -float(epsilon)) & (ray_t <= float(ray_limit) + float(epsilon))
    distances = np.where(valid, np.maximum(ray_t, 0.0), np.inf)
    return distances, u, v

def _closest_points_vectorized(xp, points, candidate_triangles, *, device=None, dtype=None):
    """Vectorized closest-point-on-triangle query for NumPy or CuPy arrays.

    Returns squared distance and barycentric coordinates (weight of B, weight
    of C) for every point/candidate-triangle pair. Degenerate triangles are
    handled by testing their three line segments rather than emitting NaNs.
    """
    compute_dtype = dtype if dtype is not None else xp.float64
    if device is None:
        query = xp.asarray(points, dtype=compute_dtype).reshape(-1, 1, 3)
        triangles = xp.asarray(candidate_triangles, dtype=compute_dtype)
    else:
        query = xp.asarray(points, dtype=compute_dtype, device=device).reshape(-1, 1, 3)
        triangles = xp.asarray(candidate_triangles, dtype=compute_dtype, device=device)
    if triangles.ndim != 4 or triangles.shape[2:] != (3, 3):
        raise ValueError("Closest-point candidates must have shape (N, K, 3, 3).")
    if query.shape[0] != triangles.shape[0]:
        raise ValueError("Closest-point query and candidate counts do not match.")

    a = triangles[:, :, 0, :]
    b = triangles[:, :, 1, :]
    c = triangles[:, :, 2, :]
    ab = b - a
    ac = c - a
    bc = c - b
    ap = query - a
    d1 = xp.sum(ab * ap, axis=-1)
    d2 = xp.sum(ac * ap, axis=-1)
    bp = query - b
    d3 = xp.sum(ab * bp, axis=-1)
    d4 = xp.sum(ac * bp, axis=-1)
    cpnt = query - c
    d5 = xp.sum(ab * cpnt, axis=-1)
    d6 = xp.sum(ac * cpnt, axis=-1)

    shape = d1.shape
    wa = xp.zeros(shape, dtype=compute_dtype)
    wb = xp.zeros_like(wa)
    wc = xp.zeros_like(wa)
    remaining = xp.ones(shape, dtype=bool)

    mask = remaining & (d1 <= 0.0) & (d2 <= 0.0)
    wa = xp.where(mask, 1.0, wa)
    remaining = remaining & ~mask

    mask = remaining & (d3 >= 0.0) & (d4 <= d3)
    wb = xp.where(mask, 1.0, wb)
    remaining = remaining & ~mask

    vc = d1 * d4 - d3 * d2
    mask = remaining & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denominator = xp.where(xp.abs(d1 - d3) > 1e-30, d1 - d3, 1.0)
    t = xp.clip(d1 / denominator, 0.0, 1.0)
    wa = xp.where(mask, 1.0 - t, wa)
    wb = xp.where(mask, t, wb)
    remaining = remaining & ~mask

    mask = remaining & (d6 >= 0.0) & (d5 <= d6)
    wc = xp.where(mask, 1.0, wc)
    remaining = remaining & ~mask

    vb = d5 * d2 - d1 * d6
    mask = remaining & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denominator = xp.where(xp.abs(d2 - d6) > 1e-30, d2 - d6, 1.0)
    t = xp.clip(d2 / denominator, 0.0, 1.0)
    wa = xp.where(mask, 1.0 - t, wa)
    wc = xp.where(mask, t, wc)
    remaining = remaining & ~mask

    va = d3 * d6 - d5 * d4
    mask = remaining & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denominator = (d4 - d3) + (d5 - d6)
    denominator = xp.where(xp.abs(denominator) > 1e-30, denominator, 1.0)
    t = xp.clip((d4 - d3) / denominator, 0.0, 1.0)
    wb = xp.where(mask, 1.0 - t, wb)
    wc = xp.where(mask, t, wc)
    remaining = remaining & ~mask

    denominator = va + vb + vc
    safe_denominator = xp.where(xp.abs(denominator) > 1e-30, denominator, 1.0)
    face_v = vb / safe_denominator
    face_w = vc / safe_denominator
    face_u = 1.0 - face_v - face_w
    wa = xp.where(remaining, face_u, wa)
    wb = xp.where(remaining, face_v, wb)
    wc = xp.where(remaining, face_w, wc)

    area_vector = xp.cross(ab, ac)
    area_squared = xp.sum(area_vector * area_vector, axis=-1)
    ab_squared = xp.sum(ab * ab, axis=-1)
    ac_squared = xp.sum(ac * ac, axis=-1)
    bc_squared = xp.sum(bc * bc, axis=-1)
    scale_squared = xp.maximum(ab_squared, xp.maximum(ac_squared, bc_squared))
    degenerate = area_squared <= xp.maximum(scale_squared * scale_squared * 1e-24, 1e-60)

    def segment(pa, delta, length_squared):
        safe_length = xp.where(length_squared > 1e-30, length_squared, 1.0)
        amount = xp.clip(xp.sum((query - pa) * delta, axis=-1) / safe_length, 0.0, 1.0)
        closest = pa + amount[..., None] * delta
        distance_squared = xp.sum((query - closest) * (query - closest), axis=-1)
        return distance_squared, amount

    dist_ab, amount_ab = segment(a, ab, ab_squared)
    dist_bc, amount_bc = segment(b, bc, bc_squared)
    dist_ca, amount_ca = segment(c, a - c, ac_squared)
    best_distance = dist_ab
    deg_wa = 1.0 - amount_ab
    deg_wb = amount_ab
    deg_wc = xp.zeros_like(amount_ab)
    choose_bc = dist_bc < best_distance
    best_distance = xp.where(choose_bc, dist_bc, best_distance)
    deg_wa = xp.where(choose_bc, 0.0, deg_wa)
    deg_wb = xp.where(choose_bc, 1.0 - amount_bc, deg_wb)
    deg_wc = xp.where(choose_bc, amount_bc, deg_wc)
    choose_ca = dist_ca < best_distance
    deg_wa = xp.where(choose_ca, amount_ca, deg_wa)
    deg_wb = xp.where(choose_ca, 0.0, deg_wb)
    deg_wc = xp.where(choose_ca, 1.0 - amount_ca, deg_wc)

    wa = xp.where(degenerate, deg_wa, wa)
    wb = xp.where(degenerate, deg_wb, wb)
    wc = xp.where(degenerate, deg_wc, wc)
    closest = wa[..., None] * a + wb[..., None] * b + wc[..., None] * c
    delta = query - closest
    distance_squared = xp.sum(delta * delta, axis=-1)
    return distance_squared, wb, wc

def _filter_mesh_step_xp(
    xp,
    vertices,
    face_normals,
    face_centers,
    face_areas,
    face_query_offsets,
    face_neighbor_ids,
    vertex_offsets,
    vertex_incident_faces,
    sigma_dist,
    theta_radians,
    *,
    array_factory=None,
    to_host=None,
    dtype=None,
):
    """Run one Li--Nan bilateral-normal and vertex-update step on an array backend.

    Topology is precomputed on the host as CSR offsets. The GPU performs the
    repeated floating-point work while the CPU retains the topology and the
    conservative triangle line search. This preserves the published neighbor
    sets and Eq. (2)--(5) semantics without constructing a dense distance matrix.
    """
    host = to_host or (lambda value: np.asarray(value))
    real_dtype = dtype or xp.float64

    def make(value, target_dtype=None):
        selected_dtype = target_dtype or real_dtype
        if array_factory is not None:
            try:
                return array_factory(value, dtype=selected_dtype)
            except TypeError:
                return array_factory(value)
        return xp.asarray(value, dtype=selected_dtype)

    v = make(vertices, real_dtype)
    normals = make(face_normals, real_dtype)
    centers = make(face_centers, real_dtype)
    areas = make(face_areas, real_dtype)
    f_offsets = make(face_query_offsets, xp.int64)
    f_neighbors = make(face_neighbor_ids, xp.int64)
    v_offsets = make(vertex_offsets, xp.int64)
    v_incident = make(vertex_incident_faces, xp.int64)

    face_count = int(normals.shape[0])
    vertex_count = int(v.shape[0])
    face_pair_counts = (f_offsets[1:] - f_offsets[:-1]).astype(xp.int64, copy=False)
    query_ids = xp.repeat(xp.arange(face_count, dtype=xp.int64), face_pair_counts)
    neighbor_ids = f_neighbors.astype(xp.int64, copy=False)

    cos_theta = float(np.cos(float(theta_radians)))
    beta_denominator = max((1.0 - cos_theta) ** 2, 1e-24)
    sigma_squared_twice = max(2.0 * float(sigma_dist) * float(sigma_dist), 1e-24)
    dots = xp.sum(normals[query_ids] * normals[neighbor_ids], axis=1)
    dots = xp.clip(dots, -1.0, 1.0)
    keep = dots > cos_theta
    q = query_ids[keep]
    n = neighbor_ids[keep]
    d = dots[keep]
    centroid_delta = centers[n] - centers[q]
    distance_squared = xp.sum(centroid_delta * centroid_delta, axis=1)
    alpha = xp.exp(-distance_squared / sigma_squared_twice)
    beta = xp.exp(-((1.0 - d) ** 2) / beta_denominator)
    weights = areas[q] * alpha * beta
    weighted_normals = normals[n] * weights[:, None]
    weighted_sum = xp.stack((
        xp.bincount(q, weights=weighted_normals[:, 0], minlength=face_count),
        xp.bincount(q, weights=weighted_normals[:, 1], minlength=face_count),
        xp.bincount(q, weights=weighted_normals[:, 2], minlength=face_count),
    ), axis=1)
    lengths = xp.sqrt(xp.sum(weighted_sum * weighted_sum, axis=1))
    valid = xp.isfinite(lengths) & (lengths > 1e-15)
    safe_lengths = xp.where(valid, lengths, 1.0)
    filtered_normals = xp.where(valid[:, None], weighted_sum / safe_lengths[:, None], normals)

    vertex_pair_counts = (v_offsets[1:] - v_offsets[:-1]).astype(xp.int64, copy=False)
    vertex_ids = xp.repeat(xp.arange(vertex_count, dtype=xp.int64), vertex_pair_counts)
    incident_faces = v_incident.astype(xp.int64, copy=False)
    offsets = centers[incident_faces] - v[vertex_ids]
    projections = xp.sum(offsets * filtered_normals[incident_faces], axis=1)
    contributions = projections[:, None] * normals[incident_faces]
    counts = vertex_pair_counts.astype(real_dtype, copy=False)
    sums = xp.stack((
        xp.bincount(vertex_ids, weights=contributions[:, 0], minlength=vertex_count),
        xp.bincount(vertex_ids, weights=contributions[:, 1], minlength=vertex_count),
        xp.bincount(vertex_ids, weights=contributions[:, 2], minlength=vertex_count),
    ), axis=1)
    safe_counts = xp.where(counts > 0, counts, 1.0)
    displacement = xp.where((counts > 0)[:, None], sums / safe_counts[:, None], 0.0)
    updated = v + displacement
    return (
        np.asarray(host(filtered_normals), dtype=np.float64),
        np.asarray(host(updated), dtype=np.float64),
        np.asarray(host(displacement), dtype=np.float64),
    )

def closest_points_cpu(
    points: np.ndarray, candidate_triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference for nearest point on candidate triangles."""
    distance_squared, weight_b, weight_c = _closest_points_vectorized(
        np, points, candidate_triangles
    )
    return (
        np.asarray(distance_squared, dtype=np.float64),
        np.asarray(weight_b, dtype=np.float64),
        np.asarray(weight_c, dtype=np.float64),
    )

class _CudaBackend:
    def __init__(self, device: GPUDevice):
        self.cp = device.native
        self.index = int(device.index)
        self.device_name = device.name
        self.precision = "float64"
        self._filter_topology_key = None
        self._filter_topology_arrays = None
        self.cp.cuda.Device(self.index).use()

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        cp = self.cp
        with cp.cuda.Device(self.index):
            o = cp.asarray(origins, dtype=cp.float64)
            d = cp.asarray(directions, dtype=cp.float64)
            tri = cp.asarray(triangles, dtype=cp.float64)
            v0 = tri[:, :, 0, :]
            edge1 = tri[:, :, 1, :] - v0
            edge2 = tri[:, :, 2, :] - v0
            pvec = cp.cross(d[:, :, None, :], edge2[:, None, :, :])
            determinant = cp.einsum("nci,nrci->nrc", edge1, pvec)
            valid = cp.abs(determinant) > 1e-12
            safe_determinant = cp.where(valid, determinant, 1.0)
            inverse = cp.where(valid, cp.divide(1.0, safe_determinant), 0.0)
            tvec = o[:, :, None, :] - v0[:, None, :, :]
            u = cp.einsum("nrci,nrci->nrc", tvec, pvec) * inverse
            qvec = cp.cross(tvec, edge1[:, None, :, :])
            v = cp.einsum("nri,nrci->nrc", d, qvec) * inverse
            ray_t = cp.einsum("nci,nrci->nrc", edge2, qvec) * inverse
            valid &= (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)
            valid &= (ray_t >= -float(epsilon)) & (ray_t <= float(ray_limit) + float(epsilon))
            distances = cp.where(valid, cp.maximum(ray_t, 0.0), cp.inf)
            return cp.asnumpy(distances), cp.asnumpy(u), cp.asnumpy(v)

    def closest_points(self, points, candidate_triangles):
        cp = self.cp
        with cp.cuda.Device(self.index):
            distance_squared, weight_b, weight_c = _closest_points_vectorized(
                cp, points, candidate_triangles
            )
            return (
                cp.asnumpy(distance_squared),
                cp.asnumpy(weight_b),
                cp.asnumpy(weight_c),
            )

    def filter_mesh_step(
        self, vertices, face_normals, face_centers, face_areas,
        face_query_offsets, face_neighbor_ids, vertex_offsets,
        vertex_incident_faces, sigma_dist, theta_radians,
    ):
        cp = self.cp
        with cp.cuda.Device(self.index):
            topology_key = tuple(map(id, (face_query_offsets, face_neighbor_ids, vertex_offsets, vertex_incident_faces)))
            if topology_key != self._filter_topology_key:
                self._filter_topology_arrays = tuple(
                    cp.asarray(item, dtype=cp.int64)
                    for item in (face_query_offsets, face_neighbor_ids, vertex_offsets, vertex_incident_faces)
                )
                self._filter_topology_key = topology_key
            return _filter_mesh_step_xp(
                cp, vertices, face_normals, face_centers, face_areas,
                *self._filter_topology_arrays, sigma_dist, theta_radians,
                dtype=cp.float64, to_host=cp.asnumpy,
            )

    def qem_candidate_batch(self, vertices, quadrics, edges):
        cp = self.cp
        with cp.cuda.Device(self.index):
            return _qem_candidate_batch_xp(
                cp, vertices, quadrics, edges, dtype=cp.float64, to_host=cp.asnumpy
            )

_OPENCL_KERNEL_TEMPLATE = r"""
{fp64_pragma}
typedef {real_type} real_t;

__kernel void intersect_rays(
    __global const real_t *origins,
    __global const real_t *directions,
    __global const real_t *triangles,
    const uint candidate_count,
    const real_t epsilon,
    const real_t ray_limit,
    __global real_t *out_t,
    __global real_t *out_u,
    __global real_t *out_v)
{{
    const size_t gid = get_global_id(0);
    const size_t candidate = gid % (size_t)candidate_count;
    const size_t ray = (gid / (size_t)candidate_count) % 2;
    const size_t sample = gid / ((size_t)candidate_count * 2);
    const size_t origin_base = (sample * 2 + ray) * 3;
    const size_t tri_base = (sample * (size_t)candidate_count + candidate) * 9;

    const real_t ox = origins[origin_base + 0];
    const real_t oy = origins[origin_base + 1];
    const real_t oz = origins[origin_base + 2];
    const real_t dx = directions[origin_base + 0];
    const real_t dy = directions[origin_base + 1];
    const real_t dz = directions[origin_base + 2];
    const real_t v0x = triangles[tri_base + 0];
    const real_t v0y = triangles[tri_base + 1];
    const real_t v0z = triangles[tri_base + 2];
    const real_t e1x = triangles[tri_base + 3] - v0x;
    const real_t e1y = triangles[tri_base + 4] - v0y;
    const real_t e1z = triangles[tri_base + 5] - v0z;
    const real_t e2x = triangles[tri_base + 6] - v0x;
    const real_t e2y = triangles[tri_base + 7] - v0y;
    const real_t e2z = triangles[tri_base + 8] - v0z;

    const real_t px = dy * e2z - dz * e2y;
    const real_t py = dz * e2x - dx * e2z;
    const real_t pz = dx * e2y - dy * e2x;
    const real_t det = e1x * px + e1y * py + e1z * pz;
    real_t hit_t = (real_t)INFINITY;
    real_t hit_u = (real_t)0;
    real_t hit_v = (real_t)0;

    if (fabs(det) > (real_t)1.0e-12) {{
        const real_t inv_det = (real_t)1 / det;
        const real_t tx = ox - v0x;
        const real_t ty = oy - v0y;
        const real_t tz = oz - v0z;
        const real_t u = (tx * px + ty * py + tz * pz) * inv_det;
        const real_t qx = ty * e1z - tz * e1y;
        const real_t qy = tz * e1x - tx * e1z;
        const real_t qz = tx * e1y - ty * e1x;
        const real_t v = (dx * qx + dy * qy + dz * qz) * inv_det;
        const real_t t = (e2x * qx + e2y * qy + e2z * qz) * inv_det;
        if (u >= (real_t)-1.0e-6 && v >= (real_t)-1.0e-6 &&
            (u + v) <= (real_t)1.000001 && t >= -epsilon &&
            t <= ray_limit + epsilon) {{
            hit_t = fmax(t, (real_t)0);
            hit_u = u;
            hit_v = v;
        }}
    }}
    out_t[gid] = hit_t;
    out_u[gid] = hit_u;
    out_v[gid] = hit_v;
}}
"""

_OPENCL_CLOSEST_KERNEL_TEMPLATE = r"""
__kernel void closest_points(
    __global const real_t *points,
    __global const real_t *triangles,
    const uint candidate_count,
    __global real_t *out_distance_squared,
    __global real_t *out_u,
    __global real_t *out_v)
{{
    const size_t gid = get_global_id(0);
    const size_t candidate = gid % (size_t)candidate_count;
    const size_t sample = gid / (size_t)candidate_count;
    const size_t point_base = sample * 3;
    const size_t tri_base = gid * 9;
    const real_t px = points[point_base + 0];
    const real_t py = points[point_base + 1];
    const real_t pz = points[point_base + 2];
    const real_t ax = triangles[tri_base + 0];
    const real_t ay = triangles[tri_base + 1];
    const real_t az = triangles[tri_base + 2];
    const real_t bx = triangles[tri_base + 3];
    const real_t by = triangles[tri_base + 4];
    const real_t bz = triangles[tri_base + 5];
    const real_t cx = triangles[tri_base + 6];
    const real_t cy = triangles[tri_base + 7];
    const real_t cz = triangles[tri_base + 8];
    const real_t abx = bx - ax, aby = by - ay, abz = bz - az;
    const real_t acx = cx - ax, acy = cy - ay, acz = cz - az;
    const real_t bcx = cx - bx, bcy = cy - by, bcz = cz - bz;
    const real_t apx = px - ax, apy = py - ay, apz = pz - az;
    const real_t d1 = abx * apx + aby * apy + abz * apz;
    const real_t d2 = acx * apx + acy * apy + acz * apz;
    const real_t bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const real_t d3 = abx * bpx + aby * bpy + abz * bpz;
    const real_t d4 = acx * bpx + acy * bpy + acz * bpz;
    const real_t cpx = px - cx, cpy = py - cy, cpz = pz - cz;
    const real_t d5 = abx * cpx + aby * cpy + abz * cpz;
    const real_t d6 = acx * cpx + acy * cpy + acz * cpz;
    real_t wa = (real_t)0, wb = (real_t)0, wc = (real_t)0;
    int assigned = 0;
    if (d1 <= (real_t)0 && d2 <= (real_t)0) {{ wa = (real_t)1; assigned = 1; }}
    if (!assigned && d3 >= (real_t)0 && d4 <= d3) {{ wb = (real_t)1; assigned = 1; }}
    const real_t vc = d1 * d4 - d3 * d2;
    if (!assigned && vc <= (real_t)0 && d1 >= (real_t)0 && d3 <= (real_t)0) {{
        const real_t den = fabs(d1 - d3) > (real_t)1e-30 ? d1 - d3 : (real_t)1;
        const real_t t = clamp(d1 / den, (real_t)0, (real_t)1);
        wa = (real_t)1 - t; wb = t; assigned = 1;
    }}
    if (!assigned && d6 >= (real_t)0 && d5 <= d6) {{ wc = (real_t)1; assigned = 1; }}
    const real_t vb = d5 * d2 - d1 * d6;
    if (!assigned && vb <= (real_t)0 && d2 >= (real_t)0 && d6 <= (real_t)0) {{
        const real_t den = fabs(d2 - d6) > (real_t)1e-30 ? d2 - d6 : (real_t)1;
        const real_t t = clamp(d2 / den, (real_t)0, (real_t)1);
        wa = (real_t)1 - t; wc = t; assigned = 1;
    }}
    const real_t va = d3 * d6 - d5 * d4;
    if (!assigned && va <= (real_t)0 && (d4 - d3) >= (real_t)0 && (d5 - d6) >= (real_t)0) {{
        const real_t den0 = (d4 - d3) + (d5 - d6);
        const real_t den = fabs(den0) > (real_t)1e-30 ? den0 : (real_t)1;
        const real_t t = clamp((d4 - d3) / den, (real_t)0, (real_t)1);
        wb = (real_t)1 - t; wc = t; assigned = 1;
    }}
    if (!assigned) {{
        const real_t den0 = va + vb + vc;
        const real_t den = fabs(den0) > (real_t)1e-30 ? den0 : (real_t)1;
        wb = vb / den; wc = vc / den; wa = (real_t)1 - wb - wc;
    }}

    const real_t nx = aby * acz - abz * acy;
    const real_t ny = abz * acx - abx * acz;
    const real_t nz = abx * acy - aby * acx;
    const real_t area2 = nx * nx + ny * ny + nz * nz;
    const real_t ab2 = abx * abx + aby * aby + abz * abz;
    const real_t ac2 = acx * acx + acy * acy + acz * acz;
    const real_t bc2 = bcx * bcx + bcy * bcy + bcz * bcz;
    const real_t scale2 = fmax(ab2, fmax(ac2, bc2));
    const int degenerate = area2 <= fmax(scale2 * scale2 * (real_t)1e-24, (real_t)1e-60);
    if (degenerate) {{
        const real_t den_ab = ab2 > (real_t)1e-30 ? ab2 : (real_t)1;
        const real_t den_bc = bc2 > (real_t)1e-30 ? bc2 : (real_t)1;
        const real_t den_ca = ac2 > (real_t)1e-30 ? ac2 : (real_t)1;
        const real_t tab = clamp((apx * abx + apy * aby + apz * abz) / den_ab, (real_t)0, (real_t)1);
        const real_t qax = ax + tab * abx, qay = ay + tab * aby, qaz = az + tab * abz;
        const real_t dax = px - qax, day = py - qay, daz = pz - qaz;
        real_t best = dax * dax + day * day + daz * daz;
        wa = (real_t)1 - tab; wb = tab; wc = (real_t)0;
        const real_t bpx2 = px - bx, bpy2 = py - by, bpz2 = pz - bz;
        const real_t tbc = clamp((bpx2 * bcx + bpy2 * bcy + bpz2 * bcz) / den_bc, (real_t)0, (real_t)1);
        const real_t qbx = bx + tbc * bcx, qby = by + tbc * bcy, qbz = bz + tbc * bcz;
        const real_t dbx = px - qbx, dby = py - qby, dbz = pz - qbz;
        const real_t dbest = dbx * dbx + dby * dby + dbz * dbz;
        if (dbest < best) {{ best = dbest; wa = (real_t)0; wb = (real_t)1 - tbc; wc = tbc; }}
        const real_t cpx2 = px - cx, cpy2 = py - cy, cpz2 = pz - cz;
        const real_t cax = ax - cx, cay = ay - cy, caz = az - cz;
        const real_t tca = clamp((cpx2 * cax + cpy2 * cay + cpz2 * caz) / den_ca, (real_t)0, (real_t)1);
        const real_t qcx = cx + tca * cax, qcy = cy + tca * cay, qcz = cz + tca * caz;
        const real_t dcx = px - qcx, dcy = py - qcy, dcz = pz - qcz;
        const real_t dcandidate = dcx * dcx + dcy * dcy + dcz * dcz;
        if (dcandidate < best) {{ wa = tca; wb = (real_t)0; wc = (real_t)1 - tca; }}
    }}
    const real_t qx = wa * ax + wb * bx + wc * cx;
    const real_t qy = wa * ay + wb * by + wc * cy;
    const real_t qz = wa * az + wb * bz + wc * cz;
    const real_t dx = px - qx, dy = py - qy, dz = pz - qz;
    out_distance_squared[gid] = dx * dx + dy * dy + dz * dz;
    out_u[gid] = wb;
    out_v[gid] = wc;
}}
"""

def _translate_opencl_source_to_hip(source: str) -> str:
    """Translate a self-contained OpenCL-C kernel source into HIPRTC C++."""
    import re

    source = re.sub(r"__global\s+", "", source)

    source = re.sub(r"__kernel\s+void", 'extern "C" __global__ void', source)
    source = source.replace(
        "get_global_id(0)",
        "(static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x)",
    )
    source = source.replace("clamp(", "pfs_clamp(")
    source = source.replace("#pragma OPENCL FP_CONTRACT OFF", "#pragma clang fp contract(off)")
    source = source.replace("__private ", "")
    source = re.sub(r"\bsize_t\b", "unsigned long long", source)
    source = re.sub(r"\bulong\b", "unsigned long long", source)
    source = re.sub(r"\bfabs\s*\(", "__builtin_fabs(", source)
    source = re.sub(r"\bfmax\s*\(", "__builtin_fmax(", source)
    source = re.sub(r"\bfmin\s*\(", "__builtin_fmin(", source)
    source = source.replace("INFINITY", "__builtin_huge_val()")
    return (
        "typedef unsigned int uint;\n"
        "__device__ inline double pfs_clamp(double value, double lower, double upper) {\n"
        "    return __builtin_fmin(__builtin_fmax(value, lower), upper);\n"
        "}\n"
        + source
    )

def _hip_kernel_source(*, include_qem: bool = True) -> str:
    """Build the base HIPRTC source; QEM can be excluded from backend startup."""
    source = (
        _OPENCL_KERNEL_TEMPLATE.format(fp64_pragma="", real_type="double")
        + _OPENCL_CLOSEST_KERNEL_TEMPLATE.format()
    )
    if include_qem:
        source += _OPENCL_QEM_CANDIDATE_TEMPLATE.format(fp64_pragma="", real_type="double")
    return _translate_opencl_source_to_hip(source)

def _hip_qem_kernel_source() -> str:
    """Return the optional QEM-only HIPRTC source for lazy compilation."""
    return _translate_opencl_source_to_hip(
        _OPENCL_QEM_CANDIDATE_TEMPLATE.format(fp64_pragma="", real_type="double")
    )

class _WindowsHipBackend:
    """Windows AMD backend using the optional native HIP/HIPRTC bridge DLL."""

    def __init__(self, device: GPUDevice):
        import ctypes

        bridge, architecture, bridge_path = device.native
        self.bridge = bridge
        self.device_name = device.name
        self.index = int(device.index)
        self.architecture = str(architecture or "unknown")
        self.bridge_path = Path(bridge_path)
        self.precision = "float64"
        self._handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(8192)
        result = self.bridge.pfhip_create_backend(
            self.index, _hip_kernel_source(include_qem=False).encode("utf-8"),
            ctypes.byref(self._handle), error, len(error),
        )
        if result != 0 or not self._handle.value:
            message = error.value.decode("utf-8", errors="replace") or f"HIP bridge initialization failed ({result})."
            raise RuntimeError(message)

    @staticmethod
    def _double_pointer(array):
        import ctypes
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    @staticmethod
    def _check(result, error, operation):
        if result != 0:
            message = error.value.decode("utf-8", errors="replace") or f"{operation} failed ({result})."
            raise RuntimeError(message)

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        import ctypes
        source_origins = np.asarray(origins, dtype=np.float64)
        source_directions = np.asarray(directions, dtype=np.float64)
        source_triangles = np.asarray(triangles, dtype=np.float64)
        if source_origins.ndim != 3 or source_origins.shape[1:] != (2, 3):
            raise ValueError("Ray origins must have shape (N, 2, 3).")
        if source_directions.shape != source_origins.shape:
            raise ValueError("Ray directions must match the origins shape.")
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Ray candidates must have shape (N, K, 3, 3).")
        sample_count, candidate_count = source_triangles.shape[:2]
        output_shape = (sample_count, 2, candidate_count)
        out_t = np.empty(output_shape, dtype=np.float64)
        out_u = np.empty(output_shape, dtype=np.float64)
        out_v = np.empty(output_shape, dtype=np.float64)
        if sample_count == 0 or candidate_count == 0:
            return out_t, out_u, out_v
        origin_shift = source_origins.reshape(-1, 3).mean(axis=0)
        host_origins = np.ascontiguousarray(source_origins - origin_shift, dtype=np.float64)
        host_directions = np.ascontiguousarray(source_directions, dtype=np.float64)
        host_triangles = np.ascontiguousarray(source_triangles - origin_shift, dtype=np.float64)
        error = ctypes.create_string_buffer(8192)
        result = self.bridge.pfhip_intersect_rays(
            self._handle, sample_count, candidate_count,
            self._double_pointer(host_origins), self._double_pointer(host_directions),
            self._double_pointer(host_triangles), float(epsilon), float(ray_limit),
            self._double_pointer(out_t), self._double_pointer(out_u), self._double_pointer(out_v),
            error, len(error),
        )
        self._check(result, error, "HIP ray intersection")
        return out_t, out_u, out_v

    def closest_points(self, points, candidate_triangles):
        import ctypes
        source_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        source_triangles = np.asarray(candidate_triangles, dtype=np.float64)
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Closest-point candidates must have shape (N, K, 3, 3).")
        if len(source_points) != len(source_triangles):
            raise ValueError("Closest-point query and candidate counts do not match.")
        sample_count, candidate_count = source_triangles.shape[:2]
        output_shape = (sample_count, candidate_count)
        out_distance = np.empty(output_shape, dtype=np.float64)
        out_u = np.empty(output_shape, dtype=np.float64)
        out_v = np.empty(output_shape, dtype=np.float64)
        if sample_count == 0 or candidate_count == 0:
            return out_distance, out_u, out_v
        origin_shift = source_points.mean(axis=0)
        host_points = np.ascontiguousarray(source_points - origin_shift, dtype=np.float64)
        host_triangles = np.ascontiguousarray(source_triangles - origin_shift, dtype=np.float64)
        error = ctypes.create_string_buffer(8192)
        result = self.bridge.pfhip_closest_points(
            self._handle, sample_count, candidate_count,
            self._double_pointer(host_points), self._double_pointer(host_triangles),
            self._double_pointer(out_distance), self._double_pointer(out_u), self._double_pointer(out_v),
            error, len(error),
        )
        self._check(result, error, "HIP closest-point search")
        return out_distance, out_u, out_v

    def qem_candidate_batch(self, vertices, quadrics, edges):
        import ctypes

        operation = getattr(self.bridge, "pfhip_qem_candidate_batch", None)
        if not callable(operation):
            raise RuntimeError(
                "The Polyfitty HIP bridge predates the QEM candidate kernel; update the HIP bridge bundled with this Polyfitty Studio environment."
            )
        source_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        source_quadrics = np.ascontiguousarray(quadrics, dtype=np.float64)
        source_edges = np.ascontiguousarray(edges, dtype=np.uint64).reshape(-1, 2)
        count_edges = len(source_edges)
        out_cost = np.empty(count_edges, dtype=np.float64)
        out_position = np.empty((count_edges, 3), dtype=np.float64)
        if count_edges == 0:
            return out_cost, out_position
        error = ctypes.create_string_buffer(8192)
        qem_source = _hip_qem_kernel_source().encode("utf-8")
        result = operation(
            self._handle,
            qem_source,
            len(source_vertices),
            count_edges,
            self._double_pointer(source_vertices),
            self._double_pointer(source_quadrics),
            source_edges.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            self._double_pointer(out_cost),
            self._double_pointer(out_position),
            error,
            len(error),
        )
        self._check(result, error, "HIP QEM candidate placement")
        return out_cost, out_position

    def __del__(self):
        handle = getattr(self, "_handle", None)
        bridge = getattr(self, "bridge", None)
        if handle is not None and getattr(handle, "value", None) and bridge is not None:
            try:
                bridge.pfhip_destroy_backend(handle)
            except Exception:
                pass
            self._handle = None

class _HipBackend:
    """AMD HIP backend using HIP Python bindings and HIPRTC JIT kernels.

    Used on platforms where AMD's HIP Python bindings are available. Windows
    the shared Windows runtime uses _WindowsHipBackend and the optional native MSVC bridge.
    """

    _BLOCK_SIZE = 256

    def __init__(self, device: GPUDevice):
        from hip import hip, hiprtc

        self.hip = hip
        self.hiprtc = hiprtc
        self.index = int(device.index)
        self.device_name = device.name
        self.precision = "float64"
        self.properties = device.native[1]
        self.arch = _decode_name(getattr(self.properties, "gcnArchName", ""), "")
        if not self.arch:
            raise RuntimeError("HIP device did not report its gcnArchName; HIPRTC cannot target this GPU safely.")
        self._check(hip.hipSetDevice(self.index), hip.hipError_t.hipSuccess, "hipSetDevice")
        self.program = None
        self.module = None
        self._kernels = {}
        self._qem_program = None
        self._qem_module = None
        self._qem_kernel = None
        self._qem_compile_error = ""
        self._compile_kernels()

    @staticmethod
    def _result(call_result, success_value, label: str):
        if not isinstance(call_result, tuple):
            call_result = (call_result,)
        if not call_result:
            raise RuntimeError(f"{label} returned no status value")
        error = call_result[0]
        if error != success_value:
            raise RuntimeError(f"{label} failed: {error}")
        values = call_result[1:]
        if len(values) == 1:
            return values[0]
        return values

    def _check(self, call_result, success_value, label: str):
        return self._result(call_result, success_value, label)

    def _compile_kernels(self):
        hip, hiprtc = self.hip, self.hiprtc
        source = _hip_kernel_source(include_qem=False).encode("utf-8")
        self.program = self._check(
            hiprtc.hiprtcCreateProgram(source, b"polyfitty_gpu_kernels.hip", 0, [], []),
            hiprtc.hiprtcResult.HIPRTC_SUCCESS,
            "hiprtcCreateProgram",
        )
        options = [b"--offload-arch=" + self.arch.encode("ascii", errors="ignore")]
        compile_result = hiprtc.hiprtcCompileProgram(self.program, len(options), options)
        compile_error = compile_result[0] if isinstance(compile_result, tuple) else compile_result
        if compile_error != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = self._check(
                hiprtc.hiprtcGetProgramLogSize(self.program),
                hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                "hiprtcGetProgramLogSize",
            )
            log_buffer = bytearray(int(log_size))
            self._check(
                hiprtc.hiprtcGetProgramLog(self.program, log_buffer),
                hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                "hiprtcGetProgramLog",
            )
            raise RuntimeError(
                "HIPRTC kernel compilation failed for " + self.arch + ": "
                + log_buffer.decode("utf-8", errors="replace")
            )
        code_size = int(self._check(
            hiprtc.hiprtcGetCodeSize(self.program),
            hiprtc.hiprtcResult.HIPRTC_SUCCESS,
            "hiprtcGetCodeSize",
        ))
        code = bytearray(code_size)
        self._check(
            hiprtc.hiprtcGetCode(self.program, code),
            hiprtc.hiprtcResult.HIPRTC_SUCCESS,
            "hiprtcGetCode",
        )
        self.module = self._check(
            hip.hipModuleLoadData(code), hip.hipError_t.hipSuccess,
            "hipModuleLoadData",
        )
        for kernel_name in (b"intersect_rays", b"closest_points"):
            self._kernels[kernel_name.decode("ascii")] = self._check(
                hip.hipModuleGetFunction(self.module, kernel_name),
                hip.hipError_t.hipSuccess,
                "hipModuleGetFunction(" + kernel_name.decode("ascii") + ")",
            )

    def _ensure_qem_kernel(self):
        if self._qem_kernel is not None:
            return
        if self._qem_compile_error:
            raise RuntimeError(self._qem_compile_error)
        hip, hiprtc = self.hip, self.hiprtc
        try:
            source = _hip_qem_kernel_source().encode("utf-8")
            program = self._check(
                hiprtc.hiprtcCreateProgram(source, b"polyfitty_qem_candidates.hip", 0, [], []),
                hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                "hiprtcCreateProgram(QEM)",
            )
            options = [b"--offload-arch=" + self.arch.encode("ascii", errors="ignore")]
            compile_result = hiprtc.hiprtcCompileProgram(program, len(options), options)
            compile_error = compile_result[0] if isinstance(compile_result, tuple) else compile_result
            if compile_error != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
                log_size = self._check(
                    hiprtc.hiprtcGetProgramLogSize(program),
                    hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                    "hiprtcGetProgramLogSize(QEM)",
                )
                log_buffer = bytearray(int(log_size))
                self._check(
                    hiprtc.hiprtcGetProgramLog(program, log_buffer),
                    hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                    "hiprtcGetProgramLog(QEM)",
                )
                raise RuntimeError(
                    "Optional HIP QEM kernel compilation failed for " + self.arch + ": "
                    + log_buffer.decode("utf-8", errors="replace")
                )
            code_size = int(self._check(
                hiprtc.hiprtcGetCodeSize(program),
                hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                "hiprtcGetCodeSize(QEM)",
            ))
            code = bytearray(code_size)
            self._check(
                hiprtc.hiprtcGetCode(program, code),
                hiprtc.hiprtcResult.HIPRTC_SUCCESS,
                "hiprtcGetCode(QEM)",
            )
            module = self._check(
                hip.hipModuleLoadData(code), hip.hipError_t.hipSuccess,
                "hipModuleLoadData(QEM)",
            )
            kernel = self._check(
                hip.hipModuleGetFunction(module, b"qem_candidate_placements"),
                hip.hipError_t.hipSuccess,
                "hipModuleGetFunction(qem_candidate_placements)",
            )
            self._qem_program = program
            self._qem_module = module
            self._qem_kernel = kernel
        except Exception as exc:
            self._qem_compile_error = (
                f"Optional HIP QEM candidate kernel is unavailable: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(self._qem_compile_error) from exc

    def _allocate_upload(self, array, allocations):
        hip = self.hip
        contiguous = np.ascontiguousarray(array)
        pointer = self._check(
            hip.hipMalloc(int(contiguous.nbytes)), hip.hipError_t.hipSuccess, "hipMalloc"
        )
        allocations.append(pointer)
        self._check(
            hip.hipMemcpy(
                pointer, contiguous, int(contiguous.nbytes),
                hip.hipMemcpyKind.hipMemcpyHostToDevice,
            ),
            hip.hipError_t.hipSuccess,
            "hipMemcpy(host-to-device)",
        )
        return pointer

    def _allocate_output(self, nbytes, allocations):
        if int(nbytes) <= 0:
            return None
        pointer = self._check(
            self.hip.hipMalloc(int(nbytes)), self.hip.hipError_t.hipSuccess, "hipMalloc(output)"
        )
        allocations.append(pointer)
        return pointer

    def _launch_function(self, kernel, label: str, total: int, arguments: tuple):
        if total <= 0:
            return
        block = self._BLOCK_SIZE
        grid = (int(total) + block - 1) // block
        self._check(
            self.hip.hipModuleLaunchKernel(
                kernel, grid, 1, 1, block, 1, 1,
                sharedMemBytes=0, stream=None, kernelParams=None, extra=arguments,
            ),
            self.hip.hipError_t.hipSuccess,
            "hipModuleLaunchKernel(" + label + ")",
        )
        self._check(self.hip.hipDeviceSynchronize(), self.hip.hipError_t.hipSuccess, "hipDeviceSynchronize")

    def _launch(self, kernel_name: str, total: int, arguments: tuple):
        self._launch_function(self._kernels[kernel_name], kernel_name, total, arguments)

    def _copy_to_host(self, output, pointer):
        if output.size:
            self._check(
                self.hip.hipMemcpy(
                    output, pointer, int(output.nbytes),
                    self.hip.hipMemcpyKind.hipMemcpyDeviceToHost,
                ),
                self.hip.hipError_t.hipSuccess,
                "hipMemcpy(device-to-host)",
            )

    def _free_all(self, allocations):
        for pointer in reversed(allocations):
            try:
                self.hip.hipFree(pointer)
            except Exception:
                pass

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        import ctypes

        source_origins = np.asarray(origins, dtype=np.float64)
        source_directions = np.asarray(directions, dtype=np.float64)
        source_triangles = np.asarray(triangles, dtype=np.float64)
        if source_origins.ndim != 3 or source_origins.shape[1:] != (2, 3):
            raise ValueError("Ray origins must have shape (N, 2, 3).")
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Ray candidates must have shape (N, K, 3, 3).")
        sample_count, candidate_count = source_triangles.shape[:2]
        output_shape = (sample_count, 2, candidate_count)
        out_t = np.empty(output_shape, dtype=np.float64)
        out_u = np.empty(output_shape, dtype=np.float64)
        out_v = np.empty(output_shape, dtype=np.float64)
        if sample_count == 0 or candidate_count == 0:
            return out_t, out_u, out_v

        origin_shift = source_origins.reshape(-1, 3).mean(axis=0)
        host_origins = np.ascontiguousarray(source_origins - origin_shift, dtype=np.float64)
        host_directions = np.ascontiguousarray(source_directions, dtype=np.float64)
        host_triangles = np.ascontiguousarray(source_triangles - origin_shift, dtype=np.float64)
        allocations = []
        try:
            d_origins = self._allocate_upload(host_origins, allocations)
            d_directions = self._allocate_upload(host_directions, allocations)
            d_triangles = self._allocate_upload(host_triangles, allocations)
            d_t = self._allocate_output(out_t.nbytes, allocations)
            d_u = self._allocate_output(out_u.nbytes, allocations)
            d_v = self._allocate_output(out_v.nbytes, allocations)
            total = sample_count * 2 * candidate_count
            self._launch("intersect_rays", total, (
                d_origins, d_directions, d_triangles, ctypes.c_uint(candidate_count),
                ctypes.c_double(float(epsilon)), ctypes.c_double(float(ray_limit)),
                d_t, d_u, d_v,
            ))
            self._copy_to_host(out_t, d_t)
            self._copy_to_host(out_u, d_u)
            self._copy_to_host(out_v, d_v)
            return out_t, out_u, out_v
        finally:
            self._free_all(allocations)

    def closest_points(self, points, candidate_triangles):
        import ctypes

        source_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        source_triangles = np.asarray(candidate_triangles, dtype=np.float64)
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Closest-point candidates must have shape (N, K, 3, 3).")
        if len(source_points) != len(source_triangles):
            raise ValueError("Closest-point query and candidate counts do not match.")
        sample_count, candidate_count = source_triangles.shape[:2]
        output_shape = (sample_count, candidate_count)
        out_distance = np.empty(output_shape, dtype=np.float64)
        out_u = np.empty(output_shape, dtype=np.float64)
        out_v = np.empty(output_shape, dtype=np.float64)
        if sample_count == 0 or candidate_count == 0:
            return out_distance, out_u, out_v

        origin_shift = source_points.mean(axis=0)
        host_points = np.ascontiguousarray(source_points - origin_shift, dtype=np.float64)
        host_triangles = np.ascontiguousarray(source_triangles - origin_shift, dtype=np.float64)
        allocations = []
        try:
            d_points = self._allocate_upload(host_points, allocations)
            d_triangles = self._allocate_upload(host_triangles, allocations)
            d_distance = self._allocate_output(out_distance.nbytes, allocations)
            d_u = self._allocate_output(out_u.nbytes, allocations)
            d_v = self._allocate_output(out_v.nbytes, allocations)
            total = sample_count * candidate_count
            self._launch("closest_points", total, (
                d_points, d_triangles, ctypes.c_uint(candidate_count), d_distance, d_u, d_v,
            ))
            self._copy_to_host(out_distance, d_distance)
            self._copy_to_host(out_u, d_u)
            self._copy_to_host(out_v, d_v)
            return out_distance, out_u, out_v
        finally:
            self._free_all(allocations)

    def qem_candidate_batch(self, vertices, quadrics, edges):
        import ctypes

        source_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        source_quadrics = np.ascontiguousarray(quadrics, dtype=np.float64)
        source_edges = np.ascontiguousarray(edges, dtype=np.uint64).reshape(-1, 2)
        count_edges = len(source_edges)
        out_cost = np.empty(count_edges, dtype=np.float64)
        out_position = np.empty((count_edges, 3), dtype=np.float64)
        if count_edges == 0:
            return out_cost, out_position
        self._ensure_qem_kernel()
        allocations = []
        try:
            d_vertices = self._allocate_upload(source_vertices, allocations)
            d_quadrics = self._allocate_upload(source_quadrics, allocations)
            d_edges = self._allocate_upload(source_edges, allocations)
            d_cost = self._allocate_output(out_cost.nbytes, allocations)
            d_position = self._allocate_output(out_position.nbytes, allocations)
            self._launch_function(self._qem_kernel, "qem_candidate_placements", count_edges, (
                d_vertices, d_quadrics, d_edges, ctypes.c_uint(count_edges),
                d_cost, d_position,
            ))
            self._copy_to_host(out_cost, d_cost)
            self._copy_to_host(out_position, d_position)
            return out_cost, out_position
        finally:
            self._free_all(allocations)

    def __del__(self):
        hip = getattr(self, "hip", None)
        if hip is None:
            return
        for module in (getattr(self, "_qem_module", None), getattr(self, "module", None)):
            if module is not None:
                try:
                    hip.hipModuleUnload(module)
                except Exception:
                    pass

_OPENCL_FILTER_KERNEL_TEMPLATE = r"""
{fp64_pragma}
typedef {real_type} filter_real_t;

__kernel void filter_face_normals_csr(
    __global const filter_real_t *normals,
    __global const filter_real_t *centers,
    __global const filter_real_t *areas,
    __global const ulong *offsets,
    __global const ulong *neighbors,
    const filter_real_t sigma_squared_twice,
    const filter_real_t cos_theta,
    const filter_real_t beta_denominator,
    __global filter_real_t *out_normals)
{{
    const ulong face = (ulong)get_global_id(0);
    filter_real_t sx = (filter_real_t)0;
    filter_real_t sy = (filter_real_t)0;
    filter_real_t sz = (filter_real_t)0;
    for (ulong i = offsets[face]; i < offsets[face + 1]; ++i) {{
        const ulong neighbor = neighbors[i];
        const filter_real_t nx = normals[face * 3 + 0];
        const filter_real_t ny = normals[face * 3 + 1];
        const filter_real_t nz = normals[face * 3 + 2];
        const filter_real_t mx = normals[neighbor * 3 + 0];
        const filter_real_t my = normals[neighbor * 3 + 1];
        const filter_real_t mz = normals[neighbor * 3 + 2];
        filter_real_t dot_value = nx * mx + ny * my + nz * mz;
        dot_value = clamp(dot_value, (filter_real_t)-1, (filter_real_t)1);
        if (dot_value <= cos_theta) continue;
        const filter_real_t dx = centers[neighbor * 3 + 0] - centers[face * 3 + 0];
        const filter_real_t dy = centers[neighbor * 3 + 1] - centers[face * 3 + 1];
        const filter_real_t dz = centers[neighbor * 3 + 2] - centers[face * 3 + 2];
        const filter_real_t distance_squared = dx * dx + dy * dy + dz * dz;
        const filter_real_t alpha = exp(-distance_squared / sigma_squared_twice);
        const filter_real_t one_minus_dot = (filter_real_t)1 - dot_value;
        const filter_real_t beta = exp(-(one_minus_dot * one_minus_dot) / beta_denominator);
        const filter_real_t weight = areas[face] * alpha * beta;
        sx += mx * weight;
        sy += my * weight;
        sz += mz * weight;
    }}
    const filter_real_t length = sqrt(sx * sx + sy * sy + sz * sz);
    const ulong base = face * 3;
    if (isfinite(length) && length > (filter_real_t)1e-15) {{
        out_normals[base + 0] = sx / length;
        out_normals[base + 1] = sy / length;
        out_normals[base + 2] = sz / length;
    }} else {{
        out_normals[base + 0] = normals[base + 0];
        out_normals[base + 1] = normals[base + 1];
        out_normals[base + 2] = normals[base + 2];
    }}
}}

__kernel void update_filter_vertices_csr(
    __global const filter_real_t *vertices,
    __global const filter_real_t *face_centers,
    __global const filter_real_t *face_normals,
    __global const filter_real_t *filtered_normals,
    __global const ulong *offsets,
    __global const ulong *incident_faces,
    __global filter_real_t *out_vertices,
    __global filter_real_t *out_displacement)
{{
    const ulong vertex = (ulong)get_global_id(0);
    filter_real_t sx = (filter_real_t)0;
    filter_real_t sy = (filter_real_t)0;
    filter_real_t sz = (filter_real_t)0;
    const ulong begin = offsets[vertex];
    const ulong end = offsets[vertex + 1];
    const filter_real_t vx = vertices[vertex * 3 + 0];
    const filter_real_t vy = vertices[vertex * 3 + 1];
    const filter_real_t vz = vertices[vertex * 3 + 2];
    for (ulong i = begin; i < end; ++i) {{
        const ulong face = incident_faces[i];
        const ulong base = face * 3;
        const filter_real_t dx = face_centers[base + 0] - vx;
        const filter_real_t dy = face_centers[base + 1] - vy;
        const filter_real_t dz = face_centers[base + 2] - vz;
        const filter_real_t projection = dx * filtered_normals[base + 0]
            + dy * filtered_normals[base + 1] + dz * filtered_normals[base + 2];
        sx += projection * face_normals[base + 0];
        sy += projection * face_normals[base + 1];
        sz += projection * face_normals[base + 2];
    }}
    const filter_real_t count = (filter_real_t)(end - begin);
    if (count > (filter_real_t)0) {{ sx /= count; sy /= count; sz /= count; }}
    const ulong out_base = vertex * 3;
    out_displacement[out_base + 0] = sx;
    out_displacement[out_base + 1] = sy;
    out_displacement[out_base + 2] = sz;
    out_vertices[out_base + 0] = vx + sx;
    out_vertices[out_base + 1] = vy + sy;
    out_vertices[out_base + 2] = vz + sz;
}}
"""

_OPENCL_QEM_CANDIDATE_TEMPLATE = r"""
{fp64_pragma}
typedef {real_type} qem_real_t;
#pragma OPENCL FP_CONTRACT OFF

inline qem_real_t qem_point_cost(__private const qem_real_t *q,
                                  qem_real_t x, qem_real_t y, qem_real_t z) {{
    qem_real_t value = q[0] * x * x + q[5] * y * y + q[10] * z * z
        + (qem_real_t)2 * q[1] * x * y + (qem_real_t)2 * q[2] * x * z
        + (qem_real_t)2 * q[6] * y * z
        + (qem_real_t)2 * q[3] * x + (qem_real_t)2 * q[7] * y
        + (qem_real_t)2 * q[11] * z + q[15];
    return isfinite(value) ? fmax(value, (qem_real_t)0) : INFINITY;
}}

__kernel void qem_candidate_placements(
    __global const qem_real_t *vertices,
    __global const qem_real_t *quadrics,
    __global const ulong *edges,
    const uint edge_count,
    __global qem_real_t *out_cost,
    __global qem_real_t *out_position)
{{
    const ulong gid = (ulong)get_global_id(0);
    if (gid >= (ulong)edge_count) return;
    const ulong first = edges[gid * 2 + 0];
    const ulong second = edges[gid * 2 + 1];
    const ulong p0 = first * 3;
    const ulong p1 = second * 3;
    const qem_real_t x = vertices[p0 + 0];
    const qem_real_t y = vertices[p0 + 1];
    const qem_real_t z = vertices[p0 + 2];
    const qem_real_t dx = vertices[p1 + 0] - x;
    const qem_real_t dy = vertices[p1 + 1] - y;
    const qem_real_t dz = vertices[p1 + 2] - z;
    const qem_real_t length_sq = dx * dx + dy * dy + dz * dz;
    const ulong q0 = first * 16;
    const ulong q1 = second * 16;
    qem_real_t q[16];
    for (int i = 0; i < 16; ++i) q[i] = quadrics[q0 + i] + quadrics[q1 + i];
    if (!isfinite(length_sq) || length_sq <= (qem_real_t)1e-24) {{
        const qem_real_t mx = (qem_real_t)0.5 * (x + vertices[p1 + 0]);
        const qem_real_t my = (qem_real_t)0.5 * (y + vertices[p1 + 1]);
        const qem_real_t mz = (qem_real_t)0.5 * (z + vertices[p1 + 2]);
        out_cost[gid] = qem_point_cost(q, mx, my, mz);
        out_position[gid * 3 + 0] = mx;
        out_position[gid * 3 + 1] = my;
        out_position[gid * 3 + 2] = mz;
        return;
    }}
    const qem_real_t h0x = q[0] * x + q[1] * y + q[2] * z + q[3];
    const qem_real_t h0y = q[4] * x + q[5] * y + q[6] * z + q[7];
    const qem_real_t h0z = q[8] * x + q[9] * y + q[10] * z + q[11];
    const qem_real_t numerator = dx * h0x + dy * h0y + dz * h0z;
    const qem_real_t qdx = q[0] * dx + q[1] * dy + q[2] * dz;
    const qem_real_t qdy = q[4] * dx + q[5] * dy + q[6] * dz;
    const qem_real_t qdz = q[8] * dx + q[9] * dy + q[10] * dz;
    const qem_real_t denominator = dx * qdx + dy * qdy + dz * qdz;
    const qem_real_t base_cost = qem_point_cost(q, x, y, z);
    qem_real_t best_cost = base_cost;
    qem_real_t best_t = (qem_real_t)0;
    const qem_real_t half_cost = base_cost + (qem_real_t)2 * numerator * (qem_real_t)0.5
        + denominator * (qem_real_t)0.5 * (qem_real_t)0.5;
    if (isfinite(half_cost) && fmax(half_cost, (qem_real_t)0) < best_cost) {{
        best_cost = fmax(half_cost, (qem_real_t)0); best_t = (qem_real_t)0.5;
    }}
    const qem_real_t end_cost = base_cost + (qem_real_t)2 * numerator
        + denominator;
    if (isfinite(end_cost) && fmax(end_cost, (qem_real_t)0) < best_cost) {{
        best_cost = fmax(end_cost, (qem_real_t)0); best_t = (qem_real_t)1;
    }}
    if (isfinite(denominator) && denominator > (qem_real_t)1e-18) {{
        const qem_real_t optimum_t = clamp(-numerator / denominator, (qem_real_t)0, (qem_real_t)1);
        const qem_real_t optimum_cost = base_cost + (qem_real_t)2 * numerator * optimum_t
            + denominator * optimum_t * optimum_t;
        if (isfinite(optimum_cost) && fmax(optimum_cost, (qem_real_t)0) < best_cost) {{
            best_cost = fmax(optimum_cost, (qem_real_t)0); best_t = optimum_t;
        }}
    }}
    out_cost[gid] = best_cost;
    out_position[gid * 3 + 0] = x + best_t * dx;
    out_position[gid * 3 + 1] = y + best_t * dy;
    out_position[gid * 3 + 2] = z + best_t * dz;
}}
"""

class _OpenCLBackend:
    def __init__(self, device: GPUDevice):
        import pyopencl as cl

        self.cl = cl
        self.device = device.native[1]
        self.device_name = device.name
        extensions = str(getattr(self.device, "extensions", ""))
        supports_fp64 = int(getattr(self.device, "double_fp_config", 0) or 0) > 0
        if supports_fp64:
            self.dtype = np.float64
            self.precision = "float64"
            pragma = (
                "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                if "cl_khr_fp64" in extensions
                else "#pragma OPENCL EXTENSION cl_amd_fp64 : enable"
            )
            real_type = "double"
        else:
            self.dtype = np.float32
            self.precision = "float32"
            pragma = ""
            real_type = "float"
        source = (
            _OPENCL_KERNEL_TEMPLATE.format(fp64_pragma=pragma, real_type=real_type)
            + _OPENCL_CLOSEST_KERNEL_TEMPLATE.format()
            + _OPENCL_FILTER_KERNEL_TEMPLATE.format(fp64_pragma=pragma, real_type=real_type)
        )
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, self.device)
        self.program = cl.Program(self.context, source).build(options=["-cl-std=CL1.2"])
        self._qem_program = None
        self._qem_program_error = ""
        self._filter_topology_key = None
        self._filter_topology_buffers = None

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        cl = self.cl
        dtype = self.dtype

        source_origins = np.asarray(origins, dtype=np.float64)
        source_triangles = np.asarray(triangles, dtype=np.float64)
        origin_shift = source_origins.reshape(-1, 3).mean(axis=0)
        origins = np.ascontiguousarray(source_origins - origin_shift, dtype=dtype)
        directions = np.ascontiguousarray(directions, dtype=dtype)
        triangles = np.ascontiguousarray(source_triangles - origin_shift, dtype=dtype)
        sample_count, candidate_count = triangles.shape[:2]
        output_shape = (sample_count, 2, candidate_count)
        out_t = np.empty(output_shape, dtype=dtype)
        out_u = np.empty(output_shape, dtype=dtype)
        out_v = np.empty(output_shape, dtype=dtype)
        flags = cl.mem_flags
        origin_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=origins)
        direction_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=directions)
        triangle_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=triangles)
        t_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_t.nbytes)
        u_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_u.nbytes)
        v_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_v.nbytes)
        self.program.intersect_rays(
            self.queue, (sample_count * 2 * candidate_count,), None,
            origin_buffer, direction_buffer, triangle_buffer,
            np.uint32(candidate_count), dtype(epsilon), dtype(ray_limit),
            t_buffer, u_buffer, v_buffer,
        )
        cl.enqueue_copy(self.queue, out_t, t_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_u, u_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_v, v_buffer, is_blocking=True)
        return out_t.astype(np.float64, copy=False), out_u.astype(np.float64, copy=False), out_v.astype(np.float64, copy=False)

    def closest_points(self, points, candidate_triangles):
        cl = self.cl
        dtype = self.dtype
        source_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        source_triangles = np.asarray(candidate_triangles, dtype=np.float64)
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Closest-point candidates must have shape (N, K, 3, 3).")
        if len(source_points) != len(source_triangles):
            raise ValueError("Closest-point query and candidate counts do not match.")
        if len(source_points) == 0 or source_triangles.shape[1] == 0:
            shape = source_triangles.shape[:2]
            empty = np.empty(shape, dtype=np.float64)
            return empty, empty.copy(), empty.copy()
        origin_shift = source_points.mean(axis=0)
        points_local = np.ascontiguousarray(source_points - origin_shift, dtype=dtype)
        triangles_local = np.ascontiguousarray(source_triangles - origin_shift, dtype=dtype)
        sample_count, candidate_count = triangles_local.shape[:2]
        output_shape = (sample_count, candidate_count)
        out_distance = np.empty(output_shape, dtype=dtype)
        out_u = np.empty(output_shape, dtype=dtype)
        out_v = np.empty(output_shape, dtype=dtype)
        flags = cl.mem_flags
        point_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=points_local)
        triangle_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=triangles_local)
        distance_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_distance.nbytes)
        u_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_u.nbytes)
        v_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_v.nbytes)
        self.program.closest_points(
            self.queue, (sample_count * candidate_count,), None,
            point_buffer, triangle_buffer, np.uint32(candidate_count),
            distance_buffer, u_buffer, v_buffer,
        )
        cl.enqueue_copy(self.queue, out_distance, distance_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_u, u_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_v, v_buffer, is_blocking=True)
        return (
            out_distance.astype(np.float64, copy=False),
            out_u.astype(np.float64, copy=False),
            out_v.astype(np.float64, copy=False),
        )

    def filter_mesh_step(
        self, vertices, face_normals, face_centers, face_areas,
        face_query_offsets, face_neighbor_ids, vertex_offsets,
        vertex_incident_faces, sigma_dist, theta_radians,
    ):
        cl = self.cl
        dtype = self.dtype
        arrays = [
            np.ascontiguousarray(item, dtype=dtype)
            for item in (vertices, face_normals, face_centers, face_areas)
        ]
        topology_key = tuple(map(id, (face_query_offsets, face_neighbor_ids, vertex_offsets, vertex_incident_faces)))
        if topology_key != self._filter_topology_key:
            f_offsets = np.ascontiguousarray(face_query_offsets, dtype=np.uint64)
            f_neighbors = np.ascontiguousarray(face_neighbor_ids, dtype=np.uint64)
            v_offsets = np.ascontiguousarray(vertex_offsets, dtype=np.uint64)
            v_incident = np.ascontiguousarray(vertex_incident_faces, dtype=np.uint64)
            flags = cl.mem_flags
            def topology_buffer(array):
                return cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array)
            self._filter_topology_buffers = tuple(
                topology_buffer(array) for array in (f_offsets, f_neighbors, v_offsets, v_incident)
            )
            self._filter_topology_key = topology_key
        f_offsets_buffer, f_neighbors_buffer, v_offsets_buffer, v_incident_buffer = self._filter_topology_buffers
        vertex_count = len(arrays[0])
        face_count = len(arrays[1])
        out_normals = np.empty((face_count, 3), dtype=dtype)
        out_vertices = np.empty((vertex_count, 3), dtype=dtype)
        out_displacement = np.empty((vertex_count, 3), dtype=dtype)
        flags = cl.mem_flags

        def input_buffer(array):
            return cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array)

        vertex_buffer, normals_buffer, centers_buffer, areas_buffer = map(input_buffer, arrays)
        out_normals_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_normals.nbytes)
        out_vertices_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_vertices.nbytes)
        out_displacement_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_displacement.nbytes)
        cos_theta = float(np.cos(float(theta_radians)))
        self.program.filter_face_normals_csr(
            self.queue, (face_count,), None,
            normals_buffer, centers_buffer, areas_buffer,
            f_offsets_buffer, f_neighbors_buffer,
            dtype(max(2.0 * float(sigma_dist) ** 2, 1e-24)), dtype(cos_theta),
            dtype(max((1.0 - cos_theta) ** 2, 1e-24)), out_normals_buffer,
        )
        self.program.update_filter_vertices_csr(
            self.queue, (vertex_count,), None,
            vertex_buffer, centers_buffer, normals_buffer, out_normals_buffer,
            v_offsets_buffer, v_incident_buffer, out_vertices_buffer, out_displacement_buffer,
        )
        cl.enqueue_copy(self.queue, out_normals, out_normals_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_vertices, out_vertices_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_displacement, out_displacement_buffer, is_blocking=True)
        return out_normals.astype(np.float64), out_vertices.astype(np.float64), out_displacement.astype(np.float64)

    def qem_candidate_batch(self, vertices, quadrics, edges):
        if self.precision != "float64":
            raise RuntimeError("Deterministic QEM candidate evaluation requires OpenCL fp64 support.")
        cl = self.cl
        if self._qem_program is None:
            if self._qem_program_error:
                raise RuntimeError(self._qem_program_error)
            try:
                extensions = str(getattr(self.device, "extensions", ""))
                pragma = (
                    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                    if "cl_khr_fp64" in extensions
                    else "#pragma OPENCL EXTENSION cl_amd_fp64 : enable"
                )
                source = _OPENCL_QEM_CANDIDATE_TEMPLATE.format(
                    fp64_pragma=pragma, real_type="double"
                )
                self._qem_program = cl.Program(self.context, source).build(
                    options=["-cl-std=CL1.2"]
                )
            except Exception as exc:
                self._qem_program_error = (
                    f"Optional OpenCL QEM candidate kernel compilation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise RuntimeError(self._qem_program_error) from exc
        source_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        source_quadrics = np.ascontiguousarray(quadrics, dtype=np.float64)
        source_edges = np.ascontiguousarray(edges, dtype=np.uint64).reshape(-1, 2)
        edge_count = len(source_edges)
        if edge_count == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        out_cost = np.empty(edge_count, dtype=np.float64)
        out_position = np.empty((edge_count, 3), dtype=np.float64)
        flags = cl.mem_flags

        def input_buffer(array):
            return cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array)

        vertex_buffer = input_buffer(source_vertices)
        quadric_buffer = input_buffer(source_quadrics)
        edge_buffer = input_buffer(source_edges)
        cost_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_cost.nbytes)
        position_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, out_position.nbytes)
        self._qem_program.qem_candidate_placements(
            self.queue,
            (edge_count,),
            None,
            vertex_buffer,
            quadric_buffer,
            edge_buffer,
            np.uint32(edge_count),
            cost_buffer,
            position_buffer,
        )
        cl.enqueue_copy(self.queue, out_cost, cost_buffer, is_blocking=True)
        cl.enqueue_copy(self.queue, out_position, position_buffer, is_blocking=True)
        return out_cost, out_position

class _SyclBackend:
    """Intel oneAPI/SYCL adapter using dpnp for vectorized device operations."""

    def __init__(self, device: GPUDevice):
        import dpctl
        import dpnp

        self.dpctl = dpctl
        self.dpnp = dpnp
        self.device = device.native
        self.queue = dpctl.SyclQueue(self.device)
        try:
            supports_fp64 = bool(getattr(self.device, "has_aspect_fp64"))
        except Exception:
            try:
                supports_fp64 = bool(self.device.has_aspect("fp64"))
            except Exception:
                supports_fp64 = False
        self.dtype = dpnp.float64 if supports_fp64 else dpnp.float32
        self.precision = "float64" if supports_fp64 else "float32"
        self.device_name = device.name

    def _device_array(self, value, dtype=None):
        return self.dpnp.asarray(value, dtype=dtype or self.dtype, device=self.queue)

    def _host_array(self, value):
        return np.asarray(self.dpnp.asnumpy(value), dtype=np.float64)

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        xp = self.dpnp
        source_origins = np.asarray(origins, dtype=np.float64)
        source_triangles = np.asarray(triangles, dtype=np.float64)
        if source_origins.ndim != 3 or source_origins.shape[1:] != (2, 3):
            raise ValueError("Ray origins must have shape (N, 2, 3).")
        if source_triangles.ndim != 4 or source_triangles.shape[2:] != (3, 3):
            raise ValueError("Ray candidates must have shape (N, K, 3, 3).")
        sample_count, candidate_count = source_triangles.shape[:2]
        output_shape = (sample_count, 2, candidate_count)
        if sample_count == 0 or candidate_count == 0:
            empty = np.empty(output_shape, dtype=np.float64)
            return empty, empty.copy(), empty.copy()

        origin_shift = source_origins.reshape(-1, 3).mean(axis=0)
        o = self._device_array(source_origins - origin_shift)
        d = self._device_array(directions)
        tri = self._device_array(source_triangles - origin_shift)
        v0 = tri[:, :, 0, :]
        edge1 = tri[:, :, 1, :] - v0
        edge2 = tri[:, :, 2, :] - v0
        pvec = xp.cross(d[:, :, None, :], edge2[:, None, :, :])
        determinant = xp.einsum("nci,nrci->nrc", edge1, pvec)
        valid = xp.abs(determinant) > 1e-12
        safe_determinant = xp.where(valid, determinant, 1.0)
        inverse = xp.where(valid, xp.divide(1.0, safe_determinant), 0.0)
        tvec = o[:, :, None, :] - v0[:, None, :, :]
        u = xp.einsum("nrci,nrci->nrc", tvec, pvec) * inverse
        qvec = xp.cross(tvec, edge1[:, None, :, :])
        v = xp.einsum("nri,nrci->nrc", d, qvec) * inverse
        ray_t = xp.einsum("nci,nrci->nrc", edge2, qvec) * inverse
        valid &= (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)
        valid &= (ray_t >= -float(epsilon)) & (ray_t <= float(ray_limit) + float(epsilon))
        distances = xp.where(valid, xp.maximum(ray_t, 0.0), xp.inf)
        return self._host_array(distances), self._host_array(u), self._host_array(v)

    def closest_points(self, points, candidate_triangles):
        distance_squared, weight_b, weight_c = _closest_points_vectorized(
            self.dpnp, points, candidate_triangles, device=self.queue, dtype=self.dtype
        )
        return (
            self._host_array(distance_squared),
            self._host_array(weight_b),
            self._host_array(weight_c),
        )

    def filter_mesh_step(
        self, vertices, face_normals, face_centers, face_areas,
        face_query_offsets, face_neighbor_ids, vertex_offsets,
        vertex_incident_faces, sigma_dist, theta_radians,
    ):
        topology_key = tuple(map(id, (face_query_offsets, face_neighbor_ids, vertex_offsets, vertex_incident_faces)))
        if topology_key != getattr(self, "_filter_topology_key", None):
            self._filter_topology_arrays = tuple(
                self._device_array(item, dtype=self.dpnp.int64)
                for item in (face_query_offsets, face_neighbor_ids, vertex_offsets, vertex_incident_faces)
            )
            self._filter_topology_key = topology_key
        return _filter_mesh_step_xp(
            self.dpnp, vertices, face_normals, face_centers, face_areas,
            *self._filter_topology_arrays, sigma_dist, theta_radians,
            array_factory=self._device_array,
            to_host=self._host_array, dtype=self.dtype,
        )

    def qem_candidate_batch(self, vertices, quadrics, edges):
        if self.precision != "float64":
            raise RuntimeError("Deterministic QEM candidate evaluation requires SYCL float64 support.")
        return _qem_candidate_batch_xp(
            self.dpnp,
            vertices,
            quadrics,
            edges,
            array_factory=self._device_array,
            to_host=self._host_array,
            dtype=self.dpnp.float64,
        )

def _verify_backend_compute(backend) -> None:
    """Execute a tiny real ray/triangle kernel before declaring a GPU READY."""
    origins = np.asarray(
        [[[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]]], dtype=np.float64
    )
    directions = np.asarray(
        [[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]], dtype=np.float64
    )
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    triangles = triangle.reshape(1, 1, 3, 3)
    distances, u, v = backend.intersect_rays(
        origins, directions, triangles, epsilon=1e-9, ray_limit=2.0
    )
    if not (
        np.asarray(distances).shape == (1, 2, 1)
        and np.allclose(distances, 1.0, rtol=1e-4, atol=1e-5)
        and np.allclose(u, 0.25, rtol=1e-4, atol=1e-5)
        and np.allclose(v, 0.25, rtol=1e-4, atol=1e-5)
    ):
        raise RuntimeError("GPU compute self-test returned an invalid ray/triangle result")
    distance_squared, closest_u, closest_v = backend.closest_points(
        np.asarray([[0.25, 0.25, 1.0]], dtype=np.float64), triangles
    )
    if not (
        np.asarray(distance_squared).shape == (1, 1)
        and np.allclose(distance_squared, 1.0, rtol=1e-4, atol=1e-5)
        and np.allclose(closest_u, 0.25, rtol=1e-4, atol=1e-5)
        and np.allclose(closest_v, 0.25, rtol=1e-4, atol=1e-5)
    ):
        raise RuntimeError("GPU compute self-test returned an invalid closest-point/triangle result")

def _emit_gpu_status(callback: ProgressCallback, payload: dict) -> None:
    """Send structured status through the existing worker-safe progress channel."""
    if callback is None:
        return
    try:
        callback(None, "__PFS_GPU_STATUS__" + json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

class GPUComputeManager:
    """Select a GPU backend and optionally switch Auto to CPU when CPU benchmarks faster."""

    AUTO_CPU_FASTER_RATIO = 1.0 / 1.10
    MIN_AUTO_BENCHMARK_OPERATIONS = 4096

    def __init__(self, selection: str = "auto", progress_callback: ProgressCallback = None):
        self.selection = str(selection or "auto").strip().lower().replace("_", "-")
        self.strict_gpu_only = self.selection in {"gpu", "gpu-only", "cuda", "hip", "rocm", "sycl", "oneapi", "opencl"}
        self.performance_fallback = False
        self.progress_callback = progress_callback
        self.backend = None
        self.backend_name = "CPU"
        self.device_name = "CPU"
        self.vendor = "CPU"
        self.memory_bytes = 0
        self.precision = "float64"
        self.fallback_reason = ""
        self.gpu_batch_count = 0
        self.devices: list[GPUDevice] = []
        self.discovery_errors: list[str] = []
        self.activation_verified = False
        self._active_notified = False
        self._benchmark_done = False
        self.cpu_reference_seconds = 0.0
        self.gpu_benchmark_seconds = 0.0
        self.speedup_ratio = 0.0
        self.speedup_gain_percent = 0.0
        self.time_saved_percent = 0.0
        self.gpu_compute_seconds_total = 0.0
        self.gpu_intersection_tests_total = 0
        self.gpu_closest_point_tests_total = 0
        self.gpu_tests_per_second = 0.0
        self._closest_benchmark_done = False
        self.filter_gpu_batches = 0
        self.filter_gpu_seconds_total = 0.0
        self.filter_cpu_reference_seconds = 0.0
        self.filter_backend_name = "CPU"
        self.filter_fallback_reason = ""
        self._filter_step_disabled = False
        self.qem_candidate_batches = 0
        self.qem_candidate_seconds_total = 0.0
        self.qem_candidate_fallback_reason = ""
        self._qem_candidate_parity_verified = False
        self._qem_candidate_disabled = False
        self.benchmark_valid = False
        self.benchmark_detail = "Waiting for the first GPU bake batch."
        self.status = "CPU · initializing"

        if self.selection in {"cpu", "cpu-only", "disabled"}:
            self.fallback_reason = "CPU-only mode selected"
            self.status = "CPU · manually selected"
            self._publish_status("cpu_selected", detail=self.fallback_reason)
            return

        self.devices, scan_errors = discover_gpu_devices(progress_callback=self.progress_callback)
        requested_backend = {
            "cuda": "cuda", "hip": "hip", "rocm": "hip",
            "sycl": "sycl", "oneapi": "sycl", "opencl": "opencl",
        }.get(self.selection)
        if requested_backend:
            self.devices = [item for item in self.devices if item.backend.lower() == requested_backend]

        for device in self.devices:
            try:
                _safe_progress(
                    self.progress_callback,
                    f"Testing {device.vendor} {device.name} ({device.backend.upper()}) compute kernels...",
                )
                backend_factory = {
                    "cuda": _CudaBackend,
                    "hip": (_WindowsHipBackend if sys.platform == "win32" and isinstance(device.native, tuple) and len(device.native) == 3 and hasattr(device.native[0], "pfhip_create_backend") else _HipBackend),
                    "sycl": _SyclBackend,
                    "opencl": _OpenCLBackend,
                }.get(device.backend.lower())
                if backend_factory is None:
                    raise RuntimeError(f"No compute adapter is registered for backend '{device.backend}'.")
                backend: GPUComputeBackend = backend_factory(device)

                _verify_backend_compute(backend)
                self.backend = backend
                self.backend_name = device.backend.upper()
                self.device_name = device.name
                self.vendor = device.vendor
                self.memory_bytes = int(device.memory_bytes)
                self.precision = backend.precision
                self.activation_verified = True
                memory_gib = device.memory_bytes / (1024 ** 3) if device.memory_bytes else 0.0
                memory_text = f" · {memory_gib:.1f} GiB" if memory_gib else ""
                self.status = (
                    f"{device.vendor} {device.name} · {self.backend_name} READY "
                    f"({self.precision}{memory_text})"
                )
                _safe_progress(self.progress_callback, f"GPU acceleration ready · {self.status}")
                self.discovery_errors = list(scan_errors)
                self._publish_status(
                    "ready", detail="GPU kernel self-test passed; waiting for bake batches.",
                    memory_bytes=int(device.memory_bytes),
                )
                break
            except Exception as exc:
                scan_errors.append(
                    f"{device.label} activation test failed ({type(exc).__name__}: {exc})"
                )
                _safe_progress(
                    self.progress_callback,
                    f"{device.label} did not pass its compute test; checking the next compatible backend...",
                )
        self.discovery_errors = list(scan_errors)
        if self.backend is None:
            self.fallback_reason = "; ".join(scan_errors) if scan_errors else "No supported GPU compute device detected"
            if self.strict_gpu_only:
                self.status = f"GPU Only unavailable · {self.fallback_reason}"
                self._publish_status("gpu_required_error", detail=self.fallback_reason)
                raise GPUComputeRequiredError(
                    f"GPU Only was selected, but no usable GPU compute backend is available: {self.fallback_reason}"
                )
            self.status = f"CPU fallback · {self.fallback_reason}"
            _safe_progress(self.progress_callback, f"GPU unavailable · CPU bake selected ({self.fallback_reason})")
            self._publish_status("cpu_fallback", detail=self.fallback_reason)

    def _publish_status(self, state: str, detail: str = "", memory_bytes: int = 0) -> None:
        _emit_gpu_status(self.progress_callback, {
            "state": str(state),
            "selection": self.selection,
            "strict_gpu_only": bool(self.strict_gpu_only),
            "performance_fallback": bool(self.performance_fallback),
            "backend": self.backend_name,
            "device": self.device_name,
            "vendor": self.vendor,
            "precision": self.precision,
            "batch_count": int(self.gpu_batch_count),
            "detected_count": int(len(self.devices)),
            "activation_verified": bool(self.activation_verified),
            "memory_bytes": int(memory_bytes or self.memory_bytes),
            "fallback_reason": self.fallback_reason,
            "cpu_reference_seconds": float(self.cpu_reference_seconds),
            "gpu_benchmark_seconds": float(self.gpu_benchmark_seconds),
            "speedup_ratio": float(self.speedup_ratio),
            "speedup_gain_percent": float(self.speedup_gain_percent),
            "time_saved_percent": float(self.time_saved_percent),
            "gpu_compute_seconds_total": float(self.gpu_compute_seconds_total),
            "gpu_intersection_tests_total": int(self.gpu_intersection_tests_total),
            "gpu_closest_point_tests_total": int(self.gpu_closest_point_tests_total),
            "gpu_workload": (
                "cage-ray/triangle" if self.gpu_intersection_tests_total and not self.gpu_closest_point_tests_total
                else "direct-nearest-triangle" if self.gpu_closest_point_tests_total and not self.gpu_intersection_tests_total
                else "mixed-ray-and-nearest-triangle" if self.gpu_intersection_tests_total or self.gpu_closest_point_tests_total
                else "not-started"
            ),
            "gpu_tests_per_second": float(self.gpu_tests_per_second),
            "reconstruction_filter_backend": str(self.filter_backend_name),
            "reconstruction_filter_gpu_batches": int(self.filter_gpu_batches),
            "reconstruction_filter_gpu_seconds": float(self.filter_gpu_seconds_total),
            "reconstruction_filter_cpu_reference_seconds": float(self.filter_cpu_reference_seconds),
            "reconstruction_filter_fallback_reason": str(self.filter_fallback_reason),
            "qem_candidate_gpu_batches": int(getattr(self, "qem_candidate_batches", 0)),
            "qem_candidate_gpu_seconds": float(getattr(self, "qem_candidate_seconds_total", 0.0)),
            "qem_candidate_gpu_fallback_reason": str(getattr(self, "qem_candidate_fallback_reason", "")),
            "benchmark_valid": bool(self.benchmark_valid),
            "benchmark_detail": str(self.benchmark_detail),
            "detail": str(detail),
        })

    @staticmethod
    def _results_match(gpu_result, cpu_result) -> bool:
        """Compare actual hit masks and hit coordinates before trusting a speed sample."""
        if len(gpu_result) != 3 or len(cpu_result) != 3:
            return False
        gpu_t, gpu_u, gpu_v = (np.asarray(item) for item in gpu_result)
        cpu_t, cpu_u, cpu_v = (np.asarray(item) for item in cpu_result)
        if gpu_t.shape != cpu_t.shape or gpu_u.shape != cpu_u.shape or gpu_v.shape != cpu_v.shape:
            return False
        gpu_hits = np.isfinite(gpu_t)
        cpu_hits = np.isfinite(cpu_t)
        if not np.array_equal(gpu_hits, cpu_hits):
            return False
        if not np.any(gpu_hits):
            return True

        return bool(
            np.allclose(gpu_t[gpu_hits], cpu_t[cpu_hits], rtol=5e-4, atol=5e-6)
            and np.allclose(gpu_u[gpu_hits], cpu_u[cpu_hits], rtol=5e-4, atol=5e-6)
            and np.allclose(gpu_v[gpu_hits], cpu_v[cpu_hits], rtol=5e-4, atol=5e-6)
        )

    def _record_speed_benchmark(self, gpu_elapsed: float, cpu_elapsed: float, detail: str) -> bool:
        """Store a same-input benchmark and switch Auto to CPU when CPU clearly wins."""
        self.cpu_reference_seconds = max(float(cpu_elapsed), 1e-9)
        self.gpu_benchmark_seconds = max(float(gpu_elapsed), 1e-9)
        self.speedup_ratio = self.cpu_reference_seconds / self.gpu_benchmark_seconds
        self.speedup_gain_percent = (self.speedup_ratio - 1.0) * 100.0
        self.time_saved_percent = (1.0 - self.gpu_benchmark_seconds / self.cpu_reference_seconds) * 100.0
        self.benchmark_valid = True
        self.benchmark_detail = str(detail)
        if (
            self.selection == "auto"
            and self.speedup_ratio < self.AUTO_CPU_FASTER_RATIO
        ):
            self.performance_fallback = True
            reason = (
                f"Auto performance switch: CPU was {1.0 / max(self.speedup_ratio, 1e-9):.2f}x faster "
                f"than {self.backend_name} on the same production bake batch "
                f"(CPU {self.cpu_reference_seconds * 1000.0:.2f} ms vs GPU "
                f"{self.gpu_benchmark_seconds * 1000.0:.2f} ms; 10% decision margin). "
                "The verified current batch is kept; remaining batches will use CPU."
            )
            self._disable_backend(reason, state="cpu_performance_fallback")
            return True
        return False

    def intersect_rays(self, origins, directions, triangles, epsilon, ray_limit):
        if self.backend is not None:
            try:
                batch_counted = False
                gpu_started = time.perf_counter()
                result = self.backend.intersect_rays(
                    origins, directions, triangles, epsilon, ray_limit
                )
                gpu_elapsed = max(time.perf_counter() - gpu_started, 1e-9)
                sample_count = int(np.asarray(triangles).shape[0])
                candidate_count = int(np.asarray(triangles).shape[1])
                intersection_tests = sample_count * 2 * candidate_count
                self.gpu_compute_seconds_total += gpu_elapsed
                self.gpu_intersection_tests_total += intersection_tests
                self.gpu_tests_per_second = (
                    (self.gpu_intersection_tests_total + self.gpu_closest_point_tests_total)
                    / max(self.gpu_compute_seconds_total, 1e-9)
                )

                if not self._benchmark_done:
                    self._benchmark_done = True
                    if self.strict_gpu_only:
                        self.benchmark_detail = "GPU Only selected; CPU comparison is skipped."
                    else:
                        cpu_started = time.perf_counter()
                        cpu_result = intersect_rays_cpu(
                            origins, directions, triangles, epsilon, ray_limit
                        )
                        cpu_elapsed = max(time.perf_counter() - cpu_started, 1e-9)
                        if not self._results_match(result, cpu_result):
                            self._disable_backend(
                                "GPU/CPU ray-intersection parity check failed on the first real bake batch; "
                                "the CPU result was used to protect output accuracy."
                            )
                            return cpu_result
                        if intersection_tests >= self.MIN_AUTO_BENCHMARK_OPERATIONS:
                            self.gpu_batch_count += 1
                            batch_counted = True
                            switched = self._record_speed_benchmark(
                                gpu_elapsed, cpu_elapsed,
                                "Measured on the same first production Ray/Triangle intersection batch; "
                                "includes GPU upload/kernel/download and one CPU reference pass. "
                                "Scope: this compute stage, not total bake time."
                            )
                            if switched:
                                return cpu_result
                        else:
                            self.benchmark_detail = (
                                f"First batch contained only {intersection_tests:,} ray/triangle operations; "
                                f"performance switching requires at least {self.MIN_AUTO_BENCHMARK_OPERATIONS:,}. "
                                "CPU/GPU parity was checked; GPU remains selected because this sample is too small."
                            )
                if not batch_counted:
                    self.gpu_batch_count += 1
                if not self._active_notified:
                    self._active_notified = True
                    self.status = (
                        f"{self.vendor} {self.device_name} · {self.backend_name} ACTIVE "
                        f"({self.precision})"
                    )
                    _safe_progress(self.progress_callback, f"GPU compute active · {self.status}")
                self._publish_status(
                    "active",
                    detail="GPU bake batches are executing.",
                )
                return result
            except Exception as exc:
                self._disable_backend(
                    f"{self.backend_name} ray-intersection compute failed "
                    f"({type(exc).__name__}: {exc})"
                )
        return intersect_rays_cpu(origins, directions, triangles, epsilon, ray_limit)

    @staticmethod
    def _closest_results_match(gpu_result, cpu_result) -> bool:
        if len(gpu_result) != 3 or len(cpu_result) != 3:
            return False
        gpu_distance, gpu_u, gpu_v = (np.asarray(item) for item in gpu_result)
        cpu_distance, cpu_u, cpu_v = (np.asarray(item) for item in cpu_result)
        if gpu_distance.shape != cpu_distance.shape or gpu_u.shape != cpu_u.shape or gpu_v.shape != cpu_v.shape:
            return False
        return bool(
            np.isfinite(gpu_distance).all()
            and np.isfinite(gpu_u).all()
            and np.isfinite(gpu_v).all()
            and np.allclose(gpu_distance, cpu_distance, rtol=5e-4, atol=5e-6)
            and np.allclose(gpu_u, cpu_u, rtol=5e-4, atol=5e-6)
            and np.allclose(gpu_v, cpu_v, rtol=5e-4, atol=5e-6)
        )

    def _disable_backend(self, reason: str, state: str = "cpu_fallback") -> None:
        self.fallback_reason = str(reason)
        self.backend = None
        if self.strict_gpu_only:
            self.status = f"GPU Only failed · {self.fallback_reason}"
            self._publish_status("gpu_required_error", detail=self.fallback_reason)
            raise GPUComputeRequiredError(
                f"GPU Only cannot continue without its selected GPU backend: {self.fallback_reason}"
            )
        self.backend_name = "CPU"
        self.device_name = "CPU"
        self.vendor = "CPU"
        self.memory_bytes = 0
        self.precision = "float64"
        self.status = f"CPU fallback · {self.fallback_reason}"
        _safe_progress(
            self.progress_callback,
            f"GPU compute switched to CPU · {self.fallback_reason}",
        )
        self._publish_status(state, detail=self.fallback_reason)

    def closest_points(self, points, candidate_triangles):
        """Evaluate nearest-point barycentrics on GPU when available, with CPU parity fallback."""
        if self.backend is not None:
            try:
                batch_counted = False
                gpu_started = time.perf_counter()
                result = self.backend.closest_points(points, candidate_triangles)
                gpu_elapsed = max(time.perf_counter() - gpu_started, 1e-9)
                point_count, candidate_count = np.asarray(candidate_triangles).shape[:2]
                test_count = int(point_count * candidate_count)
                self.gpu_compute_seconds_total += gpu_elapsed
                self.gpu_closest_point_tests_total += test_count

                if not self._closest_benchmark_done:
                    self._closest_benchmark_done = True
                    if self.strict_gpu_only:
                        self.benchmark_detail = "GPU Only selected; CPU comparison is skipped."
                    else:
                        cpu_started = time.perf_counter()
                        cpu_result = closest_points_cpu(points, candidate_triangles)
                        cpu_elapsed = max(time.perf_counter() - cpu_started, 1e-9)
                        if not self._closest_results_match(result, cpu_result):
                            self._disable_backend(
                                "GPU/CPU closest-point parity check failed on the first direct-bake batch; "
                                "the CPU result was used to protect texture sampling accuracy."
                            )
                            return cpu_result
                        total_tests = self.gpu_intersection_tests_total + self.gpu_closest_point_tests_total
                        self.gpu_tests_per_second = total_tests / max(self.gpu_compute_seconds_total, 1e-9)
                        if test_count >= self.MIN_AUTO_BENCHMARK_OPERATIONS:
                            self.gpu_batch_count += 1
                            batch_counted = True
                            switched = self._record_speed_benchmark(
                                gpu_elapsed, cpu_elapsed,
                                "Measured on the same first direct-bake nearest-point batch; "
                                "includes GPU upload/kernel/download and one CPU reference pass. "
                                "Scope: nearest-point/triangle candidate stage, not total bake time."
                            )
                            if switched:
                                return cpu_result
                        else:
                            self.benchmark_detail = (
                                f"First batch contained only {test_count:,} nearest-point/triangle operations; "
                                f"performance switching requires at least {self.MIN_AUTO_BENCHMARK_OPERATIONS:,}. "
                                "CPU/GPU parity was checked; GPU remains selected because this sample is too small."
                            )

                if not batch_counted:
                    self.gpu_batch_count += 1
                total_tests = self.gpu_intersection_tests_total + self.gpu_closest_point_tests_total
                self.gpu_tests_per_second = total_tests / max(self.gpu_compute_seconds_total, 1e-9)
                if not self._active_notified:
                    self._active_notified = True
                    self.status = f"{self.vendor} {self.device_name} · {self.backend_name} ACTIVE ({self.precision})"
                    _safe_progress(self.progress_callback, f"GPU compute active · {self.status}")
                self._publish_status("active", detail="GPU direct nearest-surface bake batches are executing.")
                return result
            except Exception as exc:
                self._disable_backend(f"{self.backend_name} closest-point compute failed ({type(exc).__name__}: {exc})")
        return closest_points_cpu(points, candidate_triangles)

    def filter_mesh_step(self, *args):
        """Execute the Li--Nan Stage-1 math on a capable GPU backend.

        Returns None when the selected adapter does not implement this workload;
        callers then keep the exact NumPy reference path. The bake backend remains
        active even when a reconstruction-specific operation is unsupported.
        """
        if self.backend is None or self._filter_step_disabled:
            return None
        operation = getattr(self.backend, "filter_mesh_step", None)
        if not callable(operation):
            self.filter_fallback_reason = (
                f"{self.backend_name} currently accelerates bake intersection kernels but "
                "does not expose the Li-Nan mesh-filter kernel; Stage 1 is using the CPU reference."
            )
            return None
        started = time.perf_counter()
        try:
            result = operation(*args)
            self.filter_gpu_seconds_total += max(time.perf_counter() - started, 0.0)
            self.filter_gpu_batches += 1
            self.filter_backend_name = self.backend_name
            return result
        except Exception as exc:
            self._filter_step_disabled = True
            self.filter_fallback_reason = (
                f"{self.backend_name} Li-Nan filter kernel failed "
                f"({type(exc).__name__}: {exc}); remaining Stage-1 iterations use CPU."
            )
            return None

    def disable_filter_kernel(self, reason: str) -> None:
        self._filter_step_disabled = True
        self.filter_fallback_reason = str(reason)

    def record_filter_cpu_reference(self, elapsed: float) -> None:
        self.filter_cpu_reference_seconds += max(0.0, float(elapsed))

    def qem_candidate_batch(self, vertices, quadrics, edges):
        """Evaluate QEM candidate placements on a bit-exact float64 GPU path.

        The first live batch is checked against the canonical CPU arithmetic.
        A backend that changes any cost or position is disabled for QEM only;
        other GPU workloads remain enabled.
        """
        if self.backend is None or self._qem_candidate_disabled:
            return None
        if str(getattr(self.backend, "precision", self.precision)).lower() != "float64":
            self.qem_candidate_fallback_reason = (
                f"{self.backend_name} QEM requires float64; CPU multi-core evaluation is active."
            )
            return None
        operation = getattr(self.backend, "qem_candidate_batch", None)
        if not callable(operation):
            self.qem_candidate_fallback_reason = (
                f"{self.backend_name} does not expose the QEM candidate kernel; CPU multi-core evaluation is active."
            )
            return None
        started = time.perf_counter()
        try:
            costs, positions = operation(vertices, quadrics, edges)
            costs = np.asarray(costs, dtype=np.float64).reshape(-1)
            positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
            if len(costs) != len(edges) or len(positions) != len(edges):
                raise RuntimeError("QEM candidate kernel returned mismatched batch dimensions.")
            if not self._qem_candidate_parity_verified and len(edges):
                try:
                    from .feature_decimation import _qem_candidate_costs_vectorized
                except ImportError:
                    from feature_decimation import _qem_candidate_costs_vectorized
                sample_count = min(32, len(edges))
                sample_indices = np.linspace(
                    0, len(edges) - 1, num=sample_count, dtype=np.int64
                )
                reference_costs, reference_positions = _qem_candidate_costs_vectorized(
                    np.asarray(vertices, dtype=np.float64),
                    np.asarray(quadrics, dtype=np.float64),
                    np.asarray(edges, dtype=np.int64)[sample_indices],
                )
                if not (
                    np.array_equal(costs[sample_indices], reference_costs)
                    and np.array_equal(positions[sample_indices], reference_positions)
                ):
                    raise RuntimeError(
                        "first-batch bit-exact parity check failed against the CPU QEM reference"
                    )
                self._qem_candidate_parity_verified = True
            self.qem_candidate_seconds_total += max(time.perf_counter() - started, 0.0)
            self.qem_candidate_batches += 1
            return costs, positions
        except Exception as exc:
            self._qem_candidate_disabled = True
            self.qem_candidate_fallback_reason = (
                f"{self.backend_name} QEM candidate kernel disabled "
                f"({type(exc).__name__}: {exc}); CPU multi-core evaluation is active."
            )
            return None

    def info(self) -> dict:
        return {
            "compute_selection": self.selection,
            "compute_strict_gpu_only": bool(self.strict_gpu_only),
            "performance_fallback": bool(self.performance_fallback),
            "compute_backend": self.backend_name,
            "compute_device": self.device_name,
            "compute_vendor": self.vendor,
            "compute_memory_bytes": int(self.memory_bytes),
            "compute_precision": self.precision,
            "compute_status": self.status,
            "compute_activation_verified": bool(self.activation_verified),
            "gpu_active": bool(
                (
                    self.gpu_batch_count > 0
                    or self.filter_gpu_batches > 0
                    or int(getattr(self, "qem_candidate_batches", 0)) > 0
                )
                and self.backend is not None
            ),
            "reconstruction_filter_active": bool(
                self.filter_gpu_batches > 0 and self.backend is not None
            ),
            "reconstruction_qem_candidate_active": bool(
                int(getattr(self, "qem_candidate_batches", 0)) > 0 and self.backend is not None
            ),
            "gpu_fallback_reason": self.fallback_reason,
            "gpu_batch_count": int(self.gpu_batch_count),
            "cpu_reference_seconds": float(self.cpu_reference_seconds),
            "gpu_benchmark_seconds": float(self.gpu_benchmark_seconds),
            "speedup_ratio": float(self.speedup_ratio),
            "speedup_gain_percent": float(self.speedup_gain_percent),
            "time_saved_percent": float(self.time_saved_percent),
            "gpu_compute_seconds_total": float(self.gpu_compute_seconds_total),
            "gpu_intersection_tests_total": int(self.gpu_intersection_tests_total),
            "gpu_closest_point_tests_total": int(self.gpu_closest_point_tests_total),
            "gpu_workload": (
                "cage-ray/triangle" if self.gpu_intersection_tests_total and not self.gpu_closest_point_tests_total
                else "direct-nearest-triangle" if self.gpu_closest_point_tests_total and not self.gpu_intersection_tests_total
                else "mixed-ray-and-nearest-triangle" if self.gpu_intersection_tests_total or self.gpu_closest_point_tests_total
                else "not-started"
            ),
            "gpu_tests_per_second": float(self.gpu_tests_per_second),
            "reconstruction_filter_backend": str(self.filter_backend_name),
            "reconstruction_filter_gpu_batches": int(self.filter_gpu_batches),
            "reconstruction_filter_gpu_seconds": float(self.filter_gpu_seconds_total),
            "reconstruction_filter_cpu_reference_seconds": float(self.filter_cpu_reference_seconds),
            "reconstruction_filter_fallback_reason": str(self.filter_fallback_reason),
            "qem_candidate_gpu_batches": int(getattr(self, "qem_candidate_batches", 0)),
            "qem_candidate_gpu_seconds": float(getattr(self, "qem_candidate_seconds_total", 0.0)),
            "qem_candidate_gpu_fallback_reason": str(getattr(self, "qem_candidate_fallback_reason", "")),
            "benchmark_valid": bool(self.benchmark_valid),
            "benchmark_detail": str(self.benchmark_detail),
        }

def probe_gpu_runtime(progress_callback: ProgressCallback = None) -> dict:
    """Enumerate GPU adapters and verify a real compute kernel with optional status updates.

    The same probe is used by the installed application's asynchronous GPU
    rescan and by the portable launcher's branded startup splash. It reports
    hardware visibility separately from compute-runtime activation so a display
    adapter is never mistaken for a usable compute device.
    """
    manager = GPUComputeManager(selection="auto", progress_callback=progress_callback)
    active_backend = str(manager.backend_name or "CPU").lower()
    active_device = str(manager.device_name or "CPU")
    devices = []
    for device in manager.devices:
        is_active = bool(
            manager.activation_verified
            and device.backend.lower() == active_backend
            and device.name == active_device
        )
        devices.append({
            "backend": device.backend.upper(),
            "index": int(device.index),
            "name": device.name,
            "vendor": device.vendor,
            "memory_bytes": int(device.memory_bytes),
            "integrated": bool(device.integrated),
            "active": is_active,
        })
    devices.sort(key=lambda item: (not item["active"], item["backend"], -item["memory_bytes"], item["index"]))
    errors = list(manager.discovery_errors)
    if not manager.activation_verified and manager.fallback_reason and not errors:
        errors.append(manager.fallback_reason)
    return {
        "devices": devices,
        "errors": errors,
        "compute_ready": bool(manager.activation_verified),
        "activation_verified": bool(manager.activation_verified),
        "active_backend": manager.backend_name,
        "active_device": manager.device_name,
        "active_vendor": manager.vendor,
        "active_memory_bytes": int(manager.memory_bytes),
        "fallback_reason": str(manager.fallback_reason),
    }
