from __future__ import annotations

import argparse
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = (ROOT / "app").resolve()

from gpu_runtime_paths import activate_optional_gpu_packages

activate_optional_gpu_packages()

def verify_numpy() -> None:
    import numpy as np

    probe = np.asarray([1, 2, 3], dtype=np.int64)
    if probe.shape != (3,) or int(probe.sum()) != 6:
        raise RuntimeError("NumPy array operations returned an unexpected result")
    print(f"NumPy: {np.__version__} | integrity: OK")

def verify_polyfit() -> None:
    from polyfit_runtime import load_polyfit_binding

    try:
        polyfit = load_polyfit_binding(ROOT, allow_repair=False)
        required = (
            "initialize",
            "read_point_set",
            "reconstruct",
            "HypothesisGenerator",
            "FaceSelection",
            "save_mesh",
            "SCIP",
        )
        missing = [name for name in required if not hasattr(polyfit, name)]
        required_methods = {
            "HypothesisGenerator": ("refine_planes", "generate", "compute_confidences"),
            "FaceSelection": ("optimize",),
        }
        missing_methods = [
            f"{class_name}.{method}"
            for class_name, methods in required_methods.items()
            for method in methods
            if not hasattr(getattr(polyfit, class_name, None), method)
        ]
        if missing or missing_methods:
            raise RuntimeError("missing API: " + ", ".join(missing + missing_methods))
        polyfit.initialize()
    except Exception as exc:
        raise RuntimeError(
            "PolyFit 1.6.0 native binding verification failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("PolyFit 1.6.0 import, native extension and Example_2 API: OK")
    print("Binding:", getattr(polyfit, "__file__", getattr(polyfit, "__name__", "native")))

def verify_runtime() -> None:
    runtime_dir = Path(sys.executable).resolve().parent
    search_paths = set()
    for item in sys.path:
        candidate = Path(item or os.getcwd())
        if candidate.is_absolute():
            search_paths.add(candidate.resolve())
        else:
            search_paths.add((runtime_dir / candidate).resolve())
            search_paths.add((Path.cwd() / candidate).resolve())

    if APP_DIR not in search_paths:
        raise RuntimeError(
            "The app directory is missing from Python's module search path. "
            r"Expected python312._pth to include '..\..\app'."
        )

    import pip
    import numpy
    import PySide6
    import vtk
    import scipy
    import networkx
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import trimesh
    import xatlas
    import PIL
    import lxml
    import psutil
    import main
    from runtime_contract import verify_runtime_manifest
    from hidden_entry import main as shared_runtime_entry
    from startup_gpu_preflight import _run_startup_gpu_preflight

    manifest = verify_runtime_manifest(require_source_match=True)
    print(
        "Runtime identity: Polyfitty Studio {0} | source {1} | {2} files".format(
            manifest["app_version"], manifest["source_fingerprint"], manifest["source_file_count"]
        )
    )
    verify_polyfit()
    if not callable(shared_runtime_entry) or not callable(_run_startup_gpu_preflight):
        raise RuntimeError("The shared installed/portable startup entry is incomplete.")
    print("Shared installed/portable application entry and GPU startup preflight: OK")
    print("Python:", sys.version.split()[0])
    print("App module path: OK")
    print("pip:", pip.__version__)
    print("NumPy:", numpy.__version__)
    print("PySide6:", PySide6.__version__)
    print("VTK:", vtk.vtkVersion.GetVTKVersion())
    print("SciPy:", scipy.__version__)
    print("NetworkX:", networkx.__version__)
    print("QVTKRenderWindowInteractor: import OK")
    print("Trimesh:", trimesh.__version__)
    print("xatlas: connected UV unwrapping import OK")
    print("Pillow:", PIL.__version__)
    print("lxml:", lxml.__version__)
    print("psutil:", psutil.__version__)
    print("Polyfitty Studio GUI module: import OK")

def _find_cuda_site_roots() -> list[Path]:
    roots = {
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.executable).resolve().parent / "Lib" / "site-packages",
    }
    roots.update(Path(entry) for entry in sys.path if entry)
    return [item for item in roots if item.is_dir()]

def _preflight_cublas() -> bool:
    paths = []
    for site_root in _find_cuda_site_roots():
        paths.extend((site_root / "nvidia" / "cublas" / "bin").glob("cublas64_12.dll"))
        paths.extend((site_root / "nvidia" / "cublas" / "bin").glob("cublasLt64_12.dll"))
    cublas_path = next((item for item in paths if item.name.lower() == "cublas64_12.dll"), None)
    cublas_lt_path = next((item for item in paths if item.name.lower() == "cublaslt64_12.dll"), None)
    if cublas_path is None or cublas_lt_path is None:
        print("[FAIL] CUDA cuBLAS DLLs are missing from the private Python runtime.")
        print("[FAIL] Required package: nvidia-cublas-cu12==12.6.4.1")
        return False
    print(f"[OK] cuBLAS DLL: {cublas_path}")
    print(f"[OK] cuBLAS Lt DLL: {cublas_lt_path}")
    if os.name == "nt":
        for dll_path in (cublas_lt_path, cublas_path):
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError as exc:
                print(f"[FAIL] Windows loader could not load {dll_path.name}: {type(exc).__name__}: {exc}")
                return False
        print("[OK] Windows loader opened cuBLAS and cuBLAS Lt successfully.")
    return True

def _run_nvidia_smi() -> None:
    candidates = []
    located = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if located:
        candidates.append(located)
    for variable in ("ProgramFiles", "ProgramW6432"):
        base = os.environ.get(variable, "")
        if base:
            candidates.append(str(Path(base) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"))
    executable = next((item for item in candidates if Path(item).is_file()), None)
    if not executable:
        print("[WARN] nvidia-smi.exe was not found.")
        return

    print(f"[INFO] nvidia-smi: {executable}")
    try:
        result = subprocess.run(
            [executable, "-q"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = result.stdout or result.stderr or "<no output>"
        print(output[:16000])
        match = re.search(r"Driver Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", output, flags=re.IGNORECASE)
        if match:
            driver = tuple(int(part) for part in match.group(1).split("."))
            print(f"[INFO] NVIDIA driver: {match.group(1)}")
            if driver < (528, 33):
                print("[FAIL] Driver is below the CUDA 12.x Windows compatibility floor (528.33).")
            elif driver < (560, 94):
                print("[WARN] Driver is below the CUDA 12.6 Update 2 driver level (560.94).")
            else:
                print("[OK] Driver meets the CUDA 12.6 Update 2 Windows driver level.")
    except Exception as exc:
        print(f"[WARN] nvidia-smi query failed: {type(exc).__name__}: {exc}")

def verify_cuda() -> int:
    print("=== Polyfitty Studio NVIDIA CUDA Self-Test ===")
    print("Python:", sys.executable)
    print("Platform:", platform.platform())
    print("CUDA_PATH:", os.environ.get("CUDA_PATH", "<not set>"))

    from gpu_compute import _prepare_cuda_dll_search_paths

    _run_nvidia_smi()
    registered_dirs = _prepare_cuda_dll_search_paths()
    print(f"[INFO] Registered CUDA DLL directories: {len(registered_dirs)}")
    for directory in registered_dirs:
        print(f"[INFO] DLL directory: {directory}")
    if os.name == "nt" and not _preflight_cublas():
        return 2

    try:
        import cupy as cp
    except Exception as exc:
        print(f"[FAIL] CuPy import failed: {type(exc).__name__}: {exc}")
        return 2

    print("[OK] CuPy version:", getattr(cp, "__version__", "unknown"))
    try:
        cp.show_config()
    except Exception as exc:
        print(f"[WARN] cupy.show_config failed: {type(exc).__name__}: {exc}")

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        print("[OK] CUDA device count:", count)
    except Exception as exc:
        print(f"[FAIL] CUDA device enumeration failed: {type(exc).__name__}: {exc}")
        return 3
    if count <= 0:
        print("[FAIL] CuPy loaded, but CUDA returned zero devices.")
        return 4

    from gpu_compute import _CudaBackend, GPUDevice, _decode_name, _verify_backend_compute

    passed = 0
    failures = 0
    for index in range(count):
        try:
            properties = cp.cuda.runtime.getDeviceProperties(index)
            name = _decode_name(properties.get("name"), f"NVIDIA GPU {index}")
            memory = int(properties.get("totalGlobalMem", 0) or 0)
            capability = f"{properties.get('major', '?')}.{properties.get('minor', '?')}"
            print(f"\n--- CUDA device {index} ---")
            print("Name:", name)
            print("Compute capability:", capability)
            print(f"Global memory: {memory / (1024 ** 3):.2f} GiB")
            with cp.cuda.Device(index):
                values = cp.arange(64, dtype=cp.float64)
                total = float(cp.asnumpy(cp.sum(values * values)))
            expected = float(sum(value * value for value in range(64)))
            if abs(total - expected) > 1e-8:
                raise RuntimeError(f"CuPy arithmetic test mismatch: {total} != {expected}")
            device = GPUDevice(
                backend="cuda", index=index, name=name, vendor="NVIDIA",
                memory_bytes=memory, native=cp,
            )
            _verify_backend_compute(_CudaBackend(device))
            passed += 1
            print("[OK] Polyfitty Studio ray/triangle kernel self-test passed.")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] Device {index} activation: {type(exc).__name__}: {exc}")

    if passed:
        print(f"[SUCCESS] {passed}/{count} CUDA device(s) passed the application kernel self-test.")
        if failures:
            print(f"[WARN] {failures} other CUDA device(s) failed and will be skipped by the application.")
        return 0
    print(f"[FAIL] No CUDA device passed the application self-test; failed devices: {failures}.")
    return 5

def verify_opencl() -> int:
    print("=== Polyfitty Studio OpenCL Self-Test ===")
    print("Python:", sys.executable)
    try:
        import pyopencl as cl
    except Exception as exc:
        print(f"[FAIL] PyOpenCL import failed: {type(exc).__name__}: {exc}")
        return 2

    from gpu_compute import GPUDevice, _OpenCLBackend, _verify_backend_compute

    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        print(f"[FAIL] OpenCL platform enumeration failed: {type(exc).__name__}: {exc}")
        return 3

    tested = 0
    passed = 0
    failures = []
    for platform_index, platform_item in enumerate(platforms):
        try:
            devices = platform_item.get_devices(device_type=cl.device_type.GPU)
        except Exception as exc:
            failures.append(f"Platform {platform_index}: {type(exc).__name__}: {exc}")
            continue
        for device in devices:
            if not bool(getattr(device, "available", True)):
                continue
            if not bool(getattr(device, "compiler_available", True)):
                failures.append(f"{getattr(device, 'name', 'Unknown GPU')}: OpenCL compiler unavailable")
                continue
            index = tested
            tested += 1
            name = str(getattr(device, "name", f"OpenCL GPU {index}")).strip()
            vendor = str(getattr(device, "vendor", "Unknown vendor")).strip()
            memory = int(getattr(device, "global_mem_size", 0) or 0)
            print(f"\n--- OpenCL device {index} ---")
            print("Platform:", getattr(platform_item, "name", "Unknown platform"))
            print("Vendor:", vendor)
            print("Name:", name)
            print(f"Global memory: {memory / (1024 ** 3):.2f} GiB")
            try:
                gpu_device = GPUDevice(
                    backend="opencl", index=index, name=name, vendor=vendor,
                    memory_bytes=memory, native=(cl, device),
                )
                backend = _OpenCLBackend(gpu_device)
                _verify_backend_compute(backend)
                passed += 1
                print(f"[OK] Polyfitty Studio ray/triangle kernel self-test passed ({backend.precision}).")
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[FAIL] OpenCL device activation failed: {message}")

    if passed:
        print(f"[SUCCESS] {passed}/{tested} OpenCL GPU device(s) passed the application kernel self-test.")
        return 0
    if not tested:
        print("[FAIL] No available OpenCL GPU device was exposed by the installed ICD.")
    for message in failures:
        print(" -", message)
    print("[FAIL] No OpenCL device passed the application kernel self-test.")
    return 4

def verify_sycl() -> int:
    print("=== Polyfitty Studio Intel oneAPI / SYCL Self-Test ===")
    print("Python:", sys.executable)
    try:
        import dpctl
    except Exception as exc:
        print(f"[FAIL] dpctl import failed: {type(exc).__name__}: {exc}")
        return 2

    dpctl_version = str(getattr(dpctl, "__version__", "unknown"))

    if tuple(dpctl_version.split(".")[:2]) != ("0", "22"):
        print(
            "[FAIL] Unsupported dpctl version:", dpctl_version,
            "| this build requires dpctl 0.22.x with dpnp 0.20.0",
        )
        return 2

    try:
        import dpnp
    except Exception as exc:
        print(f"[FAIL] dpnp import failed with dpctl {dpctl_version}: {type(exc).__name__}: {exc}")
        return 2

    dpnp_version = str(getattr(dpnp, "__version__", "unknown"))
    if dpnp_version != "0.20.0":
        print("[FAIL] Unsupported dpnp version:", dpnp_version, "| this build requires dpnp 0.20.0")
        return 2
    print("dpctl:", dpctl_version)
    print("dpnp:", dpnp_version)
    try:
        devices = list(dpctl.get_devices(device_type="gpu"))
    except Exception as exc:
        print(f"[FAIL] SYCL GPU enumeration failed: {type(exc).__name__}: {exc}")
        return 3
    if not devices:
        print("[FAIL] SYCL runtime loaded, but no GPU device was exposed by the installed Intel driver/plugin.")
        return 4

    from gpu_compute import GPUDevice, _SyclBackend, _verify_backend_compute

    passed = 0
    failures = []
    for index, native in enumerate(devices):
        name = str(getattr(native, "name", f"SYCL GPU {index}"))
        vendor = str(getattr(native, "vendor", "Intel"))
        memory = int(getattr(native, "global_mem_size", 0) or 0)
        print(f"\n--- SYCL device {index} ---")
        print("Vendor:", vendor)
        print("Name:", name)
        print(f"Global memory: {memory / (1024 ** 3):.2f} GiB")
        try:
            device = GPUDevice(
                backend="sycl", index=index, name=name, vendor=vendor,
                memory_bytes=memory, native=native,
                integrated=bool(getattr(native, "is_integrated", False)),
            )
            backend = _SyclBackend(device)
            _verify_backend_compute(backend)
            passed += 1
            print(f"[OK] Polyfitty Studio SYCL ray/triangle and closest-point self-test passed ({backend.precision}).")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] SYCL device activation failed: {failures[-1]}")

    if passed:
        print(f"[SUCCESS] {passed}/{len(devices)} SYCL GPU device(s) passed the application kernel self-test.")
        return 0
    print("[FAIL] No SYCL GPU passed the Polyfitty Studio application self-test.")
    for message in failures:
        print(" -", message)
    return 5

def verify_hip() -> int:
    print("=== Polyfitty Studio AMD HIP / ROCm Self-Test ===")
    print("Python:", sys.executable)
    from gpu_compute import GPUDevice, _HipBackend, _WindowsHipBackend, _discover_hip, _verify_backend_compute

    errors: list[str] = []
    devices = _discover_hip(errors)
    if errors:
        for message in errors:
            print("[INFO]", message)
    if not devices:
        print("[FAIL] No AMD HIP device was exposed by the installed driver/runtime or bundled bridge.")
        return 2

    passed = 0
    failures = []
    for device in devices:
        print(f"\n--- HIP device {device.index} ---")
        print("Vendor:", device.vendor)
        print("Name:", device.name)
        print(f"Global memory: {device.memory_bytes / (1024 ** 3):.2f} GiB")
        try:
            if sys.platform == "win32" and isinstance(device.native, tuple) and len(device.native) == 3:
                backend = _WindowsHipBackend(device)
            else:
                backend = _HipBackend(device)
            _verify_backend_compute(backend)
            passed += 1
            print("[OK] AMD HIP ray/triangle and closest-point kernel self-test passed.")
        except Exception as exc:
            failures.append(f"{device.name}: {type(exc).__name__}: {exc}")
            print("[FAIL] HIP device activation failed:", failures[-1])

    if passed:
        print(f"[SUCCESS] {passed}/{len(devices)} AMD HIP device(s) passed the application kernel self-test.")
        return 0
    print("[FAIL] No AMD HIP device passed the Polyfitty Studio application self-test.")
    for message in failures:
        print(" -", message)
    return 3

def main_entry() -> int:
    parser = argparse.ArgumentParser(description="Verify Polyfitty Studio installation components.")
    parser.add_argument(
        "--phase", required=True,
        choices=("polyfit", "numpy", "runtime", "cuda", "hip", "opencl", "sycl"),
    )
    phase = parser.parse_args().phase

    if phase == "polyfit":
        verify_polyfit()
    elif phase == "numpy":
        verify_numpy()
    elif phase == "runtime":
        verify_runtime()
    elif phase == "cuda":
        return verify_cuda()
    elif phase == "hip":
        return verify_hip()
    elif phase == "opencl":
        return verify_opencl()
    else:
        return verify_sycl()
    return 0

if __name__ == "__main__":
    raise SystemExit(main_entry())
