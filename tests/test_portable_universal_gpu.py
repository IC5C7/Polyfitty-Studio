from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"

def test_portable_builder_defers_recipient_gpu_setup_until_first_launch():
        gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
        guide = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "$MainRuntime" not in builder
        assert "official CPython 3.12.10 embeddable" in guide
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        assert "first launch on the recipient PC" in gui
        assert "embedded_runtime_sha256" in builder
        assert "On first launch on a recipient PC" in guide

def test_installed_and_portable_products_use_the_same_gpu_installer_with_different_triggers():
        installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
        assert "requirements_gpu_cuda12.txt" in installer
        assert "requirements_gpu_opencl.txt" in installer
        assert "requirements_gpu_sycl.txt" in installer
        assert "pip install" in installer.lower()
        assert "install_gpu.ps1" not in builder
        assert "if ($IsPortablePackage)" in startup
        assert "installer\\install_gpu.ps1" in startup
        assert "Installed product detected: GPU packages are provisioned by the installer setup-manager stage" in startup

def test_shared_runtime_sets_cuda_path_to_bundled_runtime_before_cupy_import():
    source = (APP / "gpu_compute.py").read_text(encoding="utf-8")
    start = source.index("def _prepare_cuda_dll_search_paths(")
    end = source.index("\n\n@dataclass", start)
    block = source[start:end]
    assert 'runtime_root / "bin"' in block
    assert 'runtime_root / "include"' in block
    assert 'os.environ["CUDA_PATH"] = cuda_path' in block
    discover_start = source.index("def _discover_cuda(")
    discover_end = source.index("\ndef _discover_opencl(", discover_start)
    discover = source[discover_start:discover_end]
    assert discover.index("_prepare_cuda_dll_search_paths()") < discover.index("import cupy as cp")

def test_windows_without_nvidia_driver_skips_cuda_and_allows_opencl_fallback(monkeypatch, tmp_path):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    monkeypatch.setattr(gpu_compute.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setattr(gpu_compute, "_prepare_cuda_dll_search_paths", lambda: (_ for _ in ()).throw(AssertionError("CuPy path setup should be skipped")))
    errors = []
    assert gpu_compute._discover_cuda(errors) == []
    assert any("nvcuda.dll" in message and "OpenCL discovery will continue" in message for message in errors)

def test_windows_cuda_driver_library_is_detected_from_system32(monkeypatch, tmp_path):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    system32 = tmp_path / "System32"
    system32.mkdir()
    driver = system32 / "nvcuda.dll"
    driver.write_bytes(b"test")
    monkeypatch.setattr(gpu_compute.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert gpu_compute._find_windows_cuda_driver_library() == driver

def test_amd_opencl_ready_status_explains_cuda_skip_as_expected():
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "No NVIDIA CUDA driver was found; this is expected on AMD/Intel-only systems." in main
    assert "The selected {backend} GPU backend is active." in main

def test_single_file_exe_audits_its_embedded_payload_and_runtime_tree():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
        assert "$PortableRuntime = Join-Path $Dist '.runtime\\python312'" in builder
        assert "Get-DirectoryFingerprint $PortableRuntime" in builder
        assert "Get-DirectoryFingerprint (Join-Path $zipVerifyImage '.runtime\\python312')" in builder
        assert "$runtimeFingerprint.Fingerprint -ne $zipRuntimeFingerprint.Fingerprint" in builder
        assert "embedded_runtime_sha256" in builder
        assert "PolyfittyStudio.Payload.zip" in builder
        assert "PolyfittyStudio.Payload.sha256" in builder
        assert "GetManifestResourceStream('PolyfittyStudio.Payload.zip')" in builder
        assert "GetManifestResourceStream('PolyfittyStudio.Payload.sha256')" in builder
        assert "ExtractToDirectory($PayloadVerifyZip" in builder
        assert "The portable output must contain exactly one distributable file" in builder
        assert "ExtractZipSafely" in launcher

def test_universal_gpu_discovery_prefers_cuda_then_discrete_opencl(monkeypatch):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    cuda = gpu_compute.GPUDevice(
        backend="cuda", index=0, name="RTX", vendor="NVIDIA", memory_bytes=4 * 1024**3
    )
    integrated = gpu_compute.GPUDevice(
        backend="opencl", index=0, name="Intel iGPU", vendor="Intel",
        memory_bytes=16 * 1024**3, integrated=True,
    )
    discrete = gpu_compute.GPUDevice(
        backend="opencl", index=1, name="Radeon dGPU", vendor="AMD",
        memory_bytes=8 * 1024**3, integrated=False,
    )
    monkeypatch.setattr(gpu_compute, "_discover_cuda", lambda errors: [cuda])
    monkeypatch.setattr(gpu_compute, "_discover_hip", lambda errors: [])
    monkeypatch.setattr(gpu_compute, "_discover_sycl", lambda errors: [])
    monkeypatch.setattr(gpu_compute, "_discover_opencl", lambda errors: [integrated, discrete])

    devices, errors = gpu_compute.discover_gpu_devices()
    assert errors == []
    assert [device.name for device in devices] == ["RTX", "Radeon dGPU", "Intel iGPU"]

def test_embedded_portable_gpu_setup_supports_intel_sycl_and_opencl():
        installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
        assert "requirements_gpu_sycl.txt" in installer
        assert "Install-GpuRequirements -RequirementsFile 'requirements_gpu_sycl.txt'" in installer
        assert "requirements_gpu_opencl.txt" in installer
        assert "Install-GpuRequirements -RequirementsFile 'requirements_gpu_opencl.txt'" in installer
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "Starting portable first-run GPU detection" in startup
        assert "gpu_setup = 'first_launch_on_recipient_pc'" in builder

def test_sycl_runtime_rejects_incompatible_dpctl_cython_api_before_dpnp_import():
    verifier = (INSTALLER / "verify_runtime.py").read_text(encoding="utf-8")
    assert 'tuple(dpctl_version.split(".")[:2]) != ("0", "22")' in verifier
    assert "dpctl 0.22.x with dpnp 0.20.0" in verifier
    assert 'dpctl.get_devices(device_type="gpu")' in verifier
    assert "_SyclBackend(device)" in verifier and "_verify_backend_compute(backend)" in verifier
    assert "dpctl==0.22.1" in (ROOT / "requirements_gpu_sycl.txt").read_text(encoding="utf-8")

def test_hip_kernel_source_translates_opencl_template_to_hiprtc_source():
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    source = gpu_compute._hip_kernel_source()
    assert 'extern "C" __global__ void intersect_rays' in source
    assert 'extern "C" __global__ void closest_points' in source
    assert "get_global_id(0)" not in source
    assert "__kernel" not in source
    assert "pfs_clamp(" in source

def test_sycl_backend_vectorized_operations_match_cpu_reference(monkeypatch):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute
    import numpy as np

    class FakeDevice:
        name = "Intel Arc (fake)"
        vendor = "Intel"
        global_mem_size = 4 * 1024**3
        is_integrated = False
        has_aspect_fp64 = True

    class FakeDpctl:
        __version__ = "test"
        @staticmethod
        def SyclQueue(device):
            return device

    class FakeDpnp:
        float64 = np.float64
        float32 = np.float32
        @staticmethod
        def asarray(value, dtype=None, device=None):
            return np.asarray(value, dtype=dtype)
        @staticmethod
        def asnumpy(value):
            return np.asarray(value)
        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setitem(sys.modules, "dpctl", FakeDpctl)
    monkeypatch.setitem(sys.modules, "dpnp", FakeDpnp())
    backend = gpu_compute._SyclBackend(gpu_compute.GPUDevice(
        backend="sycl", index=0, name=FakeDevice.name, vendor="Intel",
        memory_bytes=FakeDevice.global_mem_size, native=FakeDevice(),
    ))
    origins = np.asarray([[[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]]])
    directions = np.asarray([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
    triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = triangle.reshape(1, 1, 3, 3)
    expected = gpu_compute.intersect_rays_cpu(origins, directions, triangles, 1e-9, 2.0)
    actual = backend.intersect_rays(origins, directions, triangles, 1e-9, 2.0)
    for got, wanted in zip(actual, expected):
        assert np.allclose(got, wanted, equal_nan=True)
    cp_expected = gpu_compute.closest_points_cpu(np.asarray([[0.25, 0.25, 1.0]]), triangles)
    cp_actual = backend.closest_points(np.asarray([[0.25, 0.25, 1.0]]), triangles)
    for got, wanted in zip(cp_actual, cp_expected):
        assert np.allclose(got, wanted, equal_nan=True)

def test_explicit_sycl_selection_does_not_activate_cuda_first(monkeypatch):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    cuda = gpu_compute.GPUDevice("cuda", 0, "RTX", "NVIDIA", 8 * 1024**3)
    sycl = gpu_compute.GPUDevice("sycl", 0, "Arc", "Intel", 8 * 1024**3, native=object())
    monkeypatch.setattr(gpu_compute, "discover_gpu_devices", lambda progress_callback=None: ([cuda, sycl], []))
    class FakeSyclBackend:
        precision = "float32"
        device_name = "Arc"
        def __init__(self, device):
            self.device = device
    monkeypatch.setattr(gpu_compute, "_SyclBackend", FakeSyclBackend)
    monkeypatch.setattr(gpu_compute, "_verify_backend_compute", lambda backend: None)
    manager = gpu_compute.GPUComputeManager("sycl")
    assert manager.backend_name == "SYCL"
    assert manager.device_name == "Arc"
    assert [item.backend for item in manager.devices] == ["sycl"]

def test_shared_runtime_reports_missing_sycl_runtime(monkeypatch):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import builtins
    import gpu_compute

    original_import = builtins.__import__

    def missing_sycl_import(name, *args, **kwargs):
        if name in {"dpctl", "dpnp"}:
            raise ModuleNotFoundError("simulated missing Intel SYCL runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sycl_import)
    errors = []
    assert gpu_compute._discover_sycl(errors) == []
    assert any("Intel oneAPI/SYCL runtime unavailable" in message for message in errors)

def test_intel_gpu_status_explains_missing_sycl_bundle_and_driver_plugin():
        main = (APP / "main.py").read_text(encoding="utf-8")
        assert "INTEL SYCL RUNTIME MISSING" in main
        assert "embedded/installed Python environment could not load oneAPI/SYCL" in main
        assert "use the main Studio setup manager for an installed copy, or relaunch the portable copy" in main
        assert "Level Zero or OpenCL plugin" in main
        assert "INTEL SYCL DEVICE NOT EXPOSED" in main

def test_hip_kernel_source_is_self_contained_for_hiprtc():
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    source = gpu_compute._hip_kernel_source()
    assert "size_t" not in source
    assert "INFINITY" not in source
    assert "__builtin_huge_val()" in source
    assert "#include" not in source
    assert "--gpu-architecture=" in (ROOT / "native" / "hip_bridge" / "PolyfittyHipBridge.cpp").read_text(encoding="utf-8")

def test_windows_hip_discovery_uses_native_bridge_before_hip_python(monkeypatch, tmp_path):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import ctypes
    import gpu_compute

    class FakeBridge:
        def pfhip_device_count(self, count_ptr, error, capacity):
            count_ptr._obj.value = 1
            return 0
        def pfhip_device_info(self, index, name, name_capacity, arch, arch_capacity, memory_ptr, integrated_ptr, error, capacity):
            name.value = b"Radeon RX Test"
            arch.value = b"gfx1201"
            memory_ptr._obj.value = 12 * 1024**3
            integrated_ptr._obj.value = 0
            return 0

    bridge_path = tmp_path / "PolyfittyHipBridge.dll"
    bridge_path.write_bytes(b"test")
    fake_bridge = FakeBridge()
    monkeypatch.setattr(gpu_compute.sys, "platform", "win32")
    monkeypatch.setattr(gpu_compute, "_load_windows_hip_bridge", lambda: (fake_bridge, bridge_path))
    errors = []
    devices = gpu_compute._discover_hip(errors)
    assert errors == []
    assert len(devices) == 1
    assert devices[0].backend == "hip"
    assert devices[0].name == "Radeon RX Test"
    assert devices[0].native[1] == "gfx1201"

def test_windows_hip_backend_bridge_matches_cpu_reference():
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import ctypes
    import gpu_compute
    import numpy as np

    class FakeBridge:
        def pfhip_create_backend(self, index, source, handle_ptr, error, capacity):
            self.base_source = source.decode("utf-8")
            handle_ptr._obj.value = 12345
            return 0
        def pfhip_destroy_backend(self, handle):
            return None
        def pfhip_intersect_rays(self, handle, sample_count, candidate_count, origins, directions, triangles,
                                 epsilon, ray_limit, out_t, out_u, out_v, error, capacity):
            origin_array = np.ctypeslib.as_array(origins, shape=(sample_count * 6,)).reshape(sample_count, 2, 3)
            direction_array = np.ctypeslib.as_array(directions, shape=(sample_count * 6,)).reshape(sample_count, 2, 3)
            triangle_array = np.ctypeslib.as_array(triangles, shape=(sample_count * candidate_count * 9,)).reshape(sample_count, candidate_count, 3, 3)
            result = gpu_compute.intersect_rays_cpu(origin_array, direction_array, triangle_array, epsilon, ray_limit)
            for pointer, array in zip((out_t, out_u, out_v), result):
                np.ctypeslib.as_array(pointer, shape=(sample_count * 2 * candidate_count,))[:] = array.reshape(-1)
            return 0
        def pfhip_closest_points(self, handle, sample_count, candidate_count, points, triangles,
                                 out_distance, out_u, out_v, error, capacity):
            point_array = np.ctypeslib.as_array(points, shape=(sample_count * 3,)).reshape(sample_count, 3)
            triangle_array = np.ctypeslib.as_array(triangles, shape=(sample_count * candidate_count * 9,)).reshape(sample_count, candidate_count, 3, 3)
            result = gpu_compute.closest_points_cpu(point_array, triangle_array)
            for pointer, array in zip((out_distance, out_u, out_v), result):
                np.ctypeslib.as_array(pointer, shape=(sample_count * candidate_count,))[:] = array.reshape(-1)
            return 0
        def pfhip_qem_candidate_batch(self, handle, qem_source, vertex_count, edge_count,
                                      vertices, quadrics, edges, out_cost, out_position,
                                      error, capacity):
            self.qem_source = qem_source.decode("utf-8")
            vertex_array = np.ctypeslib.as_array(vertices, shape=(vertex_count * 3,)).reshape(vertex_count, 3)
            quadric_array = np.ctypeslib.as_array(quadrics, shape=(vertex_count * 16,)).reshape(vertex_count, 4, 4)
            edge_array = np.ctypeslib.as_array(edges, shape=(edge_count * 2,)).reshape(edge_count, 2)
            costs, positions = gpu_compute._qem_candidate_batch_xp(
                np, vertex_array, quadric_array, edge_array
            )
            np.ctypeslib.as_array(out_cost, shape=(edge_count,))[:] = costs
            np.ctypeslib.as_array(out_position, shape=(edge_count * 3,))[:] = positions.reshape(-1)
            return 0

    fake_bridge = FakeBridge()
    device = gpu_compute.GPUDevice(
        backend="hip", index=0, name="Radeon Test", vendor="AMD",
        native=(fake_bridge, "gfx1201", Path("PolyfittyHipBridge.dll")),
    )
    backend = gpu_compute._WindowsHipBackend(device)
    origins = np.asarray([[[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]]])
    directions = np.asarray([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
    triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = triangle.reshape(1, 1, 3, 3)
    expected = gpu_compute.intersect_rays_cpu(origins, directions, triangles, 1e-9, 2.0)
    actual = backend.intersect_rays(origins, directions, triangles, 1e-9, 2.0)
    for got, wanted in zip(actual, expected):
        assert np.allclose(got, wanted, equal_nan=True)
    closest_expected = gpu_compute.closest_points_cpu(np.asarray([[0.25, 0.25, 1.0]]), triangles)
    closest_actual = backend.closest_points(np.asarray([[0.25, 0.25, 1.0]]), triangles)
    for got, wanted in zip(closest_actual, closest_expected):
        assert np.allclose(got, wanted, equal_nan=True)

    candidate_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    candidate_quadrics = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    candidate_edges = np.asarray(((0, 1), (0, 2)), dtype=np.int64)
    expected_costs, expected_positions = gpu_compute._qem_candidate_batch_xp(
        np, candidate_vertices, candidate_quadrics, candidate_edges
    )
    actual_costs, actual_positions = backend.qem_candidate_batch(
        candidate_vertices, candidate_quadrics, candidate_edges
    )
    np.testing.assert_array_equal(actual_costs, expected_costs)
    np.testing.assert_array_equal(actual_positions, expected_positions)
    assert "qem_candidate_placements" not in fake_bridge.base_source
    assert "qem_candidate_placements" in fake_bridge.qem_source

def test_installer_checks_amd_hip_before_opencl_fallback():
    script = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
    assert "function Test-HipBackend" in script
    assert "--phase hip" in script
    assert script.index("$hipReady = Test-HipBackend") < script.index("if ($hasOpenClFallbackAdapter)")
    verify = (INSTALLER / "verify_runtime.py").read_text(encoding="utf-8")
    assert '"hip"' in verify and "def verify_hip" in verify

def test_portable_builder_does_not_select_or_rebuild_gpu_backends():
        gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        launcher = (INSTALLER / "portable_build_launcher.ps1").read_text(encoding="utf-8-sig")
        assert (ROOT / "VERSION.txt").read_text(encoding="utf-8").strip() == "0.2.131"
        for text in (gui, builder, launcher):
            assert "IncludeCuda" not in text
            assert "IncludeOpenCL" not in text
            assert "IncludeSYCL" not in text
            assert "IncludeHIP" not in text
        assert "$MainRuntime" not in builder
        assert "Get-DirectoryFingerprint" in builder
        assert "embedded_runtime_sha256" in builder
        assert "first launch" in gui.lower()

def test_portable_builder_runs_powershell_hidden_without_console_host():
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    launcher = (INSTALLER / "portable_build_launcher.ps1").read_text(encoding="utf-8-sig")
    assert "shell.Run(command, 0, false)" in gui
    assert "shell.Exec(" not in gui
    assert 'pidPath = fso.BuildPath(rootPath, ".build\\\\portable_build.pid")' in gui
    assert "WbemScripting.SWbemLocator" in gui
    assert "Win32_Process WHERE ProcessId=" in gui
    assert '" -PidFile " + quote(pidPath)' in gui
    assert "[Parameter(Mandatory = $true)][string]$PidFile" in launcher
    assert "Set-Content -LiteralPath $PidFile -Value ([string]$PID) -Encoding Unicode" in launcher
    assert "portable_build_console.log" in gui

def test_portable_builder_streams_native_process_output_and_preserves_exit_status():
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    assert "System.Diagnostics.ProcessStartInfo" in builder
    assert "$startInfo.CreateNoWindow = $true" in builder
    assert "$startInfo.RedirectStandardOutput = $true" in builder
    assert "$startInfo.RedirectStandardError = $true" in builder
    assert "ReadLineAsync()" in builder
    assert "$stdoutEnded" in builder and "$stderrEnded" in builder
    assert "$process.ExitCode" in builder
    assert "Write-BuildLog" in builder

def test_portable_builder_reports_embedded_runtime_and_verification_stages():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        for percent, stage in ((5, "Release source validation"), (12, "Prepare embedded application image"),
                               (22, "Provision embedded Python and core packages"), (66, "Verify embedded runtime"),
                               (74, "Create embedded payload"), (82, "Build single-file EXE"),
                               (88, "Audit embedded EXE payload"), (100, "Completed")):
            assert f"Set-BuildProgress {percent} '{stage}'" in builder
        assert "$MainRuntime" not in builder
        assert "-PortableBuild" in builder

def test_portable_subprocess_output_is_streamed_without_waiting_for_process_exit():
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    assert "$process.StandardOutput.ReadLineAsync()" in builder
    assert "$process.StandardError.ReadLineAsync()" in builder
    assert "Write-BuildLog $line" in builder
    assert "if ($process.HasExited -and $stdoutEnded -and $stderrEnded)" in builder

def test_gpu_discovery_reports_vendor_stages_to_startup_splash(monkeypatch):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import gpu_compute

    messages = []
    monkeypatch.setattr(gpu_compute, "_discover_cuda", lambda errors: [])
    monkeypatch.setattr(gpu_compute, "_discover_hip", lambda errors: [])
    monkeypatch.setattr(gpu_compute, "_discover_sycl", lambda errors: [])
    monkeypatch.setattr(gpu_compute, "_discover_opencl", lambda errors: [])

    devices, errors = gpu_compute.discover_gpu_devices(
        progress_callback=lambda _value, message: messages.append(message)
    )

    assert devices == []
    assert errors == []
    assert any("NVIDIA CUDA" in message for message in messages)
    assert any("AMD HIP" in message for message in messages)
    assert any("Intel oneAPI / SYCL" in message for message in messages)
    assert any("cross-vendor OpenCL" in message for message in messages)
