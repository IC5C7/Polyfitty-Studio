from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"

def test_core_reconstruction_and_bake_engines_are_present():
    expected = {
        "pipeline.py",
        "polyfit_runtime.py",
        "fallback_routing.py",
        "planar_regions.py",
        "feature_preserving.py",
        "feature_decimation.py",
        "postprocess.py",
        "texture_baking.py",
        "gpu_compute.py",
        "worker.py",
    }
    present = {path.name for path in APP.glob("*.py")}
    assert expected <= present

def test_installer_still_provisions_private_python_and_native_polyfit_binding():
    source = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    assert ".runtime\\python312" in source
    assert "python-3.12.10-embed-amd64.zip" in source
    assert "polyfit-1.6.0-cp312-cp312-win_amd64.whl" in source
    assert "verify_runtime.py" in source
    assert "--phase', 'polyfit'" in source
    assert "--phase', 'runtime'" in source

def test_optional_gpu_runtime_installers_remain_available():
    gpu_installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
    assert "requirements_gpu_cuda12.txt" in gpu_installer
    assert "requirements_gpu_opencl.txt" in gpu_installer
    assert "CUDA" in gpu_installer
    assert "OpenCL" in gpu_installer

def test_application_shows_main_window_and_emits_startup_milestones():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    entry = main[main.rfind("\ndef main(app=None, startup_splash=None, startup_gpu_probe=None):") :]
    assert "window = MainWindow(startup_gpu_probe=startup_gpu_probe)" in entry
    assert "window.showMaximized()" in entry
    assert "window.showNormal()" in entry
    assert "window.isVisible()" in entry
    assert "MAIN_WINDOW_SHOW_RETURNED visible={0}" in entry
    assert "return int(app.exec())" in entry
    assert "startup_splash.finish(window)" in entry
    assert "self._start_gpu_scan()" in main

def test_vtk_interactor_is_initialized_after_the_top_level_window_can_paint():
    vtk = (APP / "vtk_backend.py").read_text(encoding="utf-8-sig")
    show_start = vtk.index("    def showEvent(self, event):")
    paint_start = vtk.index("    def paintEvent(self, event):", show_start)
    init_start = vtk.index("    def _initialize_interactor_after_show(self):", paint_start)
    show_event = vtk[show_start:paint_start]
    paint_event = vtk[paint_start:init_start]
    initialize = vtk[init_start:vtk.index("    @staticmethod", init_start)]
    assert "QTimer.singleShot(180, self._initialize_interactor_after_show)" in show_event
    assert "self.Initialize()" not in show_event
    assert "if not self._interactor_initialized" in paint_event
    assert "return" in paint_event
    assert "self.Initialize()" in initialize
    assert "_interactor_init_failed = True" in initialize
    assert "self._renderer.AddActor(self._viewport_grid_actor)" in vtk
    assert "vtk.vtkActor2D()" not in vtk
    assert "_viewport_grid_scene_bounds" in vtk
    assert "floor_z" in vtk

def test_hidden_launcher_suppresses_only_python_console_and_preserves_python_executable():
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    assert "$startInfo.FileName = $PythonExe" in startup
    assert "$startInfo.CreateNoWindow = $true" in startup
    assert "$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Normal" in startup
    assert "$startInfo.UseShellExecute = $false" in startup
    assert "-WindowStyle Hidden -PassThru -RedirectStandardOutput $appStdout" not in startup
    assert "$startInfo.Arguments = " in startup
    assert "pythonw.exe" not in startup

def test_hidden_launcher_checks_a_real_native_windows_window_not_qt_logical_visibility():
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    assert "EnumWindows" in startup
    assert "GetWindowThreadProcessId" in startup
    assert "IsWindowVisible" in startup
    assert "IsIconic" in startup
    assert "FindMainWindow($appProcess.Id, 'Polyfitty Studio')" in startup
    assert "RestoreAndActivate($windowHandle)" in startup
    assert "IsForeground($windowHandle)" in startup
    assert "activationAttempts -eq 0" in startup
    assert "NATIVE_MAIN_WINDOW_PRESENTED" in startup
    assert "MAIN_WINDOW_SHOW_RETURNED visible=True" not in startup

def test_console_free_entry_point_captures_pythonw_stdout_and_tracebacks():
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    assert "POLYFITTY_STARTUP_STDOUT" in entry
    assert "POLYFITTY_STARTUP_STDERR" in entry
    assert "stdout_log = _open_log" in entry
    assert "stderr_log = _open_log" in entry
    assert "sys.stdout = stdout_log" in entry
    assert "sys.stderr = stderr_log" in entry
    assert "from main import main as gui_main" in entry

def test_startup_diagnostics_are_best_effort_and_do_not_block_application():
    source = (APP / "startup_diagnostics.py").read_text(encoding="utf-8-sig")
    assert "POLYFITTY_STARTUP_TRACE" in source
    assert "except Exception:" in source
    assert "if not target:" in source and "stream.write(" in source

def test_start_hidden_vbs_keeps_only_the_helper_hidden_and_waits_for_native_window_check():
    launcher = (ROOT / "START_HIDDEN.vbs").read_text(encoding="utf-16")
    assert 'exitCode = shell.Run(commandLine, 0, True)' in launcher
    assert 'installer\\start_app.ps1' in launcher

def test_product_brand_is_polyfitty_studio_and_legacy_preferences_are_migrated():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    launcher = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    workspace = (APP / "workspace.py").read_text(encoding="utf-8-sig")
    assert 'self.setWindowTitle("Polyfitty Studio | Polygonal Surface Reconstruction")' in main
    assert 'QSettings("PolyfittyStudio", "PolyfittyStudio")' in main
    assert 'QSettings("PolyfittyStudio", "PolyfittyStudio")' in main
    assert 'legacy_settings.allKeys()' in main and 'self.settings.setValue(setting_key' in main
    assert "Polyfitty Studio.lnk" in installer and "Polyfitty Studio.lnk" in installer
    assert "PolyfittyNativeWindowProbe" in launcher
    assert 'root / "PolyfittyStudio"' in workspace and 'root / "PolyfittyStudio"' in workspace

def test_legacy_temp_workspace_is_moved_into_polyfitty_folder(tmp_path):
    import sys
    app_dir = str(APP)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from workspace import mesh_work_dir

    source = tmp_path / "input_mesh.glb"
    legacy = mesh_work_dir(source, temp_root=tmp_path)
    legacy.mkdir(parents=True)
    payload = legacy / "input_mesh_reconstructed.obj"
    payload.write_text("legacy result", encoding="utf-8")

    new_root = tmp_path / "PolyfittyStudio"
    new_root.rename(tmp_path / "PolyFitStudio")
    migrated = mesh_work_dir(source, temp_root=tmp_path)
    assert migrated == tmp_path / "PolyfittyStudio" / legacy.name
    assert (migrated / payload.name).read_text(encoding="utf-8") == "legacy result"
    assert not (tmp_path / "PolyFitStudio").exists()

def test_portable_package_uses_an_independent_embedded_python_runtime_and_shared_worker_entrypoint():
        main = (APP / "main.py").read_text(encoding="utf-8-sig")
        entry = (APP / "hidden_entry.py").read_text(encoding="utf-8-sig")
        pipeline = (APP / "pipeline.py").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
        assert 'command = [sys.executable, "-u", str(worker_script), str(config_path)]' in main
        assert "IS_FROZEN_PORTABLE" not in main
        assert "sys.frozen" not in main
        assert "verify_runtime_manifest(require_source_match=True)" in entry
        assert "startup_gpu_probe = _run_startup_gpu_preflight(splash)" in entry
        assert "load_polyfit_binding(allow_repair=True)" in pipeline or "load_polyfit_binding(" in pipeline
        assert "$MainRuntime" not in builder
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        assert "requirements.txt" in installer
        assert "Get-DirectoryFingerprint $PortableRuntime" in builder
        assert "PolyfittyStudio.Runtime.exe" not in builder
        assert "PyInstaller" not in builder

def test_portable_builder_bootstraps_an_independent_embedded_python312_runtime():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
        guide = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "$MainRuntime" not in builder
        assert "The installed main runtime is not required" in builder
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        assert "requirements.txt" in installer
        assert "does not need to run `INSTALL.bat`" in guide
        assert "official CPython 3.12.10 embeddable distribution" in guide
        assert "**one file only**" in guide
        assert "Local AppData cache" in guide
        for obsolete in ("Get-Python312FromLauncher", "pymanager.exe", "-m', 'venv", "python-3.12.10-amd64.exe", "PyInstaller"):
            assert obsolete not in builder

def test_portable_build_gui_uses_a_hidden_console_vbs_launcher():
    launcher = (ROOT / "BUILD_PORTABLE_EXE.vbs").read_text(encoding="utf-8")
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    batch = (ROOT / "BUILD_PORTABLE_EXE.bat").read_text(encoding="utf-8")
    assert "mshta.exe" in launcher.lower()
    assert "portable_build_gui.hta" in launcher
    assert "wscript.exe" in batch.lower()
    assert "Start portable build" in gui
    assert "setInterval(refreshStatus, 600)" in gui
    assert "progress" in gui.lower() and "cancelBuild" in gui
    assert "-WindowStyle Hidden" in gui

def test_portable_builder_reports_embedded_runtime_milestones_logs_and_safe_cancellation():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
        assert "[string]$StatusFile" in builder
        assert "[string]$LogFile" in builder
        assert "[string]$CancelFile" in builder
        assert "function Write-BuildStatus" in builder
        assert "function Invoke-CheckedProcess" in builder
        assert "function Test-BuildCancellation" in builder
        for percent in (5, 12, 22, 66, 74, 82, 88, 100):
            assert f"Set-BuildProgress {percent}" in builder
        assert "succeeded" in builder and "failed" in builder and "cancelled" in builder
        assert "portable_build_status.json" in gui
        assert "portable_build.log" in gui
        assert "portable_build.cancel" in gui
        assert "Build log" in gui
        assert "one EXE" in gui or "single-file" in gui.lower()

def test_portable_build_has_gui_entrypoint_and_third_party_license_notices():
        guide = (ROOT / "README.md").read_text(encoding="utf-8")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        assert "BUILD_PORTABLE_EXE.vbs" in guide
        assert "BUILD_PORTABLE_EXE.bat" in guide
        assert "PolyfittyStudio.exe" in builder
        assert "THIRD_PARTY_NOTICES.txt" in guide
        assert "does not need to run `INSTALL.bat`" in guide
        assert "PolyfittyStudio.Runtime.exe" not in guide

def test_opencl_diagnostics_distinguish_missing_binding_from_missing_vendor_icd(monkeypatch):
    import sys
    app_dir = str(APP)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from gpu_compute import _discover_opencl

    monkeypatch.setitem(sys.modules, "pyopencl", None)
    errors = []
    assert _discover_opencl(errors) == []
    assert any("PyOpenCL import failed" in item and "pyopencl" in item.lower() for item in errors)

def test_opencl_device_discovery_uses_vendor_platform_and_reports_gpu_memory(monkeypatch):
    import sys
    import types
    app_dir = str(APP)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from gpu_compute import _discover_opencl

    class Device:
        name = "Discrete Test GPU"
        vendor = "Test Vendor"
        global_mem_size = 8 * 1024 ** 3
        available = True
        compiler_available = True

    class Platform:
        name = "Test OpenCL Platform"
        vendor = "Test Vendor"
        def get_devices(self, device_type):
            assert device_type == 4
            return [Device()]

    fake = types.SimpleNamespace(
        device_type=types.SimpleNamespace(GPU=4),
        get_platforms=lambda: [Platform()],
    )
    monkeypatch.setitem(sys.modules, "pyopencl", fake)
    errors = []
    devices = _discover_opencl(errors)
    assert len(devices) == 1
    assert devices[0].backend == "opencl"
    assert devices[0].name == "Discrete Test GPU"
    assert devices[0].memory_bytes == 8 * 1024 ** 3
    assert errors == []

def test_gpu_discovery_explains_missing_opencl_icd_platform(monkeypatch):
    import sys
    import types
    app_dir = str(APP)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from gpu_compute import _discover_opencl

    fake = types.SimpleNamespace(
        device_type=types.SimpleNamespace(GPU=4),
        get_platforms=lambda: (_ for _ in ()).throw(RuntimeError("PLATFORM_NOT_FOUND_KHR")),
    )
    monkeypatch.setitem(sys.modules, "pyopencl", fake)
    errors = []
    assert _discover_opencl(errors) == []
    assert any("vendor ICD platform" in item and "PLATFORM_NOT_FOUND_KHR" in item for item in errors)

def test_gpu_status_explains_missing_packages_in_the_shared_main_runtime():
        main = (APP / "main.py").read_text(encoding="utf-8-sig")
        assert "pyopencl_not_bundled" in main
        assert "OPENCL PACKAGE MISSING" in main
        assert "installed Studio setup manager" in main
        assert "relaunch the portable copy" in main
        assert "OpenCL ICD" in main

def test_opencl_dll_search_is_prepared_before_import_in_the_shared_windows_runtime():
    source = (APP / "gpu_compute.py").read_text(encoding="utf-8")
    opencl_start = source.index("def _discover_opencl(")
    opencl_end = source.index("\ndef discover_gpu_devices(", opencl_start)
    opencl_block = source[opencl_start:opencl_end]
    assert "_prepare_opencl_dll_search_paths()" in opencl_block
    assert opencl_block.index("_prepare_opencl_dll_search_paths()") < opencl_block.index("import pyopencl as cl")
    assert "_OPENCL_DLL_DIRECTORY_HANDLES.append(add_directory(resolved))" in source
    assert 'Path(sys.executable).resolve().parent' in source
    assert '_MEIPASS' not in source

def test_gpu_display_adapter_probe_uses_absolute_windows_powershell_path_first():
    source = (APP / "main.py").read_text(encoding="utf-8")
    start = source.index("def _discover_system_gpu_names(")
    end = source.index("\n\nclass GPUDiscoveryWorker", start)
    block = source[start:end]
    assert "SystemRoot" in block
    assert "WindowsPowerShell" in block
    assert "commands.append([str(powershell_exe)" in block

def test_gpu_auto_order_prefers_cuda_then_dedicated_gpu_before_unified_memory_igpu(monkeypatch):
    import sys
    app_dir = str(APP)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import gpu_compute

    shared_igpu = gpu_compute.GPUDevice(
        backend="opencl", index=0, name="Intel Integrated GPU", vendor="Intel",
        memory_bytes=16 * 1024 ** 3, native=object(), integrated=True,
    )
    discrete_egpu = gpu_compute.GPUDevice(
        backend="opencl", index=1, name="NVIDIA External GPU", vendor="NVIDIA",
        memory_bytes=8 * 1024 ** 3, native=object(), integrated=False,
    )
    cuda_egpu = gpu_compute.GPUDevice(
        backend="cuda", index=0, name="NVIDIA CUDA GPU", vendor="NVIDIA",
        memory_bytes=8 * 1024 ** 3, native=object(), integrated=False,
    )
    monkeypatch.setattr(gpu_compute, "_discover_cuda", lambda errors: [cuda_egpu])
    monkeypatch.setattr(gpu_compute, "_discover_opencl", lambda errors: [shared_igpu, discrete_egpu])

    devices, errors = gpu_compute.discover_gpu_devices()

    assert any("Intel oneAPI/SYCL runtime unavailable" in item for item in errors)
    assert [device.name for device in devices] == [
        "NVIDIA CUDA GPU", "NVIDIA External GPU", "Intel Integrated GPU",
    ]

def test_portable_builder_monitors_power_shell_startup_and_reports_early_exit():
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    launcher_path = INSTALLER / "portable_build_launcher.ps1"
    assert launcher_path.is_file()
    launcher = launcher_path.read_text(encoding="utf-8-sig")
    assert "shell.Run(command, 0, false)" in gui
    assert "Win32_Process WHERE ProcessId=" in gui
    assert "Hidden PowerShell process (PID " in gui
    assert "exited before reporting a final build state" in gui
    assert "portable_build_console.log" in gui
    assert "Waiting for PowerShell startup" in gui
    assert "-ConsoleLogFile" in gui
    assert "Convert-BuilderBoolean" not in launcher
    assert "Write-LauncherStatus" in launcher
    assert "Launcher failure:" in launcher
    assert "*>> $ConsoleLogFile" in launcher
