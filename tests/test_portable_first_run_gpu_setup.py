from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"

def test_shared_startup_installs_gpu_runtime_before_launching_python():
        startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
        gpu_installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
        setup_index = startup.index("Starting portable first-run GPU detection and matching user-space package installation.")
        app_index = startup.index("$EntryScript = Join-Path $Root 'app\\hidden_entry.py'")
        portable_gate = startup.index("if ($IsPortablePackage)")
        installed_skip = startup.index("Installed product detected: GPU packages are provisioned by the installer setup-manager stage")
        assert portable_gate < setup_index < app_index
        assert installed_skip < app_index
        assert "Get-PolyfittyGpuSetupStateDecision" in startup
        assert "installer\\install_gpu.ps1" in startup
        assert "-StartupStatusFile" in startup and "-GpuStatePath" in startup
        assert "Start-Process" in startup and "-Wait" in startup
        assert "Write-PolyfittyStartupGpuStatus" in startup
        assert "pip install" in gpu_installer.lower()
        assert "-PortablePackage" in (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
        assert "portable_runtime_manifest.json" in startup
        assert "single_file_self_extracting_embedded_portable" in startup

def test_gpu_setup_is_cached_by_hardware_driver_and_package_fingerprints():
    state = (INSTALLER / "gpu_setup_state.ps1").read_text(encoding="utf-8-sig")
    assert "Win32_VideoController" in state
    assert "PNPDeviceID" in state
    assert "DriverVersion" in state
    assert "Get-FileHash" in state and "SHA256" in state
    assert "requirements_gpu_cuda12.txt" in state
    assert "requirements_gpu_opencl.txt" in state
    assert "requirements_gpu_sycl.txt" in state
    assert "retry_after_utc" in state
    assert "hardware_fingerprint" in state
    assert "requirements_fingerprint" in state
    assert "status = $Status" in state

def test_installer_stage_persists_ready_cpu_fallback_and_failure_results():
    installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
    assert "Save-PolyfittyGpuSetupState" in installer
    assert "-Status 'ready'" in installer
    assert "-Status 'cpu_fallback'" in installer
    assert "-Status 'failed'" in installer
    assert "Write-PolyfittyStartupGpuStatus" in installer
    assert "requirements_gpu_cuda12.txt" in installer
    assert "requirements_gpu_opencl.txt" in installer
    assert "requirements_gpu_sycl.txt" in installer

def test_nvidia_cuda_user_runtime_is_deferred_until_windows_driver_is_present():
    installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
    assert "System32\\nvcuda.dll" in installer
    assert "if ($hasNvidia -and $hasNvidiaDriver)" in installer
    assert "CUDA packages will be deferred until the vendor driver is installed" in installer
    assert "Install-GpuRequirements -RequirementsFile 'requirements_gpu_cuda12.txt'" in installer

def test_driver_install_is_not_silently_performed_by_optional_gpu_setup():
    installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig").lower()
    assert "pip install" in installer
    assert "does not replace windows display drivers" in installer
    assert "start-process" not in installer
    assert "pnputil" not in installer
    assert "msiexec" not in installer

def test_portable_build_does_not_ship_per_machine_gpu_setup_cache():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8")
        startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
        assert "'.cache'" in builder or "'.cache'," in builder
        assert "gpu_setup_state.json" in startup
        assert "Remove-Item -LiteralPath $distCache -Recurse -Force" in builder
        assert "gpu_setup = 'first_launch_on_recipient_pc'" in builder
        assert "recipient-specific GPU state is created only on the target PC" in builder or "Remove-Item -LiteralPath $distCache -Recurse -Force" in builder

def test_first_run_gpu_setup_is_part_of_shared_runtime_contract():
        contract = (APP / "runtime_contract.py").read_text(encoding="utf-8")
        main = (APP / "main.py").read_text(encoding="utf-8")
        assert '"gpu_first_run_setup": "PORTABLE_GPU_FIRST_RUN_SETUP_VERSION"' in contract
        assert "PORTABLE_GPU_FIRST_RUN_SETUP_VERSION = 3" in main
        assert 'APP_VERSION = "0.2.131"' in main
        assert (ROOT / "VERSION.txt").read_text(encoding="utf-8").strip() == "0.2.131"

def test_portable_and_installed_paths_share_first_run_gpu_setup_and_preflight():
    launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    vbs = (ROOT / "START_HIDDEN.vbs").read_text(encoding="utf-16")
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    assert '"installer", "start_app.ps1"' in launcher
    assert '"installer\\start_app.ps1"' in vbs
    assert "Get-PolyfittyGpuSetupStateDecision" in startup
    assert "app\\hidden_entry.py" in startup
    assert "startup_gpu_probe = _run_startup_gpu_preflight(splash)" in entry

def test_gpu_packages_install_into_per_machine_overlay_not_shared_python_runtime():
    installer = (INSTALLER / "install_gpu.ps1").read_text(encoding="utf-8-sig")
    helper = (APP / "gpu_runtime_paths.py").read_text(encoding="utf-8")
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    verifier = (INSTALLER / "verify_runtime.py").read_text(encoding="utf-8-sig")

    assert ".cache\\gpu_runtime\\site-packages" in installer
    assert "--target $GpuPackagesRoot" in installer
    assert "activate_optional_gpu_packages" in helper
    assert "sys.path.insert(0, resolved)" in helper
    assert "activate_optional_gpu_packages()" in entry
    assert "activate_optional_gpu_packages()" in verifier
    assert "--target $GpuPackagesRoot" in installer

def test_gpu_setup_fingerprint_tracks_overlay_loader_and_compute_dll_search_logic():
    state = (INSTALLER / "gpu_setup_state.ps1").read_text(encoding="utf-8-sig")
    assert "app/gpu_runtime_paths.py" in state
    assert "app/gpu_compute.py" in state

def test_optional_gpu_overlay_path_resolves_to_cache_outside_private_python_runtime(tmp_path, monkeypatch):
    import shutil
    import sys
    sys.path.insert(0, str(APP)) if str(APP) not in sys.path else None
    from gpu_runtime_paths import activate_optional_gpu_packages, gpu_site_packages_path

    project_root = APP.parent
    site = project_root / ".cache" / "gpu_runtime" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    try:
        resolved = gpu_site_packages_path()
        assert resolved == site.resolve()
        assert ".runtime" not in str(resolved)
        activated = activate_optional_gpu_packages()
        assert activated == resolved
        assert str(resolved) in sys.path
        sys.path.remove(str(resolved))
    finally:
        shutil.rmtree(project_root / ".cache" / "gpu_runtime", ignore_errors=True)

def test_gpu_setup_state_rechecks_missing_backend_overlay_packages():
    state = (INSTALLER / "gpu_setup_state.ps1").read_text(encoding="utf-8-sig")
    assert "function Test-PolyfittyGpuOverlayForBackend" in state
    assert "The cached {0} backend is missing files from the per-machine GPU package overlay." in state
    assert "cupy\\__init__.py" in state
    assert "dpctl\\__init__.py" in state
    assert "pyopencl\\__init__.py" in state

def test_gpu_overlay_is_activated_before_numpy_and_main_module_imports():
    gpu_compute = (APP / "gpu_compute.py").read_text(encoding="utf-8")
    hidden_entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    assert gpu_compute.index("activate_optional_gpu_packages()") < gpu_compute.index("import numpy as np")
    assert hidden_entry.index("activate_optional_gpu_packages()") < hidden_entry.index("from runtime_contract import verify_runtime_manifest")

def test_shared_gpu_status_helper_is_loaded_before_installed_or_portable_branch():
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    portable_vbs = (ROOT / "START_HIDDEN.vbs").read_text(encoding="utf-16")

    helper_path = startup.index("$GpuStateHelper = Join-Path $Root 'installer\\gpu_setup_state.ps1'")
    required_paths = startup.index("foreach ($requiredPath in @($PythonExe, $PythonPathFile, $EnsurePip, $GpuStateHelper))")
    dot_source = startup.index(". $GpuStateHelper")
    installed_branch = startup.index("else {\n        Write-StartupLog 'Installed product detected:")
    assert helper_path < required_paths < dot_source < installed_branch
    assert "function Write-PolyfittyStartupGpuStatus" in (INSTALLER / "gpu_setup_state.ps1").read_text(encoding="utf-8-sig")

    assert 'portableManifestPath = fso.BuildPath(rootPath, "portable_runtime_manifest.json")' in portable_vbs
    assert 'If fso.FileExists(portableManifestPath) Then' in portable_vbs
    assert 'commandLine = commandLine & " -PortablePackage"' in portable_vbs

def test_startup_mode_is_logged_to_diagnose_portable_flag_or_manifest_detection():
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    assert "Startup mode: PortablePackage switch={0}; portable manifest present={1}; classified portable={2}." in startup
