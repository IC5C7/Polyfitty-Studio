from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"

def _runtime_contract_module():
    app_path = str(APP)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    import runtime_contract
    return runtime_contract

def test_release_version_and_application_version_are_identical():
        version = (ROOT / "VERSION.txt").read_text(encoding="utf-8-sig").strip()
        source = (APP / "main.py").read_text(encoding="utf-8-sig")
        assert f'APP_VERSION = "{version}"' in source
        assert version == "0.2.131"

def test_runtime_manifest_covers_the_single_shared_runtime_and_native_splash_sources():
    contract = _runtime_contract_module()
    manifest = contract.compute_runtime_manifest(ROOT)
    files = set(manifest["files"])
    required = {
        "app/main.py", "app/coordinate_frame.py", "app/runtime_contract.py",
        "app/hidden_entry.py", "app/portable_entry.py", "app/gpu_compute.py", "app/gpu_runtime_paths.py",
        "app/worker.py", "app/startup_gpu_preflight.py", "app/startup_splash.py",
        "installer/install.ps1", "installer/start_app.ps1", "installer/build_portable.ps1",
        "installer/native_splash_launcher.cs", "installer/verify_runtime.py",
        "installer/runtime_source_manifest.py", "installer/gpu_setup_state.ps1", "installer/portable_build_gui.hta",
        "CREATE_PORTABLE_SHORTCUT.vbs", "requirements.txt", "VERSION.txt",
    }
    assert required <= files
    assert "packaging/PolyfittyStudio.spec" not in files
    assert manifest["source_file_count"] == len(files)
    assert len(manifest["source_fingerprint"]) == 64

def test_runtime_manifest_detects_a_mixed_or_partially_updated_source_tree(tmp_path: Path):
    contract = _runtime_contract_module()
    root = tmp_path / "release"
    app_dir = root / "app"
    app_dir.mkdir(parents=True)
    (root / "VERSION.txt").write_text("9.8.7\n", encoding="utf-8")
    constants = {
        "PORTABLE_PBR_EXPORT_UI_VERSION": 2,
        "PORTABLE_VIEWPORT_COORDINATE_VERSION": 18,
        "PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION": 2,
        "PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION": 3,
        "PORTABLE_ASSET_IMPORT_PROGRESS_VERSION": 1,
        "PORTABLE_VIEWPORT_MOUSE_NAVIGATION_VERSION": 1,
        "PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION": 2,
        "PORTABLE_GPU_FIRST_RUN_SETUP_VERSION": 2,
    }
    lines = ['APP_VERSION = "9.8.7"']
    lines.extend(f"{name} = {value}" for name, value in constants.items())
    (app_dir / "main.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (app_dir / "runtime-resource.dat").write_bytes(b"release-a")
    contract.write_runtime_manifest(root)
    verified = contract.verify_source_runtime_manifest(root)
    assert verified["app_version"] == "9.8.7"
    (app_dir / "runtime-resource.dat").write_bytes(b"release-b")
    with pytest.raises(RuntimeError, match="does not match its release runtime manifest"):
        contract.verify_source_runtime_manifest(root)

def test_runtime_feature_probe_includes_all_declared_features(monkeypatch):
    import types
    contract = _runtime_contract_module()
    fake_main = types.SimpleNamespace(
        PORTABLE_PBR_EXPORT_UI_VERSION=2,
        PORTABLE_VIEWPORT_COORDINATE_VERSION=18,
        PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION=2,
        PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION=3,
        PORTABLE_ASSET_IMPORT_PROGRESS_VERSION=1,
        PORTABLE_VIEWPORT_MOUSE_NAVIGATION_VERSION=2,
        PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION=2,
        PORTABLE_GPU_FIRST_RUN_SETUP_VERSION=2,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)
    actual = contract._feature_versions_from_runtime()
    assert set(actual) == set(contract._FEATURE_CONSTANTS)
    assert actual["gpu_startup_preflight"] == 2
    assert actual == {
        "pbr_export_ui": 2,
        "viewport_coordinate": 18,
        "viewport_color_management": 2,
        "material_preview_parity": 3,
        "asset_import_progress": 1,
        "viewport_mouse_navigation": 2,
        "gpu_startup_preflight": 2,
        "gpu_first_run_setup": 2,
    }

def test_installer_verifies_source_identity_before_dependency_installation():
    install = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    assert "runtime_source_manifest.py" in install
    assert "--verify" in install
    assert install.index("runtime_source_manifest.py") < install.index("Python module installation failed")

def test_installed_runtime_verifier_checks_the_real_embedded_python_path_without_path_injection():
    verifier = (INSTALLER / "verify_runtime.py").read_text(encoding="utf-8-sig")
    assert "python312._pth" in verifier
    assert "if APP_DIR not in search_paths" in verifier
    assert "sys.path.insert" not in verifier
    assert "from hidden_entry import main as shared_runtime_entry" in verifier
    assert "from startup_gpu_preflight import _run_startup_gpu_preflight" in verifier
    assert "verify_runtime_manifest(require_source_match=True)" in verifier

def test_both_launch_paths_delegate_to_the_same_startup_helper_and_entrypoint():
    installed_vbs = (ROOT / "START_HIDDEN.vbs").read_text(encoding="utf-16")
    portable_launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    startup_helper = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    hidden_entry = (APP / "hidden_entry.py").read_text(encoding="utf-8-sig")

    assert '"installer\\start_app.ps1"' in installed_vbs
    assert 'Path.Combine(_root, "installer", "start_app.ps1")' in portable_launcher
    assert 'Path.Combine(_root, ".runtime", "python312", "python.exe")' in portable_launcher
    assert "app\\hidden_entry.py" in startup_helper
    assert "$startInfo.FileName = $PythonExe" in startup_helper
    assert "verify_runtime_manifest(require_source_match=True)" in hidden_entry
    assert "startup_gpu_probe = _run_startup_gpu_preflight(splash)" in hidden_entry
    assert "from main import main as gui_main" in hidden_entry

def test_startup_helper_does_not_mutate_the_shared_python_path_configuration():
        startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
        assert "must expose the package app directory and site-packages" in startup
        assert "Add-Content -LiteralPath $PythonPathFile" not in startup
        assert "$env:PYTHONHOME = ''" in startup
        assert "$env:PYTHONPATH = ''" in startup

def test_portable_builder_copies_the_main_runtime_byte_for_byte_and_runs_the_same_verifier():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
        assert "$MainRuntime" not in builder
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        assert "requirements.txt" in installer
        assert "verify_runtime.py" in builder and "'--phase', 'runtime'" in builder
        assert "runtime_source_manifest.py" in builder
        assert "PolyfittyStudio.Payload.zip" in builder
        assert "GetManifestResourceStream('PolyfittyStudio.Payload.zip')" in builder
        assert "ExtractToDirectory($PayloadVerifyZip" in builder
        assert "Get-DirectoryFingerprint $PortableRuntime" in builder
        assert "$runtimeFingerprint.Fingerprint -ne $zipRuntimeFingerprint.Fingerprint" in builder
        assert "embedded_runtime_sha256" in builder
        assert "The portable output must contain exactly one distributable file" in builder

def test_portable_builder_has_no_independent_interpreter_dependency_solver_or_gpu_selection():
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    launcher = (INSTALLER / "portable_build_launcher.ps1").read_text(encoding="utf-8-sig")
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    for obsolete in ("PyInstaller", "PolyfittyStudio.Runtime.exe", "IncludeCuda", "IncludeSYCL", "IncludeHIP", "PythonExe ="):
        assert obsolete not in builder
    for obsolete in ("IncludeCuda", "IncludeSYCL", "IncludeHIP", "PythonExe"):
        assert obsolete not in launcher
        assert obsolete not in gui
    assert "pythonPath" not in gui
    assert "cudaOption" not in gui and "syclOption" not in gui

def test_application_has_no_packaging_mode_specific_python_or_worker_branch():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    contract = (APP / "runtime_contract.py").read_text(encoding="utf-8-sig")
    gpu = (APP / "gpu_compute.py").read_text(encoding="utf-8-sig")
    assert "IS_FROZEN_PORTABLE" not in main
    assert "sys.frozen" not in main and "_MEIPASS" not in main
    assert "--worker" not in main
    assert "command = [sys.executable, \"-u\", str(worker_script), str(config_path)]" in main
    assert "runtime_bundle_root()" in contract
    assert "sys.frozen" not in gpu and "_MEIPASS" not in gpu

def test_current_source_runtime_manifest_is_present_and_matches_release():
        contract = _runtime_contract_module()
        manifest = contract.verify_source_runtime_manifest(ROOT)
        assert manifest["app_version"] == "0.2.131"
        assert manifest["feature_versions"]["asset_import_progress"] >= 1
        assert manifest["feature_versions"]["viewport_mouse_navigation"] == 2
        assert manifest["feature_versions"]["gpu_startup_preflight"] == 2
        assert manifest["feature_versions"]["gpu_first_run_setup"] == 3

def test_portable_distribution_embeds_its_own_python_environment_but_shares_the_app_source():
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    native_launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")
    hidden_entry = (APP / "hidden_entry.py").read_text(encoding="utf-8-sig")
    guide = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")

    assert "$MainRuntime" not in builder
    assert "The installed main runtime is not required" in builder
    assert "'.runtime'" in builder
    assert "python-3.12.10-embed-amd64.zip" in installer
    assert "-PortableBuild" in builder
    assert "package_kind = 'single_file_self_extracting_embedded_portable'" in builder
    assert "embedded_runtime_sha256" in builder
    assert 'Path.Combine(_root, ".runtime", "python312", "python.exe")' in native_launcher
    assert "-PortablePackage" in native_launcher
    assert "$IsPortablePackage" in startup and "portable_runtime_manifest.json" in startup
    assert "activate_optional_gpu_packages()" in hidden_entry
    assert "same application source" in guide
    assert "one file only" in guide
    assert "single_file_self_extracting_embedded_portable" in builder
    assert "exactly one file" in guide
    assert "Users receive one executable" in gui
