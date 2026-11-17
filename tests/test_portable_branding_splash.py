from pathlib import Path

from PIL import Image
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

def test_portable_launcher_is_a_single_file_self_extracting_app_with_embedded_runtime():
    launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    startup = (INSTALLER / "start_app.ps1").read_text(encoding="utf-8-sig")

    assert "System.IO.Compression" in launcher
    assert "PolyfittyStudio.Payload.zip" in launcher
    assert "PolyfittyStudio.Payload.sha256" in launcher
    assert "GetManifestResourceStream(PayloadResourceName)" in launcher
    assert "ComputeStreamSha256(resource)" in launcher
    assert launcher.index("if (IsCompletePackage(packageRoot, payloadHash))") < launcher.index("string actualPayloadHash = ComputeStreamSha256(resource)")
    assert "ExtractZipSafely(archive, extractionRoot, report)" in launcher
    assert '[assembly: AssemblyTitle("Polyfitty Studio")]' in launcher
    release_version = (ROOT / "VERSION.txt").read_text(encoding="utf-8").strip()
    assert f'[assembly: AssemblyInformationalVersion("{release_version}")]' in launcher
    assert f'[assembly: AssemblyVersion("{release_version}.0")]' in launcher
    assert "temporaryPayload" not in launcher
    assert "EmbeddedPortablePayload.Extract" in launcher
    assert "LocalApplicationData" in launcher
    assert '"PF"' in launcher
    assert '"P"' in launcher and '"T"' in launcher
    assert 'Guid.NewGuid().ToString("N")' in launcher
    assert 'Path.Combine(stagingRootDirectory, Guid.NewGuid().ToString("N"))' in launcher
    assert '"Packages"' not in launcher
    assert ".portable_payload_complete" in launcher
    assert "ExtractZipSafely" in launcher
    assert "Path.GetFullPath(combined)" in launcher
    assert "attempted to write outside its extraction directory" in launcher
    assert 'Path.Combine(_root, "installer", "start_app.ps1")' in launcher
    assert 'Path.Combine(_root, ".runtime", "python312", "python.exe")' in launcher
    assert "PolyfittyStudio.Runtime.exe" not in launcher
    assert "PolyfittyStudio.Splash.png" in launcher
    assert "-File " in launcher and "-RootPath " in launcher
    assert "POLYFITTY_STARTUP_READY_EVENT" in launcher
    assert "POLYFITTY_STARTUP_STATUS_FILE" in launcher
    assert "verify_runtime_manifest(require_source_match=True)" in entry
    assert "startup_gpu_probe = _run_startup_gpu_preflight(splash)" in entry
    assert "from main import main as gui_main" in entry
    assert "gui_main(startup_splash=splash, startup_gpu_probe=startup_gpu_probe)" in entry
    assert "app\\hidden_entry.py" in startup
    assert "ProcessStartInfo" in startup
    assert "$startInfo.FileName = $PythonExe" in startup
    assert "python312._pth" in startup

def test_installed_and_portable_shortcut_tools_target_the_same_single_file_executable():
    installed_vbs = (ROOT / "START_HIDDEN.vbs").read_text(encoding="utf-16")
    portable_shortcut = (ROOT / "CREATE_PORTABLE_SHORTCUT.vbs").read_text(encoding="utf-8")
    native_launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    assert '"installer\\start_app.ps1"' in installed_vbs
    assert 'Path.Combine(_root, "installer", "start_app.ps1")' in native_launcher
    assert 'fso.BuildPath(outputRoot, "PolyfittyStudio.exe")' in portable_shortcut
    assert 'shortcut.TargetPath = targetPath' in portable_shortcut
    assert 'shortcut.IconLocation = targetPath & ",0"' in portable_shortcut

def test_portable_builder_embeds_all_runtime_files_into_exactly_one_exe():
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    gui = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")

    assert "$MainRuntime" not in builder
    assert "'-File', $portableInstaller, '-PortableBuild'" in builder
    assert "python-3.12.10-embed-amd64.zip" in installer
    assert "requirements.txt" in installer
    assert "Get-DirectoryFingerprint $PortableRuntime" in builder
    assert "embedded_runtime_sha256" in builder
    assert "PolyfittyStudio.Payload.zip" in builder
    assert "PolyfittyStudio.Payload.sha256" in builder
    assert "('/resource:' + $PayloadZip + ',PolyfittyStudio.Payload.zip')" in builder
    assert "('/resource:' + $PayloadHashFile + ',PolyfittyStudio.Payload.sha256')" in builder
    assert "[Reflection.Assembly]::LoadFrom($PortableExe)" in builder
    assert "GetManifestResourceStream('PolyfittyStudio.Payload.zip')" in builder
    assert "GetManifestResourceStream('PolyfittyStudio.Payload.sha256')" in builder
    assert "release\\portable" in builder
    assert "$PortableExe = Join-Path $PortableOutput 'PolyfittyStudio.exe'" in builder
    assert "$releaseFiles.Count -ne 1" in builder
    assert "The portable output must contain exactly one distributable file" in builder
    assert "$BuildWorkRoot = Join-Path $env:TEMP ('PF' + [Guid]::NewGuid().ToString('N'))" in builder
    assert "$Dist = Join-Path $BuildWorkRoot 'i'" in builder
    assert "$zipVerifyRoot = Join-Path $BuildWorkRoot 'v'" in builder
    assert "deployment_model = 'single_exe_embedded_payload_extracted_to_local_app_data'" in builder
    assert "LocalApplicationData" in launcher
    assert "EmbeddedPortablePayload.Extract" in launcher
    assert "single-file portable executable" in gui.lower()
    assert "Users receive one executable" in gui
    assert "PyInstaller" not in builder

def test_main_window_signals_native_launcher_after_first_presentation():
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "def main(app=None, startup_splash=None, startup_gpu_probe=None):" in main
    assert "window = MainWindow(startup_gpu_probe=startup_gpu_probe)" in main
    assert 'startup_splash.update("Loading the viewport and GPU runtime adapters...")' in main
    assert "startup_splash.finish(window)" in main
    assert "return int(app.exec())" in main
    assert min(main.index("window.showMaximized()"), main.index("window.showNormal()")) < main.index("startup_splash.finish(window)")

def test_installed_and_portable_desktop_shortcuts_use_the_brand_icon():
    installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
    shortcut_script = (ROOT / "CREATE_PORTABLE_SHORTCUT.vbs").read_text(encoding="utf-8")
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    assert "app\\assets\\PolyfittyStudio.ico" in installer
    assert "$shortcut.IconLocation" in installer
    assert 'PolyfittyStudio.exe' in shortcut_script
    assert 'shortcut.IconLocation = targetPath & ",0"' in shortcut_script
    assert "app\\assets\\PolyfittyStudio.ico" in builder
    assert 'fso.BuildPath(outputRoot, "Polyfitty Studio.lnk")' in shortcut_script
    assert 'fso.BuildPath(desktopPath, "Polyfitty Studio.lnk")' in shortcut_script
    assert "shortcut.TargetPath = targetPath" in shortcut_script

def test_brand_assets_have_windows_icon_sizes_and_expected_splash_dimensions():
    icon_path = APP / "assets" / "PolyfittyStudio.ico"
    png_path = APP / "assets" / "PolyfittyStudio.png"
    splash_path = ROOT / "packaging" / "PolyfittyStudio_Splash.png"
    assert icon_path.is_file() and png_path.is_file() and splash_path.is_file()
    with Image.open(icon_path) as icon:
        sizes = set(icon.ico.sizes())
    assert {(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}.issubset(sizes)
    with Image.open(splash_path) as splash:
        assert splash.size == (760, 420)
        assert splash.mode in {"RGB", "RGBA"}

def test_portable_builder_gui_uses_windows_theme_and_brand_icon():
    hta = (INSTALLER / "portable_build_gui.hta").read_text(encoding="utf-8-sig")
    assert 'icon="..\\app\\assets\\PolyfittyStudio.ico"' in hta
    assert "AppsUseLightTheme" in hta
    assert "SystemUsesLightTheme" in hta
    assert 'document.body.className = themeName === "light" ? "theme-light" : "theme-dark"' in hta
    assert "window.setInterval(function () { applyWindowsTheme(false); }, 1500)" in hta
    assert "body.theme-light" in hta

def test_installer_and_in_app_theme_palettes_share_polyfitty_brand_colors():
    themes = (APP / "themes.py").read_text(encoding="utf-8")
    manager = (INSTALLER / "install_manager.ps1").read_text(encoding="utf-8-sig")
    assert '"graphite"' in themes and '"label": "Studio Dark"' in themes
    assert '"light"' in themes and '"label": "Studio Light"' in themes
    for color in ("#080D19", "#111D31", "#32B8FF", "#E8EFF8", "#0C73B8"):
        assert color in themes.upper()
        assert color in manager.upper()

def test_in_app_system_mode_rechecks_windows_theme_while_running():
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "self._theme_timer.timeout.connect(self._refresh_system_theme)" in main
    assert "self._theme_timer.start(1500)" in main
    assert 'self.theme_combo.currentData() != "system"' in main
    assert "detected = detect_system_theme()" in main

def test_portable_guide_explains_self_contained_embedded_runtime():
    guide = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    assert "one file only" in guide and "`PolyfittyStudio.exe`" in guide
    assert "CPython 3.12.10" in guide
    assert "same application source" in guide
    assert "Local AppData cache" in guide
    assert "BUILD_PORTABLE_EXE.bat" in guide
    assert "GPU packages" in guide
    assert "CPU mode" in guide

def test_native_splash_client_updates_status_and_signals_only_when_ready(tmp_path, monkeypatch):
    import startup_splash
    client = startup_splash.StartupSplash()
    client._enabled = True
    client._status_path = str(tmp_path / "startup.status")
    client._ready_event = "Local\\PolyfittyStudio_Test"
    signals = []
    monkeypatch.setattr(client, "_signal_ready", lambda: signals.append(True))
    client.update("Preparing the 3D workspace...")
    assert (tmp_path / "startup.status").read_text(encoding="utf-8") == "Preparing the 3D workspace..."
    client.finish(type("Window", (), {"isVisible": lambda self: True})())
    assert (tmp_path / "startup.status").read_text(encoding="utf-8") == "Polyfitty Studio is ready."
    assert signals == [True]

def test_native_splash_client_reports_startup_failures_without_signaling_ready(tmp_path, monkeypatch):
    import startup_splash
    client = startup_splash.StartupSplash()
    client._enabled = True
    client._status_path = str(tmp_path / "startup.status")
    client._ready_event = "Local\\PolyfittyStudio_Test"
    signals = []
    monkeypatch.setattr(client, "_signal_ready", lambda: signals.append(True))
    client.fail("Application startup failed.")
    assert (tmp_path / "startup.status").read_text(encoding="utf-8") == "ERROR: Application startup failed."
    assert signals == []

def test_installed_and_portable_paths_share_the_gpu_preflight_implementation():
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    preflight = (APP / "startup_gpu_preflight.py").read_text(encoding="utf-8")
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "startup_gpu_probe = _run_startup_gpu_preflight(splash)" in entry
    assert "from startup_gpu_preflight import _run_startup_gpu_preflight" in entry
    assert "self._on_gpu_scan_completed(self._startup_gpu_probe)" in main
    assert "self._start_gpu_scan()" in main
    assert "compute_ready" in preflight

def test_gpu_preflight_failure_keeps_shared_runtime_in_cpu_mode():
    import startup_gpu_preflight
    messages = []

    class FakeSplash:
        def update(self, message):
            messages.append(str(message))

    def failed_probe(progress_callback=None):
        raise RuntimeError("driver runtime unavailable")

    result = startup_gpu_preflight._run_startup_gpu_preflight(
        FakeSplash(), gpu_probe=failed_probe, display_adapter_probe=lambda: []
    )
    assert result["compute_ready"] is False
    assert result["active_backend"] == "CPU"
    assert "driver runtime unavailable" in result["fallback_reason"]
    assert result["startup_splash_summary"] == "GPU check unavailable · CPU mode"

def test_runtime_contract_and_shared_entry_are_not_split_by_packaging_mode():
    contract = (APP / "runtime_contract.py").read_text(encoding="utf-8")
    entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert '"gpu_startup_preflight": "PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION"' in contract
    assert "verify_runtime_manifest(require_source_match=True)" in entry
    assert "if sys.frozen" not in main and "IS_FROZEN_PORTABLE" not in main
    assert "--worker" not in main

def test_single_file_extraction_uses_short_paths_and_reports_expanded_vs_compressed_size():
    launcher = (INSTALLER / "native_splash_launcher.cs").read_text(encoding="utf-8")
    builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
    guide = (ROOT / "README.md").read_text(encoding="utf-8-sig")

    assert 'Path.Combine(localAppData, "PF")' in launcher
    assert 'Path.Combine(productRoot, "P")' in launcher
    assert 'Path.Combine(productRoot, "T")' in launcher
    assert 'Path.Combine(stagingRootDirectory, Guid.NewGuid().ToString("N"))' in launcher
    assert "PathTooLongException" in launcher
    assert "$BuildWorkRoot = Join-Path $env:TEMP ('PF' + [Guid]::NewGuid().ToString('N'))" in builder
    assert "Get-DirectoryByteCount $Dist" in builder
    assert "Expanded embedded image:" in builder
    assert "Compressed EXE payload:" in builder
    assert "Final single-file EXE size:" in builder
    assert "%LOCALAPPDATA%\\PF\\P\\<payload-sha256>" in guide
