from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _manager() -> str:
    return (ROOT / "installer" / "install_manager.ps1").read_text(encoding="utf-8")

def _launcher() -> str:
    return (ROOT / "INSTALL.vbs").read_text(encoding="utf-16")

def test_install_vbs_detects_windows_app_theme_and_passes_it_to_manager():
    launcher = _launcher()
    detector = launcher[launcher.index("Function DetectWindowsAppTheme"): ]
    assert 'registryPath & "AppsUseLightTheme"' in detector
    assert 'registryPath & "SystemUsesLightTheme"' in detector
    assert detector.index('"AppsUseLightTheme"') < detector.index('"SystemUsesLightTheme"')
    assert 'initialTheme = DetectWindowsAppTheme(shell)' in launcher
    assert '" -InitialTheme "' in launcher

def test_setup_manager_uses_the_same_light_and_graphite_palette_as_main_ui():
    manager = _manager()
    assert "[ValidateSet('Light', 'Graphite')][string]$InitialTheme" in manager
    assert "@('AppsUseLightTheme', 'SystemUsesLightTheme')" in manager

    for color in (
        "Window = '#E8EFF8'", "Panel = '#FFFFFF'", "Text = '#172B45'",
        "Muted = '#60758F'", "Accent = '#0C73B8'",
        "Window = '#080D19'", "Panel = '#111D31'", "Text = '#EAF2FF'",
        "Muted = '#91A8C6'", "Accent = '#32B8FF'",
    ):
        assert color in manager

def test_installer_theme_is_applied_to_all_major_controls_and_semantic_statuses():
    manager = _manager()
    apply = manager[manager.index("function Apply-InstallerTheme"):manager.index("function Set-StageState")]
    for marker in (
        "$script:Form.BackColor", "$script:Header.BackColor", "$script:TitleLabel.ForeColor",
        "$script:SubtitleLabel.ForeColor", "$script:StagePanels.Values", "$script:StageTitleLabels",
        "$script:StageDetailLabels", "$script:StageLabels.Values", "$script:LogBox.BackColor",
        "$script:OpenLogButton", "$script:RetryButton", "$script:CloseButton",
    ):
        assert marker in apply
    assert "Resolve-InstallerStatusColor" in manager
    assert "elseif ($stateText -match 'Complete')" in apply
    assert "if ($stateText -match 'Failed')" in apply

def test_setup_manager_rechecks_windows_theme_while_the_window_is_open():
    manager = _manager()
    assert "$script:Timer.Interval = 150" in manager
    assert "-ge 1.5" in manager
    assert "$detectedTheme = Get-WindowsAppTheme" in manager
    assert "Apply-InstallerTheme -ThemeName $detectedTheme" in manager
    assert "[THEME] Applied Windows $ThemeName theme" in manager

def test_theme_detection_failure_uses_graphite_fallback():
    launcher = _launcher()
    manager = _manager()
    assert 'DetectWindowsAppTheme = "Graphite"' in launcher
    assert "if ($startupTheme -notin @('Light', 'Graphite')) { $startupTheme = 'Graphite' }" in manager
    assert "if ([string]::IsNullOrWhiteSpace($startupTheme)) { $startupTheme = $InitialTheme }" in manager

def test_installation_completion_automatically_closes_manager_and_hands_off_to_hidden_launcher():
    manager = _manager()
    launcher = _launcher()
    complete = manager[manager.index("function Complete-Installation"):manager.index("function Handle-StageExit")]
    timer = manager[manager.index("$script:Timer.Add_Tick"):manager.index("$script:Timer.Start")]
    closing = manager[manager.index("$form.Add_FormClosing"):manager.index("$form.Add_FormClosed")]

    assert "$script:AutoLaunchAt = (Get-Date).AddSeconds(2.5)" in complete
    assert "Polyfitty Studio will start automatically in a moment." in complete
    assert "Automatic application launch scheduled" in complete
    assert "$now -ge $script:AutoLaunchAt" in timer
    assert "$script:Form.Close()" in timer
    assert "[IO.File]::WriteAllText($script:LaunchFlagPath, 'launch'" in closing
    assert 'If exitCode = 0 And fso.FileExists(launchFlagPath) Then' in launcher
    assert 'fso.BuildPath(rootPath, "START_HIDDEN.vbs")' in launcher
    assert 'shell.Run commandLine, 0, False' in launcher
    assert 'Press Close to launch Polyfitty Studio.' not in complete

def test_theme_refresh_errors_are_isolated_from_installation_monitor():
    manager = _manager()
    timer = manager[manager.index("$script:Timer.Add_Tick"):manager.index("$script:Timer.Start")]
    assert "[THEME WARNING] Theme refresh failed; installation continues." in timer
    assert "Read-StageOutputFiles" in timer

def test_successful_installation_sets_launch_handoff_before_form_close():
    manager = _manager()
    complete = manager[manager.index("function Complete-Installation"):manager.index("function Handle-StageExit")]
    assert "[IO.File]::WriteAllText($script:LaunchFlagPath, 'launch'" in complete
    assert "[LAUNCH] Post-install launch handoff marker created." in complete

def test_startup_launcher_captures_python_errors_without_hiding_the_gui_window():
    startup = (ROOT / "installer" / "start_app.ps1").read_text(encoding="utf-8-sig")
    assert "-WindowStyle Hidden" in startup
    assert "$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Normal" in startup
    assert "$startInfo.FileName = $PythonExe" in startup
    assert "$startInfo.CreateNoWindow = $true" in startup
    assert "$env:POLYFITTY_STARTUP_STDOUT = $appStdout" in startup
    assert "$env:POLYFITTY_STARTUP_STDERR = $appStderr" in startup
    assert "$appProcess.WaitForExit(2200)" in startup
    assert "exited during startup" in startup
    assert "IsWindowVisible(IntPtr hWnd)" in startup and "EnumWindows" in startup

def test_installer_window_uses_the_branded_application_icon():
    manager = _manager()
    assert "$brandIconPath = Join-Path $script:Root 'app\\assets\\PolyfittyStudio.ico'" in manager
    assert "$form.Icon = [Drawing.Icon]::ExtractAssociatedIcon($brandIconPath)" in manager

def test_installed_app_creates_shortcuts_in_desktop_and_project_root():
    installer = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8-sig")
    assert "(Join-Path $desktop 'Polyfitty Studio.lnk')" in installer
    assert "(Join-Path $Root 'Polyfitty Studio.lnk')" in installer
    assert "$shortcut.TargetPath = $wscript" in installer
    assert "START_HIDDEN.vbs" in installer
    assert "$shortcut.IconLocation" in installer
