param(
    [Parameter(Mandatory = $true)][string]$RootPath,
    [switch]$PortablePackage
)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($RootPath)
$LogDirectory = Join-Path $Root '.cache\logs'
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory ("startup_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$PowerShellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Runtime = Join-Path $Root '.runtime\python312'
$PythonExe = Join-Path $Runtime 'python.exe'
$PythonPathFile = Join-Path $Runtime 'python312._pth'
$EnsurePip = Join-Path $Root 'installer\ensure_pip.ps1'
$GpuStateHelper = Join-Path $Root 'installer\gpu_setup_state.ps1'
$Cache = Join-Path $Root '.cache'
$WorkingDirectory = [IO.Path]::GetTempPath()
$env:PYTHONHOME = ''
$env:PYTHONPATH = ''
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

function Write-StartupLog {
    param([string]$Message)
    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding Unicode
}

try {
    [Environment]::CurrentDirectory = $WorkingDirectory
    Set-Location -LiteralPath $WorkingDirectory
    Write-StartupLog 'Polyfitty Studio startup helper initialized.'
    Write-StartupLog ("Project root: {0}" -f $Root)
    Write-StartupLog ("Working directory: {0}" -f $WorkingDirectory)

    foreach ($requiredPath in @($PythonExe, $PythonPathFile, $EnsurePip, $GpuStateHelper)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required startup file was not found: $requiredPath"
        }
    }

    . $GpuStateHelper
    Write-StartupLog 'Shared GPU setup state/status helper loaded.'

    $PortableManifestPath = Join-Path $Root 'portable_runtime_manifest.json'
    $HasPortableManifest = Test-Path -LiteralPath $PortableManifestPath -PathType Leaf
    $IsPortablePackage = [bool]$PortablePackage -or $HasPortableManifest
    Write-StartupLog ("Startup mode: PortablePackage switch={0}; portable manifest present={1}; classified portable={2}." -f [bool]$PortablePackage, $HasPortableManifest, $IsPortablePackage)
    if ($PortablePackage -and -not $HasPortableManifest) {
        throw 'The native single-file launcher did not extract portable_runtime_manifest.json. Re-run PolyfittyStudio.exe and check its Local AppData extraction permissions.'
    }
    if ($IsPortablePackage) {
        $portableIdentity = Get-Content -LiteralPath $PortableManifestPath -Raw | ConvertFrom-Json
        if ([string]$portableIdentity.package_kind -ne 'single_file_self_extracting_embedded_portable' -or -not [bool]$portableIdentity.embedded_python -or [string]$portableIdentity.deployment_model -ne 'single_exe_embedded_payload_extracted_to_local_app_data') {
            throw "The portable package identity is invalid or belongs to an unsupported package layout: $PortableManifestPath"
        }
        if ([string]$portableIdentity.interpreter -ne '.runtime/python312/python.exe') {
            throw 'The portable package manifest points to an unexpected Python interpreter path.'
        }
        $releaseVersionPath = Join-Path $Root 'VERSION.txt'
        $sourceManifestPath = Join-Path $Root 'runtime_manifest.json'
        if (-not (Test-Path -LiteralPath $releaseVersionPath -PathType Leaf) -or -not (Test-Path -LiteralPath $sourceManifestPath -PathType Leaf)) {
            throw 'The embedded single-file payload is missing its release version or application source manifest.'
        }
        $releaseVersion = (Get-Content -LiteralPath $releaseVersionPath -Raw).Trim()
        $sourceIdentity = Get-Content -LiteralPath $sourceManifestPath -Raw | ConvertFrom-Json
        if ([string]$portableIdentity.app_version -ne $releaseVersion -or [string]$sourceIdentity.app_version -ne $releaseVersion) {
            throw 'The embedded portable manifest, source manifest and release version do not identify the same build.'
        }
        if ([string]$portableIdentity.runtime_source_fingerprint -ne [string]$sourceIdentity.source_fingerprint) {
            throw 'The embedded portable runtime was packaged with a different application source fingerprint.'
        }

        $hasVcRuntime = (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140.dll')) -and
                        (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140_1.dll'))
        if (-not $hasVcRuntime) {
            $bundledVcRedist = Join-Path $Root 'prerequisites\vc_redist.x64.exe'
            if (-not (Test-Path -LiteralPath $bundledVcRedist -PathType Leaf)) {
                throw 'The single-file portable payload is missing prerequisites\vc_redist.x64.exe, required by its embedded native Python packages.'
            }
            $vcSignature = Get-AuthenticodeSignature -FilePath $bundledVcRedist
            if ($vcSignature.Status -ne 'Valid' -or $vcSignature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
                throw 'The bundled Microsoft Visual C++ prerequisite failed its Authenticode signature check.'
            }
            Write-StartupLog 'Microsoft Visual C++ runtime is missing; requesting the signed prerequisite installer on this recipient PC.'
            Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Detail 'Preparing Microsoft native runtime prerequisite...'
            $vcProcess = Start-Process -FilePath $bundledVcRedist -ArgumentList '/install /quiet /norestart' -Verb RunAs -Wait -PassThru -ErrorAction Stop
            if ($vcProcess.ExitCode -notin @(0, 1638, 3010)) {
                throw "Microsoft Visual C++ prerequisite installation failed (exit $($vcProcess.ExitCode))."
            }
            $hasVcRuntime = (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140.dll')) -and
                            (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140_1.dll'))
            if (-not $hasVcRuntime) {
                throw 'The Microsoft Visual C++ prerequisite completed but its required runtime DLLs are still unavailable. A Windows restart may be required.'
            }
            Write-StartupLog 'Microsoft Visual C++ runtime prerequisite is ready.'
        }
    }

    $pipStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $pipStdout = Join-Path $LogDirectory ("startup_pip_{0}_stdout.log" -f $pipStamp)
    $pipStderr = Join-Path $LogDirectory ("startup_pip_{0}_stderr.log" -f $pipStamp)
    $pipArguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -RuntimePath "{1}" -CachePath "{2}"' -f $EnsurePip, $Runtime, $Cache
    Write-StartupLog 'Checking the embedded/private Python package manager.'
    $pipProcess = Start-Process -FilePath $PowerShellPath -ArgumentList $pipArguments -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $pipStdout -RedirectStandardError $pipStderr -ErrorAction Stop
    if ($pipProcess.ExitCode -ne 0) {
        $stdoutText = if (Test-Path -LiteralPath $pipStdout) { Get-Content -LiteralPath $pipStdout -Raw } else { '' }
        $stderrText = if (Test-Path -LiteralPath $pipStderr) { Get-Content -LiteralPath $pipStderr -Raw } else { '' }
        throw ("Private Python package manager repair failed with exit code {0}.`r`nSTDOUT:`r`n{1}`r`nSTDERR:`r`n{2}`r`nLogs: {3}" -f $pipProcess.ExitCode, $stdoutText, $stderrText, $LogDirectory)
    }
    Write-StartupLog 'Private Python package manager is ready.'

    $pythonPathLines = @(Get-Content -LiteralPath $PythonPathFile -Encoding ASCII)
    if ($pythonPathLines -notcontains '..\..\app' -or $pythonPathLines -notcontains 'import site') {
        throw "The embedded Python runtime configuration is incomplete: $PythonPathFile must expose the package app directory and site-packages. Re-extract one complete Polyfitty Studio release package."
    }

    if ($IsPortablePackage) {

        $GpuSetupScript = Join-Path $Root 'installer\install_gpu.ps1'
        $GpuStatePath = Join-Path $Cache 'gpu_setup_state.json'
        if (Test-Path -LiteralPath $GpuSetupScript -PathType Leaf) {
            $gpuDecision = Get-PolyfittyGpuSetupStateDecision -Root $Root -StatePath $GpuStatePath
            if (-not $gpuDecision.ShouldRun) {
                Write-StartupLog ("Portable GPU setup skipped: {0}" -f $gpuDecision.Reason)
                if ($gpuDecision.State) {
                    Write-StartupLog ("Cached portable GPU status: {0}; backend: {1}; summary: {2}" -f $gpuDecision.State.status, $gpuDecision.State.selected_backend, $gpuDecision.State.summary)
                }
                Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Detail 'Portable GPU runtime already configured for this hardware. Loading Polyfitty Studio...'
            }
            else {
                Write-StartupLog ("Portable GPU setup required: {0}" -f $gpuDecision.Reason)
                Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Detail 'First launch on this PC · detecting GPU and preparing acceleration packages...'
                $gpuStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
                $gpuStdout = Join-Path $LogDirectory ("startup_gpu_{0}_stdout.log" -f $gpuStamp)
                $gpuStderr = Join-Path $LogDirectory ("startup_gpu_{0}_stderr.log" -f $gpuStamp)
                $gpuArguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -StartupStatusFile "{1}" -GpuStatePath "{2}"' -f $GpuSetupScript, $env:POLYFITTY_STARTUP_STATUS_FILE, $GpuStatePath
                Write-StartupLog 'Starting portable first-run GPU detection and matching user-space package installation.'
                Write-StartupLog ("GPU setup stdout log: {0}" -f $gpuStdout)
                Write-StartupLog ("GPU setup stderr log: {0}" -f $gpuStderr)
                try {
                    $gpuProcess = Start-Process -FilePath $PowerShellPath -ArgumentList $gpuArguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $gpuStdout -RedirectStandardError $gpuStderr -ErrorAction Stop
                    Write-StartupLog ("Portable GPU setup process finished with exit code {0}." -f $gpuProcess.ExitCode)
                }
                catch {
                    Write-StartupLog ("[WARN] Could not launch portable GPU setup: {0}" -f $_.Exception.Message)
                }
                foreach ($gpuOutputPath in @($gpuStdout, $gpuStderr)) {
                    if (Test-Path -LiteralPath $gpuOutputPath -PathType Leaf) {
                        Add-Content -LiteralPath $LogPath -Value (Get-Content -LiteralPath $gpuOutputPath -Tail 250 -ErrorAction SilentlyContinue) -Encoding Unicode
                    }
                }
                $gpuDecisionAfter = Get-PolyfittyGpuSetupStateDecision -Root $Root -StatePath $GpuStatePath
                if ($gpuDecisionAfter.State) {
                    Write-StartupLog ("Portable GPU setup result: {0}; backend: {1}; summary: {2}" -f $gpuDecisionAfter.State.status, $gpuDecisionAfter.State.selected_backend, $gpuDecisionAfter.State.summary)
                    if ([string]$gpuDecisionAfter.State.status -eq 'ready') {
                        Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Percent 100 -Detail ("{0} acceleration ready. Loading Studio..." -f $gpuDecisionAfter.State.selected_backend)
                    }
                    else {
                        Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Percent 100 -Detail 'GPU acceleration is unavailable on this PC. Studio will continue in CPU mode.'
                    }
                }
                else {
                    Write-StartupLog '[WARN] Portable GPU setup did not produce a valid state record; CPU fallback remains available.'
                    Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Percent 100 -Detail 'GPU setup could not be verified. Loading Studio in compatibility mode...'
                }
            }
        }
        else {
            Write-StartupLog '[WARN] Portable GPU setup helper is missing. Continuing to the app GPU probe; CPU fallback remains available.'
            Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Detail 'GPU setup helper unavailable. Loading Studio in compatibility mode...'
        }
    }
    else {
        Write-StartupLog 'Installed product detected: GPU packages are provisioned by the installer setup-manager stage; skipping portable first-run installation.'
        Write-PolyfittyStartupGpuStatus -StatusPath $env:POLYFITTY_STARTUP_STATUS_FILE -Detail 'Loading configured Polyfitty Studio runtime...'
    }

    $EntryScript = Join-Path $Root 'app\hidden_entry.py'
    if (-not (Test-Path -LiteralPath $EntryScript -PathType Leaf)) {
        throw "Hidden application entry point was not found: $EntryScript"
    }

    $appStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $appStdout = Join-Path $LogDirectory ("startup_app_{0}_stdout.log" -f $appStamp)
    $appStderr = Join-Path $LogDirectory ("startup_app_{0}_stderr.log" -f $appStamp)
    $appTrace = Join-Path $LogDirectory ("startup_app_{0}_trace.log" -f $appStamp)
    $env:POLYFITTY_STARTUP_TRACE = $appTrace
    $env:POLYFITTY_STARTUP_STDOUT = $appStdout
    $env:POLYFITTY_STARTUP_STDERR = $appStderr

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.Arguments = ('"{0}"' -f $EntryScript)
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Normal
    Write-StartupLog 'Launching python.exe with CREATE_NO_WINDOW; Qt main window remains normal/visible and worker interpreter path is preserved.'
    Write-StartupLog ("Application stdout log: {0}" -f $appStdout)
    Write-StartupLog ("Application stderr log: {0}" -f $appStderr)
    $appProcess = [System.Diagnostics.Process]::Start($startInfo)
    if ($null -eq $appProcess) {
        throw 'Windows did not create the Polyfitty Studio GUI process.'
    }
    Write-StartupLog ("Polyfitty Studio GUI process created. PID={0}" -f $appProcess.Id)

    if ($appProcess.WaitForExit(2200)) {
        $stdoutText = if (Test-Path -LiteralPath $appStdout) { Get-Content -LiteralPath $appStdout -Raw -ErrorAction SilentlyContinue } else { '' }
        $stderrText = if (Test-Path -LiteralPath $appStderr) { Get-Content -LiteralPath $appStderr -Raw -ErrorAction SilentlyContinue } else { '' }
        throw ("Polyfitty Studio exited during startup (exit code {0}).`r`nSTDOUT:`r`n{1}`r`nSTDERR:`r`n{2}`r`nLogs: {3}" -f $appProcess.ExitCode, $stdoutText, $stderrText, $LogDirectory)
    }

    if (-not ('PolyfittyNativeWindowProbe' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class PolyfittyNativeWindowProbe {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern IntPtr GetWindow(IntPtr hWnd, uint command);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hWnd, int command);
    [DllImport("user32.dll")] private static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern IntPtr GetForegroundWindow();

    private const uint GW_OWNER = 4;
    private const int SW_SHOW = 5;
    private const int SW_RESTORE = 9;

    public static IntPtr FindMainWindow(int targetPid, string titleFragment) {
        IntPtr found = IntPtr.Zero;
        EnumWindowsProc callback = delegate(IntPtr hWnd, IntPtr lParam) {
            uint pid;
            GetWindowThreadProcessId(hWnd, out pid);
            if (pid != (uint)targetPid || GetWindow(hWnd, GW_OWNER) != IntPtr.Zero) return true;
            StringBuilder title = new StringBuilder(512);
            GetWindowText(hWnd, title, title.Capacity);
            if (title.ToString().IndexOf(titleFragment, StringComparison.OrdinalIgnoreCase) < 0) return true;
            RECT rect;
            if (!GetWindowRect(hWnd, out rect)) return true;
            if ((rect.Right - rect.Left) < 500 || (rect.Bottom - rect.Top) < 300) return true;
            found = hWnd;
            return false;
        };
        EnumWindows(callback, IntPtr.Zero);
        GC.KeepAlive(callback);
        return found;
    }

    public static void RestoreAndActivate(IntPtr hWnd) {
        ShowWindow(hWnd, IsIconic(hWnd) ? SW_RESTORE : SW_SHOW);
        BringWindowToTop(hWnd);
        SetForegroundWindow(hWnd);
    }

    public static bool IsForeground(IntPtr hWnd) {
        return hWnd != IntPtr.Zero && GetForegroundWindow() == hWnd;
    }

    public static bool IsPresented(IntPtr hWnd) {
        if (hWnd == IntPtr.Zero || !IsWindowVisible(hWnd) || IsIconic(hWnd)) return false;
        RECT rect;
        if (!GetWindowRect(hWnd, out rect)) return false;
        return (rect.Right - rect.Left) >= 500 && (rect.Bottom - rect.Top) >= 300;
    }

    public static string Describe(IntPtr hWnd) {
        if (hWnd == IntPtr.Zero) return "HWND=0";
        StringBuilder title = new StringBuilder(512);
        GetWindowText(hWnd, title, title.Capacity);
        RECT rect;
        GetWindowRect(hWnd, out rect);
        return String.Format("HWND=0x{0:X}; title={1}; visible={2}; minimized={3}; rect={4},{5},{6},{7}; foreground={8}",
            hWnd.ToInt64(), title.ToString(), IsWindowVisible(hWnd), IsIconic(hWnd),
            rect.Left, rect.Top, rect.Right, rect.Bottom, GetForegroundWindow() == hWnd);
    }
}
'@
    }

    Write-StartupLog 'Waiting for a real Win32 Polyfitty Studio top-level window (not QWidget.isVisible).'
    $windowPresented = $false
    $windowHandle = [IntPtr]::Zero
    $stableVisibleSamples = 0
    $activationAttempts = 0
    $startupDeadline = (Get-Date).AddSeconds(40)
    while ((Get-Date) -lt $startupDeadline) {
        $appProcess.Refresh()
        if ($appProcess.HasExited) {
            $stdoutText = if (Test-Path -LiteralPath $appStdout) { Get-Content -LiteralPath $appStdout -Raw -ErrorAction SilentlyContinue } else { '' }
            $stderrText = if (Test-Path -LiteralPath $appStderr) { Get-Content -LiteralPath $appStderr -Raw -ErrorAction SilentlyContinue } else { '' }
            $traceText = if (Test-Path -LiteralPath $appTrace) { Get-Content -LiteralPath $appTrace -Raw -ErrorAction SilentlyContinue } else { '' }
            throw ("Polyfitty Studio exited before a native main window appeared (exit code {0}).`r`nTRACE:`r`n{1}`r`nSTDOUT:`r`n{2}`r`nSTDERR:`r`n{3}" -f $appProcess.ExitCode, $traceText, $stdoutText, $stderrText)
        }

        $windowHandle = [PolyfittyNativeWindowProbe]::FindMainWindow($appProcess.Id, 'Polyfitty Studio')
        if ($windowHandle -ne [IntPtr]::Zero) {

            if ($activationAttempts -eq 0 -or (-not [PolyfittyNativeWindowProbe]::IsForeground($windowHandle) -and $activationAttempts -lt 4)) {
                [PolyfittyNativeWindowProbe]::RestoreAndActivate($windowHandle)
                $activationAttempts++
                Write-StartupLog ("Requested restore/foreground for HWND 0x{0:X}; attempt {1}." -f $windowHandle.ToInt64(), $activationAttempts)
                Start-Sleep -Milliseconds 150
            }
            if ([PolyfittyNativeWindowProbe]::IsPresented($windowHandle)) {
                $stableVisibleSamples++
                if ($stableVisibleSamples -ge 3) {
                    $windowPresented = $true
                    break
                }
            } else {
                $stableVisibleSamples = 0
            }
        }
        Start-Sleep -Milliseconds 200
    }

    if (-not $windowPresented) {
        $traceText = if (Test-Path -LiteralPath $appTrace) { Get-Content -LiteralPath $appTrace -Raw -ErrorAction SilentlyContinue } else { '(no application startup trace was written)' }
        $stdoutText = if (Test-Path -LiteralPath $appStdout) { Get-Content -LiteralPath $appStdout -Raw -ErrorAction SilentlyContinue } else { '' }
        $stderrText = if (Test-Path -LiteralPath $appStderr) { Get-Content -LiteralPath $appStderr -Raw -ErrorAction SilentlyContinue } else { '' }
        $windowState = if ($windowHandle -ne [IntPtr]::Zero) { [PolyfittyNativeWindowProbe]::Describe($windowHandle) } else { 'No matching native top-level window found.' }
        throw ("Python is running but the Windows main window was not presented within 40 seconds.`r`nNATIVE WINDOW: {0}`r`nTRACE:`r`n{1}`r`nSTDOUT:`r`n{2}`r`nSTDERR:`r`n{3}`r`nLogs: {4}" -f $windowState, $traceText, $stdoutText, $stderrText, $LogDirectory)
    }

    Write-StartupLog ("NATIVE_MAIN_WINDOW_PRESENTED {0}" -f [PolyfittyNativeWindowProbe]::Describe($windowHandle))
    Write-StartupLog 'START_HIDDEN.vbs handoff completed: the real Windows main window is visible and restored.'
    exit 0
}
catch {
    $detail = $_ | Out-String
    try { Write-StartupLog ("[ERROR] {0}" -f $detail) } catch { }
    try { Add-Content -LiteralPath (Join-Path $LogDirectory 'startup_launcher.log') -Value ("[{0}] [ERROR] {1}" -f (Get-Date -Format 'HH:mm:ss'), $detail) -Encoding Unicode } catch { }
    exit 1
}
