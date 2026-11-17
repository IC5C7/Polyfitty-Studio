param(
    [string]$StartupStatusFile = $env:POLYFITTY_STARTUP_STATUS_FILE,
    [string]$GpuStatePath = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $Root '.runtime\python312\python.exe'
$GpuPackagesRoot = Join-Path $Root '.cache\gpu_runtime\site-packages'
if (-not (Test-Path -LiteralPath $GpuPackagesRoot -PathType Container)) { New-Item -ItemType Directory -Force -Path $GpuPackagesRoot | Out-Null }
if ([string]::IsNullOrWhiteSpace($GpuStatePath)) { $GpuStatePath = Join-Path $Root '.cache\gpu_setup_state.json' }
$GpuSetupStateHelper = Join-Path $Root 'installer\gpu_setup_state.ps1'
if (Test-Path -LiteralPath $GpuSetupStateHelper -PathType Leaf) { . $GpuSetupStateHelper }
$script:SelectedBackend = 'CPU'
$script:BackendReady = $false
$script:GpuSetupSummary = 'GPU setup did not complete.'
$script:GpuSetupErrors = @()

function Say($Text, $Color = 'Cyan') {
    Write-Host "`n[Polyfitty Studio] $Text" -ForegroundColor $Color
}

$script:LastInstallProgress = -1
function Report-InstallProgress {
    param([int]$Percent, [string]$Detail)
    $Percent = [Math]::Max(0, [Math]::Min(100, $Percent))
    if ($Percent -lt $script:LastInstallProgress) { $Percent = $script:LastInstallProgress }
    if ($Percent -eq $script:LastInstallProgress -and $Detail -eq $script:LastProgressDetail) { return }
    $script:LastInstallProgress = $Percent
    $script:LastProgressDetail = $Detail
    Write-Host ("[PF_PROGRESS]|{0}|{1}" -f $Percent, ($Detail -replace '[\r\n|]', ' ')) -ForegroundColor DarkCyan
    if (Get-Command 'Write-PolyfittyStartupGpuStatus' -ErrorAction SilentlyContinue) {
        Write-PolyfittyStartupGpuStatus -StatusPath $StartupStatusFile -Percent $Percent -Detail $Detail
    }
}

function Install-GpuRequirements {
    param(
        [Parameter(Mandatory = $true)][string]$RequirementsFile,
        [Parameter(Mandatory = $true)][string]$BackendName,
        [int]$ProgressStart = 15,
        [int]$ProgressEnd = 85
    )
    $requirementsPath = Join-Path $Root $RequirementsFile
    if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
        Say "$BackendName requirements file is missing: $RequirementsFile" 'Yellow'
        return $false
    }

    Report-InstallProgress -Percent $ProgressStart -Detail ("Installing {0} compute runtime" -f $BackendName)
    Say "Installing $BackendName compute runtime..."
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'

        & $PythonExe -m pip install --disable-pip-version-check --upgrade --target $GpuPackagesRoot -r $requirementsPath 2>&1 | ForEach-Object { Write-Host $_ }
        $installExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($installExitCode -ne 0) {
        Say "$BackendName runtime installation failed (exit $installExitCode)." 'Yellow'
        Report-InstallProgress -Percent $ProgressEnd -Detail ("{0} runtime installation attempt finished" -f $BackendName)
        return $false
    }
    Say "$BackendName runtime package installation completed." 'Green'
    Report-InstallProgress -Percent $ProgressEnd -Detail ("{0} runtime installed" -f $BackendName)
    return $true
}

function Test-NvidiaCudaBackend {
    param([int]$ProgressPercent = 95)
    $diagnosticScript = Join-Path $Root 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath $diagnosticScript -PathType Leaf)) {
        Say 'NVIDIA CUDA self-test script is missing.' 'Yellow'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'NVIDIA CUDA self-test could not run'
        return $false
    }

    Say 'Running the NVIDIA CUDA allocation, arithmetic, and PolyFit ray/triangle kernel self-test...'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe $diagnosticScript --phase cuda 2>&1 | ForEach-Object { Write-Host $_ }
        $testExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($testExitCode -eq 0) {
        Say 'NVIDIA CUDA passed the real application kernel self-test.' 'Green'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'NVIDIA CUDA kernel self-test passed'
        return $true
    }
    Say "NVIDIA CUDA did not pass its self-test (exit $testExitCode)." 'Yellow'
    Report-InstallProgress -Percent $ProgressPercent -Detail 'NVIDIA CUDA self-test finished; evaluating fallback'
    return $false
}

function Test-SyclBackend {
    param([int]$ProgressPercent = 95)
    $checkScript = Join-Path $Root 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath $checkScript -PathType Leaf)) {
        Say 'Intel oneAPI/SYCL self-test script is missing.' 'Yellow'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'SYCL self-test could not run'
        return $false
    }

    Say 'Running the Intel oneAPI/SYCL device and PolyFit ray/triangle kernel self-test...'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe $checkScript --phase sycl 2>&1 | ForEach-Object { Write-Host $_ }
        $testExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($testExitCode -eq 0) {
        Say 'Intel oneAPI/SYCL passed the real application kernel self-test.' 'Green'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'Intel SYCL kernel self-test passed'
        return $true
    }
    Say "Intel oneAPI/SYCL did not pass its self-test (exit $testExitCode)." 'Yellow'
    Report-InstallProgress -Percent $ProgressPercent -Detail 'SYCL self-test finished; evaluating OpenCL fallback'
    return $false
}

function Test-HipBackend {
    param([int]$ProgressPercent = 92)
    $checkScript = Join-Path $Root 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath $checkScript -PathType Leaf)) {
        Say 'AMD HIP self-test script is missing.' 'Yellow'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'AMD HIP self-test could not run'
        return $false
    }
    Say 'AMD GPU detected. Checking the optional HIP SDK/native bridge before OpenCL fallback...'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe $checkScript --phase hip 2>&1 | ForEach-Object { Write-Host $_ }
        $testExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($testExitCode -eq 0) {
        Say 'AMD HIP passed the real application kernel self-test.' 'Green'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'AMD HIP kernel self-test passed'
        return $true
    }
    Say "AMD HIP is unavailable or did not pass its self-test (exit $testExitCode); OpenCL will be evaluated." 'Yellow'
    Report-InstallProgress -Percent $ProgressPercent -Detail 'AMD HIP self-test finished; evaluating OpenCL fallback'
    return $false
}

function Test-OpenClBackend {
    param([int]$ProgressPercent = 95)
    $checkScript = Join-Path $Root 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath $checkScript -PathType Leaf)) {
        Say 'OpenCL self-test script is missing.' 'Yellow'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'OpenCL self-test could not run'
        return $false
    }

    Say 'Running the OpenCL device and PolyFit ray/triangle kernel self-test...'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe $checkScript --phase opencl 2>&1 | ForEach-Object { Write-Host $_ }
        $testExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($testExitCode -eq 0) {
        Say 'OpenCL passed the real application kernel self-test.' 'Green'
        Report-InstallProgress -Percent $ProgressPercent -Detail 'OpenCL kernel self-test passed'
        return $true
    }
    Say "OpenCL did not pass its self-test (exit $testExitCode)." 'Yellow'
    Report-InstallProgress -Percent $ProgressPercent -Detail 'OpenCL kernel self-test finished'
    return $false
}

try {
    Report-InstallProgress -Percent 0 -Detail 'Detecting available GPU adapters'
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        Say 'Private Python runtime is missing. Run INSTALL.vbs; GPU setup is part of that installation sequence.' 'Yellow'
        Report-InstallProgress -Percent 100 -Detail 'Private Python runtime missing; run the main installer first'
        exit 0
    }

    Say 'Detecting physical graphics adapters after Core Runtime installation...'
    Say 'Automatic setup installs vendor-specific Python compute runtimes only; it does not replace Windows display drivers.' 'Gray'
    $adapters = @()
    try {
        $adapters = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | Where-Object {
            $_.Name -and $_.Name -notmatch '(?i)Microsoft Basic Display|Microsoft Remote Display|Remote Display Adapter'
        })
    }
    catch {
        Say "Windows graphics adapter query was unavailable: $($_.Exception.Message)" 'Yellow'
    }

    $hasNvidia = $false
    $hasAmd = $false
    $hasIntel = $false
    $hasOtherPciGpu = $false
    foreach ($adapter in $adapters) {
        $name = [string]$adapter.Name
        $manufacturer = [string]$adapter.AdapterCompatibility
        $pnpDeviceId = [string]$adapter.PNPDeviceID
        $signature = "$name $manufacturer $pnpDeviceId"
        Say "Detected display adapter: $name" 'Gray'

        if ($signature -match '(?i)VEN_10DE|NVIDIA') {
            $hasNvidia = $true
        }
        elseif ($signature -match '(?i)VEN_1002|Advanced Micro Devices|\bAMD\b|Radeon|ATI Technologies') {
            $hasAmd = $true
        }
        elseif ($signature -match '(?i)VEN_8086|\bIntel\b|Iris|UHD Graphics|Arc Graphics') {
            $hasIntel = $true
        }
        elseif ($pnpDeviceId -match '(?i)PCI\\VEN_[0-9A-F]{4}') {
            $hasOtherPciGpu = $true
        }
    }

    $nvidiaDriverDll = Join-Path $env:SystemRoot 'System32\nvcuda.dll'
    $hasNvidiaDriver = Test-Path -LiteralPath $nvidiaDriverDll -PathType Leaf
    if ($hasNvidia -and -not $hasNvidiaDriver) {
        Say 'NVIDIA GPU detected, but the NVIDIA display/compute driver (nvcuda.dll) is missing. CUDA packages will be deferred until the vendor driver is installed.' 'Yellow'
    }

    $nvidiaSmiPath = $null
    $nvidiaSmiCommand = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($nvidiaSmiCommand) {
        $nvidiaSmiPath = $nvidiaSmiCommand.Source
    }
    else {
        foreach ($programFilesRoot in @($env:ProgramFiles, $env:ProgramW6432)) {
            if (-not [string]::IsNullOrWhiteSpace($programFilesRoot)) {
                $candidate = Join-Path $programFilesRoot 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { $nvidiaSmiPath = $candidate; break }
            }
        }
    }

    if ($nvidiaSmiPath) {
        try {
            & $nvidiaSmiPath -L 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $hasNvidia = $true
                Say 'NVIDIA GPU confirmed by nvidia-smi.' 'Gray'
            }
        } catch { }
    }

    if ($hasNvidia -and $nvidiaSmiPath) {
        try {
            $driverVersionText = (& $nvidiaSmiPath --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
            if (-not [string]::IsNullOrWhiteSpace([string]$driverVersionText)) {
                $driverVersion = [version](([string]$driverVersionText).Trim())
                Say "NVIDIA display driver version: $driverVersion" 'Gray'
                if ($driverVersion -lt [version]'528.33') {
                    Say 'Driver is below the CUDA 12.x Windows minor-compatibility floor (528.33). Update the NVIDIA driver before expecting CUDA READY.' 'Yellow'
                }
                elseif ($driverVersion -lt [version]'560.94') {
                    Say 'Driver is below the CUDA 12.6 Update 2 driver level (560.94). NVRTC/PTX compatibility may be limited. Update the NVIDIA driver for this pinned runtime.' 'Yellow'
                }
            }
        } catch {
            Say "Could not read NVIDIA driver version: $($_.Exception.Message)" 'Yellow'
        }
    }

    Say 'GPU hardware summary:'
    Say "NVIDIA=$hasNvidia | AMD=$hasAmd | Intel=$hasIntel | Other PCI GPU=$hasOtherPciGpu" 'Gray'
    Report-InstallProgress -Percent 12 -Detail 'GPU adapter detection complete'

    $hasOpenClFallbackAdapter = $hasNvidia -or $hasAmd -or $hasIntel -or $hasOtherPciGpu
    $selectedBackend = 'CPU'
    $backendReady = $false
    $cudaInstalled = $false
    $openclInstalled = $false
    $syclInstalled = $false
    $hipReady = $false

    if ($hasIntel) {
        Say 'Intel GPU detected. Installing oneAPI/SYCL first, with OpenCL retained as a compatibility fallback.' 'Green'
        $syclInstalled = Install-GpuRequirements -RequirementsFile 'requirements_gpu_sycl.txt' -BackendName 'Intel oneAPI / SYCL (dpctl + dpnp)' -ProgressStart 15 -ProgressEnd 50
        if ($syclInstalled -and (Test-SyclBackend -ProgressPercent 58)) {
            $backendReady = $true
            $selectedBackend = 'SYCL'
        }
        elseif (-not $syclInstalled) {
            Say 'Intel oneAPI/SYCL packages could not be installed; OpenCL will be tested as a fallback.' 'Yellow'
        }
    }

    if ($hasNvidia -and $hasNvidiaDriver) {
        Say 'NVIDIA hardware and its nvcuda.dll driver were detected. Installing the CUDA 12.x user-space runtime.' 'Green'
        $cudaInstalled = Install-GpuRequirements -RequirementsFile 'requirements_gpu_cuda12.txt' -BackendName 'NVIDIA CUDA 12.x / CuPy' -ProgressStart 15 -ProgressEnd 70
        if ($cudaInstalled -and (Test-NvidiaCudaBackend -ProgressPercent 78)) {
            $backendReady = $true
            $selectedBackend = 'CUDA'
        }
        elseif (-not $cudaInstalled) {
            Say 'NVIDIA CUDA packages could not be installed; other detected GPU backends will still be tested.' 'Yellow'
        }
    }
    elseif ($hasNvidia) {
        Say 'NVIDIA hardware is present, but CUDA cannot work without the vendor driver. Install/update the NVIDIA Windows driver, then relaunch to trigger automatic CUDA setup.' 'Yellow'
    }

    if ($hasAmd) {
        $hipBridgeCandidate = Join-Path $Root 'amd_hip\bin\PolyfittyHipBridge.dll'
        if ((Test-Path -LiteralPath $hipBridgeCandidate -PathType Leaf) -or
            -not [string]::IsNullOrWhiteSpace($env:HIP_PATH) -or
            -not [string]::IsNullOrWhiteSpace($env:ROCM_PATH)) {
            $hipReady = Test-HipBackend -ProgressPercent 90
            if ($hipReady) {
                $backendReady = $true
                $selectedBackend = 'HIP'
            }
        }
    }

    if ($hasOpenClFallbackAdapter) {
        Say 'Installing the vendor-neutral OpenCL path so AMD / Intel / NVIDIA drivers can provide a fallback backend.' 'Green'
        $openclInstalled = Install-GpuRequirements -RequirementsFile 'requirements_gpu_opencl.txt' -BackendName 'OpenCL cross-vendor fallback' -ProgressStart 72 -ProgressEnd 90
        if ($openclInstalled -and (Test-OpenClBackend -ProgressPercent 96)) {
            $backendReady = $true
            if ($selectedBackend -eq 'CPU') { $selectedBackend = 'OPENCL' }
        }
    }

    if (-not ($hasNvidia -or $hasAmd -or $hasIntel -or $hasOtherPciGpu)) {
        Say 'No supported physical GPU adapter was detected. GPU packages will not be installed; CPU baking remains available.' 'Yellow'
        Report-InstallProgress -Percent 96 -Detail 'No supported GPU found; CPU mode is ready'
    }

    $finalColor = 'Yellow'
    if ($backendReady) { $finalColor = 'Green' }
    Say "Final installer backend decision: $selectedBackend | Kernel self-test passed=$backendReady" $finalColor
    Report-InstallProgress -Percent 100 -Detail ("GPU setup complete; selected backend: {0}" -f $selectedBackend)
    if ($backendReady) {
        Say "The selected compute backend is READY: $selectedBackend. No separate GPU installer is required." 'Green'
        $script:SelectedBackend = $selectedBackend
        $script:BackendReady = $true
        $script:GpuSetupSummary = "GPU compute backend $selectedBackend passed the real application-kernel self-test."
        if (Get-Command 'Save-PolyfittyGpuSetupState' -ErrorAction SilentlyContinue) {
            Save-PolyfittyGpuSetupState -Root $Root -StatePath $GpuStatePath -Status 'ready' -Backend $selectedBackend -Summary $script:GpuSetupSummary -RetryHours 24
        }
        Write-PolyfittyStartupGpuStatus -StatusPath $StartupStatusFile -Percent 100 -Detail $script:GpuSetupSummary
        exit 0
    }

    if ($hasNvidia -and -not $hasNvidiaDriver) {
        Say 'NVIDIA was detected without nvcuda.dll; CUDA runtime installation was deferred until the Windows driver is present.' 'Yellow'
    }
    elseif ($hasNvidia -and -not $cudaInstalled) {
        Say 'NVIDIA hardware was detected, but its CUDA runtime could not be installed.' 'Yellow'
    }
    elseif ($hasNvidia) {
        Say 'NVIDIA hardware was detected, but CUDA did not pass the application kernel self-test.' 'Yellow'
    }
    elseif ($hasAmd -or $hasIntel -or $hasOtherPciGpu) {
        Say 'No detected CUDA, AMD HIP, oneAPI/SYCL or OpenCL backend passed the application kernel self-test.' 'Yellow'
    }
    Say 'No supported GPU backend became READY. CPU baking remains available until a GPU backend passes its kernel self-test.' 'Yellow'
    $script:SelectedBackend = 'CPU'
    $script:BackendReady = $false
    $script:GpuSetupSummary = 'No detected GPU backend passed validation; CPU mode remains available.'
    $retryHours = if ($adapters.Count -eq 0) { 168 } else { 24 }
    if (Get-Command 'Save-PolyfittyGpuSetupState' -ErrorAction SilentlyContinue) {
        Save-PolyfittyGpuSetupState -Root $Root -StatePath $GpuStatePath -Status 'cpu_fallback' -Backend 'CPU' -Summary $script:GpuSetupSummary -RetryHours $retryHours -Errors @($script:GpuSetupErrors)
    }
    Write-PolyfittyStartupGpuStatus -StatusPath $StartupStatusFile -Percent 100 -Detail $script:GpuSetupSummary
    exit 0
}
catch {
    $errorMessage = $_.Exception.Message
    $script:GpuSetupErrors += $errorMessage
    Say "Automatic GPU setup encountered an optional error: $errorMessage" 'Yellow'
    Say 'Continuing with CPU baking; the main application installation is not blocked.' 'Yellow'
    Report-InstallProgress -Percent 100 -Detail 'GPU setup ended with a warning; CPU mode remains available'
    if (Get-Command 'Save-PolyfittyGpuSetupState' -ErrorAction SilentlyContinue) {
        try {
            Save-PolyfittyGpuSetupState -Root $Root -StatePath $GpuStatePath -Status 'failed' -Backend 'CPU' -Summary $errorMessage -RetryHours 6 -Errors @($script:GpuSetupErrors)
        }
        catch { Say "Could not persist GPU setup retry state: $($_.Exception.Message)" 'Yellow' }
    }
    Write-PolyfittyStartupGpuStatus -StatusPath $StartupStatusFile -Percent 100 -Detail 'GPU runtime setup could not be completed; CPU mode remains available'
    exit 0
}
