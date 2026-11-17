

function Get-PolyfittyGpuHardwareSnapshot {
    param([Parameter(Mandatory = $true)][string]$Root)

    $adapters = @()
    $available = $true
    try {
        $adapters = @(
            Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
                Where-Object {
                    $_.Name -and $_.Name -notmatch '(?i)Microsoft Basic Display|Microsoft Remote Display|Remote Display Adapter'
                } |
                ForEach-Object {
                    [ordered]@{
                        name = ([string]$_.Name).Trim()
                        manufacturer = ([string]$_.AdapterCompatibility).Trim()
                        pnp_device_id = ([string]$_.PNPDeviceID).Trim()
                        driver_version = ([string]$_.DriverVersion).Trim()
                    }
                } |
                Sort-Object -Property pnp_device_id, name
        )
    }
    catch {
        $available = $false
        $adapters = @()
    }

    $json = ConvertTo-Json -InputObject @($adapters) -Depth 6 -Compress
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($json)
        $fingerprint = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally { $sha.Dispose() }

    return [pscustomobject]@{
        Available = $available
        Fingerprint = $fingerprint
        AdapterCount = $adapters.Count
        Adapters = @($adapters)
        Json = $json
    }
}

function Get-PolyfittyGpuRequirementsFingerprint {
    param([Parameter(Mandatory = $true)][string]$Root)

    $relativePaths = @(
        'installer/gpu_setup_state.ps1',
        'installer/install_gpu.ps1',
        'installer/verify_runtime.py',
        'app/gpu_runtime_paths.py',
        'app/gpu_compute.py',
        'requirements_gpu_cuda12.txt',
        'requirements_gpu_opencl.txt',
        'requirements_gpu_sycl.txt'
    )
    $builder = New-Object System.Text.StringBuilder
    foreach ($relative in $relativePaths) {
        $fullPath = Join-Path $Root ($relative.Replace('/', '\\'))
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            [void]$builder.Append($relative).Append([char]0).Append('missing').Append("`n")
            continue
        }
        $hash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
        [void]$builder.Append($relative).Append([char]0).Append($hash).Append("`n")
    }

    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($builder.ToString())
        return [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Test-PolyfittyGpuOverlayForBackend {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Backend
    )

    $site = Join-Path $Root '.cache\gpu_runtime\site-packages'
    switch ($Backend.ToUpperInvariant()) {
        'CUDA' {
            return (Test-Path -LiteralPath (Join-Path $site 'cupy\__init__.py') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $site 'nvidia\cuda_runtime\bin') -PathType Container)
        }
        'SYCL' {
            return (Test-Path -LiteralPath (Join-Path $site 'dpctl\__init__.py') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $site 'dpnp\__init__.py') -PathType Leaf)
        }
        'OPENCL' {
            return (Test-Path -LiteralPath (Join-Path $site 'pyopencl\__init__.py') -PathType Leaf)
        }
        'HIP' { return $true }
        default { return $false }
    }
}

function Get-PolyfittyGpuSetupStateDecision {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$StatePath
    )

    $snapshot = Get-PolyfittyGpuHardwareSnapshot -Root $Root
    $requirementsFingerprint = Get-PolyfittyGpuRequirementsFingerprint -Root $Root
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
        return [pscustomobject]@{ ShouldRun = $true; Reason = 'First launch on this extracted runtime.'; State = $null; Hardware = $snapshot }
    }

    try {
        $state = Get-Content -LiteralPath $StatePath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        return [pscustomobject]@{ ShouldRun = $true; Reason = 'The previous GPU setup state is unreadable and will be rebuilt.'; State = $null; Hardware = $snapshot }
    }

    if ([int]$state.schema_version -ne 1) {
        return [pscustomobject]@{ ShouldRun = $true; Reason = 'GPU setup state schema changed.'; State = $state; Hardware = $snapshot }
    }
    if ([string]$state.hardware_fingerprint -ne $snapshot.Fingerprint) {
        return [pscustomobject]@{ ShouldRun = $true; Reason = 'The detected GPU adapter or driver configuration changed.'; State = $state; Hardware = $snapshot }
    }
    if ([string]$state.requirements_fingerprint -ne $requirementsFingerprint) {
        return [pscustomobject]@{ ShouldRun = $true; Reason = 'GPU runtime requirements or installer logic changed.'; State = $state; Hardware = $snapshot }
    }

    $status = [string]$state.status
    if ($status -eq 'ready') {
        $backend = [string]$state.selected_backend
        if (-not (Test-PolyfittyGpuOverlayForBackend -Root $Root -Backend $backend)) {
            return [pscustomobject]@{ ShouldRun = $true; Reason = ('The cached {0} backend is missing files from the per-machine GPU package overlay.' -f $backend); State = $state; Hardware = $snapshot }
        }
        return [pscustomobject]@{ ShouldRun = $false; Reason = 'GPU runtime was installed and kernel-verified for this hardware.'; State = $state; Hardware = $snapshot }
    }

    $retryAfter = [DateTimeOffset]::MinValue
    try { $retryAfter = [DateTimeOffset]::Parse([string]$state.retry_after_utc).ToUniversalTime() } catch { }
    if ($retryAfter -gt [DateTimeOffset]::UtcNow) {
        return [pscustomobject]@{ ShouldRun = $false; Reason = ('Previous optional GPU setup ended in {0}; retry is scheduled after {1}.' -f $status, $retryAfter.ToString('u')); State = $state; Hardware = $snapshot }
    }

    return [pscustomobject]@{ ShouldRun = $true; Reason = 'The previous GPU setup fallback retry window has expired.'; State = $state; Hardware = $snapshot }
}

function Save-PolyfittyGpuSetupState {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$StatePath,
        [Parameter(Mandatory = $true)][ValidateSet('ready', 'cpu_fallback', 'failed')][string]$Status,
        [string]$Backend = 'CPU',
        [string]$Summary = '',
        [int]$RetryHours = 24,
        [string[]]$Errors = @()
    )

    $snapshot = Get-PolyfittyGpuHardwareSnapshot -Root $Root
    $requirementsFingerprint = Get-PolyfittyGpuRequirementsFingerprint -Root $Root
    $now = [DateTimeOffset]::UtcNow
    $retryAfter = if ($Status -eq 'ready') { [DateTimeOffset]::MaxValue } else { $now.AddHours([Math]::Max(1, $RetryHours)) }
    $payload = [ordered]@{
        schema_version = 1
        status = $Status
        selected_backend = $Backend
        summary = $Summary
        attempted_utc = $now.ToString('o')
        retry_after_utc = $retryAfter.ToString('o')
        hardware_fingerprint = $snapshot.Fingerprint
        hardware_inventory_available = $snapshot.Available
        hardware_adapter_count = $snapshot.AdapterCount
        hardware_adapters = @($snapshot.Adapters)
        requirements_fingerprint = $requirementsFingerprint
        errors = @($Errors)
    }

    $directory = Split-Path -Parent $StatePath
    if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    $temporary = "$StatePath.tmp"
    $json = $payload | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Write-PolyfittyStartupGpuStatus {
    param(
        [string]$StatusPath = $env:POLYFITTY_STARTUP_STATUS_FILE,
        [int]$Percent = -1,
        [string]$Detail = ''
    )
    if ([string]::IsNullOrWhiteSpace($StatusPath)) { return }
    try {
        $directory = Split-Path -Parent $StatusPath
        if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
        $safeDetail = ($Detail -replace '[\r\n]', ' ').Trim()
        $message = if ($Percent -ge 0) { ('GPU setup {0}% · {1}' -f [Math]::Max(0, [Math]::Min(100, $Percent)), $safeDetail) } else { $safeDetail }
        $temporary = "$StatusPath.tmp"
        [IO.File]::WriteAllText($temporary, $message, (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
    }
    catch { }
}
