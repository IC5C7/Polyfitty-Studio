param(
    [Parameter(Mandatory = $true)][string]$RuntimePath,
    [Parameter(Mandatory = $true)][string]$CachePath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RuntimePath = [IO.Path]::GetFullPath($RuntimePath)
$CachePath = [IO.Path]::GetFullPath($CachePath)
$PythonExe = Join-Path $RuntimePath 'python.exe'
$SitePackages = Join-Path $RuntimePath 'Lib\site-packages'
$GetPip = Join-Path $CachePath 'get-pip.py'
$GetPipUrl = 'https://bootstrap.pypa.io/get-pip.py'
$script:LastPipProbeOutput = ''

function Test-PipModule {

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $probeOutput = @(& $PythonExe -c 'import pip; print(pip.__version__)' 2>&1)
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($probeExitCode -eq 0) {
        $script:LastPipProbeOutput = ''
        return $true
    }
    $script:LastPipProbeOutput = (($probeOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
    return $false
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Private Python executable was not found: $PythonExe"
    }
    New-Item -ItemType Directory -Force -Path $CachePath, $SitePackages | Out-Null

    if (Test-PipModule) {
        Write-Host '[Polyfitty Studio] Private Python pip module: OK' -ForegroundColor Green
        exit 0
    }

    Write-Host '[Polyfitty Studio] pip is missing or damaged; restoring it...' -ForegroundColor Yellow
    if ($script:LastPipProbeOutput) {
        Write-Host '[Polyfitty Studio] pip import diagnostic:' -ForegroundColor DarkYellow
        Write-Host $script:LastPipProbeOutput -ForegroundColor DarkGray
    }

    $stalePip = Join-Path $SitePackages 'pip'
    if (Test-Path -LiteralPath $stalePip) {
        Remove-Item -LiteralPath $stalePip -Recurse -Force
    }
    Get-ChildItem -LiteralPath $SitePackages -Directory -Filter 'pip-*.dist-info' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force

    if (-not (Test-Path -LiteralPath $GetPip -PathType Leaf) -or (Get-Item -LiteralPath $GetPip).Length -lt 50000) {
        if (Test-Path -LiteralPath $GetPip) {
            Remove-Item -LiteralPath $GetPip -Force
        }
        Write-Host '[Polyfitty Studio] Downloading official pip bootstrap...' -ForegroundColor Cyan
        Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPip -UseBasicParsing
    }

    if (-not (Test-Path -LiteralPath $GetPip -PathType Leaf) -or (Get-Item -LiteralPath $GetPip).Length -lt 50000) {
        throw 'The pip bootstrap download is missing or incomplete.'
    }

    Write-Host '[Polyfitty Studio] Installing pip into the private Python runtime...' -ForegroundColor Cyan
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe $GetPip --disable-pip-version-check --no-warn-script-location 2>&1 |
            ForEach-Object { Write-Host $_ }
        $bootstrapExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($bootstrapExitCode -ne 0) {
        throw "pip bootstrap exited with code $bootstrapExitCode. See the Python error output above."
    }

    if (-not (Test-PipModule)) {
        if ($script:LastPipProbeOutput) { Write-Host $script:LastPipProbeOutput -ForegroundColor DarkGray }
        throw 'The bootstrap process ended, but `import pip` still fails in the private Python runtime.'
    }

    Write-Host '[Polyfitty Studio] Private Python pip module restored successfully.' -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "[Polyfitty Studio] pip check/repair failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
