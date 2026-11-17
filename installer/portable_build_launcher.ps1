param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$StatusFile,
    [Parameter(Mandatory = $true)][string]$LogFile,
    [Parameter(Mandatory = $true)][string]$ConsoleLogFile,
    [Parameter(Mandatory = $true)][string]$CancelFile,
    [Parameter(Mandatory = $true)][string]$PidFile
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-LauncherLog([string]$Message) {
    try {
        $directory = Split-Path -Parent $LogFile
        if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
        Add-Content -LiteralPath $LogFile -Value ("[{0}] [launcher] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message) -Encoding Unicode
    }
    catch { }
}

function Write-LauncherStatus([string]$State, [int]$Percent, [string]$Stage, [string]$Message, [string]$ErrorText = '') {
    try {
        $directory = Split-Path -Parent $StatusFile
        if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
        $payload = [ordered]@{
            state = $State
            percent = [Math]::Max(0, [Math]::Min(100, $Percent))
            stage = $Stage
            message = $Message
            error = $ErrorText
            updated = (Get-Date).ToString('o')
        }
        $temporary = "$StatusFile.launcher.tmp"
        $payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $temporary -Encoding Unicode
        Move-Item -LiteralPath $temporary -Destination $StatusFile -Force
    }
    catch {
        Write-LauncherLog ("Could not write build status: {0}" -f $_.Exception.Message)
    }
}

try {

    $pidDirectory = Split-Path -Parent $PidFile
    if ($pidDirectory) { New-Item -ItemType Directory -Force -Path $pidDirectory | Out-Null }
    Set-Content -LiteralPath $PidFile -Value ([string]$PID) -Encoding Unicode

    if (-not (Test-Path -LiteralPath $BuildScript -PathType Leaf)) {
        throw "Portable build script was not found: $BuildScript"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -PathType Leaf)) {
        throw 'Windows PowerShell was not found in the expected system location.'
    }

    $consoleDirectory = Split-Path -Parent $ConsoleLogFile
    if ($consoleDirectory) { New-Item -ItemType Directory -Force -Path $consoleDirectory | Out-Null }
    Set-Content -LiteralPath $ConsoleLogFile -Value ("[{0}] PowerShell build launcher initialized." -f (Get-Date -Format 'o')) -Encoding Unicode
    Write-LauncherLog 'PowerShell build launcher initialized.'
    Write-LauncherStatus -State 'running' -Percent 0 -Stage 'PowerShell startup' -Message 'PowerShell launcher started; building an independent embedded portable runtime from this release source.'

    $buildParameters = @{
        StatusFile = $StatusFile
        LogFile = $LogFile
        CancelFile = $CancelFile
    }

    Write-LauncherLog ("Starting build script: {0}" -f $BuildScript)
    Write-LauncherStatus -State 'running' -Percent 0 -Stage 'Starting builder' -Message 'Build script is starting.'

    & $BuildScript @buildParameters *>> $ConsoleLogFile

    $finalStatus = $null
    if (Test-Path -LiteralPath $StatusFile -PathType Leaf) {
        try { $finalStatus = Get-Content -LiteralPath $StatusFile -Raw | ConvertFrom-Json } catch { }
    }
    if (-not $finalStatus -or $finalStatus.state -notin @('succeeded', 'failed', 'cancelled')) {
        throw 'The build script returned without writing a succeeded, failed, or cancelled status.'
    }
    if ($finalStatus.state -eq 'succeeded') {
        Write-LauncherLog 'Build script completed successfully.'
        exit 0
    }
    if ($finalStatus.state -eq 'cancelled') {
        Write-LauncherLog 'Build script reported cancellation.'
        exit 2
    }
    Write-LauncherLog ("Build script reported failure: {0}" -f $finalStatus.message)
    exit 1
}
catch {
    $message = if ($_.Exception) { $_.Exception.Message } else { [string]$_ }
    Write-LauncherLog ("Launcher failure: {0}" -f $message)
    try {
        Add-Content -LiteralPath $ConsoleLogFile -Value ("[{0}] Launcher failure: {1}" -f (Get-Date -Format 'o'), $message) -Encoding Unicode
    }
    catch { }
    Write-LauncherStatus -State 'failed' -Percent 0 -Stage 'PowerShell launch failed' -Message $message -ErrorText $message
    exit 1
}
