
param(
    [string]$LogPath = '',
    [ValidateSet('Light', 'Graphite')][string]$InitialTheme = 'Graphite'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$script:Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$script:LogDirectory = Join-Path $script:Root '.cache\logs'
$script:LaunchFlagPath = Join-Path $script:Root '.cache\launch_after_setup.flag'
New-Item -ItemType Directory -Force -Path $script:LogDirectory | Out-Null
if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $LogPath = Join-Path $script:LogDirectory ("install_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$script:LogPath = [IO.Path]::GetFullPath($LogPath)

function Add-ManagerLog {
    param([AllowEmptyString()][string]$Message)

    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message
    try { Add-Content -LiteralPath $script:LogPath -Value $line -Encoding Unicode } catch { }
    if ($script:LogBox -and -not $script:LogBox.IsDisposed) {
        try {
            $script:LogBox.AppendText($line + [Environment]::NewLine)
            if ($script:LogBox.TextLength -gt 90000) {
                $script:LogBox.Text = $script:LogBox.Text.Substring($script:LogBox.TextLength - 70000)
            }
            $script:LogBox.SelectionStart = $script:LogBox.TextLength
            $script:LogBox.ScrollToCaret()
        } catch { }
    }
}

Add-ManagerLog '[BOOT] Setup manager process entered.'
Add-ManagerLog ("[BOOT] PowerShell version: {0}; process architecture: {1}-bit" -f $PSVersionTable.PSVersion, ([IntPtr]::Size * 8))
Add-ManagerLog ("[BOOT] Manager script: {0}" -f $PSCommandPath)
Add-ManagerLog ("[BOOT] Installation root: {0}" -f $script:Root)
Add-ManagerLog ("[BOOT] Initial theme hint from INSTALL.vbs: {0}" -f $InitialTheme)

try {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
    Add-Type -AssemblyName System.Drawing -ErrorAction Stop
    Add-ManagerLog '[BOOT] Windows Forms assemblies loaded.'
    [System.Windows.Forms.Application]::SetUnhandledExceptionMode([System.Windows.Forms.UnhandledExceptionMode]::CatchException)
    [System.Windows.Forms.Application]::EnableVisualStyles()
    Add-ManagerLog '[BOOT] Windows Forms exception mode configured.'
}
catch {
    $startupError = $_ | Out-String
    Add-ManagerLog "[FATAL] Windows Forms startup failed: $startupError"
    try {
        [System.Windows.Forms.MessageBox]::Show("Unable to initialize the setup manager.`r`n`r`n$startupError`r`n`r`nLog: $script:LogPath", 'Polyfitty Studio - Startup error') | Out-Null
    } catch { }
    exit 2
}
$script:ActiveProcess = $null
$script:ActiveStage = ''
$script:OutputQueue = [System.Collections.Concurrent.ConcurrentQueue[string]]::new()
$script:StageOutputFiles = @{
    stdout = @{ Path = ''; Offset = [int64]0; Remainder = ''; Encoding = [Text.Encoding]::Default; PrefixChecked = $false }
    stderr = @{ Path = ''; Offset = [int64]0; Remainder = ''; Encoding = [Text.Encoding]::Default; PrefixChecked = $false }
}
$script:Timer = $null
$script:Form = $null
$script:LogBox = $null
$script:StatusLabel = $null
$script:DetailLabel = $null
$script:ProgressBar = $null
$script:PercentLabel = $null
$script:CurrentOverallPercent = 0
$script:RetryButton = $null
$script:CloseButton = $null
$script:StageLabels = @{}
$script:StagePanels = @{}
$script:GpuWarning = $false
$script:InstallationFlowActive = $true
$script:CleanupInProgress = $false
$script:ManagerFatalError = $false
$script:CanLaunchApp = $false
$script:AutoLaunchAt = $null
$script:CurrentTheme = ''
$script:ThemePalette = $null
$script:StatusSemanticColor = '#E8EDF5'
$script:LastThemeCheck = [DateTime]::MinValue
$script:Header = $null
$script:TitleLabel = $null
$script:SubtitleLabel = $null
$script:StatusTitleLabel = $null
$script:LogTitleLabel = $null
$script:OpenLogButton = $null
$script:StageTitleLabels = @()
$script:StageDetailLabels = @()

function Get-WindowsAppTheme {

    $personalizeKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize'
    try {
        $settings = Get-ItemProperty -LiteralPath $personalizeKey -ErrorAction Stop
        foreach ($valueName in @('AppsUseLightTheme', 'SystemUsesLightTheme')) {
            $property = $settings.PSObject.Properties[$valueName]
            if ($null -eq $property) { continue }
            try {
                if ([int]$property.Value -eq 1) { return 'Light' }
                return 'Graphite'
            }
            catch { continue }
        }
    }
    catch { }
    return $null
}

function Get-InstallerThemePalette {
    param([ValidateSet('Light', 'Graphite')][string]$ThemeName)

    if ($ThemeName -eq 'Light') {
        return @{
            Name = 'Studio Light'; Window = '#E8EFF8'; Base = '#F3F7FC'; Panel = '#FFFFFF'
            PanelAlt = '#EAF1FA'; Border = '#C8D8EA'; Text = '#172B45'; Muted = '#60758F'
            Accent = '#0C73B8'; AccentHover = '#095F9F'; Button = '#E3EDF8'; ButtonHover = '#D2E4F5'
            Log = '#F6F9FD'; LogText = '#172B45'; Success = '#087F68'; Warning = '#94600A'
            Error = '#B4233A'; Info = '#096FAF'
        }
    }
    return @{
        Name = 'Studio Dark'; Window = '#080D19'; Base = '#0C1423'; Panel = '#111D31'
        PanelAlt = '#182941'; Border = '#29415F'; Text = '#EAF2FF'; Muted = '#91A8C6'
        Accent = '#32B8FF'; AccentHover = '#69CCFF'; Button = '#1A304B'; ButtonHover = '#254463'
        Log = '#080D19'; LogText = '#EAF2FF'; Success = '#53D6B0'; Warning = '#FFC56E'
        Error = '#FF858F'; Info = '#73C9FF'
    }
}

function ConvertTo-InstallerColor {
    param([Parameter(Mandatory = $true)][string]$Hex)
    return [Drawing.ColorTranslator]::FromHtml($Hex)
}

function Resolve-InstallerStatusColor {
    param([string]$RequestedColor = '')

    $palette = $script:ThemePalette
    if ($null -eq $palette) { return [Drawing.Color]::DimGray }
    switch ($RequestedColor.ToUpperInvariant()) {
        '#FF7777' { return (ConvertTo-InstallerColor $palette.Error) }
        '#78D6A3' { return (ConvertTo-InstallerColor $palette.Success) }
        '#E8B45F' { return (ConvertTo-InstallerColor $palette.Warning) }
        '#82C7FF' { return (ConvertTo-InstallerColor $palette.Info) }
        '#AAB4C4' { return (ConvertTo-InstallerColor $palette.Muted) }
        '#E8EDF5' { return (ConvertTo-InstallerColor $palette.Text) }
        default { return (ConvertTo-InstallerColor $RequestedColor) }
    }
}

function Apply-InstallerTheme {
    param([ValidateSet('Light', 'Graphite')][string]$ThemeName)

    if ($script:CurrentTheme -eq $ThemeName -or $null -eq $script:Form -or $script:Form.IsDisposed) { return }
    $palette = Get-InstallerThemePalette -ThemeName $ThemeName
    $script:ThemePalette = $palette

    $script:Form.BackColor = ConvertTo-InstallerColor $palette.Base
    $script:Form.ForeColor = ConvertTo-InstallerColor $palette.Text
    if ($script:Header) { $script:Header.BackColor = ConvertTo-InstallerColor $palette.Panel }
    if ($script:TitleLabel) { $script:TitleLabel.ForeColor = ConvertTo-InstallerColor $palette.Text }
    if ($script:SubtitleLabel) { $script:SubtitleLabel.ForeColor = ConvertTo-InstallerColor $palette.Muted }

    foreach ($panel in $script:StagePanels.Values) {
        if ($panel -and -not $panel.IsDisposed) {
            $panel.BackColor = ConvertTo-InstallerColor $palette.Panel
            $panel.ForeColor = ConvertTo-InstallerColor $palette.Text
        }
    }
    foreach ($label in $script:StageTitleLabels) {
        if ($label -and -not $label.IsDisposed) { $label.ForeColor = ConvertTo-InstallerColor $palette.Text }
    }
    foreach ($label in $script:StageDetailLabels) {
        if ($label -and -not $label.IsDisposed) { $label.ForeColor = ConvertTo-InstallerColor $palette.Muted }
    }
    foreach ($label in $script:StageLabels.Values) {
        if ($null -eq $label -or $label.IsDisposed) { continue }
        $stateText = [string]$label.Text
        if ($stateText -match 'Failed') { $color = $palette.Error }
        elseif ($stateText -match 'Complete') { $color = $palette.Success }
        elseif ($stateText -match 'Warning|CPU mode available') { $color = $palette.Warning }
        elseif ($stateText -match 'In progress|Retrying') { $color = $palette.Info }
        else { $color = $palette.Muted }
        $label.ForeColor = ConvertTo-InstallerColor $color
    }

    if ($script:StatusLabel) { $script:StatusLabel.ForeColor = Resolve-InstallerStatusColor -RequestedColor $script:StatusSemanticColor }
    if ($script:DetailLabel) { $script:DetailLabel.ForeColor = ConvertTo-InstallerColor $palette.Muted }
    if ($script:PercentLabel) { $script:PercentLabel.ForeColor = ConvertTo-InstallerColor $palette.Accent }
    if ($script:ProgressBar) { $script:ProgressBar.ForeColor = ConvertTo-InstallerColor $palette.Accent }
    if ($script:LogTitleLabel) { $script:LogTitleLabel.ForeColor = ConvertTo-InstallerColor $palette.Text }
    if ($script:LogBox) {
        $script:LogBox.BackColor = ConvertTo-InstallerColor $palette.Log
        $script:LogBox.ForeColor = ConvertTo-InstallerColor $palette.LogText
    }

    foreach ($button in @($script:OpenLogButton, $script:RetryButton, $script:CloseButton)) {
        if ($null -eq $button -or $button.IsDisposed) { continue }
        $button.FlatStyle = [Windows.Forms.FlatStyle]::Flat
        $button.UseVisualStyleBackColor = $false
        $button.FlatAppearance.BorderColor = ConvertTo-InstallerColor $palette.Border
        $button.FlatAppearance.MouseOverBackColor = ConvertTo-InstallerColor $palette.ButtonHover
        $button.FlatAppearance.MouseDownBackColor = ConvertTo-InstallerColor $palette.AccentHover
        $button.BackColor = ConvertTo-InstallerColor $palette.Button
        $button.ForeColor = ConvertTo-InstallerColor $palette.Text
    }
    if ($script:RetryButton) {
        $script:RetryButton.BackColor = ConvertTo-InstallerColor $palette.Accent
        if ($ThemeName -eq 'Light') { $retryTextColor = [Drawing.Color]::White } else { $retryTextColor = ConvertTo-InstallerColor $palette.Window }
        $script:RetryButton.ForeColor = $retryTextColor
        $script:RetryButton.FlatAppearance.BorderColor = ConvertTo-InstallerColor $palette.AccentHover
    }
    if ($script:CloseButton) {
        $script:CloseButton.BackColor = ConvertTo-InstallerColor $palette.PanelAlt
    }

    $script:CurrentTheme = $ThemeName
    Add-ManagerLog "[THEME] Applied Windows $ThemeName theme to the setup manager."
}

function Set-StageState {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('core', 'gpu', 'launch')][string]$Stage,
        [Parameter(Mandatory = $true)][string]$State,
        [string]$Color = '#AAB4C4'
    )
    $script:StageLabels[$Stage].Text = $State
    $script:StageLabels[$Stage].ForeColor = Resolve-InstallerStatusColor -RequestedColor $Color
}

function Set-ManagerStatus {
    param([string]$Title, [string]$Detail, [string]$Color = '#E8EDF5')
    $script:StatusLabel.Text = $Title
    $script:StatusSemanticColor = $Color
    $script:StatusLabel.ForeColor = Resolve-InstallerStatusColor -RequestedColor $Color
    $script:DetailLabel.Text = $Detail
}

function Set-OverallProgress {
    param([int]$Percent, [string]$Detail = '')
    $Percent = [Math]::Max(0, [Math]::Min(100, $Percent))
    if ($Percent -lt $script:CurrentOverallPercent) { $Percent = $script:CurrentOverallPercent }
    $script:CurrentOverallPercent = $Percent
    if ($script:ProgressBar -and -not $script:ProgressBar.IsDisposed) {
        $script:ProgressBar.Style = [Windows.Forms.ProgressBarStyle]::Blocks
        $script:ProgressBar.MarqueeAnimationSpeed = 0
        $script:ProgressBar.Value = $Percent
    }
    if ($script:PercentLabel -and -not $script:PercentLabel.IsDisposed) {
        $script:PercentLabel.Text = ('{0}%' -f $Percent)
    }
    if (-not [string]::IsNullOrWhiteSpace($Detail) -and $script:DetailLabel -and -not $script:DetailLabel.IsDisposed) {
        $script:DetailLabel.Text = $Detail
    }
}

function Update-StageProgress {
    param([ValidateSet('core', 'gpu')][string]$Stage, [int]$StagePercent, [string]$Detail = '')
    $StagePercent = [Math]::Max(0, [Math]::Min(100, $StagePercent))
    if ($Stage -eq 'core') {
        $overall = [int][Math]::Round($StagePercent * 0.85, 0, [MidpointRounding]::AwayFromZero)
    }
    else {
        $overall = 85 + [int][Math]::Round($StagePercent * 0.15, 0, [MidpointRounding]::AwayFromZero)
    }
    Set-OverallProgress -Percent $overall -Detail $Detail
}

function Set-RunningUi {
    $script:ProgressBar.Style = [Windows.Forms.ProgressBarStyle]::Blocks
    $script:ProgressBar.MarqueeAnimationSpeed = 0
    $script:RetryButton.Visible = $false
    $script:CloseButton.Enabled = $false
}

function Set-FinishedUi {
    param([switch]$Completed)
    if ($script:Form) { $script:Form.TopMost = $false }
    $script:ProgressBar.Style = [Windows.Forms.ProgressBarStyle]::Blocks
    $script:ProgressBar.MarqueeAnimationSpeed = 0
    if ($Completed) { Set-OverallProgress -Percent 100 -Detail 'Installation tasks completed.' }
    $script:CloseButton.Enabled = $true
}

function Stop-ActiveInstallProcessTree {
    if ($script:CleanupInProgress) { return }
    $script:CleanupInProgress = $true
    try {
        $process = $script:ActiveProcess
        if ($null -eq $process) { return }
        $workerPid = 0
        try { $workerPid = $process.Id } catch { }
        if ($workerPid -le 0) { return }

        $isRunning = $false
        try { $isRunning = -not $process.HasExited } catch { $isRunning = $true }
        if (-not $isRunning) { return }

        Add-ManagerLog "[SAFETY] The installer manager exited unexpectedly; stopping the installation process tree. PID=$workerPid"
        $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        if (Test-Path -LiteralPath $taskkill -PathType Leaf) {
            & $taskkill /PID $workerPid /T /F 2>&1 | Out-Null
        }
        try {
            if (-not $process.HasExited) { $process.Kill() }
            $process.WaitForExit(5000) | Out-Null
        }
        catch { }
        Add-ManagerLog '[SAFETY] The child installation process cleanup request completed.'
    }
    catch {
        try { Add-ManagerLog "[SAFETY] Additional process cleanup error: $($_.Exception.Message)" } catch { }
    }
    finally {
        $script:CleanupInProgress = $false
    }
}

function Start-InstallStage {
    param([Parameter(Mandatory = $true)][ValidateSet('core', 'gpu')][string]$Stage)

    $relativeScript = if ($Stage -eq 'core') { 'installer\install.ps1' } else { 'installer\install_gpu.ps1' }
    $stageScript = Join-Path $script:Root $relativeScript
    if (-not (Test-Path -LiteralPath $stageScript -PathType Leaf)) {
        Add-ManagerLog "[ERROR] Installer script not found: $relativeScript"
        if ($Stage -eq 'core') {
            $script:InstallationFlowActive = $false
            Set-StageState -Stage core -State 'Failed' -Color '#FF7777'
            Set-ManagerStatus 'The installer script could not be found.' $relativeScript '#FF7777'
            Set-FinishedUi
            $script:RetryButton.Visible = $true
        }
        else {
            $script:GpuWarning = $true
            Set-StageState -Stage gpu -State 'Skipped' -Color '#E8B45F'
            Complete-Installation
        }
        return
    }

    $script:ActiveStage = $Stage
    $stageTitle = if ($Stage -eq 'core') { 'Core runtime installation' } else { 'GPU compute setup' }
    if ($Stage -eq 'core' -and $script:CurrentOverallPercent -eq 0) { Set-OverallProgress -Percent 0 -Detail 'Preparing Core Runtime installation.' }
    if ($Stage -eq 'gpu') { Set-OverallProgress -Percent 85 -Detail 'Core Runtime complete - preparing GPU setup.' }
    Set-ManagerStatus $stageTitle 'Installation is in progress. Current tasks and download activity are shown in the log below.' '#82C7FF'
    Set-RunningUi
    if ($Stage -eq 'core') {
        Set-StageState -Stage core -State 'In progress' -Color '#82C7FF'
    }
    else {
        Set-StageState -Stage gpu -State 'In progress' -Color '#82C7FF'
    }
    Add-ManagerLog "========== $stageTitle started =========="

    $powerShellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $outputStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdoutPath = Join-Path $script:LogDirectory ("stage_{0}_{1}_stdout.log" -f $Stage, $outputStamp)
    $stderrPath = Join-Path $script:LogDirectory ("stage_{0}_{1}_stderr.log" -f $Stage, $outputStamp)
    $script:StageOutputFiles.stdout = @{ Path = $stdoutPath; Offset = [int64]0; Remainder = ''; Encoding = [Text.Encoding]::Default; PrefixChecked = $false }
    $script:StageOutputFiles.stderr = @{ Path = $stderrPath; Offset = [int64]0; Remainder = ''; Encoding = [Text.Encoding]::Default; PrefixChecked = $false }
    $arguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $stageScript
    $process = $null

    try {
        $process = Start-Process -FilePath $powerShellPath -ArgumentList $arguments -WorkingDirectory $script:Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -ErrorAction Stop
        if ($null -eq $process) { throw 'PowerShell did not return an installation process handle.' }
        $script:ActiveProcess = $process
        Add-ManagerLog "Running: $relativeScript"
        Add-ManagerLog "[BOOT] Child process PID: $($process.Id)"
        Add-ManagerLog "[BOOT] Child stdout: $stdoutPath"
        Add-ManagerLog "[BOOT] Child stderr: $stderrPath"
    }
    catch {
        Add-ManagerLog "[ERROR] Unable to start $relativeScript`: $($_.Exception.Message)"
        if ($_.ScriptStackTrace) { Add-ManagerLog "[ERROR] Startup stack: $($_.ScriptStackTrace)" }
        if ($process) { try { $process.Dispose() } catch { } }
        $script:ActiveProcess = $null
        if ($Stage -eq 'core') {
            $script:InstallationFlowActive = $false
            Set-StageState core 'Failed' '#FF7777'
            Set-ManagerStatus 'Unable to start core installation.' 'Review the log and try again.' '#FF7777'
            Set-FinishedUi
            $script:RetryButton.Visible = $true
        }
        else {
            $script:GpuWarning = $true
            Set-StageState gpu 'Warning - CPU mode available' '#E8B45F'
            Complete-Installation
        }
    }
}

function Read-StageOutputFile {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('stdout', 'stderr')][string]$StreamName,
        [switch]$FlushRemainder
    )

    $state = $script:StageOutputFiles[$StreamName]
    $path = $state.Path
    if ([string]::IsNullOrWhiteSpace($path) -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return }

    $file = $null
    try {
        $file = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        if ($file.Length -lt $state.Offset) {
            $state.Offset = [int64]0
            $state.Remainder = ''
            $state.PrefixChecked = $false
            $state.Encoding = [Text.Encoding]::Default
        }
        if ($file.Length -le $state.Offset) {
            if ($FlushRemainder -and $state.Remainder.Length -gt 0) {
                $prefix = if ($StreamName -eq 'stderr') { '[stderr] ' } else { '' }
                $script:OutputQueue.Enqueue($prefix + $state.Remainder)
                $state.Remainder = ''
            }
            return
        }

        $file.Position = $state.Offset
        $count = [int]($file.Length - $file.Position)
        $bytes = New-Object byte[] $count
        $read = 0
        while ($read -lt $count) {
            $n = $file.Read($bytes, $read, $count - $read)
            if ($n -le 0) { break }
            $read += $n
        }
        $state.Offset = $file.Position
        $skip = 0
        if (-not $state.PrefixChecked) {
            $state.PrefixChecked = $true
            if ($read -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
                $state.Encoding = [Text.Encoding]::Unicode
                $skip = 2
            }
            elseif ($read -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
                $state.Encoding = [Text.Encoding]::UTF8
                $skip = 3
            }
            elseif ($read -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
                $state.Encoding = [Text.Encoding]::BigEndianUnicode
                $skip = 2
            }
        }

        $text = $state.Remainder + $state.Encoding.GetString($bytes, $skip, [Math]::Max(0, $read - $skip))
        $parts = [regex]::Split($text, "`r`n|`n|`r")
        $hasTrailingBreak = $text.EndsWith("`n") -or $text.EndsWith("`r")
        if ($hasTrailingBreak) {
            $state.Remainder = ''
            $lineCount = $parts.Length - 1
        }
        else {
            $state.Remainder = $parts[$parts.Length - 1]
            $lineCount = $parts.Length - 1
        }
        $prefix = if ($StreamName -eq 'stderr') { '[stderr] ' } else { '' }
        for ($i = 0; $i -lt $lineCount; $i++) {
            if ($parts[$i].Length -gt 0) { $script:OutputQueue.Enqueue($prefix + $parts[$i]) }
        }
        if ($FlushRemainder -and $state.Remainder.Length -gt 0) {
            $script:OutputQueue.Enqueue($prefix + $state.Remainder)
            $state.Remainder = ''
        }
    }
    catch {
        Add-ManagerLog "[ERROR] Unable to read child $StreamName output: $($_.Exception.Message)"
    }
    finally {
        if ($file) { $file.Dispose() }
    }
}

function Read-StageOutputFiles {
    param([switch]$FlushRemainder)
    Read-StageOutputFile -StreamName stdout -FlushRemainder:$FlushRemainder
    Read-StageOutputFile -StreamName stderr -FlushRemainder:$FlushRemainder
}

function Drain-OutputQueue {
    $entry = $null
    while ($script:OutputQueue.TryDequeue([ref]$entry)) {
        $message = [string]$entry
        if ($message -match '^\[PF_PROGRESS\]\|(\d{1,3})\|(.*)$') {
            $stagePercent = [int]$Matches[1]
            $progressDetail = $Matches[2].Trim()
            if ($script:ActiveStage -eq 'core' -or $script:ActiveStage -eq 'gpu') {
                Update-StageProgress -Stage $script:ActiveStage -StagePercent $stagePercent -Detail $progressDetail
                Add-ManagerLog ('[PROGRESS] {0}% - {1}' -f $script:CurrentOverallPercent, $progressDetail)
            }
            else {
                Add-ManagerLog $progressDetail
            }
            $entry = $null
            continue
        }
        if ($script:ActiveStage -eq 'gpu' -and $message -match '(?i)No supported physical GPU adapter|No supported GPU backend became READY|Automatic GPU setup encountered an optional error|GPU runtime scan returned exit') {
            $script:GpuWarning = $true
        }
        Add-ManagerLog $message
        if ($script:ActiveStage -eq 'core' -or $script:ActiveStage -eq 'gpu') {
            if ($message -match '(?i)Downloading (.+?)(?: via| from| \(attempt|\.\.\.)') {
                $itemName = $Matches[1].Trim()
                if ($itemName.Length -gt 82) { $itemName = $itemName.Substring(0, 79) + '...' }
                Set-ManagerStatus 'Downloading required files' $itemName '#82C7FF'
            }
            elseif ($message -match '(?i)Installing (.+?)(?:\.\.\.|$)') {
                $itemName = $Matches[1].Trim()
                if ($itemName.Length -gt 82) { $itemName = $itemName.Substring(0, 79) + '...' }
                Set-ManagerStatus 'Installing components' $itemName '#82C7FF'
            }
            elseif ($message -match '(?i)Extracting (.+?)(?:\.\.\.|$)') {
                $itemName = $Matches[1].Trim()
                Set-ManagerStatus 'Extracting runtime' $itemName '#82C7FF'
            }
            elseif ($message -match '(?i)Download verified: (.+)$') {
                Set-ManagerStatus 'Download verification complete' $Matches[1].Trim() '#78D6A3'
            }
        }
        $entry = $null
    }
}

function Complete-Installation {
    $script:InstallationFlowActive = $false
    $script:CanLaunchApp = $true
    $script:AutoLaunchAt = (Get-Date).AddSeconds(2.5)
    Set-FinishedUi -Completed
    $script:RetryButton.Visible = $false
    $script:CloseButton.Text = 'Launch now'
    if ($script:GpuWarning) {
        Set-ManagerStatus 'Installation complete - CPU baking is available' 'Polyfitty Studio will start automatically in a moment.' '#E8B45F'
    }
    else {
        Set-ManagerStatus 'Polyfitty Studio installation is complete.' 'Polyfitty Studio will start automatically in a moment.' '#78D6A3'
    }

    try {
        [IO.File]::WriteAllText($script:LaunchFlagPath, 'launch', [Text.Encoding]::ASCII)
        Add-ManagerLog '[LAUNCH] Post-install launch handoff marker created.'
    }
    catch {
        Add-ManagerLog "[LAUNCH WARNING] Could not create the initial launch marker: $($_.Exception.Message)"
    }
    Add-ManagerLog '========== Installer manager finished =========='
    Add-ManagerLog 'Automatic application launch scheduled. The VBS launcher will start START_HIDDEN.vbs when this manager closes.'
}

function Handle-StageExit {
    param([string]$Stage, [int]$ExitCode)
    $script:ActiveProcess = $null
    if ($Stage -eq 'core') {
        if ($ExitCode -ne 0) {
            $script:InstallationFlowActive = $false
            Set-StageState -Stage core -State "Failed (code $ExitCode)" -Color '#FF7777'
            Set-ManagerStatus 'Core installation failed.' 'Review the log and select [Retry]. Existing installation files will be preserved.' '#FF7777'
            Set-FinishedUi
            $script:RetryButton.Visible = $true
            Add-ManagerLog "[ERROR] Core installation process exited with code $ExitCode."
            Add-ManagerLog ("[ERROR] Captured stdout: {0}" -f $script:StageOutputFiles.stdout.Path)
            Add-ManagerLog ("[ERROR] Captured stderr: {0}" -f $script:StageOutputFiles.stderr.Path)
            Add-ManagerLog "[ERROR] Review the child output above for the first failing command and exception."
            return
        }
        Set-StageState -Stage core -State 'Complete' -Color '#78D6A3'
        Add-ManagerLog 'Core runtime installation completed successfully. Starting automatic GPU vendor detection and matching runtime setup.'
        Start-InstallStage -Stage gpu
        return
    }

    if ($ExitCode -eq 0 -and -not $script:GpuWarning) {
        Set-StageState -Stage gpu -State 'Complete' -Color '#78D6A3'
        Add-ManagerLog 'The automatic GPU setup script completed successfully.'
    }
    elseif ($ExitCode -eq 0) {
        Set-StageState -Stage gpu -State 'CPU mode available' -Color '#E8B45F'
        Add-ManagerLog 'No usable GPU device or runtime was found. CPU baking will remain available.'
    }
    else {
        $script:GpuWarning = $true
        Set-StageState -Stage gpu -State 'Warning - CPU mode available' -Color '#E8B45F'
        Add-ManagerLog "[WARNING] GPU setup exited with code $ExitCode. Continuing with CPU baking."
    }
    Complete-Installation
}

$form = [Windows.Forms.Form]::new()
$script:Form = $form
$form.Text = 'Polyfitty Studio | Setup Manager'
$form.ClientSize = [Drawing.Size]::new(900, 700)
$form.BackColor = [Drawing.Color]::FromArgb(18, 20, 22)
$form.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$form.Font = [Drawing.Font]::new('Segoe UI', 9)
$form.FormBorderStyle = [Windows.Forms.FormBorderStyle]::FixedDialog
$form.MaximizeBox = $false
$form.ShowInTaskbar = $true
$form.TopMost = $true
$form.StartPosition = [Windows.Forms.FormStartPosition]::CenterScreen
$brandIconPath = Join-Path $script:Root 'app\assets\PolyfittyStudio.ico'
$brandImagePath = Join-Path $script:Root 'app\assets\PolyfittyStudio.png'
if (Test-Path -LiteralPath $brandIconPath -PathType Leaf) {
    try { $form.Icon = [Drawing.Icon]::ExtractAssociatedIcon($brandIconPath) } catch { Add-ManagerLog "[THEME WARNING] Brand icon could not be loaded: $($_.Exception.Message)" }
}

$header = [Windows.Forms.Panel]::new()
$header.Location = [Drawing.Point]::new(0, 0)
$header.Size = [Drawing.Size]::new(900, 112)
$header.BackColor = [Drawing.Color]::FromArgb(35, 39, 45)
$form.Controls.Add($header)
$script:Header = $header
if (Test-Path -LiteralPath $brandImagePath -PathType Leaf) {
    try {
        $brandMark = [Windows.Forms.PictureBox]::new()
        $brandMark.Location = [Drawing.Point]::new(20, 18)
        $brandMark.Size = [Drawing.Size]::new(76, 76)
        $brandMark.SizeMode = [Windows.Forms.PictureBoxSizeMode]::Zoom
        $brandMark.Image = [Drawing.Image]::FromFile($brandImagePath)
        $header.Controls.Add($brandMark)
    }
    catch { Add-ManagerLog "[THEME WARNING] Brand mark could not be loaded: $($_.Exception.Message)" }
}

$title = [Windows.Forms.Label]::new()
$title.Text = 'POLYFITTY STUDIO'
$title.Font = [Drawing.Font]::new('Segoe UI Semibold', 20, [Drawing.FontStyle]::Bold)
$title.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$title.AutoSize = $true
$title.Location = [Drawing.Point]::new(110, 20)
$header.Controls.Add($title)
$script:TitleLabel = $title

$subtitle = [Windows.Forms.Label]::new()
$subtitle.Text = 'Setup manager - Monitor download and installation progress in this window.'
$subtitle.Font = [Drawing.Font]::new('Segoe UI', 9)
$subtitle.ForeColor = [Drawing.Color]::FromArgb(160, 168, 178)
$subtitle.AutoSize = $true
$subtitle.Location = [Drawing.Point]::new(112, 65)
$header.Controls.Add($subtitle)
$script:SubtitleLabel = $subtitle

$cardSpecs = @(
    @{ Key = 'core'; X = 24; Title = '01  CORE RUNTIME'; Detail = 'Python / GUI / Geometry' },
    @{ Key = 'gpu'; X = 318; Title = '02  GPU COMPUTE'; Detail = 'Auto-detect vendor and test matching runtime' },
    @{ Key = 'launch'; X = 612; Title = '03  LAUNCH'; Detail = 'Launch Polyfitty Studio' }
)
foreach ($spec in $cardSpecs) {
    $panel = [Windows.Forms.Panel]::new()
    $panel.Location = [Drawing.Point]::new($spec.X, 132)
    $panel.Size = [Drawing.Size]::new(264, 92)
    $panel.BackColor = [Drawing.Color]::FromArgb(35, 39, 45)
    $panel.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
    $form.Controls.Add($panel)
    $script:StagePanels[$spec.Key] = $panel

    $stageTitle = [Windows.Forms.Label]::new()
    $stageTitle.Text = $spec.Title
    $stageTitle.Location = [Drawing.Point]::new(14, 14)
    $stageTitle.AutoSize = $true
    $stageTitle.Font = [Drawing.Font]::new('Segoe UI Semibold', 9, [Drawing.FontStyle]::Bold)
    $stageTitle.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
    $panel.Controls.Add($stageTitle)
    $script:StageTitleLabels += $stageTitle

    $stageDetail = [Windows.Forms.Label]::new()
    $stageDetail.Text = $spec.Detail
    $stageDetail.Location = [Drawing.Point]::new(14, 39)
    $stageDetail.AutoSize = $true
    $stageDetail.Font = [Drawing.Font]::new('Segoe UI', 8)
    $stageDetail.ForeColor = [Drawing.Color]::FromArgb(160, 168, 178)
    $panel.Controls.Add($stageDetail)
    $script:StageDetailLabels += $stageDetail

    $state = [Windows.Forms.Label]::new()
    $state.Text = 'Pending'
    $state.Location = [Drawing.Point]::new(14, 64)
    $state.AutoSize = $true
    $state.Font = [Drawing.Font]::new('Segoe UI Semibold', 9, [Drawing.FontStyle]::Bold)
    $state.ForeColor = [Drawing.Color]::FromArgb(160, 168, 178)
    $panel.Controls.Add($state)
    $script:StageLabels[$spec.Key] = $state
}

$script:StatusLabel = [Windows.Forms.Label]::new()
$script:StatusLabel.Text = 'Preparing setup manager...'
$script:StatusLabel.Location = [Drawing.Point]::new(27, 246)
$script:StatusLabel.Size = [Drawing.Size]::new(840, 28)
$script:StatusLabel.Font = [Drawing.Font]::new('Segoe UI Semibold', 13, [Drawing.FontStyle]::Bold)
$form.Controls.Add($script:StatusLabel)

$script:DetailLabel = [Windows.Forms.Label]::new()
$script:DetailLabel.Text = 'Checking installation files and runtime requirements.'
$script:DetailLabel.Location = [Drawing.Point]::new(29, 278)
$script:DetailLabel.Size = [Drawing.Size]::new(840, 22)
$script:DetailLabel.ForeColor = [Drawing.Color]::FromArgb(160, 168, 178)
$form.Controls.Add($script:DetailLabel)

$script:ProgressBar = [Windows.Forms.ProgressBar]::new()
$script:ProgressBar.Location = [Drawing.Point]::new(28, 307)
$script:ProgressBar.Size = [Drawing.Size]::new(790, 16)
$script:ProgressBar.Minimum = 0
$script:ProgressBar.Maximum = 100
$script:ProgressBar.Value = 0
$script:ProgressBar.Style = [Windows.Forms.ProgressBarStyle]::Blocks
$script:ProgressBar.MarqueeAnimationSpeed = 0
$form.Controls.Add($script:ProgressBar)

$script:PercentLabel = [Windows.Forms.Label]::new()
$script:PercentLabel.Text = '0%'
$script:PercentLabel.Location = [Drawing.Point]::new(825, 303)
$script:PercentLabel.Size = [Drawing.Size]::new(48, 22)
$script:PercentLabel.TextAlign = [Drawing.ContentAlignment]::MiddleRight
$script:PercentLabel.Font = [Drawing.Font]::new('Segoe UI Semibold', 10, [Drawing.FontStyle]::Bold)
$script:PercentLabel.ForeColor = [Drawing.Color]::FromArgb(174, 184, 196)
$form.Controls.Add($script:PercentLabel)

$logTitle = [Windows.Forms.Label]::new()
$logTitle.Text = 'Installation log'
$logTitle.Location = [Drawing.Point]::new(28, 340)
$logTitle.AutoSize = $true
$logTitle.Font = [Drawing.Font]::new('Segoe UI Semibold', 9, [Drawing.FontStyle]::Bold)
$logTitle.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$form.Controls.Add($logTitle)
$script:LogTitleLabel = $logTitle

$script:LogBox = [Windows.Forms.TextBox]::new()
$script:LogBox.Location = [Drawing.Point]::new(28, 366)
$script:LogBox.Size = [Drawing.Size]::new(844, 238)
$script:LogBox.Multiline = $true
$script:LogBox.ReadOnly = $true
$script:LogBox.ScrollBars = [Windows.Forms.ScrollBars]::Vertical
$script:LogBox.WordWrap = $false
$script:LogBox.BackColor = [Drawing.Color]::FromArgb(18, 20, 22)
$script:LogBox.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$script:LogBox.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
$script:LogBox.Font = [Drawing.Font]::new('Consolas', 8.5)
$script:LogBox.MaxLength = 100000
$form.Controls.Add($script:LogBox)

$openLogButton = [Windows.Forms.Button]::new()
$openLogButton.Text = 'Open log folder'
$openLogButton.Location = [Drawing.Point]::new(545, 624)
$openLogButton.Size = [Drawing.Size]::new(112, 34)
$openLogButton.FlatStyle = [Windows.Forms.FlatStyle]::Flat
$openLogButton.FlatAppearance.BorderColor = [Drawing.Color]::FromArgb(68, 83, 104)
$openLogButton.BackColor = [Drawing.Color]::FromArgb(52, 58, 66)
$openLogButton.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$openLogButton.Add_Click({ Start-Process explorer.exe -ArgumentList ('"{0}"' -f $script:LogDirectory) })
$form.Controls.Add($openLogButton)
$script:OpenLogButton = $openLogButton

$script:RetryButton = [Windows.Forms.Button]::new()
$script:RetryButton.Text = 'Retry'
$script:RetryButton.Location = [Drawing.Point]::new(668, 624)
$script:RetryButton.Size = [Drawing.Size]::new(92, 34)
$script:RetryButton.FlatStyle = [Windows.Forms.FlatStyle]::Flat
$script:RetryButton.FlatAppearance.BorderColor = [Drawing.Color]::FromArgb(53, 128, 181)
$script:RetryButton.BackColor = [Drawing.Color]::FromArgb(174, 184, 196)
$script:RetryButton.ForeColor = [Drawing.Color]::FromArgb(18, 20, 22)
$script:RetryButton.Visible = $false
$script:RetryButton.Add_Click({
    $script:InstallationFlowActive = $true
    $script:CanLaunchApp = $false
    $script:AutoLaunchAt = $null
    $script:CloseButton.Text = 'Close'
    $script:GpuWarning = $false
    $script:CurrentOverallPercent = 0
    Set-OverallProgress -Percent 0 -Detail 'Restarting Core Runtime installation.'
    Set-StageState -Stage core -State 'Retrying' -Color '#82C7FF'
    Start-InstallStage -Stage core
})
$form.Controls.Add($script:RetryButton)

$script:CloseButton = [Windows.Forms.Button]::new()
$script:CloseButton.Text = 'Close'
$script:CloseButton.Location = [Drawing.Point]::new(772, 624)
$script:CloseButton.Size = [Drawing.Size]::new(100, 34)
$script:CloseButton.FlatStyle = [Windows.Forms.FlatStyle]::Flat
$script:CloseButton.FlatAppearance.BorderColor = [Drawing.Color]::FromArgb(68, 83, 104)
$script:CloseButton.BackColor = [Drawing.Color]::FromArgb(52, 58, 66)
$script:CloseButton.ForeColor = [Drawing.Color]::FromArgb(228, 231, 235)
$script:CloseButton.Enabled = $false
$script:CloseButton.Add_Click({ $script:Form.Close() })
$form.Controls.Add($script:CloseButton)

$startupTheme = Get-WindowsAppTheme
if ([string]::IsNullOrWhiteSpace($startupTheme)) { $startupTheme = $InitialTheme }
if ($startupTheme -notin @('Light', 'Graphite')) { $startupTheme = 'Graphite' }
try {
    Apply-InstallerTheme -ThemeName $startupTheme
}
catch {
    $script:CurrentTheme = 'Graphite'
    Add-ManagerLog "[THEME WARNING] Initial theme application failed; keeping safe default styling. $($_ | Out-String)"
}
$script:LastThemeCheck = Get-Date

$script:Timer = [Windows.Forms.Timer]::new()
$script:Timer.Interval = 150
$script:Timer.Add_Tick({
    try {
        $now = Get-Date
        if (($now - $script:LastThemeCheck).TotalSeconds -ge 1.5) {
            $script:LastThemeCheck = $now

            try {
                $detectedTheme = Get-WindowsAppTheme
                if (-not [string]::IsNullOrWhiteSpace($detectedTheme) -and $detectedTheme -ne $script:CurrentTheme) {
                    Apply-InstallerTheme -ThemeName $detectedTheme
                }
            }
            catch {
                Add-ManagerLog "[THEME WARNING] Theme refresh failed; installation continues. $($_ | Out-String)"
                $script:LastThemeCheck = $now
            }
        }
        Read-StageOutputFiles
        Drain-OutputQueue
        if ($script:ActiveProcess -and $script:ActiveProcess.HasExited) {
            $script:ActiveProcess.WaitForExit()
            Read-StageOutputFiles -FlushRemainder
            Drain-OutputQueue
            $stage = $script:ActiveStage
            $exitCode = $script:ActiveProcess.ExitCode
            Add-ManagerLog "[PROCESS] $stage child process exited with code $exitCode."
            $script:ActiveProcess.Dispose()
            $script:ActiveProcess = $null
            $script:ActiveStage = ''
            Handle-StageExit -Stage $stage -ExitCode $exitCode
        }
        if ($script:CanLaunchApp -and $null -ne $script:AutoLaunchAt -and $now -ge $script:AutoLaunchAt) {
            $script:AutoLaunchAt = $null
            Add-ManagerLog 'Automatic launch countdown completed. Closing setup manager to hand off to START_HIDDEN.vbs.'
            if ($script:Form -and -not $script:Form.IsDisposed) { $script:Form.Close() }
        }
    }
    catch {
        $errorText = $_ | Out-String
        Add-ManagerLog "[FATAL] Installation progress monitor failed: $errorText"
        try { Stop-ActiveInstallProcessTree } catch { }
        $script:InstallationFlowActive = $false
        $script:AutoLaunchAt = $null
        $script:CanLaunchApp = $false
        Set-FinishedUi
        Set-ManagerStatus 'The installation monitor encountered an error.' 'Open the log folder and review the diagnostic output.' '#FF7777'
        $script:RetryButton.Visible = $true
    }
})
$script:Timer.Start()

$form.Add_FormClosing({
    param($sender, $eventArgs)
    if ($script:InstallationFlowActive) {
        $eventArgs.Cancel = $true
        [Windows.Forms.MessageBox]::Show(
            'The setup manager cannot be closed until all installation stages are complete.',
            'Polyfitty Studio - Installation in progress',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }
    if ($script:CanLaunchApp) {
        try {
            [IO.File]::WriteAllText($script:LaunchFlagPath, 'launch', [Text.Encoding]::ASCII)
            Add-ManagerLog 'Setup manager closing. The VBS launcher will start Polyfitty Studio after this manager exits.'
        }
        catch {
            $eventArgs.Cancel = $true
            Add-ManagerLog "[ERROR] Unable to save launch-on-close request: $($_.Exception.Message)"
            [Windows.Forms.MessageBox]::Show(
                "The application launch request could not be saved.`r`n`r`n$($_.Exception.Message)",
                'Polyfitty Studio - Launch error',
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    }
})
$form.Add_FormClosed({
    if ($script:Timer) { $script:Timer.Stop(); $script:Timer.Dispose() }
    if ($script:ActiveProcess) {
        Stop-ActiveInstallProcessTree
        try { $script:ActiveProcess.Dispose() } catch { }
        $script:ActiveProcess = $null
    }
})

$script:UiExceptionHandler = [System.Threading.ThreadExceptionEventHandler]{
    param($sender, $eventArgs)
    $errorText = [string]$eventArgs.Exception
    $script:ManagerFatalError = $true
    $script:CanLaunchApp = $false
    try { Add-ManagerLog "[FATAL] Setup manager UI error: $errorText" } catch { }
    Stop-ActiveInstallProcessTree
    $script:InstallationFlowActive = $false
    try {
        [Windows.Forms.MessageBox]::Show(
            "An error occurred in the setup window. The active installation task has been stopped.`r`n`r`n$errorText`r`n`r`nLog: $script:LogPath",
            'Polyfitty Studio - Setup error',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch { }
    if ($script:Form -and -not $script:Form.IsDisposed) { $script:Form.Close() }
}
[Windows.Forms.Application]::add_ThreadException($script:UiExceptionHandler)

trap {
    $fatalText = [string]$_.Exception
    $script:CanLaunchApp = $false
    try { Add-ManagerLog "[FATAL] Setup manager error: $fatalText" } catch { }
    try { Stop-ActiveInstallProcessTree } catch { }
    try {
        [Windows.Forms.MessageBox]::Show(
            "The setup manager encountered an error. Background installation tasks were stopped.`r`n`r`n$fatalText`r`n`r`nLog: $script:LogPath",
            'Polyfitty Studio - Setup error',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch { }
    exit 1
}

$form.Add_Shown({
    try {
        Add-ManagerLog 'Polyfitty Studio GUI setup manager started.'
        Add-ManagerLog ("Version: {0}" -f ((Get-Content -LiteralPath (Join-Path $script:Root 'VERSION.txt') -Raw).Trim()))
        Start-InstallStage -Stage core
    }
    catch {
        $fatalText = $_ | Out-String
        Add-ManagerLog "[FATAL] Unable to initialize the Core installation stage: $fatalText"
        $script:InstallationFlowActive = $false
        Set-FinishedUi
        Set-ManagerStatus 'Unable to start Core installation.' 'Review the diagnostic log and retry.' '#FF7777'
        $script:RetryButton.Visible = $true
    }
})

$form.Show()
$form.BringToFront()
$form.Activate()
[Windows.Forms.Application]::DoEvents()
[Windows.Forms.Application]::Run($form)
Add-ManagerLog '[EXIT] Windows Forms message loop returned normally.'
if ($script:ManagerFatalError) { exit 1 }
exit 0
