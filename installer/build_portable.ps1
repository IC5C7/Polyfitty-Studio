param(
    [string]$StatusFile,
    [string]$LogFile,
    [string]$CancelFile
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONHOME = ''
$env:PYTHONPATH = ''
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$PortableOutput = Join-Path $Root 'release\portable'
$PortableExe = Join-Path $PortableOutput 'PolyfittyStudio.exe'

$BuildWorkRoot = Join-Path $env:TEMP ('PF' + [Guid]::NewGuid().ToString('N'))
$Dist = Join-Path $BuildWorkRoot 'i'
$PayloadZip = Join-Path $BuildWorkRoot 'PolyfittyStudio_embedded_payload.zip'
$PayloadVerifyZip = Join-Path $BuildWorkRoot 'PolyfittyStudio_embedded_payload_verify.zip'
$PayloadHashFile = Join-Path $BuildWorkRoot 'PolyfittyStudio_embedded_payload.sha256'
$NativeLauncherSource = Join-Path $Root 'installer\native_splash_launcher.cs'
$NativeSplashImage = Join-Path $Root 'packaging\PolyfittyStudio_Splash.png'
$NativeIcon = Join-Path $Root 'app\assets\PolyfittyStudio.ico'
$RuntimeManifestPath = Join-Path $Root 'runtime_manifest.json'
$script:BuildPercent = 0
$script:BuildStage = 'Starting'

function Write-BuildStatus([string]$State, [int]$Percent, [string]$Stage, [string]$Message, [string]$ErrorText = '') {
    if ([string]::IsNullOrWhiteSpace($StatusFile)) { return }
    try {
        $statusDirectory = Split-Path -Parent $StatusFile
        if ($statusDirectory) { New-Item -ItemType Directory -Force -Path $statusDirectory | Out-Null }
        $payload = [ordered]@{
            state = $State
            percent = [Math]::Max(0, [Math]::Min(100, $Percent))
            stage = $Stage
            message = $Message
            error = $ErrorText
            updated = (Get-Date).ToString('o')
        }
        $temporary = "$StatusFile.tmp"
        $payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $temporary -Encoding Unicode
        Move-Item -LiteralPath $temporary -Destination $StatusFile -Force
    } catch { }
}

function Write-BuildLog([string]$Message) {
    if ([string]::IsNullOrWhiteSpace($LogFile)) { return }
    try {
        $directory = Split-Path -Parent $LogFile
        if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
        Add-Content -LiteralPath $LogFile -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message) -Encoding Unicode
    } catch { }
}

function Set-BuildProgress([int]$Percent, [string]$Stage, [string]$Message) {
    $script:BuildPercent = [Math]::Max($script:BuildPercent, $Percent)
    $script:BuildStage = $Stage
    Write-BuildStatus -State 'running' -Percent $script:BuildPercent -Stage $Stage -Message $Message
    Write-BuildLog $Message
    Write-Host ("[{0}%] {1} - {2}" -f $script:BuildPercent, $Stage, $Message) -ForegroundColor Cyan
}

function Test-BuildCancellation {
    if (-not [string]::IsNullOrWhiteSpace($CancelFile) -and (Test-Path -LiteralPath $CancelFile)) {
        throw [System.OperationCanceledException]::new('Portable packaging cancelled at a safe checkpoint.')
    }
}

function Invoke-CheckedProcess([string]$Executable, [string[]]$Arguments, [string]$FailureMessage, [string]$WorkingDirectory = $Root) {
    $quoted = foreach ($argument in $Arguments) {
        $value = [string]$argument
        if ($value -match '[\s"]') { '"' + $value.Replace('"', '\\"') + '"' } else { $value }
    }
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = ($quoted -join ' ')
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) { throw "Could not start process: $Executable" }
        $stdoutTask = $process.StandardOutput.ReadLineAsync()
        $stderrTask = $process.StandardError.ReadLineAsync()
        $stdoutEnded = $false
        $stderrEnded = $false
        while ($true) {
            $received = $false
            if (-not $stdoutEnded -and $stdoutTask.IsCompleted) {
                $line = $stdoutTask.GetAwaiter().GetResult()
                if ($null -eq $line) { $stdoutEnded = $true }
                else { Write-Host $line; Write-BuildLog $line; $stdoutTask = $process.StandardOutput.ReadLineAsync(); $received = $true }
            }
            if (-not $stderrEnded -and $stderrTask.IsCompleted) {
                $line = $stderrTask.GetAwaiter().GetResult()
                if ($null -eq $line) { $stderrEnded = $true }
                else { Write-Host ("[stderr] {0}" -f $line) -ForegroundColor DarkYellow; Write-BuildLog ("[stderr] {0}" -f $line); $stderrTask = $process.StandardError.ReadLineAsync(); $received = $true }
            }
            if ($process.HasExited -and $stdoutEnded -and $stderrEnded) { break }
            Test-BuildCancellation
            if (-not $received) { [void]$process.WaitForExit(100) }
        }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw ("{0} (exit code {1})." -f $FailureMessage, $process.ExitCode) }
    }
    finally { $process.Dispose() }
}

function Get-DirectoryByteCount([string]$Path) {
    $total = [Int64]0
    foreach ($file in (Get-ChildItem -LiteralPath $Path -File -Recurse -Force)) {
        $total += [Int64]$file.Length
    }
    return $total
}

function Get-DirectoryFingerprint([string]$Path) {
    $base = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $files = @(Get-ChildItem -LiteralPath $base -File -Recurse -Force | Sort-Object { $_.FullName.Substring($base.Length).ToLowerInvariant() })
    $builder = New-Object System.Text.StringBuilder
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($base.Length).TrimStart('\').Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        [void]$builder.Append($relative).Append([char]0).Append($hash).Append("`n")
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($builder.ToString())
        $digest = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally { $sha.Dispose() }
    return [pscustomobject]@{ Fingerprint = $digest; FileCount = $files.Count }
}

function Test-SourceManifest([string]$ProjectRoot) {
    $manifestPath = Join-Path $ProjectRoot 'runtime_manifest.json'
    $versionPath = Join-Path $ProjectRoot 'VERSION.txt'
    $mainPath = Join-Path $ProjectRoot 'app\main.py'
    foreach ($required in @($manifestPath, $versionPath, $mainPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Release source identity file is missing: $required" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $releaseVersion = (Get-Content -LiteralPath $versionPath -Raw).Trim()
    if ([string]$manifest.app_version -ne $releaseVersion) { throw 'runtime_manifest.json and VERSION.txt identify different releases.' }
    $mainText = [IO.File]::ReadAllText($mainPath)
    if ($mainText -notmatch '(?m)^APP_VERSION\s*=\s*["'']([^"'']+)["'']\s*$' -or $Matches[1] -ne $releaseVersion) {
        throw 'app/main.py APP_VERSION does not match VERSION.txt.'
    }
    $properties = @($manifest.files.PSObject.Properties | Sort-Object Name)
    if ($properties.Count -ne [int]$manifest.source_file_count) { throw 'runtime_manifest.json source file count is inconsistent.' }
    foreach ($property in $properties) {
        $relative = ([string]$property.Name).Replace('/', '\\')
        $path = Join-Path $ProjectRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Release source file is missing: $($property.Name)" }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$property.Value).ToLowerInvariant()) { throw "Release source file does not match runtime_manifest.json: $($property.Name)" }
    }
    return $manifest
}

trap {
    $failure = $_
    $message = if ($failure.Exception) { $failure.Exception.Message } else { [string]$failure }
    $isCancelled = $failure.Exception -is [System.OperationCanceledException]
    $state = if ($isCancelled) { 'cancelled' } else { 'failed' }
    $stage = if ($isCancelled) { 'Cancelled' } else { 'Failed' }
    Write-BuildLog ("{0}: {1}" -f $stage, $message)
    Write-BuildStatus -State $state -Percent $script:BuildPercent -Stage $stage -Message $message -ErrorText $message
    exit $(if ($isCancelled) { 2 } else { 1 })
}

Set-BuildProgress 0 'Starting' 'Preparing one self-contained EXE with the complete application and embedded Python runtime.'
Test-BuildCancellation
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'The portable launcher and CPython runtime must be packaged on Windows x64.'
}
foreach ($required in @($NativeLauncherSource, $NativeSplashImage, $NativeIcon, (Join-Path $Root 'installer\install.ps1'), (Join-Path $Root 'requirements.txt'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "A required portable build input is missing: $required" }
}

Set-BuildProgress 5 'Release source validation' 'Verifying that this source tree is one complete release. The installed main runtime is not required.'
$sourceManifest = Test-SourceManifest -ProjectRoot $Root
Test-BuildCancellation

Set-BuildProgress 12 'Prepare embedded application image' 'Preparing an isolated build image. The recipient receives one EXE; Python, Qt, VTK, application source and resources will be embedded inside it.'
if (Test-Path -LiteralPath $Dist) { Remove-Item -LiteralPath $Dist -Recurse -Force }
if (Test-Path -LiteralPath $PortableOutput) { Remove-Item -LiteralPath $PortableOutput -Recurse -Force }
if (Test-Path -LiteralPath $PayloadZip) { Remove-Item -LiteralPath $PayloadZip -Force }
if (Test-Path -LiteralPath $PayloadHashFile) { Remove-Item -LiteralPath $PayloadHashFile -Force }
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
New-Item -ItemType Directory -Force -Path $PortableOutput | Out-Null
New-Item -ItemType Directory -Force -Path $BuildWorkRoot | Out-Null
$robocopy = Join-Path $env:SystemRoot 'System32\robocopy.exe'
$copyArgs = @(
    $Root, $Dist, '/E', '/COPY:DAT', '/DCOPY:DAT', '/R:2', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS',
    '/XD', '.git', '.cache', '.build', '.pytest_cache', '.mypy_cache', '.ruff_cache', '__pycache__', 'release', 'tests', '.runtime',
    '/XF', '*.pyc', '*.pyo'
)
& $robocopy @copyArgs | Out-Null
$copyExitCode = $LASTEXITCODE
if ($copyExitCode -ge 8) { throw "Portable source image copy failed (robocopy exit code $copyExitCode)." }

Set-BuildProgress 22 'Provision embedded Python and core packages' 'Provisioning CPython 3.12.10 and all core requirements into the temporary build image that will be compressed inside the final EXE.'
Test-BuildCancellation
$powerShellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$portableInstaller = Join-Path $Dist 'installer\install.ps1'
Invoke-CheckedProcess -Executable $powerShellPath -Arguments @(
    '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
    '-File', $portableInstaller, '-PortableBuild'
) -FailureMessage 'Could not provision the independent embedded Python runtime and core application dependencies.' -WorkingDirectory $Dist

Set-BuildProgress 66 'Verify embedded runtime' 'Testing the embedded Python interpreter, VTK, Qt, NumPy and PolyFit before the complete image is compressed into one EXE.'
Test-BuildCancellation
$PortableRuntime = Join-Path $Dist '.runtime\python312'
$portablePython = Join-Path $PortableRuntime 'python.exe'
$pythonPathFile = Join-Path $PortableRuntime 'python312._pth'
foreach ($required in @($portablePython, $pythonPathFile, (Join-Path $Dist 'prerequisites\vc_redist.x64.exe'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "The embedded portable runtime is incomplete: $required" }
}
$pthText = [IO.File]::ReadAllText($pythonPathFile)
if ($pthText -notmatch '(?m)^\.\.\\\.\.\\app\s*$' -or $pthText -notmatch '(?m)^Lib\\site-packages\s*$' -or $pthText -notmatch '(?m)^import site\s*$') {
    throw 'The single-file payload python312._pth is not configured for the packaged app and site-packages.'
}
$pythonProbe = & $portablePython -c "import sys,struct; assert sys.version_info[:3] == (3,12,10), sys.version; assert struct.calcsize('P') == 8; import pip,numpy,PySide6,vtk,trimesh,xatlas,scipy,polyfit; print(sys.version); print(sys.executable)" 2>&1
if ($LASTEXITCODE -ne 0) { throw "The embedded single-file runtime failed its native package import probe: $pythonProbe" }
$portableManifestVerifier = Join-Path $Dist 'installer\runtime_source_manifest.py'
Invoke-CheckedProcess -Executable $portablePython -Arguments @($portableManifestVerifier, '--verify', '--root', $Dist) -FailureMessage 'The application source inside the single-file staging image does not match its release manifest.' -WorkingDirectory $Dist
$portableVerifier = Join-Path $Dist 'installer\verify_runtime.py'
Invoke-CheckedProcess -Executable $portablePython -Arguments @($portableVerifier, '--phase', 'runtime') -FailureMessage 'The embedded single-file runtime failed the standard runtime self-test.' -WorkingDirectory $Dist
$runtimeFingerprint = Get-DirectoryFingerprint $PortableRuntime

$distCache = Join-Path $Dist '.cache'
if (Test-Path -LiteralPath $distCache) { Remove-Item -LiteralPath $distCache -Recurse -Force }

$runtimeIdentity = [ordered]@{
    schema_version = 3
    package_kind = 'single_file_self_extracting_embedded_portable'
    app_version = (Get-Content -LiteralPath (Join-Path $Dist 'VERSION.txt') -Raw).Trim()
    python_version = '3.12.10'
    runtime_source_fingerprint = $sourceManifest.source_fingerprint
    embedded_runtime_sha256 = $runtimeFingerprint.Fingerprint
    runtime_file_count = $runtimeFingerprint.FileCount
    embedded_python = $true
    core_requirements = 'requirements.txt'
    gpu_setup = 'first_launch_on_recipient_pc'
    deployment_model = 'single_exe_embedded_payload_extracted_to_local_app_data'
    entrypoint = 'app/hidden_entry.py'
    interpreter = '.runtime/python312/python.exe'
    startup_helper = 'installer/start_app.ps1'
    gpu_overlay = '.cache/gpu_runtime/site-packages'
    extraction_root = '%LOCALAPPDATA%/PF/P/<payload-sha256>'
}
$runtimeIdentity | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Dist 'portable_runtime_manifest.json') -Encoding UTF8

Set-BuildProgress 74 'Create embedded payload'  'Compressing the complete application image, embedded CPython, Qt, VTK and core libraries for inclusion inside the single EXE.'
Test-BuildCancellation
if (Test-Path -LiteralPath $PayloadZip) { Remove-Item -LiteralPath $PayloadZip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($Dist, $PayloadZip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
if (-not (Test-Path -LiteralPath $PayloadZip -PathType Leaf) -or (Get-Item -LiteralPath $PayloadZip).Length -le 0) {
    throw 'The embedded application payload archive was not generated.'
}
$payloadLength = (Get-Item -LiteralPath $PayloadZip).Length
$expandedLength = Get-DirectoryByteCount $Dist
$compressionPercent = if ($expandedLength -gt 0) { [Math]::Round((1.0 - ($payloadLength / [double]$expandedLength)) * 100.0, 1) } else { 0 }
Write-BuildLog ("Expanded embedded image: {0:N1} MB across {1} files." -f ($expandedLength / 1MB), (Get-ChildItem -LiteralPath $Dist -File -Recurse -Force).Count)
Write-BuildLog ("Compressed EXE payload: {0:N1} MB ({1}% smaller after ZIP compression)." -f ($payloadLength / 1MB), $compressionPercent)
if ($payloadLength -gt 1800000000) {
    throw 'The embedded payload exceeds the safe managed-resource size limit for the native single-file launcher.'
}
$payloadHash = (Get-FileHash -LiteralPath $PayloadZip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($PayloadHashFile, $payloadHash, [Text.Encoding]::ASCII)

Set-BuildProgress 82 'Build single-file EXE' 'Compiling one branded executable with the complete compressed application payload and splash image embedded as assembly resources.'
Test-BuildCancellation
$cscCandidates = @(
    (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'),
    (Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
)
$csc = $cscCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $csc) { throw 'The Windows .NET Framework C# compiler (csc.exe) is required to build the single-file launcher.' }
if (Test-Path -LiteralPath $PortableExe) { Remove-Item -LiteralPath $PortableExe -Force }
Invoke-CheckedProcess -Executable $csc -Arguments @(
    '/nologo', '/target:winexe', '/platform:x64', '/optimize+', ('/out:' + $PortableExe),
    ('/win32icon:' + $NativeIcon), ('/resource:' + $NativeSplashImage + ',PolyfittyStudio.Splash.png'),
    ('/resource:' + $PayloadZip + ',PolyfittyStudio.Payload.zip'),
    ('/resource:' + $PayloadHashFile + ',PolyfittyStudio.Payload.sha256'),
    '/reference:System.dll', '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
    '/reference:System.IO.Compression.dll',
    $NativeLauncherSource
) -FailureMessage 'The self-extracting single-file Polyfitty Studio EXE could not be compiled.'
if (-not (Test-Path -LiteralPath $PortableExe -PathType Leaf)) { throw 'The single-file Polyfitty Studio EXE was not generated.' }

Set-BuildProgress 88 'Audit embedded EXE payload' 'Opening the compiled EXE resource, comparing its payload SHA-256, and extracting the embedded image into a temporary verification directory.'
Test-BuildCancellation
$zipVerifyRoot = Join-Path $BuildWorkRoot 'v'
try {
    New-Item -ItemType Directory -Force -Path $zipVerifyRoot | Out-Null
    $launcherAssembly = [Reflection.Assembly]::LoadFrom($PortableExe)
    $resourceNames = @($launcherAssembly.GetManifestResourceNames())
    if ($resourceNames -notcontains 'PolyfittyStudio.Payload.zip' -or $resourceNames -notcontains 'PolyfittyStudio.Payload.sha256' -or $resourceNames -notcontains 'PolyfittyStudio.Splash.png') {
        throw 'The generated single-file EXE is missing its embedded payload, payload hash, or splash image resource.'
    }
    $payloadHashResource = $launcherAssembly.GetManifestResourceStream('PolyfittyStudio.Payload.sha256')
    if ($null -eq $payloadHashResource) { throw 'The generated EXE payload hash resource could not be opened.' }
    try {
        $payloadHashReader = New-Object System.IO.StreamReader($payloadHashResource, [Text.Encoding]::ASCII)
        try { $embeddedDeclaredHash = $payloadHashReader.ReadToEnd().Trim() }
        finally { $payloadHashReader.Dispose() }
    }
    catch { throw }
    if ($embeddedDeclaredHash -ne $payloadHash) { throw 'The payload SHA-256 resource embedded in PolyfittyStudio.exe does not match the build image.' }
    $payloadResource = $launcherAssembly.GetManifestResourceStream('PolyfittyStudio.Payload.zip')
    if ($null -eq $payloadResource) { throw 'The generated EXE payload resource could not be opened.' }
    try {
        $verifyPayloadStream = [IO.File]::Open($PayloadVerifyZip, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $payloadResource.CopyTo($verifyPayloadStream) }
        finally { $verifyPayloadStream.Dispose() }
    }
    finally { $payloadResource.Dispose() }
    $embeddedPayloadHash = (Get-FileHash -LiteralPath $PayloadVerifyZip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($embeddedPayloadHash -ne $payloadHash) {
        throw 'The payload embedded in PolyfittyStudio.exe does not match the verified build image.'
    }

    $zipVerifyImage = Join-Path $zipVerifyRoot 'image'
    [System.IO.Compression.ZipFile]::ExtractToDirectory($PayloadVerifyZip, $zipVerifyImage)
    $zipPython = Join-Path $zipVerifyImage '.runtime\python312\python.exe'
    $zipManifestVerifier = Join-Path $zipVerifyImage 'installer\runtime_source_manifest.py'
    $zipRuntimeVerifier = Join-Path $zipVerifyImage 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath (Join-Path $zipVerifyImage 'prerequisites\vc_redist.x64.exe') -PathType Leaf)) {
        throw 'The EXE embedded payload is missing the signed VC++ prerequisite installer.'
    }
    Invoke-CheckedProcess -Executable $zipPython -Arguments @($zipManifestVerifier, '--verify', '--root', $zipVerifyImage) -FailureMessage 'The EXE embedded application source manifest failed after extraction.' -WorkingDirectory $zipVerifyImage
    Invoke-CheckedProcess -Executable $zipPython -Arguments @($zipRuntimeVerifier, '--phase', 'runtime') -FailureMessage 'The EXE embedded Python runtime failed after extraction.' -WorkingDirectory $zipVerifyImage
    $zipRuntimeFingerprint = Get-DirectoryFingerprint (Join-Path $zipVerifyImage '.runtime\python312')
    if ($runtimeFingerprint.Fingerprint -ne $zipRuntimeFingerprint.Fingerprint -or $runtimeFingerprint.FileCount -ne $zipRuntimeFingerprint.FileCount) {
        throw 'The embedded Python runtime changed while being packaged into the single-file EXE.'
    }
    if (Test-Path -LiteralPath (Join-Path $zipVerifyImage '.cache')) {
        throw 'The EXE unexpectedly contains builder-machine cache or GPU setup state.'
    }

    $releaseFiles = @(Get-ChildItem -LiteralPath $PortableOutput -File -Force)
    $releaseDirectories = @(Get-ChildItem -LiteralPath $PortableOutput -Directory -Force)
    if ($releaseFiles.Count -ne 1 -or $releaseFiles[0].Name -ne 'PolyfittyStudio.exe' -or $releaseDirectories.Count -ne 0) {
        throw 'The portable output must contain exactly one distributable file: PolyfittyStudio.exe.'
    }
}
finally {
    if (Test-Path -LiteralPath $zipVerifyRoot) { Remove-Item -LiteralPath $zipVerifyRoot -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $PayloadVerifyZip) { Remove-Item -LiteralPath $PayloadVerifyZip -Force -ErrorAction SilentlyContinue }
}

if (Test-Path -LiteralPath $Dist) { Remove-Item -LiteralPath $Dist -Recurse -Force }
if (Test-Path -LiteralPath $PayloadZip) { Remove-Item -LiteralPath $PayloadZip -Force }
if (Test-Path -LiteralPath $PayloadHashFile) { Remove-Item -LiteralPath $PayloadHashFile -Force }
if (Test-Path -LiteralPath $BuildWorkRoot) { Remove-Item -LiteralPath $BuildWorkRoot -Recurse -Force }

Set-BuildProgress 100 'Completed' 'Single-file portable build completed. The EXE embeds the application, CPython and core libraries; recipient-specific GPU packages are prepared on first launch.'
Write-BuildStatus -State 'succeeded' -Percent 100 -Stage 'Completed' -Message ("Single-file portable EXE created: {0}" -f $PortableExe)
$finalExeLength = (Get-Item -LiteralPath $PortableExe).Length
Write-BuildLog ("Final single-file EXE size: {0:N1} MB." -f ($finalExeLength / 1MB))
Write-BuildLog ("Embedded portable runtime SHA-256: {0}" -f $runtimeFingerprint.Fingerprint)
Write-BuildLog ("Application source fingerprint: {0}" -f $sourceManifest.source_fingerprint)
Write-BuildLog ("Single-file executable: {0}" -f $PortableExe)
Write-BuildLog ("Embedded payload SHA-256: {0}" -f $payloadHash)
