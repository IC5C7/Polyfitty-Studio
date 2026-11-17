param([switch]$PortableBuild)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root '.runtime\python312'
$Cache = Join-Path $Root '.cache'
$PythonExe = Join-Path $Runtime 'python.exe'
$RuntimeManifestPath = Join-Path $Root 'runtime_manifest.json'
$RuntimeManifestTool = Join-Path $Root 'installer\runtime_source_manifest.py'
$RuntimeContractSource = Join-Path $Root 'app\runtime_contract.py'
$PythonZip = Join-Path $Cache 'python-3.12.10-embed-amd64.zip'
$Wheel = Join-Path $Cache 'polyfit-1.6.0-cp312-cp312-win_amd64.whl'
$VcRedist = if ($PortableBuild) { Join-Path $Root 'prerequisites\vc_redist.x64.exe' } else { Join-Path $Cache 'vc_redist.x64.exe' }
$PythonUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip'
$VcUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'

function Say($Text, $Color = 'Cyan') { Write-Host "`n[Polyfitty Studio] $Text" -ForegroundColor $Color }

$script:LastInstallProgress = -1
function Report-InstallProgress {
    param([int]$Percent, [string]$Detail)
    $Percent = [Math]::Max(0, [Math]::Min(100, $Percent))
    if ($Percent -lt $script:LastInstallProgress) { $Percent = $script:LastInstallProgress }
    if ($Percent -eq $script:LastInstallProgress -and $Detail -eq $script:LastProgressDetail) { return }
    $script:LastInstallProgress = $Percent
    $script:LastProgressDetail = $Detail
    Write-Host ("[PF_PROGRESS]|{0}|{1}" -f $Percent, ($Detail -replace '[\r\n|]', ' ')) -ForegroundColor DarkCyan
}

function Invoke-DownloadWithProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [hashtable]$Headers = @{},
        [int]$ProgressStart = 0,
        [int]$ProgressEnd = 0,
        [string]$Name = 'download'
    )

    $request = [System.Net.HttpWebRequest]::Create($Uri)
    $request.Method = 'GET'
    $request.Timeout = 600000
    $request.ReadWriteTimeout = 600000
    $request.AllowAutoRedirect = $true
    foreach ($key in $Headers.Keys) {
        if ($key -ieq 'User-Agent') {
            $request.UserAgent = [string]$Headers[$key]
        }
        else {
            $request.Headers[[string]$key] = [string]$Headers[$key]
        }
    }

    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        $response = $request.GetResponse()
        $totalBytes = [int64]$response.ContentLength
        $inputStream = $response.GetResponseStream()
        $outputStream = [System.IO.FileStream]::new($OutFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        $buffer = New-Object byte[] (1024 * 1024)
        [int64]$downloadedBytes = 0
        $lastReported = -1
        while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outputStream.Write($buffer, 0, $read)
            $downloadedBytes += $read
            if ($totalBytes -gt 0 -and $ProgressEnd -gt $ProgressStart) {
                $fraction = [Math]::Min(1.0, ($downloadedBytes / [double]$totalBytes))
                $current = $ProgressStart + [int][Math]::Floor(($ProgressEnd - $ProgressStart) * $fraction)
                if ($current -gt $lastReported) {
                    $lastReported = $current
                    $detail = if ($totalBytes -gt 0) { '{0} - {1:N0} / {2:N0} MB' -f $Name, ($downloadedBytes / 1MB), ($totalBytes / 1MB) } else { $Name }
                    Report-InstallProgress -Percent $current -Detail $detail
                }
            }
        }
        $outputStream.Flush()
        if ($ProgressEnd -gt 0) { Report-InstallProgress -Percent $ProgressEnd -Detail ("Downloaded {0}" -f $Name) }
    }
    finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
    }
}

function Test-PythonWheel {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedPackage,
        [Parameter(Mandatory = $true)][string]$ExpectedVersion,
        [string]$ExpectedSha256 = ''
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $archive = $null
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $metadataEntry = $archive.Entries | Where-Object { $_.FullName -match '(?i)\.dist-info/METADATA$' } | Select-Object -First 1
        $wheelEntry = $archive.Entries | Where-Object { $_.FullName -match '(?i)\.dist-info/WHEEL$' } | Select-Object -First 1
        $nativeEntry = $archive.Entries | Where-Object { $_.FullName -match '(?i)\.pyd$' } | Select-Object -First 1
        if (-not $metadataEntry -or -not $wheelEntry -or -not $nativeEntry) { return $false }

        $reader = [System.IO.StreamReader]::new($metadataEntry.Open())
        try { $metadata = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $normalizedExpected = ($ExpectedPackage -replace '[-_.]+', '-').ToLowerInvariant()
        $nameMatch = [regex]::Match($metadata, '(?im)^Name:\s*(.+?)\s*$')
        $versionMatch = [regex]::Match($metadata, '(?im)^Version:\s*(.+?)\s*$')
        if (-not $nameMatch.Success -or -not $versionMatch.Success) { return $false }
        $actualName = ($nameMatch.Groups[1].Value -replace '[-_.]+', '-').ToLowerInvariant()
        if ($actualName -ne $normalizedExpected) { return $false }
        if ($versionMatch.Groups[1].Value -ne $ExpectedVersion) { return $false }

        if ($ExpectedSha256) {
            $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) { return $false }
        }
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($archive) { $archive.Dispose() }
    }
}

function Test-PolyFitWheel {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Test-PythonWheel -Path $Path -ExpectedPackage 'polyfit' -ExpectedVersion '1.6.0')
}

function Test-ExpectedWheel {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$Version,
        [string]$Sha256 = ''
    )
    if ($Package -ieq 'polyfit') { return (Test-PolyFitWheel -Path $Path) }
    return (Test-PythonWheel -Path $Path -ExpectedPackage $Package -ExpectedVersion $Version -ExpectedSha256 $Sha256)
}

function Get-GitHubReleaseAssetDownloadUrl {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$AssetName
    )

    $apiUrl = "https://api.github.com/repos/$Repository/releases/tags/$Tag"
    try {
        $release = Invoke-RestMethod -Method Get -Uri $apiUrl -TimeoutSec 45 -UserAgent 'PolyfittyStudio-Installer/0.2.88' -Headers @{
            'X-GitHub-Api-Version' = '2022-11-28'
        }
        $asset = $release.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
        if ($asset -and $asset.browser_download_url) {
            $downloadUri = [Uri][string]$asset.browser_download_url
            if ($downloadUri.Scheme -eq 'https' -and $downloadUri.Host -ieq 'github.com') {
                return $downloadUri.AbsoluteUri
            }
            Say "GitHub returned an unexpected release asset URL for '$AssetName'." 'Yellow'
        }
        Say "GitHub release API did not list a usable browser download URL for '$AssetName'." 'Yellow'
    }
    catch {
        Say "GitHub release asset lookup failed: $($_.Exception.Message)" 'Yellow'
    }
    return $null
}

function Download-File {
    param(
        [string]$Url = '',
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name,
        [int64]$MinBytes = 100000,
        [string]$GitHubRepository = '',
        [string]$GitHubTag = '',
        [string]$GitHubAssetName = '',
        [string]$ExpectedPackage = '',
        [string]$ExpectedVersion = '',
        [string]$ExpectedSha256 = '',
        [int]$ProgressStart = 0,
        [int]$ProgressEnd = 0
    )

    $isWheel = -not [string]::IsNullOrWhiteSpace($ExpectedPackage)
    $isGitHubAsset = -not [string]::IsNullOrWhiteSpace($GitHubRepository) -and
                     -not [string]::IsNullOrWhiteSpace($GitHubTag) -and
                     -not [string]::IsNullOrWhiteSpace($GitHubAssetName)

    if ((-not $isGitHubAsset) -and [string]::IsNullOrWhiteSpace($Url)) {
        throw "No download source was configured for $Name."
    }
    if ($isWheel -and (-not $ExpectedVersion)) {
        throw "Wheel version verification was not configured for $Name."
    }

    if (Test-Path -LiteralPath $Path) {
        $cachedIsValid = (Get-Item -LiteralPath $Path).Length -ge $MinBytes
        if ($cachedIsValid -and $isWheel) {
            $cachedIsValid = Test-ExpectedWheel -Path $Path -Package $ExpectedPackage -Version $ExpectedVersion -Sha256 $ExpectedSha256
        }
        if ($cachedIsValid) {
            Say "Using verified cached download: $Name" 'Green'
            if ($ProgressEnd -gt 0) { Report-InstallProgress -Percent $ProgressEnd -Detail ("Using cached {0}" -f $Name) }
            return
        }
        Say "Removing incomplete or invalid cached download: $Name" 'Yellow'
        Remove-Item -LiteralPath $Path -Force
    }

    $partialPath = "$Path.download"
    if (Test-Path -LiteralPath $partialPath) { Remove-Item -LiteralPath $partialPath -Force }
    $lastError = 'Unknown download error.'

    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            if (Test-Path -LiteralPath $partialPath) { Remove-Item -LiteralPath $partialPath -Force }
            if ($isGitHubAsset) {
                Say "Downloading $Name from the official GitHub release URL (attempt $attempt/5)..."
                $downloadUrl = Get-GitHubReleaseAssetDownloadUrl -Repository $GitHubRepository -Tag $GitHubTag -AssetName $GitHubAssetName
                if ([string]::IsNullOrWhiteSpace($downloadUrl)) {
                    throw "GitHub Releases API did not return asset '$GitHubAssetName' from $GitHubRepository tag $GitHubTag."
                }
                $headers = @{ 'User-Agent' = 'PolyfittyStudio-Installer/0.2.88' }
            }
            else {
                Say "Downloading $Name from its configured official source (attempt $attempt/5)..."
                $downloadUrl = $Url
                $headers = @{ 'User-Agent' = 'PolyfittyStudio-Installer/0.2.88' }
            }

            if ($ProgressStart -gt 0) { Report-InstallProgress -Percent $ProgressStart -Detail ("Starting download: {0}" -f $Name) }
            Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $partialPath -Headers $headers -ProgressStart $ProgressStart -ProgressEnd $ProgressEnd -Name $Name

            if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf) -or (Get-Item -LiteralPath $partialPath).Length -lt $MinBytes) {
                throw "Downloaded file is missing or smaller than $MinBytes bytes."
            }
            if ($isWheel -and -not (Test-ExpectedWheel -Path $partialPath -Package $ExpectedPackage -Version $ExpectedVersion -Sha256 $ExpectedSha256)) {
                throw "Downloaded file failed the $ExpectedPackage $ExpectedVersion wheel ZIP/metadata/native-module/checksum verification."
            }

            Move-Item -LiteralPath $partialPath -Destination $Path -Force
            Say "Download verified: $Name" 'Green'
            return
        }
        catch {
            $lastError = $_.Exception.Message
            if (Test-Path -LiteralPath $partialPath) {
                Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
            }
            if ($attempt -lt 5) {
                $delaySeconds = [Math]::Min(24, [int](3 * [Math]::Pow(2, $attempt - 1)))
                Write-Host "[Polyfitty Studio] Download attempt failed: $lastError. Retrying in $delaySeconds seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds $delaySeconds
            }
            else {
                Write-Host "[Polyfitty Studio] $Name download failed after 5 attempts: $lastError" -ForegroundColor Yellow
            }
        }
    }

    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
    }
    throw "Unable to download $Name through its configured official source. Last error: $lastError"
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($pythonExitCode -ne 0) {
        throw "$Description (Python exit code $pythonExitCode)."
    }
}

try {
    Report-InstallProgress -Percent 0 -Detail 'Preparing the Core Runtime installation'
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows 64-bit is required.' }
    New-Item -ItemType Directory -Force -Path $Cache, $Runtime | Out-Null

    $LegacyVoxelRuntime = Join-Path $Root 'vendor\openvdb_runtime'
    if (Test-Path -LiteralPath $LegacyVoxelRuntime) {
        Remove-Item -LiteralPath $LegacyVoxelRuntime -Recurse -Force
    }

    if (!(Test-Path $PythonExe)) {
        Download-File -Url $PythonUrl -Path $PythonZip -Name 'CPython 3.12.10 x64 embedded runtime' -MinBytes 5000000 -ProgressStart 2 -ProgressEnd 18
        Say 'Extracting private Python runtime...'
        Expand-Archive -LiteralPath $PythonZip -DestinationPath $Runtime -Force
        New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'Lib\site-packages') | Out-Null
        $pth = Get-ChildItem -LiteralPath $Runtime -Filter 'python312._pth' | Select-Object -First 1
        if (-not $pth) { throw 'python312._pth was not found in the embedded Python package.' }
        @("python312.zip", ".", "Lib\site-packages", "..\..\app", "import site") | Set-Content -LiteralPath $pth.FullName -Encoding ASCII
        if (!(Test-Path $PythonExe)) { throw 'python.exe was not found after extracting embedded Python.' }
        Report-InstallProgress -Percent 22 -Detail 'Private Python runtime extracted and configured'
    } else { Say 'Private Python runtime already exists.' 'Green'; Report-InstallProgress -Percent 22 -Detail 'Existing private Python runtime validated' }

    $pth = Get-ChildItem -LiteralPath $Runtime -Filter 'python312._pth' | Select-Object -First 1
    if (-not $pth) { throw 'python312._pth was not found in the private Python runtime.' }
    @("python312.zip", ".", "Lib\site-packages", "..\..\app", "import site") | Set-Content -LiteralPath $pth.FullName -Encoding ASCII
    New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'Lib\site-packages') | Out-Null

    foreach ($requiredRuntimeFile in @($RuntimeManifestPath, $RuntimeManifestTool, $RuntimeContractSource, (Join-Path $Root 'VERSION.txt'))) {
        if (-not (Test-Path -LiteralPath $requiredRuntimeFile -PathType Leaf)) {
            throw "The complete shared installed/portable runtime source is required. Missing: $requiredRuntimeFile. Extract the complete release ZIP; do not combine files from different versions."
        }
    }
    Say 'Verifying the shared installed/portable source manifest before installation...'
    Invoke-PythonChecked -Description 'The project folder does not match the shared installed/portable release manifest' -Arguments @($RuntimeManifestTool, '--verify', '--root', $Root)
    Report-InstallProgress -Percent 22 -Detail 'Shared runtime source identity verified'

    $hasVcRuntime = (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140.dll')) -and (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140_1.dll'))
    if ($PortableBuild) {

        $portablePrerequisiteDirectory = Split-Path -Parent $VcRedist
        New-Item -ItemType Directory -Force -Path $portablePrerequisiteDirectory | Out-Null
        Download-File -Url $VcUrl -Path $VcRedist -Name 'Microsoft Visual C++ Redistributable x64 (portable prerequisite)' -MinBytes 1000000 -ProgressStart 23 -ProgressEnd 31
        $vcSig = Get-AuthenticodeSignature -FilePath $VcRedist
        if ($vcSig.Status -ne 'Valid' -or $vcSig.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
            throw "Bundled VC++ Redistributable signature check failed. Status: $($vcSig.Status)"
        }
        Say 'Microsoft Visual C++ runtime installer has been bundled for first launch on the recipient PC.' 'Green'
    }
    elseif (-not $hasVcRuntime) {
        Download-File -Url $VcUrl -Path $VcRedist -Name 'Microsoft Visual C++ Redistributable x64' -MinBytes 1000000 -ProgressStart 23 -ProgressEnd 31
        $vcSig = Get-AuthenticodeSignature -FilePath $VcRedist
        if ($vcSig.Status -ne 'Valid' -or $vcSig.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
            throw "VC++ Redistributable signature check failed. Status: $($vcSig.Status)"
        }
        Say 'Installing Microsoft Visual C++ runtime...'
        $vc = Start-Process -FilePath $VcRedist -ArgumentList '/install /quiet /norestart' -Verb RunAs -Wait -PassThru
        if ($vc.ExitCode -notin @(0, 1638, 3010)) { throw "VC++ Redistributable installation failed (exit $($vc.ExitCode))." }
    }
    Report-InstallProgress -Percent 35 -Detail 'Microsoft runtime prerequisite check complete'

    $PipBootstrap = Join-Path $PSScriptRoot 'ensure_pip.ps1'
    if (-not (Test-Path -LiteralPath $PipBootstrap)) { throw 'The pip bootstrap helper is missing.' }
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $PipBootstrap -RuntimePath $Runtime -CachePath $Cache
    if ($LASTEXITCODE -ne 0) { throw 'pip verification/bootstrap failed in the private Python runtime.' }
    Report-InstallProgress -Percent 42 -Detail 'Private Python package manager ready'

    Say 'Installing GUI, geometry and monitoring modules...'
    $CoreRequirements = Join-Path $Root 'requirements.txt'
    if (!(Test-Path -LiteralPath $CoreRequirements)) { throw 'Unified requirements.txt was not found in the application folder.' }
    Invoke-PythonChecked -Description 'Python module installation failed' -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '-r', $CoreRequirements)
    Report-InstallProgress -Percent 57 -Detail 'GUI, geometry and monitoring modules installed'

    Download-File -Path $Wheel -Name 'PolyFit 1.6.0 native Windows Python binding' -MinBytes 1000000 -GitHubRepository 'LiangliangNan/PolyFit' -GitHubTag 'v1.6.0' -GitHubAssetName 'polyfit-1.6.0-cp312-cp312-win_amd64.whl' -ExpectedPackage 'polyfit' -ExpectedVersion '1.6.0' -ProgressStart 58 -ProgressEnd 64
    Say 'Installing official PolyFit wheel...'
    Invoke-PythonChecked -Description 'PolyFit wheel installation failed' -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', $Wheel)
    $runtimeVerifier = Join-Path $Root 'installer\verify_runtime.py'
    if (-not (Test-Path -LiteralPath $runtimeVerifier)) { throw 'Runtime verification helper is missing.' }
    Invoke-PythonChecked -Description 'PolyFit native binding verification failed' -Arguments @($runtimeVerifier, '--phase', 'polyfit')
    Report-InstallProgress -Percent 82 -Detail 'PolyFit native binding installed and verified'

    Say 'Removing the retired Open3D package from an existing private runtime, when present...'
    & $PythonExe -m pip uninstall -y open3d | Out-Null
    if ($LASTEXITCODE -ne 0) { Say 'Open3D cleanup was skipped; it is not used by this application.' 'Yellow' }

    Say 'Checking main application NumPy package integrity...'
    Invoke-PythonChecked -Description 'Main NumPy integrity verification failed after installation' -Arguments @($runtimeVerifier, '--phase', 'numpy')
    Report-InstallProgress -Percent 91 -Detail 'NumPy integrity verified'

    Say 'Running import and runtime checks...'
    Invoke-PythonChecked -Description 'Runtime verification failed' -Arguments @($runtimeVerifier, '--phase', 'runtime')
    Report-InstallProgress -Percent 98 -Detail 'Core runtime checks passed'

    if (-not $PortableBuild) {
        try {
            $desktop = [Environment]::GetFolderPath('Desktop')
            $legacyShortcut = Join-Path $desktop 'PolyFit Studio.lnk'
            if (Test-Path -LiteralPath $legacyShortcut) {
                Remove-Item -LiteralPath $legacyShortcut -Force -ErrorAction SilentlyContinue
            }
            $shell = New-Object -ComObject WScript.Shell
            $appIcon = Join-Path $Root 'app\assets\PolyfittyStudio.ico'
            $launcher = Join-Path $Root 'START_HIDDEN.vbs'
            $wscript = Join-Path $env:WINDIR 'System32\wscript.exe'
            $shortcutTargets = @(
                (Join-Path $desktop 'Polyfitty Studio.lnk'),
                (Join-Path $Root 'Polyfitty Studio.lnk')
            )
            foreach ($shortcutPath in $shortcutTargets) {
                $shortcut = $shell.CreateShortcut($shortcutPath)
                $shortcut.TargetPath = $wscript
                $shortcut.Arguments = ('//B //NoLogo "{0}"' -f $launcher)
                $shortcut.WorkingDirectory = $Root
                $shortcut.Description = 'Polyfitty Studio - 3D mesh reconstruction workspace'
                if (Test-Path -LiteralPath $appIcon -PathType Leaf) {
                    $shortcut.IconLocation = ('{0},0' -f $appIcon)
                }
                $shortcut.Save()
                Write-Host ("Created application shortcut: {0}" -f $shortcutPath) -ForegroundColor Green
            }
        } catch { Write-Host "Application shortcut creation skipped: $($_.Exception.Message)" -ForegroundColor Yellow }
    }
    else {
        Say 'Portable build mode: no desktop shortcuts or host-machine installation state is created.' 'Green'
    }

    Report-InstallProgress -Percent 100 -Detail $(if ($PortableBuild) { 'Self-contained portable core runtime provisioned' } else { 'Core Runtime installation complete' })
    if ($PortableBuild) { Say 'Embedded portable Python runtime and core dependencies installed. GPU setup is deferred to the portable first launch.' 'Green' } else { Say 'Core installation complete. The setup manager will configure optional GPU compute.' 'Green' }
    exit 0
}
catch {
    $failure = $_
    Write-Host "`n[INSTALL ERROR] $($failure.Exception.ToString())" -ForegroundColor Red
    if ($failure.InvocationInfo -and $failure.InvocationInfo.ScriptName) {
        Write-Host ("[INSTALL ERROR] File: {0}" -f $failure.InvocationInfo.ScriptName) -ForegroundColor Red
    }
    if ($failure.InvocationInfo -and $failure.InvocationInfo.ScriptLineNumber) {
        Write-Host ("[INSTALL ERROR] Line: {0}" -f $failure.InvocationInfo.ScriptLineNumber) -ForegroundColor Red
        Write-Host ("[INSTALL ERROR] Command: {0}" -f $failure.InvocationInfo.Line.Trim()) -ForegroundColor Red
    }
    if ($failure.ScriptStackTrace) {
        Write-Host ("[INSTALL ERROR] Stack: {0}" -f $failure.ScriptStackTrace) -ForegroundColor DarkYellow
    }
    Write-Host 'Installed Python packages and the private runtime are reused on retry; incomplete downloads are discarded.' -ForegroundColor Yellow
    Write-Host 'Review the captured stdout and stderr files in .cache/logs.' -ForegroundColor Yellow
    exit 1
}
