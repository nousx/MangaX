# Builds MangaX-Setup.exe with the C# compiler bundled in .NET Framework 4.x.
# Usage: pwsh packaging/installer/build.ps1 [-Output <path>]
param(
    [string]$Output = (Join-Path $PSScriptRoot 'build\MangaX-Setup.exe')
)

$ErrorActionPreference = 'Stop'

# Pinned 7-Zip extractor (LGPL), embedded so the setup can unpack .7z volumes.
$extractorUrl = 'https://github.com/ip7z/7zip/releases/download/26.04/7zr.exe'
$extractorSha256 = '256feca8e274e5da655e2a284fabafd9f554365eb164862089dacd4e8276d282'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$buildDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

$extractor = Join-Path $buildDir '7zr.exe'
if (-not (Test-Path $extractor) -or (Get-FileHash $extractor -Algorithm SHA256).Hash -ne $extractorSha256) {
    Invoke-WebRequest -Uri $extractorUrl -OutFile $extractor
}
if ((Get-FileHash $extractor -Algorithm SHA256).Hash -ne $extractorSha256) {
    Remove-Item $extractor -Force
    throw '7zr.exe does not match the pinned SHA-256.'
}

$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { throw "C# compiler not found at $csc." }

$icon = Join-Path $repoRoot 'desktop_qt_ui\ui\icons\icon.ico'
$source = Join-Path $PSScriptRoot 'MangaXSetup.cs'
& $csc /nologo /target:winexe /optimize+ /platform:anycpu /codepage:65001 `
    "/out:$Output" "/win32icon:$icon" "/resource:$extractor,7zr.exe" `
    /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll `
    /reference:System.Windows.Forms.dll /reference:System.Management.dll `
    /reference:System.Web.Extensions.dll `
    $source
if ($LASTEXITCODE -ne 0) { throw 'Compiling MangaX-Setup.exe failed.' }

$log = Join-Path $buildDir 'self-test.log'
Remove-Item $log -Force -ErrorAction SilentlyContinue
$test = Start-Process -FilePath $Output -ArgumentList '--self-test', '--log', "`"$log`"" -Wait -PassThru
Get-Content $log
if ($test.ExitCode -ne 0) { throw 'MangaX-Setup.exe self-test failed.' }

Write-Host "Built $Output ($((Get-Item $Output).Length) bytes)"
