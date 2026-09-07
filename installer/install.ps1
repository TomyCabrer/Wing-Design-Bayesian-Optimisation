# Get AeroBO onto this machine and start it. Windows.
#
#   irm https://raw.githubusercontent.com/TomyCabrer/AeroBO/main/installer/install.ps1 | iex
#
# Downloads the app into %USERPROFILE%\AeroBO and runs AeroBO.bat, which
# builds its own private Python environment there. Nothing is installed
# system-wide; deleting the folder removes everything.

$ErrorActionPreference = "Stop"

$Repo   = if ($env:AEROBO_REPO)   { $env:AEROBO_REPO }   else { "https://github.com/TomyCabrer/AeroBO" }
$Dir    = if ($env:AEROBO_DIR)    { $env:AEROBO_DIR }    else { Join-Path $HOME "AeroBO" }
$Branch = if ($env:AEROBO_BRANCH) { $env:AEROBO_BRANCH } else { "main" }

Write-Host ""
Write-Host "  AeroBO - install"
Write-Host ""

if (Test-Path (Join-Path $Dir ".git")) {
    Write-Host "  updating the copy already in $Dir"
    git -C $Dir pull --ff-only
} elseif (Test-Path (Join-Path $Dir "launch.py")) {
    Write-Host "  using the copy already in $Dir"
} elseif (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "  downloading into $Dir"
    try {
        git clone --depth 1 --branch $Branch $Repo $Dir
    } catch {
        Write-Host ""
        Write-Host "  Could not download it. If the repository is still private,"
        Write-Host "  GitHub refuses an anonymous clone - ask for access, or download"
        Write-Host "  the zip from the repository page, unpack it, and double-click"
        Write-Host "  AeroBO.bat."
        Write-Host ""
        exit 1
    }
} else {
    Write-Host "  downloading into $Dir (no git here - fetching the zip)"
    $tmp = Join-Path $env:TEMP "aerobo.zip"
    Invoke-WebRequest -Uri "$Repo/archive/refs/heads/$Branch.zip" -OutFile $tmp
    $stage = Join-Path $env:TEMP "aerobo-unpack"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    Expand-Archive -Path $tmp -DestinationPath $stage
    $inner = Get-ChildItem $stage | Select-Object -First 1
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Copy-Item -Path (Join-Path $inner.FullName "*") -Destination $Dir -Recurse -Force
    Remove-Item -Recurse -Force $stage, $tmp
}

Write-Host "  starting it - the first run installs what it needs and takes a few minutes"
Write-Host ""
& (Join-Path $Dir "AeroBO.bat")
