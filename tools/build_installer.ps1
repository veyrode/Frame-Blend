# Builds the Frame Blend installer: build\installer\FrameBlend-Setup-<version>.exe
#
# - Builds the Rust core and the Adobe plugin (Release), as in
#   docs\plugin-build.md;
# - signs the Premiere companion (tools\sign_panel.ps1) unless
#   build\panel\FrameBlend is already signed and newer than its sources;
# - compiles installer\FrameBlend.iss with Inno Setup 6 (ISCC.exe).
#
# The Adobe SDKs are not in the repository (Adobe's licence); see
# docs\plugin-build.md. -AeSdk is the After Effects SDK's Examples folder
# (Headers\AE_Effect.h, Resources\PiPLtool.exe), -PrSdk the Premiere Pro SDK
# folder holding Headers\PrSDKGPUFilter.h.
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$AeSdk = $(if ($env:FRAMEBLEND_AE_SDK) { $env:FRAMEBLEND_AE_SDK } else { Join-Path (Split-Path -Parent $PSScriptRoot) 'dependence\sdks\Adobe\AfterEffectsSDK_25.6_61_win\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK\Examples' }),
    [string]$PrSdk = $(if ($env:FRAMEBLEND_PR_SDK) { $env:FRAMEBLEND_PR_SDK } else { Join-Path (Split-Path -Parent $PSScriptRoot) 'dependence\sdks\Adobe\PremierePro_24.0_SDK_win' }),
    [string]$Cmake = $(if (Get-Command cmake -ErrorAction SilentlyContinue) { (Get-Command cmake).Source } else { 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' })
)
$ErrorActionPreference = 'Stop'
Set-Location $Repo

$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 not found (winget install JRSoftware.InnoSetup)' }

cargo build --release
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
$coreLib = Join-Path $Repo 'target\release\frame_blend.lib'
if (-not (Test-Path -LiteralPath (Join-Path $AeSdk 'Headers\AE_Effect.h'))) { throw "After Effects SDK not found at $AeSdk (-AeSdk)" }
if (-not (Test-Path -LiteralPath (Join-Path $PrSdk 'Headers\PrSDKGPUFilter.h'))) { throw "Premiere Pro SDK not found at $PrSdk (-PrSdk)" }
& $Cmake -S plugins/adobe -B build/adobe -DAE_SDK_PATH="$AeSdk" -DPR_SDK_PATH="$PrSdk" -DFRAME_BLEND_CORE_LIB="$coreLib" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
& $Cmake --build build/adobe --config Release
if ($LASTEXITCODE -ne 0) { throw 'plugin build failed' }

$panel = Join-Path $Repo 'build\panel\FrameBlend'
$signature = Join-Path $panel 'META-INF\signatures.xml'
$newestSource = Get-ChildItem (Join-Path $Repo 'plugins\premiere-panel') -Recurse -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not (Test-Path -LiteralPath $signature) -or (Get-Item $signature).LastWriteTime -lt $newestSource.LastWriteTime) {
    & (Join-Path $PSScriptRoot 'sign_panel.ps1') -Repo $Repo
    if ($LASTEXITCODE -ne 0) { throw 'signing the companion failed' }
}

foreach ($dll in 'avutil-60.dll', 'swresample-6.dll', 'avcodec-62.dll', 'avformat-62.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $Repo "dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\$dll"))) {
        throw "missing FFmpeg DLL $dll (see dependence\sdks\FFmpeg_8.1_lgpl_win64\README.md)"
    }
}

& $iscc /Q "/DRepo=$Repo" (Join-Path $Repo 'installer\FrameBlend.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
Get-ChildItem (Join-Path $Repo 'build\installer') -Filter 'FrameBlend-Setup-*.exe' | ForEach-Object { "built $($_.FullName) $($_.Length) bytes" }
