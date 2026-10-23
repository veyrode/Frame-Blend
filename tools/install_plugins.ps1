# Installs the Release plugin to the standard Windows location. Run elevated:
#   Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File tools\install_plugins.ps1'
#
# - Adobe: C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\FrameBlend\FrameBlend.aex
#   plus the FFmpeg DLLs it decodes recordings with. Premiere Pro and After
#   Effects (2020 and later) both load effects from this MediaCore folder.
# - Creates FrameBlend-log.txt beside the plugin, writable by BUILTIN\Users
#   because the hosts run without elevation; its existence turns the debug log
#   on (a release install from installer\FrameBlend.iss writes no log).
# For other people's machines, build the installer (tools\build_installer.ps1).
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Transcript = ''
)
$ErrorActionPreference = 'Stop'
if ($Transcript) { Start-Transcript -Path $Transcript -Force | Out-Null }
try {
    $adobeDir = 'C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\FrameBlend'
    New-Item -ItemType Directory -Force -Path $adobeDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $Repo 'build\adobe\Release\FrameBlend.aex') -Destination $adobeDir -Force
    # FFmpeg runtime for the Premiere file reader (loaded from the plugin's own folder).
    foreach ($dll in 'avutil-60.dll', 'swresample-6.dll', 'avcodec-62.dll', 'avformat-62.dll') {
        Copy-Item -LiteralPath (Join-Path $Repo "dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\$dll") -Destination $adobeDir -Force
    }
    # Frame Blend panel for Premiere (per user), the signed copy made by
    # tools\sign_panel.ps1. The folder is replaced as a whole: any file that is
    # not part of the signature makes Premiere reject the panel.
    $panelDir = Join-Path $env:APPDATA 'Adobe\CEP\extensions\FrameBlend'
    $signedPanel = Join-Path $Repo 'build\panel\FrameBlend'
    if (-not (Test-Path -LiteralPath (Join-Path $signedPanel 'META-INF\signatures.xml'))) {
        throw "no signed panel in $signedPanel; run tools\sign_panel.ps1 first"
    }
    if (Test-Path -LiteralPath $panelDir) { Remove-Item -LiteralPath $panelDir -Recurse -Force }
    # A release install (installer\FrameBlend.iss) puts the companion in the
    # machine-wide folder; both copies would run.
    $machinePanel = Join-Path ${env:CommonProgramFiles(x86)} 'Adobe\CEP\extensions\FrameBlend'
    if (Test-Path -LiteralPath $machinePanel) { Remove-Item -LiteralPath $machinePanel -Recurse -Force; "removed $machinePanel" }
    Copy-Item -LiteralPath $signedPanel -Destination $panelDir -Recurse
    Get-ChildItem -LiteralPath $panelDir -Recurse -File | ForEach-Object { "installed $($_.FullName)" }
    $log = Join-Path $adobeDir 'FrameBlend-log.txt'
    if (-not (Test-Path -LiteralPath $log)) { New-Item -ItemType File -Path $log | Out-Null }
    icacls $log /grant '*S-1-5-32-545:(M)' | Out-Null
    Get-Item -LiteralPath (Join-Path $adobeDir '*') | ForEach-Object { "installed $($_.FullName) $($_.Length) bytes" }
    'done'
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
}
