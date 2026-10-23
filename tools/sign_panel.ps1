# Signs the Premiere panel (plugins\premiere-panel) so Premiere loads it without
# CEP debug mode, which Premiere ignores when started normally from Windows.
#
# - Creates a fresh self-signed certificate with a random password (nothing to
#   keep: every signature carries its own certificate).
# - Signs with Adobe's ZXPSignCmd (-Tool; download it from
#   https://github.com/Adobe-CEP/CEP-Resources, folder ZXPSignCMD), timestamped
#   when that works, and verifies it.
# - Output: build\panel\FrameBlend.zxp and its contents unpacked in
#   build\panel\FrameBlend, which tools\install_plugins.ps1 installs as is
#   (the installed files must match the signature exactly).
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Tool = (Join-Path (Split-Path -Parent $PSScriptRoot) 'dependence\sdks\ZXPSignCmd_4.1.103_win64\ZXPSignCmd.exe')
)
$ErrorActionPreference = 'Stop'
$tool = $Tool
if (-not (Test-Path -LiteralPath $tool)) { throw "ZXPSignCmd not found at $tool (-Tool)" }
$source = Join-Path $Repo 'plugins\premiere-panel'
$out = Join-Path $Repo 'build\panel'
$zxp = Join-Path $out 'FrameBlend.zxp'
$cert = Join-Path $out 'FrameBlend.p12'
$unpacked = Join-Path $out 'FrameBlend'

if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path $out | Out-Null
# ZXPSignCmd writes META-INF and mimetype into the folder it signs: sign a copy.
$staging = Join-Path $out 'staging'
Copy-Item -LiteralPath $source -Destination $staging -Recurse
$source = $staging
$password = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })

& $tool -selfSignedCert US CA 'Frame Blend' 'Frame Blend' $password $cert -validityDays 3650
if ($LASTEXITCODE -ne 0) { throw "certificate creation failed ($LASTEXITCODE)" }
& $tool -sign $source $zxp $cert $password -tsa 'http://timestamp.digicert.com'
if ($LASTEXITCODE -ne 0) {
    # ZXPSignCmd 4.1.103 crashes on current timestamp servers (exit 139 with
    # DigiCert and Sectigo, 2026-10-05).
    'timestamping failed; signing without a timestamp (valid while the certificate is, 10 years)'
    if (Test-Path -LiteralPath $zxp) { Remove-Item -LiteralPath $zxp -Force }
    & $tool -sign $source $zxp $cert $password
    if ($LASTEXITCODE -ne 0) { throw "signing failed ($LASTEXITCODE)" }
}
& $tool -verify $zxp
if ($LASTEXITCODE -ne 0) { throw "signature verification failed ($LASTEXITCODE)" }
Remove-Item -LiteralPath $cert -Force
Remove-Item -LiteralPath $staging -Recurse -Force

$zip = Join-Path $out 'FrameBlend.zip'
Copy-Item -LiteralPath $zxp -Destination $zip
Expand-Archive -LiteralPath $zip -DestinationPath $unpacked
Remove-Item -LiteralPath $zip
Get-ChildItem -LiteralPath $unpacked -Recurse -File | ForEach-Object { "signed $($_.FullName.Substring($unpacked.Length + 1))" }
