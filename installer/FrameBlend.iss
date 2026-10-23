; Frame Blend installer (Inno Setup 6). Build with tools\build_installer.ps1,
; which compiles the plugin first and passes the repository path as Repo.
;
; Installs, for all users:
; - FrameBlend.aex and the FFmpeg DLLs its Premiere file reader loads, in
;   Adobe's shared MediaCore plug-in folder (Premiere Pro, After Effects and
;   Media Encoder 2020 and later all load effects from there);
; - the signed Premiere companion (build\panel\FrameBlend, made by
;   tools\sign_panel.ps1) in the machine-wide CEP extensions folder.
; The uninstaller and the licence texts go to Program Files\Frame Blend.

#define AppVersion "1.4.0"
#ifndef Repo
  #define Repo ".."
#endif

[Setup]
AppId={{94BDF341-F24C-43C8-B396-07915CDE38CD}
AppName=Frame Blend
AppVersion={#AppVersion}
AppVerName=Frame Blend {#AppVersion}
VersionInfoVersion={#AppVersion}
UninstallDisplayName=Frame Blend {#AppVersion} (Premiere Pro / After Effects)
DefaultDirName={commonpf64}\Adobe\Common\Plug-ins\7.0\MediaCore\FrameBlend
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=no
UninstallFilesDir={commonpf64}\Frame Blend
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#Repo}\build\installer
OutputBaseFilename=FrameBlend-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The hosts are checked for in InitializeSetup; the Restart Manager would
; offer to close them too, which can lose unsaved work.
CloseApplications=no

[Files]
Source: "{#Repo}\build\adobe\Release\FrameBlend.aex"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Repo}\dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\avutil-60.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Repo}\dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\swresample-6.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Repo}\dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\avcodec-62.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Repo}\dependence\sdks\FFmpeg_8.1_lgpl_win64\bin\avformat-62.dll"; DestDir: "{app}"; Flags: ignoreversion
; The companion's files must match its signature exactly: the folder is
; replaced as a whole (InstallDelete below).
Source: "{#Repo}\build\panel\FrameBlend\*"; DestDir: "{commoncf32}\Adobe\CEP\extensions\FrameBlend"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Repo}\installer\README.txt"; DestDir: "{commonpf64}\Frame Blend"; Flags: ignoreversion isreadme
Source: "{#Repo}\dependence\sdks\FFmpeg_8.1_lgpl_win64\LICENSE.txt"; DestDir: "{commonpf64}\Frame Blend"; DestName: "FFmpeg-LICENSE.txt"; Flags: ignoreversion

[InstallDelete]
Type: filesandordirs; Name: "{commoncf32}\Adobe\CEP\extensions\FrameBlend"
; A per-user copy of the companion (development installs) would run beside
; this one.
Type: filesandordirs; Name: "{userappdata}\Adobe\CEP\extensions\FrameBlend"

[UninstallDelete]
Type: filesandordirs; Name: "{commoncf32}\Adobe\CEP\extensions\FrameBlend"
Type: files; Name: "{app}\FrameBlend-log.txt"
Type: dirifempty; Name: "{app}"

[Code]
{ Names of the Adobe hosts that are running, one per line. }
function RunningHosts(): String;
var
  Locator, Service, Found: Variant;
  Names: TStringList;
  I: Integer;
begin
  Result := '';
  Names := TStringList.Create;
  try
    Names.CommaText := '"Adobe Premiere Pro.exe","AfterFX.exe","Adobe Media Encoder.exe"';
    try
      Locator := CreateOleObject('WbemScripting.SWbemLocator');
      Service := Locator.ConnectServer('.', 'root\CIMV2');
      for I := 0 to Names.Count - 1 do
      begin
        Found := Service.ExecQuery('SELECT Name FROM Win32_Process WHERE Name = ''' + Names[I] + '''');
        if Found.Count > 0 then
          Result := Result + '  ' + Names[I] + #13#10;
      end;
    except
      Result := '';
    end;
  finally
    Names.Free;
  end;
end;

{ Waits for the user to close the Adobe hosts; False when they cancel. }
function HostsClosed(Action: String): Boolean;
var
  Running: String;
begin
  Result := True;
  Running := RunningHosts();
  while Running <> '' do
  begin
    if SuppressibleMsgBox('Please close these programs before Frame Blend is ' + Action + ':' + #13#10#13#10 +
                          Running + #13#10 + 'Click Retry once they are closed.',
                          mbError, MB_RETRYCANCEL, IDCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
    Running := RunningHosts();
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := HostsClosed('installed');
end;

function InitializeUninstall(): Boolean;
begin
  Result := HostsClosed('removed');
end;
