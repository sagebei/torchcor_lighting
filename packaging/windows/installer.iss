; Inno Setup script for TorchCor-Lighting-windows-<arch>-setup.exe.
;
; Compiled by `python packaging/build.py --archive` on Windows, which passes the values
; below with /D (from packaging/appinfo.py and the PyInstaller bundle in dist/).
;
; The installer asks whether to install for the current user only (no administrator
; rights; %LOCALAPPDATA%\Programs) or for all users (Program Files). Settings, history
; and keychain entries live in the user's profile and are kept on uninstall.

#ifndef AppVersion
  #error Compile with packaging/build.py --archive: it defines AppVersion, SourceDir, ...
#endif

[Setup]
; Windows recognises an installed copy, and upgrades it in place, by AppId: never change it.
AppId={{8A9A7BEB-B346-4A5F-B40F-D7C83F3B5119}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#Publisher}
AppCopyright={#Copyright}
VersionInfoVersion={#FileVersion}
VersionInfoProductName={#AppName}
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed={#Architectures}
ArchitecturesInstallIn64BitMode={#Architectures}
; Qt 6 needs Windows 10 or newer
MinVersion=10.0
WizardStyle=modern
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#ExeName}
UninstallDisplayName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
Compression=lzma2
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade replaces the bundle as a whole: no module of the previous version is left behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#ExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
