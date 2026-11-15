; Baja MV installer (Inno Setup 6). Compiled by build_installer.ps1, which stages everything first:
;   iscc /DAppVersion=<version> /DStageDir=<repo>\build\stage /DOutputDir=<repo>\dist BajaMV.iss
;
; Installs, per machine, into Program Files\BajaMV: python\ (private Python 3.12 + packages), ffmpeg\bin\,
; app\ (the code + models), BajaMV.cmd. Writable data goes to a data folder chosen here (default C:\BajaMV), kept on
; uninstall and on upgrade. Silent install: BajaMV-Setup-x.exe /VERYSILENT /DATADIR=D:\BajaMV /TASKS=firewall

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef StageDir
  #define StageDir "..\..\build\stage"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
; never change AppId: it is how an upgrade finds the installed copy
AppId={{6B1E4C1E-6C1B-4E0D-9C1A-BA1A5A0C2026}
AppName=Baja MV
AppVersion={#AppVersion}
AppVerName=Baja MV {#AppVersion}
AppPublisher=Baja SAE Brasil / RD Tests
DefaultDirName={autopf}\BajaMV
DefaultGroupName=Baja MV
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=BajaMV-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\app\bajamv.ico
UninstallDisplayName=Baja MV
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes

[Languages]
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "firewall"; Description: "Liberar as portas do Baja MV no firewall (redes privadas: 8765, 8090, 8890, 8780)"
Name: "startatlogin"; Description: "Abrir o Baja MV ao entrar no Windows"; Flags: unchecked

[Dirs]
; the data folder: everyone on this PC may write to it; never removed by the uninstaller
Name: "{code:GetDataDir}"; Permissions: users-modify; Flags: uninsneveruninstall

[InstallDelete]
; an upgrade replaces the program completely (no stale files from an older version); the data folder is not touched
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\ffmpeg"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[INI]
; tells the launcher where the data folder is (launcher/paths.py)
Filename: "{app}\app\install.ini"; Section: "paths"; Key: "data_root"; String: "{code:GetDataDir}"

[Icons]
Name: "{group}\Baja MV"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m launcher"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\bajamv.ico"
Name: "{group}\Baja MV (autoteste)"; Filename: "{app}\BajaMV.cmd"; Parameters: "--self-test"; WorkingDir: "{app}\app"
Name: "{group}\Pasta de dados do Baja MV"; Filename: "{code:GetDataDir}"
Name: "{commonstartup}\Baja MV"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m launcher --no-browser"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\bajamv.ico"; Tasks: startatlogin

[Run]
; firewall: delete first, so an upgrade never doubles the rules
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Baja MV (portas)"""; Flags: runhidden; Tasks: firewall
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Baja MV (portas)"" dir=in action=allow protocol=TCP localport=8765,8090,8890,8780 profile=private"; Flags: runhidden; Tasks: firewall
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Baja MV (python)"""; Flags: runhidden; Tasks: firewall
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Baja MV (python)"" dir=in action=allow program=""{app}\python\python.exe"" profile=private"; Flags: runhidden; Tasks: firewall
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Baja MV (python)"" dir=in action=allow program=""{app}\python\pythonw.exe"" profile=private"; Flags: runhidden; Tasks: firewall
Filename: "{app}\python\pythonw.exe"; Parameters: "-m launcher"; WorkingDir: "{app}\app"; Description: "Abrir o Baja MV agora"; Flags: postinstall nowait skipifsilent runasoriginaluser

[UninstallRun]
; stop a running launcher and both roles first (they hold files of the program)
Filename: "{app}\python\python.exe"; Parameters: "-m launcher --shutdown"; WorkingDir: "{app}\app"; Flags: runhidden waituntilterminated; RunOnceId: "StopBajaMV"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Baja MV (portas)"""; Flags: runhidden; RunOnceId: "FirewallPorts"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Baja MV (python)"""; Flags: runhidden; RunOnceId: "FirewallPython"

[UninstallDelete]
; files Python may have written next to the code (bytecode caches); the data folder is elsewhere and is kept
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\python"

[Code]
var
  DataDirPage: TInputDirWizardPage;

procedure InitializeWizard;
var
  Suggested: String;
begin
  Suggested := ExpandConstant('{param:DATADIR}');
  if Suggested = '' then
    Suggested := 'C:\BajaMV';
  DataDirPage := CreateInputDirPage(wpSelectDir,
    'Pasta de dados', 'Onde o Baja MV guarda configurações, gravações e logs?',
    'Escolha uma pasta com bastante espaço (as gravações ocupam ~4 GB por hora). Ela fica fora de Arquivos de ' +
    'Programas porque o programa escreve nela, e é mantida ao atualizar ou desinstalar.',
    False, '');
  DataDirPage.Add('');
  { order: the previous install's choice, then /DATADIR=... on the command line, then C:\BajaMV }
  DataDirPage.Values[0] := GetPreviousData('DataDir', Suggested);
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  { an upgrade keeps the data folder it already has }
  Result := (PageID = DataDirPage.ID) and (GetPreviousData('DataDir', '') <> '');
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'DataDir', DataDirPage.Values[0]);
end;

function GetDataDir(Param: String): String;
begin
  Result := DataDirPage.Values[0];
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  OldPython: String;
begin
  { upgrade: ask the running launcher (if any) to stop both roles and exit, so no file is in use }
  OldPython := ExpandConstant('{app}\python\python.exe');
  if FileExists(OldPython) then
    Exec(OldPython, '-m launcher --shutdown', ExpandConstant('{app}\app'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := '';
end;
