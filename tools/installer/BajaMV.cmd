@echo off
rem Baja MV from a console window (installed next to python\ and app\ in Program Files\BajaMV).
rem   BajaMV.cmd                 the launcher (tray + page), with its messages in this window
rem   BajaMV.cmd --self-test     check the installation (exit code 0 = all good)
rem   BajaMV.cmd --shutdown      stop a running launcher and both roles
rem -P: never put the current folder on the import path, so this always runs the installed code (app\, found
rem through the .pth file), even when started from a folder that has its own launcher\ (e.g. a repo checkout)
"%~dp0python\python.exe" -P -m launcher %*
set "RC=%ERRORLEVEL%"
if /i "%~1"=="--self-test" if not defined CI pause
exit /b %RC%
