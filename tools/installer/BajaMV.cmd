@echo off
rem Baja MV from a console window (installed next to python\ and app\ in Program Files\BajaMV).
rem   BajaMV.cmd                 the launcher (tray + page), with its messages in this window
rem   BajaMV.cmd --self-test     check the installation (exit code 0 = all good)
rem   BajaMV.cmd --shutdown      stop a running launcher and both roles
"%~dp0python\python.exe" -m launcher %*
set "RC=%ERRORLEVEL%"
if /i "%~1"=="--self-test" if not defined CI pause
exit /b %RC%
