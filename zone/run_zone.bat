@echo off
rem Zone node launcher (spotter + PTZ tracking + page + recordings).
rem   zone\run_zone.bat                      uses zone\configs\site.toml if it exists, else the defaults
rem   zone\run_zone.bat zone\configs\x.toml  uses that config
rem Two windows: "PTZ recorder" (ffmpeg -c copy of the PTZ USB, only if recording.ptz_usb = true) and this one
rem (the tracker). Each restarts with backoff if it dies; the tracker always comes back in STOPPED.
rem Page: http://localhost:8090/   Stop everything: close both windows (or Ctrl-C in each).

setlocal
cd /d "%~dp0\.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "CFG=%~1"
if "%CFG%"=="" if exist "zone\configs\site.toml" set "CFG=zone\configs\site.toml"
set "CFGARG="
if not "%CFG%"=="" set CFGARG=--config "%CFG%"

rem the tracker must stay off the NVIDIA GPU (CPU ONNX Runtime); an empty value would be dropped by Windows
set "CUDA_VISIBLE_DEVICES=-1"

start "PTZ recorder" "%PY%" -m zone.supervisor --name ptz_recorder %CFGARG% -- "%PY%" -m zone.tools.record_ptz %CFGARG%
"%PY%" -m zone.supervisor --name zone %CFGARG% -- "%PY%" -m zone.app %CFGARG%
endlocal
