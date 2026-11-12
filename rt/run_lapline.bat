@echo off
rem Lap-line node: starts rt with the lap-line config and restarts it if it dies or stalls (rt\supervise.py).
rem usage: rt\run_lapline.bat [config]      (default rt\configs\lapline.toml)
rem Stop: Ctrl+C in this window (stops rt too). The status page: http://localhost:8765/
setlocal
cd /d "%~dp0.."
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=rt\configs\lapline.toml"
if not exist "%CONFIG%" (
  echo Config %CONFIG% not found.
  echo Copy rt\configs\lapline.example.toml to rt\configs\lapline.toml and set launch.event and camera.device.
  pause
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set CUDA_VISIBLE_DEVICES=-1
"%PY%" rt\supervise.py --config "%CONFIG%"
pause
