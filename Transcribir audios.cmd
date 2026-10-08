@echo off
setlocal
title Transcripcion local de audios
pushd "%~dp0"
if errorlevel 1 goto ubicacion_error
if not exist ".venv\Scripts\python.exe" goto entorno_error
if not exist "transcribir.py" goto entorno_error
if not exist "configuracion.json" goto entorno_error
echo Revisando la carpeta configurada. Solo se transcribiran los pendientes.
echo No cierre esta ventana mientras trabaja.
echo.
".venv\Scripts\python.exe" -B "transcribir.py" --todos
set "transcripcion_exit=%ERRORLEVEL%"
echo.
echo Codigo de salida: %transcripcion_exit%
if not "%transcripcion_exit%"=="0" echo Hay errores. Consulte el registro indicado arriba.
echo Pulse una tecla para cerrar.
pause >nul
popd
exit /b %transcripcion_exit%

:entorno_error
echo Falta Python de .venv, transcribir.py o configuracion.json.
echo Ejecute Instalar.cmd y prepare el modelo siguiendo README.md.
echo Si .venv ya existe, no reinstale: revise los pasos de configuracion del README.
echo Extraiga todos los archivos y mantenga este lanzador dentro del proyecto.
pause
popd
exit /b 1

:ubicacion_error
echo No se pudo abrir la carpeta del proyecto.
pause
exit /b 1
