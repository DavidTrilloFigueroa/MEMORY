@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
where py.exe >nul 2>nul
if errorlevel 1 goto python_directo
py -3.11 -B instalar.py
set "instalacion_exit=%ERRORLEVEL%"
goto final
:python_directo
where python.exe >nul 2>nul
if errorlevel 1 goto falta_python
python -B instalar.py
set "instalacion_exit=%ERRORLEVEL%"
goto final
:falta_python
echo Instale Python 3.11 x64 segun README. No se ha instalado nada.
set "instalacion_exit=1"
:final
echo Codigo de salida: %instalacion_exit%
echo Si no se encuentra Python, consulte la ejecucion con ruta completa en README.
pause
popd
exit /b %instalacion_exit%
