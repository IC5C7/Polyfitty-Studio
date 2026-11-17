@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%TEMP%"
if not exist "%PROJECT_ROOT%INSTALL.vbs" exit /b 1
start "" /b "%SystemRoot%\System32\wscript.exe" "%PROJECT_ROOT%INSTALL.vbs"
exit /b 0
