@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%TEMP%"
if not exist "%PROJECT_ROOT%START_HIDDEN.vbs" exit /b 1
"%SystemRoot%\System32\wscript.exe" "%PROJECT_ROOT%START_HIDDEN.vbs"
exit /b %ERRORLEVEL%
