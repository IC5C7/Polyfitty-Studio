@echo off
setlocal

wscript.exe //B //NoLogo "%~dp0BUILD_PORTABLE_EXE.vbs"
exit /b %ERRORLEVEL%
