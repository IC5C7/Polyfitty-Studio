@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%TEMP%"
echo This removes the private Python runtime, installed modules and cached installers.
echo Your input files and generated output files are NOT deleted.
choice /m "Continue"
if errorlevel 2 exit /b 0
rmdir /s /q "%PROJECT_ROOT%.runtime" 2>nul
rmdir /s /q "%PROJECT_ROOT%.cache" 2>nul
rmdir /s /q "%PROJECT_ROOT%vendor\openvdb_runtime" 2>nul
if exist "%USERPROFILE%\Desktop\Polyfitty Studio.lnk" del /q "%USERPROFILE%\Desktop\Polyfitty Studio.lnk"
if exist "%USERPROFILE%\Desktop\PolyFit Studio.lnk" del /q "%USERPROFILE%\Desktop\PolyFit Studio.lnk"
echo Runtime, retired engine runtime and desktop shortcut removed. Project source and outputs were kept.
exit /b 0
