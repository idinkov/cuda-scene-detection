@echo off
setlocal

echo [*] Checking for Visual Studio C++ Build Tools...

REM Check for cl.exe (MSVC compiler)
where cl >nul 2>&1
if not errorlevel 1 (
    echo [OK] MSVC compiler found on PATH.
    exit /b 0
)

REM Check common VS 2022 install locations
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" goto :vs_found
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" goto :vs_found
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" goto :vs_found
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" goto :vs_found

echo [!] Visual Studio 2022 not found. Installing Build Tools via winget...
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements --accept-source-agreements -e
if errorlevel 1 (
    echo [!] Installation failed. Install manually: https://visualstudio.microsoft.com/downloads/
    exit /b 1
)
echo [OK] Visual Studio 2022 Build Tools installed. Restart terminal for PATH changes.
exit /b 0

:vs_found
echo [OK] Visual Studio 2022 found.
exit /b 0
