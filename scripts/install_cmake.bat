@echo off
setlocal

echo [*] Checking for CMake...
where cmake >nul 2>&1
if %errorlevel%==0 (
    echo [OK] CMake found:
    cmake --version | findstr /R "cmake version"
    exit /b 0
)

echo [!] CMake not found. Installing via winget...
winget install Kitware.CMake --accept-package-agreements --accept-source-agreements -e

if errorlevel 1 (
    echo [!] CMake installation failed.
    echo     Try installing manually from https://cmake.org/download/
    exit /b 1
)

REM Refresh PATH so cmake is available in this session
set "PATH=%PATH%;C:\Program Files\CMake\bin"

echo [OK] CMake installed.
cmake --version | findstr /R "cmake version"
exit /b 0
