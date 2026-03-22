@echo off
setlocal

echo [*] Checking for CUDA Toolkit (nvcc)...
where nvcc >nul 2>&1
if not errorlevel 1 (
    echo [OK] CUDA Toolkit found:
    nvcc --version | findstr /R "release"
    exit /b 0
)

REM Check common install paths even if not on PATH
for /d %%v in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
    if exist "%%v\bin\nvcc.exe" (
        echo [OK] CUDA Toolkit found at %%v
        REM Write path to temp file so caller can pick it up
        echo %%v> "%TEMP%\cuda_path.txt"
        exit /b 0
    )
)

echo [!] CUDA Toolkit not found. Installing via winget...
winget install Nvidia.CUDA --accept-package-agreements --accept-source-agreements -e

if errorlevel 1 (
    echo [!] CUDA Toolkit installation failed.
    echo     Try installing manually from https://developer.nvidia.com/cuda-toolkit
    exit /b 1
)

REM Find the newly installed CUDA path
for /d %%v in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
    if exist "%%v\bin\nvcc.exe" (
        echo %%v> "%TEMP%\cuda_path.txt"
    )
)

echo [OK] CUDA Toolkit installed.
echo [!] You may need to restart your terminal for PATH changes to take effect.
exit /b 0
