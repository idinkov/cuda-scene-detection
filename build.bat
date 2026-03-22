@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "SCRIPTS_DIR=%PROJECT_ROOT%scripts"
set BUILD_DIR=build
set GENERATOR="Visual Studio 17 2022"
set ARCH=x64
set CONFIG=Release

echo.
echo ============================================
echo   nvdec_scene_detect - Build Script
echo ============================================
echo.

REM === DEPENDENCY CHECKS ===
echo [*] Checking dependencies...
echo.

REM --- Visual Studio Build Tools ---
call "%SCRIPTS_DIR%\install_vs_buildtools.bat"
if errorlevel 1 (
    echo [!] Cannot proceed without Visual Studio Build Tools.
    exit /b 1
)

REM --- CMake ---
call "%SCRIPTS_DIR%\install_cmake.bat"
if errorlevel 1 (
    echo [!] Cannot proceed without CMake.
    exit /b 1
)
REM Ensure cmake is on PATH after install
where cmake >nul 2>&1 || set "PATH=%PATH%;C:\Program Files\CMake\bin"

REM --- CUDA Toolkit ---
call "%SCRIPTS_DIR%\install_cuda.bat"
if errorlevel 1 (
    echo [!] Cannot proceed without CUDA Toolkit.
    exit /b 1
)
REM Pick up CUDA path for this session (installer writes it to temp file)
if exist "%TEMP%\cuda_path.txt" (
    set /p CUDA_FOUND_PATH=<"%TEMP%\cuda_path.txt"
)
REM Also scan directly for CUDA path (prefer version matching driver)
if not defined CUDA_PATH (
    for /d %%v in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1") do (
        if exist "%%~v\bin\nvcc.exe" set "CUDA_PATH=%%~v"
    )
)
if not defined CUDA_PATH (
    for /d %%v in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        set "CUDA_PATH=%%~v"
    )
)
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
    set "CudaToolkitDir=%CUDA_PATH%"
    echo [*] CUDA_PATH set to %CUDA_PATH%
)

REM --- FFmpeg ---
call "%SCRIPTS_DIR%\install_ffmpeg.bat"
if errorlevel 1 (
    echo [!] Cannot proceed without FFmpeg.
    exit /b 1
)

echo.
echo [*] All dependencies satisfied.
echo.

REM === Clean old build (optional) ===
if exist %BUILD_DIR% (
    echo [*] Removing old build directory...
    rmdir /s /q %BUILD_DIR%
)

REM === Create build folder ===
mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM === Run CMake configure ===
echo [*] Configuring project with CMake...
echo [*] Using CUDA Toolkit: %CUDA_PATH%
cmake .. -G %GENERATOR% -A %ARCH% -DCMAKE_CUDA_ARCHITECTURES=120 -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe"

if errorlevel 1 (
    echo [!] CMake configuration failed.
    cd ..
    exit /b 1
)

REM === Build project ===
echo [*] Building project (%CONFIG%)...
cmake --build . --config %CONFIG% -j

if errorlevel 1 (
    echo [!] Build failed.
    cd ..
    exit /b 1
)

REM --- After successful build ---
set OUTPUT_DIR=%CONFIG%
echo [*] Copying FFmpeg DLLs...
xcopy /Y /I "%PROJECT_ROOT%dependencies\ffmpeg\bin\*.dll" "%OUTPUT_DIR%" 2>nul

echo [*] Copying CUDA DLLs...
REM Copy cudart DLL (shared runtime) and nvcuvid
if defined CUDA_PATH (
    xcopy /Y /I "%CUDA_PATH%\bin\x64\cudart64_*.dll" "%OUTPUT_DIR%" 2>nul
    xcopy /Y /I "%CUDA_PATH%\bin\cudart64_*.dll" "%OUTPUT_DIR%" 2>nul
)
xcopy /Y /I "C:\Windows\System32\nvcuvid.dll" "%OUTPUT_DIR%" 2>nul

echo.
echo [OK] Build completed successfully!
echo [*] Executable: %cd%\%CONFIG%\nvdec_scene_detect.exe
echo.

cd ..
endlocal
