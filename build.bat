@echo off
setlocal

REM === CONFIGURATION ===
set BUILD_DIR=build
set GENERATOR="Visual Studio 17 2022"
set ARCH=x64
set CONFIG=Release

echo.
echo [*] Starting build for nvdec_scene_detect
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
cmake .. -G %GENERATOR% -A %ARCH%

if errorlevel 1 (
    echo [!] CMake configuration failed.
    exit /b 1
)

REM === Build project ===
echo [*] Building project (%CONFIG%)...
cmake --build . --config %CONFIG% -j

if errorlevel 1 (
    echo [!] Build failed.
    exit /b 1
)

REM --- After successful build ---
set OUTPUT_DIR=%CONFIG%
echo [*] Copying FFmpeg DLLs...
xcopy /Y /I "C:\Users\thanks\CLionProjects\cuda-scene-detection\dependencies\ffmpeg\bin\*.dll" "%OUTPUT_DIR%"

echo [*] Copying CUDA DLLs...
xcopy /Y /I "C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common\cudart64_*.dll" "%OUTPUT_DIR%" 2>nul
xcopy /Y /I "C:\Windows\System32\nvcuvid.dll" "%OUTPUT_DIR%" 2>nul


echo.
echo [✓] Build completed successfully!
echo [*] Executable should be here: %cd%\%CONFIG%\nvdec_scene_detect.exe
echo.

cd ..
endlocal
