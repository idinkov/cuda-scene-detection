@echo off
setlocal

REM Install FFmpeg shared dev build into dependencies/ffmpeg/ (relative to project root)
set "PROJECT_ROOT=%~dp0.."
set "FFMPEG_DIR=%PROJECT_ROOT%\dependencies\ffmpeg"
set "FFMPEG_ZIP=%TEMP%\ffmpeg-shared.zip"
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"

echo [*] Checking for FFmpeg dev files...
if exist "%FFMPEG_DIR%\include\libavformat\avformat.h" (
    if exist "%FFMPEG_DIR%\lib\avformat.lib" (
        echo [OK] FFmpeg dev files found at %FFMPEG_DIR%.
        exit /b 0
    )
)

echo [!] FFmpeg dev files not found. Downloading shared build...
echo     URL: %FFMPEG_URL%

REM Create directories
if not exist "%PROJECT_ROOT%\dependencies" mkdir "%PROJECT_ROOT%\dependencies"

REM Download
echo [*] Downloading (this may take a minute)...
powershell -NoProfile -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing }"

if not exist "%FFMPEG_ZIP%" (
    echo [!] Download failed.
    exit /b 1
)

REM Extract to temp location
set "EXTRACT_DIR=%TEMP%\ffmpeg_extract"
if exist "%EXTRACT_DIR%" rmdir /s /q "%EXTRACT_DIR%"

echo [*] Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%EXTRACT_DIR%' -Force"

if errorlevel 1 (
    echo [!] Extraction failed.
    del "%FFMPEG_ZIP%" 2>nul
    exit /b 1
)

REM Find the extracted folder (name varies, e.g. ffmpeg-master-latest-win64-gpl-shared)
set "SRC_DIR="
for /d %%d in ("%EXTRACT_DIR%\ffmpeg-*") do set "SRC_DIR=%%d"

if "%SRC_DIR%"=="" (
    echo [!] Could not find extracted FFmpeg folder.
    del "%FFMPEG_ZIP%" 2>nul
    rmdir /s /q "%EXTRACT_DIR%" 2>nul
    exit /b 1
)

REM Copy to project dependencies
if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%"
echo [*] Copying to %FFMPEG_DIR%...
xcopy "%SRC_DIR%\include" "%FFMPEG_DIR%\include\" /E /I /Y /Q >nul
xcopy "%SRC_DIR%\lib" "%FFMPEG_DIR%\lib\" /E /I /Y /Q >nul
xcopy "%SRC_DIR%\bin" "%FFMPEG_DIR%\bin\" /E /I /Y /Q >nul

REM Cleanup
del "%FFMPEG_ZIP%" 2>nul
rmdir /s /q "%EXTRACT_DIR%" 2>nul

REM Verify
if exist "%FFMPEG_DIR%\include\libavformat\avformat.h" (
    echo [OK] FFmpeg dev files installed to %FFMPEG_DIR%
    exit /b 0
) else (
    echo [!] Installation verification failed. Check %FFMPEG_DIR% manually.
    exit /b 1
)
