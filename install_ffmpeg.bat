@echo off
setlocal

set FFMPEG_DIR=C:\ffmpeg
set FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z
set FFMPEG_ARCHIVE=ffmpeg-release-full-shared.7z

echo [*] Checking for FFmpeg in %FFMPEG_DIR% ...

if exist "%FFMPEG_DIR%\include\libavformat\avformat.h" (
    echo [✓] FFmpeg already installed at %FFMPEG_DIR%.
    exit /b 0
)

echo [!] FFmpeg not found. Installing...

REM Create folder
if not exist %FFMPEG_DIR% (
    mkdir %FFMPEG_DIR%
)

REM Download archive (requires PowerShell)
echo [*] Downloading FFmpeg from %FFMPEG_URL% ...
powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ARCHIVE%'"

if not exist "%FFMPEG_ARCHIVE%" (
    echo [!] Download failed.
    exit /b 1
)

REM Extract (requires 7-Zip in PATH)
echo [*] Extracting FFmpeg...
7z x %FFMPEG_ARCHIVE% -o%FFMPEG_DIR% > nul

REM Move headers/libs up (depends on archive structure)
for /d %%i in (%FFMPEG_DIR%\ffmpeg-*) do (
    xcopy "%%i\include" "%FFMPEG_DIR%\include" /E /I /Y
    xcopy "%%i\lib" "%FFMPEG_DIR%\lib" /E /I /Y
)

echo [*] Cleaning up...
del %FFMPEG_ARCHIVE%

echo [✓] FFmpeg installed to %FFMPEG_DIR%
exit /b 0
