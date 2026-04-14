@echo off
rem Build with MSVC (cl) if available, otherwise fall back to MinGW gcc.
rem Usage: build.bat
rem Output: ImageSaver.dll

where cl >nul 2>&1
if %ERRORLEVEL% == 0 (
    cl /LD /O2 /I src src\imagesaver.c /Fe:ImageSaver.dll
    echo Built: ImageSaver.dll  (MSVC^)
) else (
    gcc -shared -O2 -o ImageSaver.dll src/imagesaver.c -Isrc
    echo Built: ImageSaver.dll  (MinGW^)
)
