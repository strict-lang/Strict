@echo off
rem Build with MSVC (cl) if available, otherwise fall back to MinGW gcc.
rem Usage: build.bat
rem Output: ImageLoader.dll

where cl >nul 2>&1
if %ERRORLEVEL% == 0 (
    cl /LD /O2 /I src src\imageloader.c /Fe:ImageLoader.dll
    echo Built: ImageLoader.dll  (MSVC^)
) else (
    gcc -shared -O2 -o ImageLoader.dll src/imageloader.c -Isrc
    echo Built: ImageLoader.dll  (MinGW^)
)
