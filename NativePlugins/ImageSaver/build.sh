#!/usr/bin/env sh
# Quick build without CMake.
# Usage: sh build.sh
# Outputs:
#   Linux   → ImageSaver.so
#   macOS   → ImageSaver.dylib
#   Windows → ImageSaver.dll  (use build.bat instead)
set -e
OS=$(uname -s)
case "$OS" in
  Linux*)
    gcc -shared -fPIC -O2 -o ImageSaver.so src/imagesaver.c -Isrc -lm
    echo "Built: ImageSaver.so"
    ;;
  Darwin*)
    gcc -dynamiclib -O2 -o ImageSaver.dylib src/imagesaver.c -Isrc
    echo "Built: ImageSaver.dylib"
    ;;
  *)
    echo "Unknown OS: $OS — try build.bat on Windows"
    exit 1
    ;;
esac
