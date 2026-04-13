#!/usr/bin/env sh
# Quick build without CMake.
# Usage: sh build.sh
# Outputs:
#   Linux   → ImageLoader.so
#   macOS   → ImageLoader.dylib
#   Windows → ImageLoader.dll  (use build.bat instead)
set -e
OS=$(uname -s)
case "$OS" in
  Linux*)
    gcc -shared -fPIC -O2 -o ImageLoader.so src/imageloader.c -Isrc -lm
    echo "Built: ImageLoader.so"
    ;;
  Darwin*)
    gcc -dynamiclib -O2 -o ImageLoader.dylib src/imageloader.c -Isrc
    echo "Built: ImageLoader.dylib"
    ;;
  *)
    echo "Unknown OS: $OS — try build.bat on Windows"
    exit 1
    ;;
esac
