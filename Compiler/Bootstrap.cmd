@echo off
rem Builds strict.exe in the Strict folder from Bootstrap.mlir without any Strict binary, needs mlir-opt, mlir-translate and clang
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\msys64\ucrt64\bin;C:\Program Files\LLVM\bin"
(type Bootstrap.mlir & type NativeRuntime.mlir & echo. & type NativeRuntimeWindows.mlir & echo. & echo }) > Strict.mlir
set "PASSES=--convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt Strict.mlir --mlir-disable-threading %PASSES% -o Strict.llvm.mlir || exit /b 1
mlir-translate --mlir-disable-threading --mlir-to-llvmir Strict.llvm.mlir -o Strict.ll || exit /b 1
clang Strict.ll -o ..\strict.exe -O2 -march=native -Wno-override-module -Wl,-Brepro || exit /b 1
echo Built %~dp0..\strict.exe
