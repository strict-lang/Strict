@echo off
rem Builds a seed compiler from Bootstrap.mlir without any Strict binary and with it strict.exe in the Strict folder,
rem needs mlir-opt, mlir-translate and clang; the seed has no GPU code, so its compiler runs on the CPU
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\msys64\ucrt64\bin;C:\Program Files\LLVM\bin"
(type Bootstrap.mlir & type NativeRuntime.mlir & echo. & type NativeRuntimeWindows.mlir & echo. & echo }) > NativeCompiler.mlir
set "PASSES=--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt NativeCompiler.mlir %PASSES% -o NativeCompiler.llvm.mlir || exit /b 1
mlir-translate --mlir-to-llvmir NativeCompiler.llvm.mlir -o NativeCompiler.ll || exit /b 1
clang NativeCompiler.ll -o NativeCompiler.exe -O3 -march=native -Wno-override-module -Wl,-Brepro -Wl,/STACK:67108864 || exit /b 1
cd ..
set "STRICT_GPU=0"
Compiler\NativeCompiler.exe build Strict.strict . || exit /b 1
del Compiler\NativeCompiler.exe Compiler\NativeCompiler.ll Compiler\NativeCompiler.llvm.mlir Compiler\NativeCompiler.mlir
echo Built %~dp0..\strict.exe
