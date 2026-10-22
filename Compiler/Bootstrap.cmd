@echo off
rem Builds NativeCompiler.exe from Bootstrap.mlir without any Strict binary and with it strict.exe in the Strict folder,
rem needs mlir-opt, mlir-translate and clang
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\msys64\ucrt64\bin;C:\Program Files\LLVM\bin"
(type Bootstrap.mlir & type NativeRuntime.mlir & echo. & type NativeRuntimeWindows.mlir & echo. & echo }) > NativeCompiler.mlir
set "PASSES=--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt NativeCompiler.mlir --mlir-disable-threading %PASSES% -o NativeCompiler.llvm.mlir || exit /b 1
mlir-translate --mlir-disable-threading --mlir-to-llvmir NativeCompiler.llvm.mlir -o NativeCompiler.ll || exit /b 1
clang NativeCompiler.ll -o NativeCompiler.exe -O2 -march=native -Wno-override-module -Wl,-Brepro -Wl,/STACK:67108864 || exit /b 1
cd ..
Compiler\NativeCompiler.exe build Strict.strict . || exit /b 1
echo Built %~dp0NativeCompiler.exe and %~dp0..\strict.exe
