@echo off
rem Builds a seed compiler from Bootstrap.mlir without any Strict binary, with it strict.exe in the Strict folder and with that one itself,
rem needs mlir-opt, mlir-translate and clang; the seed has no GPU code, so its compiler runs on the CPU
setlocal
cd /d "%~dp0"
rem an old strict server would answer the new strict.exe
if exist ..\strict.exe ..\strict.exe stop >nul 2>&1
set "PATH=%PATH%;C:\msys64\ucrt64\bin;C:\Program Files\LLVM\bin"
(type Bootstrap.mlir & type NativeRuntime.mlir & echo. & type NativeRuntimeWindows.mlir & echo. & echo }) > NativeCompiler.mlir
set "PASSES=--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt NativeCompiler.mlir %PASSES% -o NativeCompiler.llvm.mlir || exit /b 1
mlir-translate --mlir-to-llvmir NativeCompiler.llvm.mlir -o NativeCompiler.ll || exit /b 1
clang NativeCompiler.ll ..\Networking\native\Socket.c -I . -lws2_32 -o NativeCompiler.exe -O3 -march=native -Wno-override-module -Wl,-Brepro -Wl,/STACK:67108864 || exit /b 1
cd ..
set "GPU=%STRICT_GPU%"
set "STRICT_GPU=0"
rem windows builds a release compiler: a long garbage collection warns instead of stopping
Compiler\NativeCompiler.exe build Compiler/Strict.strict . windows || exit /b 1
rem a seed older than Compiler/Strict.strict puts strict.exe next to it
if exist Compiler\strict.exe move /y Compiler\strict.exe strict.exe >nul
del Compiler\NativeCompiler.exe Compiler\NativeCompiler.ll Compiler\NativeCompiler.llvm.mlir Compiler\NativeCompiler.mlir
set "STRICT_GPU=%GPU%"
rem the seed compiler is old and builds without GPU code, the new strict.exe builds itself fully
.\strict.exe build strict || exit /b 1
echo Built %CD%\strict.exe
