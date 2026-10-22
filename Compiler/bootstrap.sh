#!/bin/sh
# Builds NativeCompiler from Bootstrap.mlir without any Strict binary and with it strict in the Strict folder, needs mlir-opt, mlir-translate and clang
set -e
cd "$(dirname "$0")"
platform=Linux
linked="-lm -lpthread"
[ "$(uname)" = Darwin ] && platform=MacOS && linked="$linked -Wl,-stack_size,0x4000000"
architecture=""
[ $platform = MacOS ] && architecture=X64 && [ "$(uname -m)" = arm64 ] && architecture=Arm64
PATH="$(cd .. && pwd)/Tools/LLVM-22.1.0-Linux-X64/bin:$(cd .. && pwd)/Tools/LLVM-22.1.8-MacOS/bin:$PATH"
{ cat Bootstrap.mlir NativeRuntime.mlir; echo; cat NativeRuntime$platform.mlir; echo; [ -n "$architecture" ] && cat NativeRuntime$platform$architecture.mlir && echo; echo "}"; } > NativeCompiler.$platform.mlir
passes="--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt NativeCompiler.$platform.mlir --mlir-disable-threading $passes -o NativeCompiler.$platform.llvm.mlir
mlir-translate --mlir-disable-threading --mlir-to-llvmir NativeCompiler.$platform.llvm.mlir -o NativeCompiler.$platform.ll
clang NativeCompiler.$platform.ll -o NativeCompiler -O2 -march=native -Wno-override-module $linked
cd ..
./Compiler/NativeCompiler build Strict.strict .
echo "Built $(pwd)/Compiler/NativeCompiler and $(pwd)/strict"
