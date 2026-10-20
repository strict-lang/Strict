#!/bin/sh
# Builds strict in the Strict folder from Bootstrap.mlir without any Strict binary, needs mlir-opt, mlir-translate and clang
set -e
cd "$(dirname "$0")"
platform=Linux
linked="-lm -lpthread"
[ "$(uname)" = Darwin ] && platform=MacOS && linked="$linked -Wl,-stack_size,0x4000000"
architecture=""
[ $platform = MacOS ] && architecture=X64 && [ "$(uname -m)" = arm64 ] && architecture=Arm64
PATH="$(cd .. && pwd)/Tools/LLVM-22.1.0-Linux-X64/bin:$(cd .. && pwd)/Tools/LLVM-22.1.8-MacOS/bin:$PATH"
{ cat Bootstrap.mlir NativeRuntime.mlir; echo; cat NativeRuntime$platform.mlir; echo; [ -n "$architecture" ] && cat NativeRuntime$platform$architecture.mlir && echo; echo "}"; } > Strict.$platform.mlir
passes="--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt Strict.$platform.mlir --mlir-disable-threading $passes -o Strict.$platform.llvm.mlir
mlir-translate --mlir-disable-threading --mlir-to-llvmir Strict.$platform.llvm.mlir -o Strict.$platform.ll
clang Strict.$platform.ll -o ../strict -O2 -march=native -Wno-override-module $linked
echo "Built $(cd .. && pwd)/strict"
