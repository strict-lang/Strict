#!/bin/sh
# Builds strict in the Strict folder from Bootstrap.mlir without any Strict binary, needs mlir-opt, mlir-translate and clang
set -e
cd "$(dirname "$0")"
PATH="$PATH:$(cd .. && pwd)/Tools/LLVM-22.1.0-Linux-X64/bin"
{ cat Bootstrap.mlir NativeRuntime.mlir; echo; cat NativeRuntimeLinux.mlir; echo; echo "}"; } > Strict.linux.mlir
passes="--symbol-dce --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt Strict.linux.mlir --mlir-disable-threading $passes -o Strict.linux.llvm.mlir
mlir-translate --mlir-disable-threading --mlir-to-llvmir Strict.linux.llvm.mlir -o Strict.linux.ll
clang Strict.linux.ll -o ../strict -O2 -march=native -Wno-override-module -lm -lpthread
echo "Built $(cd .. && pwd)/strict"
