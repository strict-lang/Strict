#!/bin/sh
# Builds Compiler/NativeCompiler from Bootstrap.mlir without any Strict binary, needs mlir-opt, mlir-translate and clang
set -e
cd "$(dirname "$0")"
PATH="$PATH:$(cd .. && pwd)/Tools/LLVM-22.1.0-Linux-X64/bin"
{ cat Bootstrap.mlir NativeRuntime.mlir; echo; cat NativeRuntimeLinux.mlir; echo; echo "}"; } > NativeCompiler.linux.mlir
passes="--convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
mlir-opt NativeCompiler.linux.mlir --mlir-disable-threading $passes -o NativeCompiler.linux.llvm.mlir
mlir-translate --mlir-disable-threading --mlir-to-llvmir NativeCompiler.linux.llvm.mlir -o NativeCompiler.linux.ll
clang NativeCompiler.linux.ll -o NativeCompiler -O2 -Wno-override-module -lm
echo "Built $(pwd)/NativeCompiler"
