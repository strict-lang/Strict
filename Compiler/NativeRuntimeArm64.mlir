  // x19 to x28 and d8 to d15 are saved by the callee, so values can stay in them across the collector's call
  llvm.mlir.global internal @strict_spilled_registers(dense<0> : tensor<18xi64>) : !llvm.array<18 x i64>
  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    %yield = llvm.mlir.constant(1 : i32) : i32
    llvm.call_intrinsic "llvm.aarch64.hint"(%yield) : (i32) -> ()
    llvm.return
  }
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %buffer = llvm.mlir.addressof @strict_spilled_registers : !llvm.ptr
    llvm.inline_asm has_side_effects "stp x19, x20, [$0]\0Astp x21, x22, [$0, #16]\0Astp x23, x24, [$0, #32]\0Astp x25, x26, [$0, #48]\0Astp x27, x28, [$0, #64]\0Astp d8, d9, [$0, #80]\0Astp d10, d11, [$0, #96]\0Astp d12, d13, [$0, #112]\0Astp d14, d15, [$0, #128]", "r,~{memory}" %buffer : (!llvm.ptr) -> ()
    %start = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %bytes = arith.constant 144 : i64
    %end = arith.addi %start, %bytes : i64
    return %start, %end : i64, i64
  }
