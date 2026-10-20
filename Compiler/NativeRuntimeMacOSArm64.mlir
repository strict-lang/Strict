  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    %yield = llvm.mlir.constant(1 : i32) : i32
    llvm.call_intrinsic "llvm.aarch64.hint"(%yield) : (i32) -> ()
    llvm.return
  }
  // x19 to x28 and d8 to d15 are saved by the callee, so values can stay in them across the collector's call
  llvm.mlir.global internal @strict_spilled_registers(dense<0> : tensor<18xi64>) : !llvm.array<18 x i64>
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %buffer = llvm.mlir.addressof @strict_spilled_registers : !llvm.ptr
    llvm.inline_asm has_side_effects "stp x19, x20, [$0]\0Astp x21, x22, [$0, #16]\0Astp x23, x24, [$0, #32]\0Astp x25, x26, [$0, #48]\0Astp x27, x28, [$0, #64]\0Astp d8, d9, [$0, #80]\0Astp d10, d11, [$0, #96]\0Astp d12, d13, [$0, #112]\0Astp d14, d15, [$0, #128]", "r,~{memory}" %buffer : (!llvm.ptr) -> ()
    %start = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %bytes = arith.constant 144 : i64
    %end = arith.addi %start, %bytes : i64
    return %start, %end : i64, i64
  }
  llvm.func @opendir(!llvm.ptr) -> !llvm.ptr
  llvm.func @readdir(!llvm.ptr) -> !llvm.ptr
  llvm.func @strict_opendir(%name: !llvm.ptr) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %directory = llvm.call @opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %directory : !llvm.ptr
  }
  llvm.func @strict_readdir(%directory: !llvm.ptr) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %entry = llvm.call @readdir(%directory) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %entry : !llvm.ptr
  }
  // MTLSize has 24 bytes, more than 16 go as a pointer to a copy
  llvm.func @strict_metal_dispatch(%send: !llvm.ptr, %encoder: !llvm.ptr, %selector: !llvm.ptr, %grid: !llvm.ptr, %group: !llvm.ptr) {
    llvm.call %send(%encoder, %selector, %grid, %group) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
