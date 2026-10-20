  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    llvm.return
  }
  // every xmm register is saved by the caller, values live across the collector's call are on the stack
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %none = arith.constant 0 : i64
    return %none, %none : i64, i64
  }
  // the plain opendir and readdir on Intel are the ones with 32 bit inodes
  llvm.func @"opendir$INODE64"(!llvm.ptr) -> !llvm.ptr
  llvm.func @"readdir$INODE64"(!llvm.ptr) -> !llvm.ptr
  llvm.func @strict_opendir(%name: !llvm.ptr) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %directory = llvm.call @"opendir$INODE64"(%name) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %directory : !llvm.ptr
  }
  llvm.func @strict_readdir(%directory: !llvm.ptr) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %entry = llvm.call @"readdir$INODE64"(%directory) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %entry : !llvm.ptr
  }
  // MTLSize has 24 bytes, the caller passes a copy on the stack
  llvm.func @strict_metal_dispatch(%send: !llvm.ptr, %encoder: !llvm.ptr, %selector: !llvm.ptr, %grid: !llvm.ptr, %group: !llvm.ptr) {
    llvm.call %send(%encoder, %selector, %grid, %group) {arg_attrs = [{}, {}, {llvm.byval = !llvm.array<3 x i64>}, {llvm.byval = !llvm.array<3 x i64>}]} : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
