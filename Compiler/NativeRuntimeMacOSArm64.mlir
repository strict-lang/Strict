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
