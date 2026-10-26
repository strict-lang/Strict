  // opendir and readdir with 64 bit inodes, macOS on Intel has its own (MacOSX64)
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