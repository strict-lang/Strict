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