  llvm.mlir.global internal constant @strict_linux("Android\00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_temp_template("./strictXXXXXX\00") : !llvm.array<15 x i8>
  llvm.mlir.global internal constant @strict_shell("/system/bin/sh\00") : !llvm.array<15 x i8>
  llvm.mlir.global internal constant @strict_online(97 : i32) : i32
  llvm.mlir.global internal constant @strict_physical_pages(98 : i32) : i32
  llvm.mlir.global internal constant @strict_page_size(39 : i32) : i32
  func.func private @strict_guard() {
    %one = arith.constant 1 : i64
    %stackBytes = arith.constant 65536 : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %stack = func.call @strict_malloc(%stackBytes) : (i64) -> !llvm.ptr
    %alternate = llvm.alloca %one x !llvm.array<24 x i8> : (i64) -> !llvm.ptr
    llvm.store %stack, %alternate : !llvm.ptr, !llvm.ptr
    %stackFlags = llvm.getelementptr %alternate[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %noFlags = arith.constant 0 : i32
    llvm.store %noFlags, %stackFlags : i32, !llvm.ptr
    %stackSize = llvm.getelementptr %alternate[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %stackBytes, %stackSize : i64, !llvm.ptr
    %alternated = llvm.call @sigaltstack(%alternate, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    %action = llvm.alloca %one x !llvm.array<32 x i8> : (i64) -> !llvm.ptr
    %actionBytes = arith.constant 32 : i64
    %cleared = llvm.call @memset(%action, %noFlags, %actionBytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    %handlerSlot = llvm.getelementptr %action[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %handler, %handlerSlot : !llvm.ptr, !llvm.ptr
    %actionFlags = llvm.getelementptr %action[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %resetOnStackWithInfo = arith.constant -2013265916 : i32
    llvm.store %resetOnStackWithInfo, %actionFlags : i32, !llvm.ptr
    %segmentation = arith.constant 11 : i32
    %installed = llvm.call @sigaction(%segmentation, %action, %null) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    return
  }
  func.func private @strict_temp_file() -> f64 {
    %size = arith.constant 15 : i64
    %template = llvm.mlir.addressof @strict_temp_template : !llvm.ptr
    %name = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%name, %template, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %descriptor = llvm.call @mkstemp(%name) : (!llvm.ptr) -> i32
    %closed = llvm.call @close(%descriptor) : (i32) -> i32
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %lowest = func.call @strict_thread_stack_lowest(%stack) : (!llvm.ptr) -> i64
    return %lowest : i64
  }
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  }
