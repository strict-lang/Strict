  llvm.mlir.global internal constant @strict_linux("Linux\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @strict_temp_template("/tmp/strictXXXXXX\00") : !llvm.array<18 x i8>
  llvm.mlir.global internal constant @strict_shell("/bin/sh\00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_online(84 : i32) : i32
  llvm.mlir.global internal constant @strict_physical_pages(85 : i32) : i32
  llvm.mlir.global internal constant @strict_page_size(30 : i32) : i32
  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    llvm.return
  }
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %none = arith.constant 0 : i64
    return %none, %none : i64, i64
  }
  llvm.func @getpid() -> i32
  llvm.func @gettid() -> i32
  // the main stack grows up to its limit (RLIMIT_STACK) below where it started, raised to 64 MB like on Windows and
  // macOS (8 MB is the usual default, the kernel keeps at least 128 MB free below the stack for growing)
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %process = llvm.call @getpid() : () -> i32
    %self = llvm.call @gettid() : () -> i32
    %isMain = arith.cmpi eq, %process, %self : i32
    cf.cond_br %isMain, ^main, ^worker
  ^worker:
    %workerLowest = func.call @strict_thread_stack_lowest(%stack) : (!llvm.ptr) -> i64
    return %workerLowest : i64
  ^main:
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %success = arith.constant 0 : i32
    %stackResource = arith.constant 3 : i32
    %most = arith.constant 67108864 : i64
    %limits = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %limits : i64, !llvm.ptr
    %got = llvm.call @getrlimit(%stackResource, %limits) : (i32, !llvm.ptr) -> i32
    %current = llvm.load %limits : !llvm.ptr -> i64
    %hardSlot = llvm.getelementptr %limits[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %hard = llvm.load %hardSlot : !llvm.ptr -> i64
    %allowed = arith.minui %most, %hard : i64
    %wanted = arith.maxui %current, %allowed : i64
    llvm.store %wanted, %limits : i64, !llvm.ptr
    %set = llvm.call @setrlimit(%stackResource, %limits) : (i32, !llvm.ptr) -> i32
    %isRaised = arith.cmpi eq, %set, %success : i32
    %limit = arith.select %isRaised, %wanted, %current : i64
    %size = arith.minui %limit, %most : i64
    %top = llvm.ptrtoint %stack : !llvm.ptr to i64
    %lowest = arith.subi %top, %size : i64
    return %lowest : i64
  }
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
    %action = llvm.alloca %one x !llvm.array<152 x i8> : (i64) -> !llvm.ptr
    %actionBytes = arith.constant 152 : i64
    %cleared = llvm.call @memset(%action, %noFlags, %actionBytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    llvm.store %handler, %action : !llvm.ptr, !llvm.ptr
    %actionFlags = llvm.getelementptr %action[136] : (!llvm.ptr) -> !llvm.ptr, i8
    %resetOnStackWithInfo = arith.constant -2013265916 : i32
    llvm.store %resetOnStackWithInfo, %actionFlags : i32, !llvm.ptr
    %segmentation = arith.constant 11 : i32
    %installed = llvm.call @sigaction(%segmentation, %action, %null) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    return
  }
  func.func private @strict_temp_file() -> f64 {
    %size = arith.constant 18 : i64
    %template = llvm.mlir.addressof @strict_temp_template : !llvm.ptr
    %name = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%name, %template, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %descriptor = llvm.call @mkstemp(%name) : (!llvm.ptr) -> i32
    %closed = llvm.call @close(%descriptor) : (i32) -> i32
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %nvidia = llvm.mlir.addressof @strict_gpu_driver : !llvm.ptr
    %amd = llvm.mlir.addressof @strict_gpu_amd_driver : !llvm.ptr
    %name = arith.select %isAmd, %amd, %nvidia : !llvm.ptr
    %now = arith.constant 2 : i32
    %library = llvm.call @dlopen(%name, %now) : (!llvm.ptr, i32) -> !llvm.ptr
    return %library : !llvm.ptr
  }
