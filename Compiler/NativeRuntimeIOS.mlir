  llvm.mlir.global internal constant @strict_heap_bytes(1073741824 : i64) : i64
  llvm.mlir.global internal constant @strict_macos("iOS\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_temp_template("/strictXXXXXX\00") : !llvm.array<14 x i8>
  // the web runs kernels as WebGPU shaders (NativeRuntimeWeb.mlir), everywhere else the device code takes them
  func.func private @strict_gpu_shaded(%name: !llvm.ptr, %count: i64, %source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64, %format: i64) -> f64 {
    %none = arith.constant 0.0 : f64
    return %none : f64
  }
  // lists of class 255 live in WebGPU buffers (NativeRuntimeWeb.mlir), there are none here
  func.func private @strict_gpu_web_get(%holder: !llvm.ptr, %device: i64) -> !llvm.ptr {
    %none = llvm.mlir.zero : !llvm.ptr
    return %none : !llvm.ptr
  }
  func.func private @strict_gpu_web_give(%device: i64) {
    return
  }
  // 8 bit rows that only are in WebGPU buffers joined in a new one (NativeRuntimeWeb.mlir), 0 here
  func.func private @strict_gpu_web_gathered(%items: !llvm.ptr, %count: i64, %shape: !llvm.ptr) -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  // a WebGPU buffer of the words at data (NativeRuntimeWeb.mlir), 0 here
  func.func private @strict_gpu_web_uploaded(%data: !llvm.ptr, %words: i64, %halves: i64) -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  // two lists in WebGPU buffers joined there (NativeRuntimeWeb.mlir), 0.0 here
  func.func private @strict_gpu_web_joined(%left: f64, %right: f64) -> f64 {
    %none = arith.constant 0.0 : f64
    return %none : f64
  }
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  }
  llvm.mlir.global internal constant @strict_temp_variable("TMPDIR\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_temp_failed("TemporaryFileCreateFailed\00") : !llvm.array<26 x i8>
  func.func private @strict_temp_file() -> f64 {
    %variable = llvm.mlir.addressof @strict_temp_variable : !llvm.ptr
    %directory = llvm.call @getenv(%variable) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %directory, %null : !llvm.ptr
    cf.cond_br %missing, ^failed, ^create
  ^create:
    %folder = func.call @strict_bits(%directory) : (!llvm.ptr) -> f64
    %template = llvm.mlir.addressof @strict_temp_template : !llvm.ptr
    %ending = func.call @strict_bits(%template) : (!llvm.ptr) -> f64
    %path = func.call @strict_text_concat(%folder, %ending) : (f64, f64) -> f64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %descriptor = llvm.call @mkstemp(%name) : (!llvm.ptr) -> i32
    %zero = arith.constant 0 : i32
    %invalid = arith.cmpi slt, %descriptor, %zero : i32
    cf.cond_br %invalid, ^failed, ^close
  ^close:
    %closed = llvm.call @close(%descriptor) : (i32) -> i32
    return %path : f64
  ^failed:
    %message = llvm.mlir.addressof @strict_temp_failed : !llvm.ptr
    %error = func.call @strict_error_named(%message) : (!llvm.ptr) -> f64
    return %error : f64
  }
