  llvm.mlir.global internal constant @strict_heap_bytes(68719476736 : i64) : i64
  llvm.mlir.global internal constant @strict_macos("MacOS\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @strict_temp_template("/tmp/strictXXXXXX\00") : !llvm.array<18 x i8>
  // the web runs kernels as WebGPU shaders (NativeRuntimeWeb.mlir), everywhere else the device code takes them
  func.func private @strict_gpu_shaded(%name: !llvm.ptr, %count: i64, %source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64) -> f64 {
    %none = arith.constant 0.0 : f64
    return %none : f64
  }
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    cf.cond_br %isAmd, ^failed, ^open
  ^open:
    %path = llvm.mlir.addressof @strict_metal_n_library : !llvm.ptr
    %now = arith.constant 2 : i32
    %library = llvm.call @dlopen(%path, %now) : (!llvm.ptr, i32) -> !llvm.ptr
    %missing = llvm.icmp "eq" %library, %null : !llvm.ptr
    cf.cond_br %missing, ^failed, ^runtime
  ^runtime:
    %libraryBits = llvm.ptrtoint %library : !llvm.ptr to i64
    %libraryIndex = arith.constant 14 : i64
    llvm.call @strict_metal_put(%libraryIndex, %libraryBits) : (i64, i64) -> ()
    %everywhere = arith.constant -2 : i64
    %default = llvm.inttoptr %everywhere : i64 to !llvm.ptr
    %send = llvm.mlir.addressof @strict_metal_n_send : !llvm.ptr
    %register = llvm.mlir.addressof @strict_metal_n_register : !llvm.ptr
    %class = llvm.mlir.addressof @strict_metal_n_class : !llvm.ptr
    %push = llvm.mlir.addressof @strict_metal_n_push : !llvm.ptr
    %pop = llvm.mlir.addressof @strict_metal_n_pop : !llvm.ptr
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %found = func.call @strict_metal_resolve(%default, %send, %zero) : (!llvm.ptr, !llvm.ptr, i64) -> i1
    %registered = func.call @strict_metal_resolve(%default, %register, %one) : (!llvm.ptr, !llvm.ptr, i64) -> i1
    %classes = func.call @strict_metal_resolve(%default, %class, %two) : (!llvm.ptr, !llvm.ptr, i64) -> i1
    %pushes = func.call @strict_metal_resolve(%default, %push, %three) : (!llvm.ptr, !llvm.ptr, i64) -> i1
    %pops = func.call @strict_metal_resolve(%default, %pop, %four) : (!llvm.ptr, !llvm.ptr, i64) -> i1
    %both = arith.andi %found, %registered : i1
    %withClasses = arith.andi %both, %classes : i1
    %withPush = arith.andi %withClasses, %pushes : i1
    %all = arith.andi %withPush, %pops : i1
    cf.cond_br %all, ^done, ^failed
  ^done:
    return %library : !llvm.ptr
  ^failed:
    return %null : !llvm.ptr
  }
  func.func private @strict_temp_file() -> f64 {
    %template = llvm.mlir.addressof @strict_temp_template : !llvm.ptr
    %length = llvm.call @strlen(%template) : (!llvm.ptr) -> i64
    %one = arith.constant 1 : i64
    %size = arith.addi %length, %one : i64
    %name = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%name, %template, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %descriptor = llvm.call @mkstemp(%name) : (!llvm.ptr) -> i32
    %closed = llvm.call @close(%descriptor) : (i32) -> i32
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
