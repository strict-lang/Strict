  // the target specific operations of the device runtime and of the kernels generated from loops, NVVM for NVIDIA
  func.func private @strict_device_thread_index() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    return %thread : i32
  }
  func.func private @strict_device_block_index() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    return %block : i32
  }
  func.func private @strict_device_block_size() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %width = nvvm.read.ptx.sreg.ntid.x : i32
    return %width : i32
  }
  func.func private @strict_device_block_count() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %blocks = nvvm.read.ptx.sreg.nctaid.x : i32
    return %blocks : i32
  }
  // the bits of the lane whose number differs in the bit of offset, a step of a butterfly over a warp of 32 lanes
  func.func private @strict_device_swap(%bits: i32, %offset: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %other = nvvm.shfl.sync bfly %all, %bits, %offset, %clamp : i32 -> i32
    return %other : i32
  }
  func.func private @strict_device_barrier() attributes {passthrough = ["alwaysinline"]} {
    nvvm.barrier0
    return
  }
  func.func private @strict_device_binary_logarithm(%number: f32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %binary = llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"(%number) : (f32) -> f32
    return %binary : f32
  }
  func.func private @strict_device_rounded(%number: f32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %rounded = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%number) : (f32) -> i32
    return %rounded : i32
  }
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %result = llvm.call_intrinsic "llvm.nvvm.idp4a.s.s"(%left, %right, %sum) : (i32, i32, i32) -> i32
    return %result : i32
  }
