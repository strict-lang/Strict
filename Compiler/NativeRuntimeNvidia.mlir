  // the target specific operations of the device runtime and of the kernels generated from loops, NVVM for NVIDIA
  func.func private @strict_device_thread_index() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    return %thread : i32
  }
  func.func private @strict_device_block_index() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    return %block : i32
  }
  func.func private @strict_device_block_size() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %width = nvvm.read.ptx.sreg.ntid.x : i32
    return %width : i32
  }
  func.func private @strict_device_block_count() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %blocks = nvvm.read.ptx.sreg.nctaid.x : i32
    return %blocks : i32
  }
  // the bits of the lane whose number differs in the bit of offset, a step of a butterfly over a warp of 32 lanes
  func.func private @strict_device_swap(%bits: i32, %offset: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %other = nvvm.shfl.sync bfly %all, %bits, %offset, %clamp : i32 -> i32
    return %other : i32
  }
  func.func private @strict_device_barrier() attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    nvvm.barrier0
    return
  }
  func.func private @strict_device_binary_logarithm(%number: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %binary = llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"(%number) : (f32) -> f32
    return %binary : f32
  }
  func.func private @strict_device_rounded(%number: f32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %rounded = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%number) : (f32) -> i32
    return %rounded : i32
  }
  func.func private @strict_device_fence() attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    llvm.call_intrinsic "llvm.nvvm.membar.gl"() : () -> ()
    return
  }
  // clang replaces __nvvm_reflect of this with the compute capability it compiles for (750 or 800, see DeviceBuild)
  llvm.mlir.global private constant @strict_device_arch_name("__CUDA_ARCH\00") {addr_space = 0 : i32} : !llvm.array<12 x i8>
  llvm.func @__nvvm_reflect(!llvm.ptr) -> i32
  func.func private @strict_device_arch() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %name = llvm.mlir.addressof @strict_device_arch_name : !llvm.ptr
    %arch = llvm.call @__nvvm_reflect(%name) : (!llvm.ptr) -> i32
    return %arch : i32
  }
  // the tensor core operations of a warp: 8 bit products of 8 x 16 by 16 x 8 (two of them along k and rows are one
  // m16n8k32 from sm_80 on), 16 bit floats of 16 x 8 by 8 x 8 and 4 matrices of 8 x 16 bytes loaded from shared memory
  func.func private @strict_device_mma8(%a: i32, %b: i32, %c0: i32, %c1: i32) -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%a, %b, %c0, %c1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    return %d : !llvm.struct<(i32, i32)>
  }
  func.func private @strict_device_mma16(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %c0: i32, %c1: i32, %c2: i32, %c3: i32) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k32.row.col.s8"(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    return %d : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_halves_mma(%a0: vector<2xf16>, %a1: vector<2xf16>, %b: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f32.f32"(%a0, %a1, %b, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %d : !llvm.struct<(f32, f32, f32, f32)>
  }
  func.func private @strict_device_matrices(%slot: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %loaded = llvm.call_intrinsic "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3"(%slot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    return %loaded : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %result = llvm.call_intrinsic "llvm.nvvm.idp4a.s.s"(%left, %right, %sum) : (i32, i32, i32) -> i32
    return %result : i32
  }
