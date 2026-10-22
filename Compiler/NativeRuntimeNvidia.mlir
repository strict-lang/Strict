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
  // 2 to the power of number (flushing results below the smallest normal float to 0)
  func.func private @strict_device_binary_power(%number: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %power = llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"(%number) : (f32) -> f32
    return %power : f32
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
  // before sm_80 m16n8k32 is four m8n8k16: rows g and g + 8 (a0, a1) of k 0 to 15 (b0), then of k 16 to 31 (a2, a3, b1)
  func.func private @strict_device_mma16(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %c0: i32, %c1: i32, %c2: i32, %c3: i32) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isWide = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isWide, ^wide, ^narrow
  ^wide:
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k32.row.col.s8"(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    return %d : !llvm.struct<(i32, i32, i32, i32)>
  ^narrow:
    %upper = func.call @strict_device_mma8(%a0, %b0, %c0, %c1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %upper0 = llvm.extractvalue %upper[0] : !llvm.struct<(i32, i32)>
    %upper1 = llvm.extractvalue %upper[1] : !llvm.struct<(i32, i32)>
    %upperDone = func.call @strict_device_mma8(%a2, %b1, %upper0, %upper1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lower = func.call @strict_device_mma8(%a1, %b0, %c2, %c3) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lower0 = llvm.extractvalue %lower[0] : !llvm.struct<(i32, i32)>
    %lower1 = llvm.extractvalue %lower[1] : !llvm.struct<(i32, i32)>
    %lowerDone = func.call @strict_device_mma8(%a3, %b1, %lower0, %lower1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %d0 = llvm.extractvalue %upperDone[0] : !llvm.struct<(i32, i32)>
    %d1 = llvm.extractvalue %upperDone[1] : !llvm.struct<(i32, i32)>
    %d2 = llvm.extractvalue %lowerDone[0] : !llvm.struct<(i32, i32)>
    %d3 = llvm.extractvalue %lowerDone[1] : !llvm.struct<(i32, i32)>
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %with0 = llvm.insertvalue %d0, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %with1 = llvm.insertvalue %d1, %with0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %with2 = llvm.insertvalue %d2, %with1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %with3 = llvm.insertvalue %d3, %with2[3] : !llvm.struct<(i32, i32, i32, i32)>
    return %with3 : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_halves_mma(%a0: vector<2xf16>, %a1: vector<2xf16>, %b: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f32.f32"(%a0, %a1, %b, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %d : !llvm.struct<(f32, f32, f32, f32)>
  }
  // m16n8k16 of 16 bit floats (a0, a1 numbers 0 to 7 of rows g and g + 8, a2, a3 numbers 8 to 15, b0 and b1 those of
  // column g), one mma from sm_80 on, two m16n8k8 before
  func.func private @strict_device_halves_mma16(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isWide = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isWide, ^wide, ^narrow
  ^wide:
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k16.row.col.f32.f32"(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %d : !llvm.struct<(f32, f32, f32, f32)>
  ^narrow:
    %halves = func.call @strict_device_halves_mma8x2(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %halves : !llvm.struct<(f32, f32, f32, f32)>
  }
  // strict_device_halves_mma16 adding to 16 bit sums (c0 the two of row g, c1 those of row g + 8), twice as fast as 32
  // bit sums on consumer GPUs
  func.func private @strict_device_halves_mma16_halves(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: vector<2xf16>, %c1: vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isWide = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isWide, ^wide, ^narrow
  ^wide:
    %d = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k16.row.col.f16.f16"(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    return %d : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  ^narrow:
    %low = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f16.f16"(%a0, %a1, %b0, %c0, %c1) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %low0 = llvm.extractvalue %low[0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %low1 = llvm.extractvalue %low[1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %high = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f16.f16"(%a2, %a3, %b1, %low0, %low1) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    return %high : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  }
  // ldmatrix.x4.trans: lane l gets the numbers of rows 2 * (l % 4) and the next in column l / 4 of each matrix
  func.func private @strict_device_matrices_transposed(%slot: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %loaded = llvm.call_intrinsic "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3"(%slot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    return %loaded : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_matrices(%slot: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %loaded = llvm.call_intrinsic "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3"(%slot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    return %loaded : !llvm.struct<(i32, i32, i32, i32)>
  }
  // 16 bytes of global memory into shared memory without waiting for them (cp.async from sm_80 on, a load and a store
  // before), the copies of a thread since the last commit are one group, waited for until at most one group is pending
  func.func private @strict_device_copy(%shared: !llvm.ptr<3>, %global: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isAsync = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isAsync, ^async, ^plain
  ^async:
    %source = llvm.addrspacecast %global : !llvm.ptr to !llvm.ptr<1>
    llvm.call_intrinsic "llvm.nvvm.cp.async.cg.shared.global.16"(%shared, %source) : (!llvm.ptr<3>, !llvm.ptr<1>) -> ()
    return
  ^plain:
    %words = llvm.load %global invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    llvm.store %words, %shared {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    return
  }
  func.func private @strict_device_copies_commit() attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isAsync = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isAsync, ^async, ^done
  ^async:
    llvm.call_intrinsic "llvm.nvvm.cp.async.commit.group"() : () -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_device_copies_wait() attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isAsync = arith.cmpi sge, %arch, %c800 : i32
    cf.cond_br %isAsync, ^async, ^done
  ^async:
    %pending = llvm.mlir.constant(1 : i32) : i32
    llvm.call_intrinsic "llvm.nvvm.cp.async.wait.group"(%pending) : (i32) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, passthrough = ["alwaysinline"]} {
    %result = llvm.call_intrinsic "llvm.nvvm.idp4a.s.s"(%left, %right, %sum) : (i32, i32, i32) -> i32
    return %result : i32
  }
