  // the target specific operations of the device runtime and of the kernels generated from loops, AMDGPU for AMD: the
  // block size and count come from the dispatch packet (16 bit workgroup size x at 4, 32 bit grid size x at 12)
  func.func private @strict_device_thread_index() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %thread = llvm.call_intrinsic "llvm.amdgcn.workitem.id.x"() : () -> i32
    return %thread : i32
  }
  func.func private @strict_device_block_index() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %block = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
    return %block : i32
  }
  func.func private @strict_device_block_size() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %packet = llvm.call_intrinsic "llvm.amdgcn.dispatch.ptr"() : () -> !llvm.ptr<4>
    %sizeSlot = llvm.getelementptr %packet[4] : (!llvm.ptr<4>) -> !llvm.ptr<4>, i8
    %size = llvm.load %sizeSlot : !llvm.ptr<4> -> i16
    %width = arith.extui %size : i16 to i32
    return %width : i32
  }
  func.func private @strict_device_block_count() -> i32 attributes {passthrough = ["alwaysinline"]} {
    %packet = llvm.call_intrinsic "llvm.amdgcn.dispatch.ptr"() : () -> !llvm.ptr<4>
    %gridSlot = llvm.getelementptr %packet[12] : (!llvm.ptr<4>) -> !llvm.ptr<4>, i8
    %grid = llvm.load %gridSlot : !llvm.ptr<4> -> i32
    %width = func.call @strict_device_block_size() : () -> i32
    %blocks = arith.divui %grid, %width : i32
    return %blocks : i32
  }
  // the bits of the lane whose number differs in the bit of offset, a step of a butterfly over 32 lanes (offsets below
  // 32 stay inside each half of a 64 lane wavefront)
  func.func private @strict_device_swap(%bits: i32, %offset: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %all = arith.constant -1 : i32
    %zero = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %low = llvm.call_intrinsic "llvm.amdgcn.mbcnt.lo"(%all, %zero) : (i32, i32) -> i32
    %lane = llvm.call_intrinsic "llvm.amdgcn.mbcnt.hi"(%all, %low) : (i32, i32) -> i32
    %source = arith.xori %lane, %offset : i32
    %address = arith.shli %source, %two : i32
    %other = llvm.call_intrinsic "llvm.amdgcn.ds.bpermute"(%address, %bits) : (i32, i32) -> i32
    return %other : i32
  }
  func.func private @strict_device_barrier() attributes {passthrough = ["alwaysinline"]} {
    llvm.fence syncscope("workgroup") release
    llvm.call_intrinsic "llvm.amdgcn.s.barrier"() : () -> ()
    llvm.fence syncscope("workgroup") acquire
    return
  }
  func.func private @strict_device_binary_logarithm(%number: f32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %binary = llvm.call_intrinsic "llvm.log2.f32"(%number) : (f32) -> f32
    return %binary : f32
  }
  func.func private @strict_device_rounded(%number: f32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %even = llvm.call_intrinsic "llvm.rint.f32"(%number) : (f32) -> f32
    %rounded = arith.fptosi %even : f32 to i32
    return %rounded : i32
  }
  // older GPUs have no 8 bit dot product instruction (sdot4 needs gfx906), four multiplies of the signed bytes
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %leftBytes = llvm.bitcast %left : i32 to vector<4xi8>
    %rightBytes = llvm.bitcast %right : i32 to vector<4xi8>
    %leftWide = arith.extsi %leftBytes : vector<4xi8> to vector<4xi32>
    %rightWide = arith.extsi %rightBytes : vector<4xi8> to vector<4xi32>
    %products = arith.muli %leftWide, %rightWide : vector<4xi32>
    %added = llvm.call_intrinsic "llvm.vector.reduce.add.v4i32"(%products) : (vector<4xi32>) -> i32
    %result = arith.addi %added, %sum : i32
    return %result : i32
  }
