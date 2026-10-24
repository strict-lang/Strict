  // the target specific operations of the device runtime and of the kernels generated from loops, AMDGPU for AMD: the
  // block size and count come from the dispatch packet (16 bit workgroup size x at 4, 32 bit grid size x at 12)
  func.func private @strict_device_thread_index() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %thread = llvm.call_intrinsic "llvm.amdgcn.workitem.id.x"() : () -> i32
    return %thread : i32
  }
  func.func private @strict_device_block_index() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %block = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
    return %block : i32
  }
  func.func private @strict_device_block_size() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %packet = llvm.call_intrinsic "llvm.amdgcn.dispatch.ptr"() : () -> !llvm.ptr<4>
    %sizeSlot = llvm.getelementptr %packet[4] : (!llvm.ptr<4>) -> !llvm.ptr<4>, i8
    %size = llvm.load %sizeSlot : !llvm.ptr<4> -> i16
    %width = arith.extui %size : i16 to i32
    return %width : i32
  }
  func.func private @strict_device_block_count() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %packet = llvm.call_intrinsic "llvm.amdgcn.dispatch.ptr"() : () -> !llvm.ptr<4>
    %gridSlot = llvm.getelementptr %packet[12] : (!llvm.ptr<4>) -> !llvm.ptr<4>, i8
    %grid = llvm.load %gridSlot : !llvm.ptr<4> -> i32
    %width = func.call @strict_device_block_size() : () -> i32
    %blocks = arith.divui %grid, %width : i32
    return %blocks : i32
  }
  // a warp is each half of a 64 lane wavefront, lanes 0 to 31 of it
  func.func private @strict_device_lane_index() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %all = arith.constant -1 : i32
    %zero = arith.constant 0 : i32
    %thirtyOne = arith.constant 31 : i32
    %low = llvm.call_intrinsic "llvm.amdgcn.mbcnt.lo"(%all, %zero) : (i32, i32) -> i32
    %lane = llvm.call_intrinsic "llvm.amdgcn.mbcnt.hi"(%all, %low) : (i32, i32) -> i32
    %inWarp = arith.andi %lane, %thirtyOne : i32
    return %inWarp : i32
  }
  // the bits lane source of the own warp has
  func.func private @strict_device_lane(%bits: i32, %source: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %all = arith.constant -1 : i32
    %zero = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %thirtyTwo = arith.constant 32 : i32
    %low = llvm.call_intrinsic "llvm.amdgcn.mbcnt.lo"(%all, %zero) : (i32, i32) -> i32
    %lane = llvm.call_intrinsic "llvm.amdgcn.mbcnt.hi"(%all, %low) : (i32, i32) -> i32
    %half = arith.andi %lane, %thirtyTwo : i32
    %at = arith.addi %half, %source : i32
    %address = arith.shli %at, %two : i32
    %other = llvm.call_intrinsic "llvm.amdgcn.ds.bpermute"(%address, %bits) : (i32, i32) -> i32
    return %other : i32
  }
  // the bits of the lane whose number differs in the bit of offset, a step of a butterfly over a warp
  func.func private @strict_device_swap(%bits: i32, %offset: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %lane = func.call @strict_device_lane_index() : () -> i32
    %source = arith.xori %lane, %offset : i32
    %other = func.call @strict_device_lane(%bits, %source) : (i32, i32) -> i32
    return %other : i32
  }
  func.func private @strict_device_fence() attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    llvm.fence syncscope("agent") seq_cst
    return
  }
  // the kernels take the paths of NVIDIA GPUs before sm_80 (one m8n8k16 product at a time)
  func.func private @strict_device_arch() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %arch = arith.constant 750 : i32
    return %arch : i32
  }
  // the tensor core operations of an NVIDIA warp with the same fragments per lane (lane l has row l / 4 and the k or
  // columns of l % 4), every lane gathers its row and columns from the lanes holding them
  func.func private @strict_device_mma8(%a: i32, %b: i32, %c0: i32, %c1: i32) -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0i = arith.constant 0 : i32
    %c1i = arith.constant 1 : i32
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %lane = func.call @strict_device_lane_index() : () -> i32
    %group = arith.shrui %lane, %c2i : i32
    %quad = arith.andi %lane, %c3i : i32
    %rowLanes = arith.shli %group, %c2i : i32
    %firstColumn = arith.shli %quad, %c3i : i32
    %secondColumn = arith.addi %firstColumn, %c4i : i32
    cf.br ^step(%c0i, %c0, %c1 : i32, i32, i32)
  ^step(%k: i32, %first: i32, %second: i32):
    %more = arith.cmpi slt, %k, %c4i : i32
    cf.cond_br %more, ^add, ^done
  ^add:
    %aLane = arith.addi %rowLanes, %k : i32
    %bFirstLane = arith.addi %firstColumn, %k : i32
    %bSecondLane = arith.addi %secondColumn, %k : i32
    %aBits = func.call @strict_device_lane(%a, %aLane) : (i32, i32) -> i32
    %bFirst = func.call @strict_device_lane(%b, %bFirstLane) : (i32, i32) -> i32
    %bSecond = func.call @strict_device_lane(%b, %bSecondLane) : (i32, i32) -> i32
    %firstSum = func.call @strict_device_dp4a(%aBits, %bFirst, %first) : (i32, i32, i32) -> i32
    %secondSum = func.call @strict_device_dp4a(%aBits, %bSecond, %second) : (i32, i32, i32) -> i32
    %next = arith.addi %k, %c1i : i32
    cf.br ^step(%next, %firstSum, %secondSum : i32, i32, i32)
  ^done:
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %withFirst = llvm.insertvalue %first, %empty[0] : !llvm.struct<(i32, i32)>
    %both = llvm.insertvalue %second, %withFirst[1] : !llvm.struct<(i32, i32)>
    return %both : !llvm.struct<(i32, i32)>
  }
  // m16n8k32: rows g and g + 8 (a0, a1) of k 0 to 15 (b0) and then of k 16 to 31 (a2, a3 with b1)
  func.func private @strict_device_mma16(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %c0: i32, %c1: i32, %c2: i32, %c3: i32) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
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
  // m16n8k8 of 16 bit floats: a0 has 2 numbers of row g, a1 of row g + 8, b 2 numbers of column g, k is 2 * (l % 4)
  func.func private @strict_device_halves_mma(%a0: vector<2xf16>, %a1: vector<2xf16>, %b: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0i = arith.constant 0 : i32
    %c1i = arith.constant 1 : i32
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %lane = func.call @strict_device_lane_index() : () -> i32
    %group = arith.shrui %lane, %c2i : i32
    %quad = arith.andi %lane, %c3i : i32
    %rowLanes = arith.shli %group, %c2i : i32
    %firstColumn = arith.shli %quad, %c3i : i32
    %secondColumn = arith.addi %firstColumn, %c4i : i32
    %upperBits = llvm.bitcast %a0 : vector<2xf16> to i32
    %lowerBits = llvm.bitcast %a1 : vector<2xf16> to i32
    %columnBits = llvm.bitcast %b : vector<2xf16> to i32
    cf.br ^step(%c0i, %c0, %c1, %c2, %c3 : i32, f32, f32, f32, f32)
  ^step(%k: i32, %d0: f32, %d1: f32, %d2: f32, %d3: f32):
    %more = arith.cmpi slt, %k, %c4i : i32
    cf.cond_br %more, ^add, ^done
  ^add:
    %aLane = arith.addi %rowLanes, %k : i32
    %bFirstLane = arith.addi %firstColumn, %k : i32
    %bSecondLane = arith.addi %secondColumn, %k : i32
    %upper = func.call @strict_device_lane(%upperBits, %aLane) : (i32, i32) -> i32
    %lower = func.call @strict_device_lane(%lowerBits, %aLane) : (i32, i32) -> i32
    %first = func.call @strict_device_lane(%columnBits, %bFirstLane) : (i32, i32) -> i32
    %second = func.call @strict_device_lane(%columnBits, %bSecondLane) : (i32, i32) -> i32
    %e0 = func.call @strict_device_halves_dot(%upper, %first, %d0) : (i32, i32, f32) -> f32
    %e1 = func.call @strict_device_halves_dot(%upper, %second, %d1) : (i32, i32, f32) -> f32
    %e2 = func.call @strict_device_halves_dot(%lower, %first, %d2) : (i32, i32, f32) -> f32
    %e3 = func.call @strict_device_halves_dot(%lower, %second, %d3) : (i32, i32, f32) -> f32
    %next = arith.addi %k, %c1i : i32
    cf.br ^step(%next, %e0, %e1, %e2, %e3 : i32, f32, f32, f32, f32)
  ^done:
    %empty = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
    %with0 = llvm.insertvalue %d0, %empty[0] : !llvm.struct<(f32, f32, f32, f32)>
    %with1 = llvm.insertvalue %d1, %with0[1] : !llvm.struct<(f32, f32, f32, f32)>
    %with2 = llvm.insertvalue %d2, %with1[2] : !llvm.struct<(f32, f32, f32, f32)>
    %with3 = llvm.insertvalue %d3, %with2[3] : !llvm.struct<(f32, f32, f32, f32)>
    return %with3 : !llvm.struct<(f32, f32, f32, f32)>
  }
  // sum plus the products of the two 16 bit floats in the bits of left and right
  func.func private @strict_device_halves_dot(%left: i32, %right: i32, %sum: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %leftHalves = llvm.bitcast %left : i32 to vector<2xf16>
    %rightHalves = llvm.bitcast %right : i32 to vector<2xf16>
    %leftWide = arith.extf %leftHalves : vector<2xf16> to vector<2xf32>
    %rightWide = arith.extf %rightHalves : vector<2xf16> to vector<2xf32>
    %products = arith.mulf %leftWide, %rightWide : vector<2xf32>
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %first = llvm.extractelement %products[%c0 : i32] : vector<2xf32>
    %second = llvm.extractelement %products[%c1 : i32] : vector<2xf32>
    %partial = arith.addf %sum, %first : f32
    %total = arith.addf %partial, %second : f32
    return %total : f32
  }
  // ldmatrix.x4: lane j * 8 + r has the address of row r of matrix j, lane l gets 4 bytes of row l / 4 of each matrix
  // m16n8k16 of 16 bit floats as two m16n8k8
  func.func private @strict_device_halves_mma16(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %halves = func.call @strict_device_halves_mma8x2(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %halves : !llvm.struct<(f32, f32, f32, f32)>
  }
  func.func private @strict_device_attention_mma16(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32, %brain: i1) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %halves = func.call @strict_device_halves_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %halves : !llvm.struct<(f32, f32, f32, f32)>
  }
  func.func private @strict_device_matrices(%slot: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c24 = arith.constant 24 : i32
    %lane = func.call @strict_device_lane_index() : () -> i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %column = arith.shli %quad, %c2 : i32
    %address = llvm.ptrtoint %slot : !llvm.ptr<3> to i32
    %second = arith.addi %row, %c8 : i32
    %third = arith.addi %row, %c16 : i32
    %fourth = arith.addi %row, %c24 : i32
    %m0 = func.call @strict_device_matrix_part(%address, %row, %column) : (i32, i32, i32) -> i32
    %m1 = func.call @strict_device_matrix_part(%address, %second, %column) : (i32, i32, i32) -> i32
    %m2 = func.call @strict_device_matrix_part(%address, %third, %column) : (i32, i32, i32) -> i32
    %m3 = func.call @strict_device_matrix_part(%address, %fourth, %column) : (i32, i32, i32) -> i32
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %with0 = llvm.insertvalue %m0, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %with1 = llvm.insertvalue %m1, %with0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %with2 = llvm.insertvalue %m2, %with1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %with3 = llvm.insertvalue %m3, %with2[3] : !llvm.struct<(i32, i32, i32, i32)>
    return %with3 : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_matrix_part(%address: i32, %source: i32, %column: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %rowAddress = func.call @strict_device_lane(%address, %source) : (i32, i32) -> i32
    %at = arith.addi %rowAddress, %column : i32
    %pointer = llvm.inttoptr %at : i32 to !llvm.ptr<3>
    %bits = llvm.load %pointer {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    return %bits : i32
  }
  // strict_device_halves_mma16 adding to 16 bit sums (c0 the two of row g, c1 those of row g + 8)
  func.func private @strict_device_halves_mma16_halves(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: vector<2xf16>, %c1: vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %lowWide = arith.extf %c0 : vector<2xf16> to vector<2xf32>
    %highWide = arith.extf %c1 : vector<2xf16> to vector<2xf32>
    %sum0 = llvm.extractelement %lowWide[%i0 : i32] : vector<2xf32>
    %sum1 = llvm.extractelement %lowWide[%i1 : i32] : vector<2xf32>
    %sum2 = llvm.extractelement %highWide[%i0 : i32] : vector<2xf32>
    %sum3 = llvm.extractelement %highWide[%i1 : i32] : vector<2xf32>
    %wide = func.call @strict_device_halves_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %d0 = llvm.extractvalue %wide[0] : !llvm.struct<(f32, f32, f32, f32)>
    %d1 = llvm.extractvalue %wide[1] : !llvm.struct<(f32, f32, f32, f32)>
    %d2 = llvm.extractvalue %wide[2] : !llvm.struct<(f32, f32, f32, f32)>
    %d3 = llvm.extractvalue %wide[3] : !llvm.struct<(f32, f32, f32, f32)>
    %empty = llvm.mlir.undef : vector<2xf32>
    %low0 = llvm.insertelement %d0, %empty[%i0 : i32] : vector<2xf32>
    %low = llvm.insertelement %d1, %low0[%i1 : i32] : vector<2xf32>
    %high0 = llvm.insertelement %d2, %empty[%i0 : i32] : vector<2xf32>
    %high = llvm.insertelement %d3, %high0[%i1 : i32] : vector<2xf32>
    %lowHalves = arith.truncf %low : vector<2xf32> to vector<2xf16>
    %highHalves = arith.truncf %high : vector<2xf32> to vector<2xf16>
    %none = llvm.mlir.undef : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %withLow = llvm.insertvalue %lowHalves, %none[0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %both = llvm.insertvalue %highHalves, %withLow[1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    return %both : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  }
  // ldmatrix.x4.trans: lane l gets the numbers of rows 2 * (l % 4) and the next in column l / 4 of each matrix
  func.func private @strict_device_matrices_transposed(%slot: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c24 = arith.constant 24 : i32
    %lane = func.call @strict_device_lane_index() : () -> i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %column = arith.shli %row, %c1 : i32
    %pair = arith.shli %quad, %c1 : i32
    %address = llvm.ptrtoint %slot : !llvm.ptr<3> to i32
    %second = arith.addi %pair, %c8 : i32
    %third = arith.addi %pair, %c16 : i32
    %fourth = arith.addi %pair, %c24 : i32
    %m0 = func.call @strict_device_matrix_column(%address, %pair, %column) : (i32, i32, i32) -> i32
    %m1 = func.call @strict_device_matrix_column(%address, %second, %column) : (i32, i32, i32) -> i32
    %m2 = func.call @strict_device_matrix_column(%address, %third, %column) : (i32, i32, i32) -> i32
    %m3 = func.call @strict_device_matrix_column(%address, %fourth, %column) : (i32, i32, i32) -> i32
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %with0 = llvm.insertvalue %m0, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %with1 = llvm.insertvalue %m1, %with0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %with2 = llvm.insertvalue %m2, %with1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %with3 = llvm.insertvalue %m3, %with2[3] : !llvm.struct<(i32, i32, i32, i32)>
    return %with3 : !llvm.struct<(i32, i32, i32, i32)>
  }
  // the 16 bit numbers at column (bytes) of the rows lanes source and source + 1 point to, as low and high half
  func.func private @strict_device_matrix_column(%address: i32, %source: i32, %column: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %next = arith.addi %source, %c1 : i32
    %firstRow = func.call @strict_device_lane(%address, %source) : (i32, i32) -> i32
    %secondRow = func.call @strict_device_lane(%address, %next) : (i32, i32) -> i32
    %firstAt = arith.addi %firstRow, %column : i32
    %secondAt = arith.addi %secondRow, %column : i32
    %firstPointer = llvm.inttoptr %firstAt : i32 to !llvm.ptr<3>
    %secondPointer = llvm.inttoptr %secondAt : i32 to !llvm.ptr<3>
    %low = llvm.load %firstPointer {alignment = 2 : i64} : !llvm.ptr<3> -> i16
    %high = llvm.load %secondPointer {alignment = 2 : i64} : !llvm.ptr<3> -> i16
    %lowWide = arith.extui %low : i16 to i32
    %highWide = arith.extui %high : i16 to i32
    %shifted = arith.shli %highWide, %c16 : i32
    %bits = arith.ori %lowWide, %shifted : i32
    return %bits : i32
  }
  func.func private @strict_device_barrier() attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    llvm.fence syncscope("workgroup") release
    llvm.call_intrinsic "llvm.amdgcn.s.barrier"() : () -> ()
    llvm.fence syncscope("workgroup") acquire
    return
  }
  // 2 to the power of number (flushing results below the smallest normal float to 0)
  func.func private @strict_device_binary_power(%number: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %power = llvm.call_intrinsic "llvm.exp2.f32"(%number) : (f32) -> f32
    return %power : f32
  }
  func.func private @strict_device_binary_logarithm(%number: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %binary = llvm.call_intrinsic "llvm.log2.f32"(%number) : (f32) -> f32
    return %binary : f32
  }
  func.func private @strict_device_radian_sine(%radians: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %sine = llvm.call_intrinsic "llvm.sin.f32"(%radians) : (f32) -> f32
    return %sine : f32
  }
  func.func private @strict_device_radian_cosine(%radians: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %cosine = llvm.call_intrinsic "llvm.cos.f32"(%radians) : (f32) -> f32
    return %cosine : f32
  }
  func.func private @strict_device_rounded(%number: f32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %even = llvm.call_intrinsic "llvm.rint.f32"(%number) : (f32) -> f32
    %rounded = arith.fptosi %even : f32 to i32
    return %rounded : i32
  }
  // the copies of strict_device_copy are a load and a store, done when they return
  func.func private @strict_device_copy(%shared: !llvm.ptr<3>, %global: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %words = llvm.load %global invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    llvm.store %words, %shared {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    return
  }
  func.func private @strict_device_copy_filled(%shared: !llvm.ptr<3>, %global: !llvm.ptr, %isCopied: i1) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %zeros = arith.constant dense<0> : vector<4xi32>
    %words = llvm.load %global invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %filled = arith.select %isCopied, %words, %zeros : vector<4xi32>
    llvm.store %filled, %shared {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    return
  }
  func.func private @strict_device_copy_word(%shared: !llvm.ptr<3>, %global: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %word = llvm.load %global invariant : !llvm.ptr -> i32
    llvm.store %word, %shared : i32, !llvm.ptr<3>
    return
  }
  func.func private @strict_device_copies_commit() attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    return
  }
  func.func private @strict_device_copies_wait() attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    return
  }
  // older GPUs have no 8 bit dot product instruction (sdot4 needs gfx906), four multiplies of the signed bytes
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %leftBytes = llvm.bitcast %left : i32 to vector<4xi8>
    %rightBytes = llvm.bitcast %right : i32 to vector<4xi8>
    %leftWide = arith.extsi %leftBytes : vector<4xi8> to vector<4xi32>
    %rightWide = arith.extsi %rightBytes : vector<4xi8> to vector<4xi32>
    %products = arith.muli %leftWide, %rightWide : vector<4xi32>
    %added = llvm.call_intrinsic "llvm.vector.reduce.add.v4i32"(%products) : (vector<4xi32>) -> i32
    %result = arith.addi %added, %sum : i32
    return %result : i32
  }
