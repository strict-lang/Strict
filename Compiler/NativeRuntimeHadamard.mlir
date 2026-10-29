  // the 256 numbers of the group of a block for strict_device_hadamard_run
  llvm.mlir.global internal @strict_device_hadamard_numbers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<256 x f32>
  // one radix-4 butterfly of the 256 numbers of a block at step (1, 4, 16, 64) for its thread: the sum of the 4 numbers
  // of its quad halved, the one at 3 - digit negated, like `Convrot.Butterfly`
  func.func private @strict_device_hadamard_stage(%value: f32, %thread: i64, %step: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c3 = arith.constant 3 : i64
    %shared = llvm.mlir.addressof @strict_device_hadamard_numbers : !llvm.ptr<3>
    %own = llvm.getelementptr %shared[%thread] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %value, %own : f32, !llvm.ptr<3>
    func.call @strict_device_barrier() : () -> ()
    %quotient = arith.divui %thread, %step : i64
    %digit = arith.andi %quotient, %c3 : i64
    %offset = arith.muli %digit, %step : i64
    %first = arith.subi %thread, %offset : i64
    %partner = arith.subi %c3, %digit : i64
    %sum = func.call @strict_device_hadamard_quad(%shared, %first, %step, %partner) : (!llvm.ptr<3>, i64, i64, i64) -> f32
    func.call @strict_device_barrier() : () -> ()
    return %sum : f32
  }
  // the quad from first at step in shared memory through strict_device_hadamard_sum
  func.func private @strict_device_hadamard_quad(%shared: !llvm.ptr<3>, %first: i64, %step: i64, %partner: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %at1 = arith.addi %first, %step : i64
    %at2 = arith.addi %at1, %step : i64
    %at3 = arith.addi %at2, %step : i64
    %slot0 = llvm.getelementptr %shared[%first] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %slot1 = llvm.getelementptr %shared[%at1] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %slot2 = llvm.getelementptr %shared[%at2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %slot3 = llvm.getelementptr %shared[%at3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %x0 = llvm.load %slot0 : !llvm.ptr<3> -> f32
    %x1 = llvm.load %slot1 : !llvm.ptr<3> -> f32
    %x2 = llvm.load %slot2 : !llvm.ptr<3> -> f32
    %x3 = llvm.load %slot3 : !llvm.ptr<3> -> f32
    %sum = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %partner) : (f32, f32, f32, f32, i64) -> f32
    return %sum : f32
  }
  // ((x0 s0 + x1 s1) + x2 s2) + x3 s3 of a quad, s is -0.5 at partner and 0.5 elsewhere
  func.func private @strict_device_hadamard_sum(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %partner: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c3 = arith.constant 3 : i64
    %half = arith.constant 0.5 : f32
    %negativeHalf = arith.constant -0.5 : f32
    %is0 = arith.cmpi eq, %partner, %c0 : i64
    %is1 = arith.cmpi eq, %partner, %c1 : i64
    %is2 = arith.cmpi eq, %partner, %c2 : i64
    %is3 = arith.cmpi eq, %partner, %c3 : i64
    %s0 = arith.select %is0, %negativeHalf, %half : f32
    %s1 = arith.select %is1, %negativeHalf, %half : f32
    %s2 = arith.select %is2, %negativeHalf, %half : f32
    %s3 = arith.select %is3, %negativeHalf, %half : f32
    %p0 = arith.mulf %x0, %s0 : f32
    %p1 = arith.mulf %x1, %s1 : f32
    %p2 = arith.mulf %x2, %s2 : f32
    %p3 = arith.mulf %x3, %s3 : f32
    %low = arith.addf %p0, %p1 : f32
    %lowHigh = arith.addf %low, %p2 : f32
    %sum = arith.addf %lowHigh, %p3 : f32
    return %sum : f32
  }
  // `Convrot.Rotated`: a block of 256 threads per group of 256 numbers (count a multiple of 256), each thread one number
  // through the 4 radix-4 butterflies in shared memory, written as doubles when wide (to the host), else in the format
  // results (32 or 17, strict_device_stored)
  func.func private @strict_device_hadamard_run(%count: i64, %result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %results: i64) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    %c16 = arith.constant 16 : i64
    %c64 = arith.constant 64 : i64
    %c256 = arith.constant 256 : i64
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %start = arith.muli %block, %c256 : i64
    %at = arith.addi %start, %thread : i64
    %inside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %inside, ^rotate, ^done
  ^rotate:
    %value = func.call @strict_device_elements(%values, %valuesBits, %at, %valuesCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %first = func.call @strict_device_hadamard_stage(%value, %thread, %c1) : (f32, i64, i64) -> f32
    %second = func.call @strict_device_hadamard_stage(%first, %thread, %c4) : (f32, i64, i64) -> f32
    %third = func.call @strict_device_hadamard_stage(%second, %thread, %c16) : (f32, i64, i64) -> f32
    %fourth = func.call @strict_device_hadamard_stage(%third, %thread, %c64) : (f32, i64, i64) -> f32
    func.call @strict_device_result(%result, %at, %fourth, %wide, %results) : (!llvm.ptr, i64, f32, i1, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_hadamard(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %results: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_hadamard_run(%count, %result, %wide, %values, %valuesCount, %valuesBits, %results) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_hadamards(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %results: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_hadamard_run(%count, %result, %wide, %values, %valuesCount, %valuesBits, %results) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
  // the largest magnitude of each warp of a block of strict_device_hadamard_tokens
  llvm.mlir.global internal @strict_device_hadamard_largest() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<32 x f32>
  // the numbers at first, second, third and fourth of numbers (digits 0 to 3 of a quad) through the radix-4
  // butterfly of strict_device_hadamard_sum
  func.func private @strict_device_hadamard_quad_of(%numbers: vector<8xf32>, %first: i32, %second: i32, %third: i32, %fourth: i32) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %x0 = llvm.extractelement %numbers[%first : i32] : vector<8xf32>
    %x1 = llvm.extractelement %numbers[%second : i32] : vector<8xf32>
    %x2 = llvm.extractelement %numbers[%third : i32] : vector<8xf32>
    %x3 = llvm.extractelement %numbers[%fourth : i32] : vector<8xf32>
    %d0 = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %three) : (f32, f32, f32, f32, i64) -> f32
    %d1 = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %two) : (f32, f32, f32, f32, i64) -> f32
    %d2 = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %one) : (f32, f32, f32, f32, i64) -> f32
    %d3 = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %none) : (f32, f32, f32, f32, i64) -> f32
    %v0 = llvm.insertelement %d0, %numbers[%first : i32] : vector<8xf32>
    %v1 = llvm.insertelement %d1, %v0[%second : i32] : vector<8xf32>
    %v2 = llvm.insertelement %d2, %v1[%third : i32] : vector<8xf32>
    %v3 = llvm.insertelement %d3, %v2[%fourth : i32] : vector<8xf32>
    return %v3 : vector<8xf32>
  }
  // numbers after bit apart of their positions and bit mask of the lane trade places: of each pair low and low + apart
  // the lane without the mask bit keeps low and gets the low of the lane mask apart, that one keeps its high
  func.func private @strict_device_hadamard_traded(%numbers: vector<8xf32>, %apart: i32, %mask: i32, %lane: i32) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %laneBit = arith.andi %lane, %mask : i32
    %set = arith.cmpi ne, %laneBit, %c0 : i32
    %above = arith.subi %c0, %apart : i32
    %upper1 = arith.andi %c1, %above : i32
    %upper2 = arith.andi %c2, %above : i32
    %upper3 = arith.andi %c3, %above : i32
    %low1 = arith.addi %c1, %upper1 : i32
    %low2 = arith.addi %c2, %upper2 : i32
    %low3 = arith.addi %c3, %upper3 : i32
    %traded0 = func.call @strict_device_hadamard_exchange(%numbers, %c0, %apart, %mask, %set) : (vector<8xf32>, i32, i32, i32, i1) -> vector<8xf32>
    %traded1 = func.call @strict_device_hadamard_exchange(%traded0, %low1, %apart, %mask, %set) : (vector<8xf32>, i32, i32, i32, i1) -> vector<8xf32>
    %traded2 = func.call @strict_device_hadamard_exchange(%traded1, %low2, %apart, %mask, %set) : (vector<8xf32>, i32, i32, i32, i1) -> vector<8xf32>
    %traded3 = func.call @strict_device_hadamard_exchange(%traded2, %low3, %apart, %mask, %set) : (vector<8xf32>, i32, i32, i32, i1) -> vector<8xf32>
    return %traded3 : vector<8xf32>
  }
  func.func private @strict_device_hadamard_exchange(%numbers: vector<8xf32>, %low: i32, %apart: i32, %mask: i32, %set: i1) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %high = arith.addi %low, %apart : i32
    %lowNumber = llvm.extractelement %numbers[%low : i32] : vector<8xf32>
    %highNumber = llvm.extractelement %numbers[%high : i32] : vector<8xf32>
    %sent = arith.select %set, %lowNumber, %highNumber : f32
    %received = func.call @strict_device_shuffles(%sent, %mask) : (f32, i32) -> f32
    %newLow = arith.select %set, %received, %lowNumber : f32
    %newHigh = arith.select %set, %highNumber, %received : f32
    %withLow = llvm.insertelement %newLow, %numbers[%low : i32] : vector<8xf32>
    %withHigh = llvm.insertelement %newHigh, %withLow[%high : i32] : vector<8xf32>
    return %withHigh : vector<8xf32>
  }
  // the 8 numbers from at of source (bits 32, 17 or 16), with 16 byte loads when they are aligned
  func.func private @strict_device_hadamard_eight(%source: !llvm.ptr, %bits: i64, %at: i64) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %fifteen = arith.constant 15 : i64
    %thirtyTwo = arith.constant 32 : i64
    %sixteen = arith.constant 16 : i64
    %brainBits = arith.constant 17 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    %isHalf = arith.cmpi eq, %bits, %sixteen : i64
    %isNarrow = arith.ori %isBrain, %isHalf : i1
    cf.cond_br %isSingle, ^single, ^narrowing
  ^single:
    %singleSlot = llvm.getelementptr %source[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleAddress = llvm.ptrtoint %singleSlot : !llvm.ptr to i64
    %singleRest = arith.andi %singleAddress, %fifteen : i64
    %singleAligned = arith.cmpi eq, %singleRest, %none : i64
    cf.cond_br %singleAligned, ^singles, ^each
  ^singles:
    %lower = llvm.load %singleSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %upperSlot = llvm.getelementptr %singleSlot[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %upper = llvm.load %upperSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %joined = llvm.shufflevector %lower, %upper [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
    return %joined : vector<8xf32>
  ^narrowing:
    cf.cond_br %isNarrow, ^narrow, ^each
  ^narrow:
    %narrowSlot = llvm.getelementptr %source[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %narrowAddress = llvm.ptrtoint %narrowSlot : !llvm.ptr to i64
    %narrowRest = arith.andi %narrowAddress, %fifteen : i64
    %narrowAligned = arith.cmpi eq, %narrowRest, %none : i64
    cf.cond_br %narrowAligned, ^narrows, ^each
  ^narrows:
    %words = llvm.load %narrowSlot {alignment = 16 : i64} : !llvm.ptr -> vector<8xi16>
    %brains = arith.bitcast %words : vector<8xi16> to vector<8xbf16>
    %halves = arith.bitcast %words : vector<8xi16> to vector<8xf16>
    %fromBrains = arith.extf %brains : vector<8xbf16> to vector<8xf32>
    %fromHalves = arith.extf %halves : vector<8xf16> to vector<8xf32>
    %narrowNumbers = arith.select %isBrain, %fromBrains, %fromHalves : vector<8xf32>
    return %narrowNumbers : vector<8xf32>
  ^each:
    %empty = llvm.mlir.undef : vector<8xf32>
    cf.br ^next(%none, %empty : i64, vector<8xf32>)
  ^next(%index: i64, %gathered: vector<8xf32>):
    %eight = arith.constant 8 : i64
    %more = arith.cmpi slt, %index, %eight : i64
    cf.cond_br %more, ^gather, ^gathered
  ^gather:
    %numberAt = arith.addi %at, %index : i64
    %number = func.call @strict_device_floats(%source, %bits, %numberAt) : (!llvm.ptr, i64, i64) -> f32
    %position = arith.trunci %index : i64 to i32
    %withNumber = llvm.insertelement %number, %gathered[%position : i32] : vector<8xf32>
    %following = arith.constant 1 : i64
    %indexNext = arith.addi %index, %following : i64
    llvm.br ^next(%indexNext, %withNumber : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^gathered:
    return %gathered : vector<8xf32>
  }
  // the 8 numbers of a lane of strict_device_hadamard_tokens rotated through the 4 radix-4 butterflies of its group of
  // 256 from start: it reads numbers 8 lane to 8 lane + 7, each butterfly (steps 1, 4, 16 and 64) runs on positions of
  // the lane after strict_device_hadamard_traded moved the bits of its digit there, in the order of
  // strict_device_hadamard_sum; then position p holds number (lane & 7) + 2 (lane & 24) + 8 p0 + 64 p1 + 128 p2
  func.func private @strict_device_hadamard_group(%source: !llvm.ptr, %bits: i64, %start: i64, %lane: i32) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %three = arith.constant 3 : i64
    %lane64 = arith.extui %lane : i32 to i64
    %laneFirst = arith.shli %lane64, %three : i64
    %at = arith.addi %start, %laneFirst : i64
    %read = func.call @strict_device_hadamard_eight(%source, %bits, %at) : (!llvm.ptr, i64, i64) -> vector<8xf32>
    %stepOneLow = func.call @strict_device_hadamard_quad_of(%read, %c0, %c1, %c2, %c3) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %stepOne = func.call @strict_device_hadamard_quad_of(%stepOneLow, %c4, %c5, %c6, %c7) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %eightIn = func.call @strict_device_hadamard_traded(%stepOne, %c1, %c1, %lane) : (vector<8xf32>, i32, i32, i32) -> vector<8xf32>
    %stepFourLow = func.call @strict_device_hadamard_quad_of(%eightIn, %c0, %c4, %c1, %c5) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %stepFour = func.call @strict_device_hadamard_quad_of(%stepFourLow, %c2, %c6, %c3, %c7) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %sixteenIn = func.call @strict_device_hadamard_traded(%stepFour, %c2, %c2, %lane) : (vector<8xf32>, i32, i32, i32) -> vector<8xf32>
    %thirtyTwoIn = func.call @strict_device_hadamard_traded(%sixteenIn, %c4, %c4, %lane) : (vector<8xf32>, i32, i32, i32) -> vector<8xf32>
    %stepSixteenLow = func.call @strict_device_hadamard_quad_of(%thirtyTwoIn, %c0, %c2, %c4, %c6) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %stepSixteen = func.call @strict_device_hadamard_quad_of(%stepSixteenLow, %c1, %c3, %c5, %c7) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %sixtyFourIn = func.call @strict_device_hadamard_traded(%stepSixteen, %c2, %c8, %lane) : (vector<8xf32>, i32, i32, i32) -> vector<8xf32>
    %hundredTwentyEightIn = func.call @strict_device_hadamard_traded(%sixtyFourIn, %c4, %c16, %lane) : (vector<8xf32>, i32, i32, i32) -> vector<8xf32>
    %stepSixtyFourLow = func.call @strict_device_hadamard_quad_of(%hundredTwentyEightIn, %c0, %c2, %c4, %c6) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    %rotated = func.call @strict_device_hadamard_quad_of(%stepSixtyFourLow, %c1, %c3, %c5, %c7) : (vector<8xf32>, i32, i32, i32, i32) -> vector<8xf32>
    return %rotated : vector<8xf32>
  }
  // the largest magnitude of the 8 numbers of a lane
  func.func private @strict_device_hadamard_magnitude(%numbers: vector<8xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %i0 = arith.constant 0 : i32
    %n0 = llvm.extractelement %numbers[%i0 : i32] : vector<8xf32>
    %m0 = llvm.call_intrinsic "llvm.fabs.f32"(%n0) : (f32) -> f32
    %i1 = arith.constant 1 : i32
    %n1 = llvm.extractelement %numbers[%i1 : i32] : vector<8xf32>
    %m1 = llvm.call_intrinsic "llvm.fabs.f32"(%n1) : (f32) -> f32
    %i2 = arith.constant 2 : i32
    %n2 = llvm.extractelement %numbers[%i2 : i32] : vector<8xf32>
    %m2 = llvm.call_intrinsic "llvm.fabs.f32"(%n2) : (f32) -> f32
    %i3 = arith.constant 3 : i32
    %n3 = llvm.extractelement %numbers[%i3 : i32] : vector<8xf32>
    %m3 = llvm.call_intrinsic "llvm.fabs.f32"(%n3) : (f32) -> f32
    %i4 = arith.constant 4 : i32
    %n4 = llvm.extractelement %numbers[%i4 : i32] : vector<8xf32>
    %m4 = llvm.call_intrinsic "llvm.fabs.f32"(%n4) : (f32) -> f32
    %i5 = arith.constant 5 : i32
    %n5 = llvm.extractelement %numbers[%i5 : i32] : vector<8xf32>
    %m5 = llvm.call_intrinsic "llvm.fabs.f32"(%n5) : (f32) -> f32
    %i6 = arith.constant 6 : i32
    %n6 = llvm.extractelement %numbers[%i6 : i32] : vector<8xf32>
    %m6 = llvm.call_intrinsic "llvm.fabs.f32"(%n6) : (f32) -> f32
    %i7 = arith.constant 7 : i32
    %n7 = llvm.extractelement %numbers[%i7 : i32] : vector<8xf32>
    %m7 = llvm.call_intrinsic "llvm.fabs.f32"(%n7) : (f32) -> f32
    %largest01 = arith.maxnumf %m0, %m1 : f32
    %largest23 = arith.maxnumf %m2, %m3 : f32
    %largest45 = arith.maxnumf %m4, %m5 : f32
    %largest67 = arith.maxnumf %m6, %m7 : f32
    %largestLow = arith.maxnumf %largest01, %largest23 : f32
    %largestHigh = arith.maxnumf %largest45, %largest67 : f32
    %largest = arith.maxnumf %largestLow, %largestHigh : f32
    return %largest : f32
  }
  // the 8 numbers of a lane times factor rounded to 8 bit numbers of target, position 0 at at, the others 8, 64, 72,
  // 128, 136, 192 and 200 after it (strict_device_hadamard_group)
  func.func private @strict_device_hadamard_codes(%target: !llvm.ptr, %at: i64, %numbers: vector<8xf32>, %factor: f32, %residual: i1) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %bytes = llvm.getelementptr %target[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %i0 = arith.constant 0 : i32
    %n0 = llvm.extractelement %numbers[%i0 : i32] : vector<8xf32>
    %scaled0 = arith.mulf %n0, %factor : f32
    %byte0 = func.call @strict_device_code(%scaled0, %residual) : (f32, i1) -> i8
    %byteSlot0 = llvm.getelementptr %bytes[0] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte0, %byteSlot0 : i8, !llvm.ptr
    %i1 = arith.constant 1 : i32
    %n1 = llvm.extractelement %numbers[%i1 : i32] : vector<8xf32>
    %scaled1 = arith.mulf %n1, %factor : f32
    %byte1 = func.call @strict_device_code(%scaled1, %residual) : (f32, i1) -> i8
    %byteSlot1 = llvm.getelementptr %bytes[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte1, %byteSlot1 : i8, !llvm.ptr
    %i2 = arith.constant 2 : i32
    %n2 = llvm.extractelement %numbers[%i2 : i32] : vector<8xf32>
    %scaled2 = arith.mulf %n2, %factor : f32
    %byte2 = func.call @strict_device_code(%scaled2, %residual) : (f32, i1) -> i8
    %byteSlot2 = llvm.getelementptr %bytes[64] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte2, %byteSlot2 : i8, !llvm.ptr
    %i3 = arith.constant 3 : i32
    %n3 = llvm.extractelement %numbers[%i3 : i32] : vector<8xf32>
    %scaled3 = arith.mulf %n3, %factor : f32
    %byte3 = func.call @strict_device_code(%scaled3, %residual) : (f32, i1) -> i8
    %byteSlot3 = llvm.getelementptr %bytes[72] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte3, %byteSlot3 : i8, !llvm.ptr
    %i4 = arith.constant 4 : i32
    %n4 = llvm.extractelement %numbers[%i4 : i32] : vector<8xf32>
    %scaled4 = arith.mulf %n4, %factor : f32
    %byte4 = func.call @strict_device_code(%scaled4, %residual) : (f32, i1) -> i8
    %byteSlot4 = llvm.getelementptr %bytes[128] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte4, %byteSlot4 : i8, !llvm.ptr
    %i5 = arith.constant 5 : i32
    %n5 = llvm.extractelement %numbers[%i5 : i32] : vector<8xf32>
    %scaled5 = arith.mulf %n5, %factor : f32
    %byte5 = func.call @strict_device_code(%scaled5, %residual) : (f32, i1) -> i8
    %byteSlot5 = llvm.getelementptr %bytes[136] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte5, %byteSlot5 : i8, !llvm.ptr
    %i6 = arith.constant 6 : i32
    %n6 = llvm.extractelement %numbers[%i6 : i32] : vector<8xf32>
    %scaled6 = arith.mulf %n6, %factor : f32
    %byte6 = func.call @strict_device_code(%scaled6, %residual) : (f32, i1) -> i8
    %byteSlot6 = llvm.getelementptr %bytes[192] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte6, %byteSlot6 : i8, !llvm.ptr
    %i7 = arith.constant 7 : i32
    %n7 = llvm.extractelement %numbers[%i7 : i32] : vector<8xf32>
    %scaled7 = arith.mulf %n7, %factor : f32
    %byte7 = func.call @strict_device_code(%scaled7, %residual) : (f32, i1) -> i8
    %byteSlot7 = llvm.getelementptr %bytes[200] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte7, %byteSlot7 : i8, !llvm.ptr
    return
  }
  // strict_device_quantize_tokens of `Convrot.Rotated` of 32 or 16 bit source numbers (bits) without writing the rotated
  // ones: a block per token (width a multiple of 256 up to 16384, strict_gpu_hadamard_threads), a warp per group of 256
  // numbers (strict_device_hadamard_group) and a second one warps groups later when there are more groups than warps
  func.func private @strict_device_hadamard_tokens_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %residual, %format = func.call @strict_device_code_format(%bits) : (i64) -> (i1, i64)
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c31 = arith.constant 31 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %blockWidth = func.call @strict_device_block_size() : () -> i32
    %lane = arith.andi %thread, %c31 : i32
    %warp = arith.shrui %thread, %c5 : i32
    %blockWarps = arith.shrui %blockWidth, %c5 : i32
    %thread64 = arith.extui %thread : i32 to i64
    %lane64 = arith.extui %lane : i32 to i64
    %warp64 = arith.extui %warp : i32 to i64
    %warps = arith.extui %blockWarps : i32 to i64
    %token = arith.extui %block : i32 to i64
    %start = arith.muli %token, %width : i64
    %seven = arith.constant 7 : i64
    %twentyFour = arith.constant 24 : i64
    %laneLow = arith.andi %lane64, %seven : i64
    %laneHigh = arith.andi %lane64, %twentyFour : i64
    %laneHighTwice = arith.addi %laneHigh, %laneHigh : i64
    %laneFirst = arith.addi %laneLow, %laneHighTwice : i64
    %groups = arith.shrui %width, %eight : i64
    %firstOffset = arith.shli %warp64, %eight : i64
    %firstStart = arith.addi %start, %firstOffset : i64
    %firstAt = arith.addi %firstStart, %laneFirst : i64
    %first = func.call @strict_device_hadamard_group(%source, %format, %firstStart, %lane) : (!llvm.ptr, i64, i64, i32) -> vector<8xf32>
    %firstLargest = func.call @strict_device_hadamard_magnitude(%first) : (vector<8xf32>) -> f32
    %secondGroup = arith.addi %warp64, %warps : i64
    %hasSecond = arith.cmpi ult, %secondGroup, %groups : i64
    %secondOffset = arith.shli %secondGroup, %eight : i64
    %secondStart = arith.addi %start, %secondOffset : i64
    %secondAt = arith.addi %secondStart, %laneFirst : i64
    cf.cond_br %hasSecond, ^second, ^largest(%first, %firstLargest : vector<8xf32>, f32)
  ^second:
    %second = func.call @strict_device_hadamard_group(%source, %format, %secondStart, %lane) : (!llvm.ptr, i64, i64, i32) -> vector<8xf32>
    %secondLargest = func.call @strict_device_hadamard_magnitude(%second) : (vector<8xf32>) -> f32
    %bothLargest = arith.maxnumf %firstLargest, %secondLargest : f32
    cf.br ^largest(%second, %bothLargest : vector<8xf32>, f32)
  ^largest(%other: vector<8xf32>, %largest: f32):
    %warpLargest = func.call @strict_device_largest(%largest) : (f32) -> f32
    %shared = llvm.mlir.addressof @strict_device_hadamard_largest : !llvm.ptr<3>
    %isFirst = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %isFirst, ^share, ^shared
  ^share:
    %sharedSlot = llvm.getelementptr %shared[%warp64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %warpLargest, %sharedSlot : f32, !llvm.ptr<3>
    cf.br ^shared
  ^shared:
    func.call @strict_device_barrier() : () -> ()
    cf.br ^gather(%none, %zero : i64, f32)
  ^gather(%index: i64, %gathered: f32):
    %moreOthers = arith.cmpi slt, %index, %warps : i64
    cf.cond_br %moreOthers, ^gatherBody, ^scaled
  ^gatherBody:
    %otherSlot = llvm.getelementptr %shared[%index] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %otherLargest = llvm.load %otherSlot : !llvm.ptr<3> -> f32
    %gatheredNext = arith.maxnumf %gathered, %otherLargest : f32
    %indexNext = arith.addi %index, %one : i64
    cf.br ^gather(%indexNext, %gatheredNext : i64, f32)
  ^scaled:
    %largestScale = arith.divf %gathered, %most : f32
    %scale = func.call @strict_device_code_scale(%largestScale, %residual) : (f32, i1) -> f32
    %isEmpty = arith.cmpf oeq, %gathered, %zero : f32
    %inverse = arith.divf %most, %gathered : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    func.call @strict_device_hadamard_codes(%target, %firstAt, %first, %factor, %residual) : (!llvm.ptr, i64, vector<8xf32>, f32, i1) -> ()
    cf.cond_br %hasSecond, ^secondCodes, ^scales
  ^secondCodes:
    func.call @strict_device_hadamard_codes(%target, %secondAt, %other, %factor, %residual) : (!llvm.ptr, i64, vector<8xf32>, f32, i1) -> ()
    cf.br ^scales
  ^scales:
    %parts = arith.shrui %width, %five : i64
    %isScaling = arith.cmpi slt, %thread64, %parts : i64
    cf.cond_br %isScaling, ^scale, ^exit
  ^scale:
    %scales = llvm.getelementptr %target[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %tokenParts = arith.shrui %start, %five : i64
    %part = arith.addi %tokenParts, %thread64 : i64
    %scaleSlot = llvm.getelementptr %scales[%part] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %scaleSlot : f32, !llvm.ptr
    cf.br ^exit
  ^exit:
    return
  }
  llvm.func @strict_device_hadamard_tokens(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 1024, 1, 1>} {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^other
  ^brain:
    func.call @strict_device_hadamard_tokens_run(%source, %count, %width, %target, %brainBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^other:
    func.call @strict_device_hadamard_tokens_run(%source, %count, %width, %target, %bits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  }
