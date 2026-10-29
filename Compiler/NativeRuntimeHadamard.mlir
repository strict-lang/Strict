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
  // the quad number of a lane whose digit is at shift of its lane number, the lanes xor 1, 2 and 3 times 1 << shift hold
  // the other numbers of the quad (steps 1 and 4 of strict_device_hadamard_tokens)
  func.func private @strict_device_hadamard_lanes(%value: f32, %lane: i32, %shift: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %three = arith.constant 3 : i64
    %oneApart = arith.shli %c1, %shift : i32
    %twoApart = arith.shli %c2, %shift : i32
    %threeApart = arith.shli %c3, %shift : i32
    %v1 = func.call @strict_device_shuffles(%value, %oneApart) : (f32, i32) -> f32
    %v2 = func.call @strict_device_shuffles(%value, %twoApart) : (f32, i32) -> f32
    %v3 = func.call @strict_device_shuffles(%value, %threeApart) : (f32, i32) -> f32
    %shifted = arith.shrui %lane, %shift : i32
    %digit = arith.andi %shifted, %c3 : i32
    %digit1 = arith.xori %digit, %c1 : i32
    %digit2 = arith.xori %digit, %c2 : i32
    %digit3 = arith.xori %digit, %c3 : i32
    %x0 = func.call @strict_device_hadamard_pick(%digit, %value, %v1, %v2, %v3) : (i32, f32, f32, f32, f32) -> f32
    %x1 = func.call @strict_device_hadamard_pick(%digit1, %value, %v1, %v2, %v3) : (i32, f32, f32, f32, f32) -> f32
    %x2 = func.call @strict_device_hadamard_pick(%digit2, %value, %v1, %v2, %v3) : (i32, f32, f32, f32, f32) -> f32
    %x3 = func.call @strict_device_hadamard_pick(%digit3, %value, %v1, %v2, %v3) : (i32, f32, f32, f32, f32) -> f32
    %digitLong = arith.extui %digit : i32 to i64
    %partner = arith.subi %three, %digitLong : i64
    %sum = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %partner) : (f32, f32, f32, f32, i64) -> f32
    return %sum : f32
  }
  func.func private @strict_device_hadamard_pick(%index: i32, %v0: f32, %v1: f32, %v2: f32, %v3: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %odd = arith.andi %index, %c1 : i32
    %isEven = arith.cmpi ne, %odd, %c1 : i32
    %isHigh = arith.cmpi uge, %index, %c2 : i32
    %low = arith.select %isEven, %v0, %v1 : f32
    %high = arith.select %isEven, %v2, %v3 : f32
    %picked = arith.select %isHigh, %high, %low : f32
    return %picked : f32
  }
  // the quad number at step 16 of the even or odd j (isOdd 0 or 1) of a lane from its even and odd numbers and those of
  // the lane 16 apart (high is bit 4 of the lane)
  func.func private @strict_device_hadamard_pair(%even: f32, %odd: f32, %otherEven: f32, %otherOdd: f32, %high: i1, %isOdd: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %x0 = arith.select %high, %otherEven, %even : f32
    %x1 = arith.select %high, %even, %otherEven : f32
    %x2 = arith.select %high, %otherOdd, %odd : f32
    %x3 = arith.select %high, %odd, %otherOdd : f32
    %highLong = arith.extui %high : i1 to i64
    %oddTwice = arith.shli %isOdd, %one : i64
    %digit = arith.addi %highLong, %oddTwice : i64
    %partner = arith.subi %three, %digit : i64
    %sum = func.call @strict_device_hadamard_sum(%x0, %x1, %x2, %x3, %partner) : (f32, f32, f32, f32, i64) -> f32
    return %sum : f32
  }
  // the largest magnitude of each warp of a block of strict_device_hadamard_tokens
  llvm.mlir.global internal @strict_device_hadamard_largest() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<32 x f32>
  // the 8 numbers of a lane of strict_device_hadamard_tokens rotated through the 4 radix-4 butterflies of its group of
  // 256 (lane holds number at + 32 j, j 0 to 7), steps 1 and 4 over lanes, 16 over lanes and j, 64 over j
  func.func private @strict_device_hadamard_group(%source: !llvm.ptr, %bits: i64, %at: i64, %lane: i32) -> vector<8xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %r0 = func.call @strict_device_floats(%source, %bits, %at) : (!llvm.ptr, i64, i64) -> f32
    %apart1 = arith.constant 32 : i64
    %at1 = arith.addi %at, %apart1 : i64
    %r1 = func.call @strict_device_floats(%source, %bits, %at1) : (!llvm.ptr, i64, i64) -> f32
    %apart2 = arith.constant 64 : i64
    %at2 = arith.addi %at, %apart2 : i64
    %r2 = func.call @strict_device_floats(%source, %bits, %at2) : (!llvm.ptr, i64, i64) -> f32
    %apart3 = arith.constant 96 : i64
    %at3 = arith.addi %at, %apart3 : i64
    %r3 = func.call @strict_device_floats(%source, %bits, %at3) : (!llvm.ptr, i64, i64) -> f32
    %apart4 = arith.constant 128 : i64
    %at4 = arith.addi %at, %apart4 : i64
    %r4 = func.call @strict_device_floats(%source, %bits, %at4) : (!llvm.ptr, i64, i64) -> f32
    %apart5 = arith.constant 160 : i64
    %at5 = arith.addi %at, %apart5 : i64
    %r5 = func.call @strict_device_floats(%source, %bits, %at5) : (!llvm.ptr, i64, i64) -> f32
    %apart6 = arith.constant 192 : i64
    %at6 = arith.addi %at, %apart6 : i64
    %r6 = func.call @strict_device_floats(%source, %bits, %at6) : (!llvm.ptr, i64, i64) -> f32
    %apart7 = arith.constant 224 : i64
    %at7 = arith.addi %at, %apart7 : i64
    %r7 = func.call @strict_device_floats(%source, %bits, %at7) : (!llvm.ptr, i64, i64) -> f32
    %a0 = func.call @strict_device_hadamard_lanes(%r0, %lane, %c0) : (f32, i32, i32) -> f32
    %a1 = func.call @strict_device_hadamard_lanes(%r1, %lane, %c0) : (f32, i32, i32) -> f32
    %a2 = func.call @strict_device_hadamard_lanes(%r2, %lane, %c0) : (f32, i32, i32) -> f32
    %a3 = func.call @strict_device_hadamard_lanes(%r3, %lane, %c0) : (f32, i32, i32) -> f32
    %a4 = func.call @strict_device_hadamard_lanes(%r4, %lane, %c0) : (f32, i32, i32) -> f32
    %a5 = func.call @strict_device_hadamard_lanes(%r5, %lane, %c0) : (f32, i32, i32) -> f32
    %a6 = func.call @strict_device_hadamard_lanes(%r6, %lane, %c0) : (f32, i32, i32) -> f32
    %a7 = func.call @strict_device_hadamard_lanes(%r7, %lane, %c0) : (f32, i32, i32) -> f32
    %b0 = func.call @strict_device_hadamard_lanes(%a0, %lane, %c2) : (f32, i32, i32) -> f32
    %b1 = func.call @strict_device_hadamard_lanes(%a1, %lane, %c2) : (f32, i32, i32) -> f32
    %b2 = func.call @strict_device_hadamard_lanes(%a2, %lane, %c2) : (f32, i32, i32) -> f32
    %b3 = func.call @strict_device_hadamard_lanes(%a3, %lane, %c2) : (f32, i32, i32) -> f32
    %b4 = func.call @strict_device_hadamard_lanes(%a4, %lane, %c2) : (f32, i32, i32) -> f32
    %b5 = func.call @strict_device_hadamard_lanes(%a5, %lane, %c2) : (f32, i32, i32) -> f32
    %b6 = func.call @strict_device_hadamard_lanes(%a6, %lane, %c2) : (f32, i32, i32) -> f32
    %b7 = func.call @strict_device_hadamard_lanes(%a7, %lane, %c2) : (f32, i32, i32) -> f32
    %laneHigh = arith.andi %lane, %c16 : i32
    %high = arith.cmpi ne, %laneHigh, %c0 : i32
    %o0 = func.call @strict_device_shuffles(%b0, %c16) : (f32, i32) -> f32
    %o1 = func.call @strict_device_shuffles(%b1, %c16) : (f32, i32) -> f32
    %o2 = func.call @strict_device_shuffles(%b2, %c16) : (f32, i32) -> f32
    %o3 = func.call @strict_device_shuffles(%b3, %c16) : (f32, i32) -> f32
    %o4 = func.call @strict_device_shuffles(%b4, %c16) : (f32, i32) -> f32
    %o5 = func.call @strict_device_shuffles(%b5, %c16) : (f32, i32) -> f32
    %o6 = func.call @strict_device_shuffles(%b6, %c16) : (f32, i32) -> f32
    %o7 = func.call @strict_device_shuffles(%b7, %c16) : (f32, i32) -> f32
    %e0 = func.call @strict_device_hadamard_pair(%b0, %b1, %o0, %o1, %high, %none) : (f32, f32, f32, f32, i1, i64) -> f32
    %e1 = func.call @strict_device_hadamard_pair(%b0, %b1, %o0, %o1, %high, %one) : (f32, f32, f32, f32, i1, i64) -> f32
    %e2 = func.call @strict_device_hadamard_pair(%b2, %b3, %o2, %o3, %high, %none) : (f32, f32, f32, f32, i1, i64) -> f32
    %e3 = func.call @strict_device_hadamard_pair(%b2, %b3, %o2, %o3, %high, %one) : (f32, f32, f32, f32, i1, i64) -> f32
    %e4 = func.call @strict_device_hadamard_pair(%b4, %b5, %o4, %o5, %high, %none) : (f32, f32, f32, f32, i1, i64) -> f32
    %e5 = func.call @strict_device_hadamard_pair(%b4, %b5, %o4, %o5, %high, %one) : (f32, f32, f32, f32, i1, i64) -> f32
    %e6 = func.call @strict_device_hadamard_pair(%b6, %b7, %o6, %o7, %high, %none) : (f32, f32, f32, f32, i1, i64) -> f32
    %e7 = func.call @strict_device_hadamard_pair(%b6, %b7, %o6, %o7, %high, %one) : (f32, f32, f32, f32, i1, i64) -> f32
    %d0 = func.call @strict_device_hadamard_sum(%e0, %e2, %e4, %e6, %three) : (f32, f32, f32, f32, i64) -> f32
    %d1 = func.call @strict_device_hadamard_sum(%e1, %e3, %e5, %e7, %three) : (f32, f32, f32, f32, i64) -> f32
    %d2 = func.call @strict_device_hadamard_sum(%e0, %e2, %e4, %e6, %two) : (f32, f32, f32, f32, i64) -> f32
    %d3 = func.call @strict_device_hadamard_sum(%e1, %e3, %e5, %e7, %two) : (f32, f32, f32, f32, i64) -> f32
    %d4 = func.call @strict_device_hadamard_sum(%e0, %e2, %e4, %e6, %one) : (f32, f32, f32, f32, i64) -> f32
    %d5 = func.call @strict_device_hadamard_sum(%e1, %e3, %e5, %e7, %one) : (f32, f32, f32, f32, i64) -> f32
    %d6 = func.call @strict_device_hadamard_sum(%e0, %e2, %e4, %e6, %none) : (f32, f32, f32, f32, i64) -> f32
    %d7 = func.call @strict_device_hadamard_sum(%e1, %e3, %e5, %e7, %none) : (f32, f32, f32, f32, i64) -> f32
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %i4 = arith.constant 4 : i32
    %i5 = arith.constant 5 : i32
    %i6 = arith.constant 6 : i32
    %i7 = arith.constant 7 : i32
    %vz = llvm.mlir.undef : vector<8xf32>
    %v0 = llvm.insertelement %d0, %vz[%i0 : i32] : vector<8xf32>
    %v1 = llvm.insertelement %d1, %v0[%i1 : i32] : vector<8xf32>
    %v2 = llvm.insertelement %d2, %v1[%i2 : i32] : vector<8xf32>
    %v3 = llvm.insertelement %d3, %v2[%i3 : i32] : vector<8xf32>
    %v4 = llvm.insertelement %d4, %v3[%i4 : i32] : vector<8xf32>
    %v5 = llvm.insertelement %d5, %v4[%i5 : i32] : vector<8xf32>
    %v6 = llvm.insertelement %d6, %v5[%i6 : i32] : vector<8xf32>
    %v7 = llvm.insertelement %d7, %v6[%i7 : i32] : vector<8xf32>
    return %v7 : vector<8xf32>
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
  // the 8 numbers of a lane times factor rounded to 8 bit numbers at at + 32 j of target
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
    %byteSlot1 = llvm.getelementptr %bytes[32] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %byteSlot3 = llvm.getelementptr %bytes[96] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %byteSlot5 = llvm.getelementptr %bytes[160] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %byteSlot7 = llvm.getelementptr %bytes[224] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %laneStart = arith.addi %start, %lane64 : i64
    %groups = arith.shrui %width, %eight : i64
    %firstOffset = arith.shli %warp64, %eight : i64
    %firstAt = arith.addi %laneStart, %firstOffset : i64
    %first = func.call @strict_device_hadamard_group(%source, %format, %firstAt, %lane) : (!llvm.ptr, i64, i64, i32) -> vector<8xf32>
    %firstLargest = func.call @strict_device_hadamard_magnitude(%first) : (vector<8xf32>) -> f32
    %secondGroup = arith.addi %warp64, %warps : i64
    %hasSecond = arith.cmpi ult, %secondGroup, %groups : i64
    %secondOffset = arith.shli %secondGroup, %eight : i64
    %secondAt = arith.addi %laneStart, %secondOffset : i64
    cf.cond_br %hasSecond, ^second, ^largest(%first, %firstLargest : vector<8xf32>, f32)
  ^second:
    %second = func.call @strict_device_hadamard_group(%source, %format, %secondAt, %lane) : (!llvm.ptr, i64, i64, i32) -> vector<8xf32>
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
