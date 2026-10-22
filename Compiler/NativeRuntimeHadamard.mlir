  // the 256 numbers of the group of a block for strict_device_hadamard_run
  llvm.mlir.global internal @strict_device_hadamard_numbers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<256 x f32>
  // one radix-4 butterfly of the 256 numbers of a block at step (1, 4, 16, 64) for its thread: the sum of the 4 numbers
  // of its quad halved, the one at 3 - digit negated, like `Convrot.Butterfly`
  func.func private @strict_device_hadamard_stage(%value: f32, %thread: i64, %step: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_hadamard_quad(%shared: !llvm.ptr<3>, %first: i64, %step: i64, %partner: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_hadamard_sum(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %partner: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_hadamard_run(%count: i64, %result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %results: i64) {
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
    cf.cond_br %wide, ^double, ^single
  ^double:
    %double = arith.extf %fourth : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    cf.br ^done
  ^single:
    func.call @strict_device_stored(%result, %at, %fourth, %results) : (!llvm.ptr, i64, f32, i64) -> ()
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
  func.func private @strict_device_hadamard_lanes(%value: f32, %lane: i32, %shift: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_hadamard_pick(%index: i32, %v0: f32, %v1: f32, %v2: f32, %v3: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_hadamard_pair(%even: f32, %odd: f32, %otherEven: f32, %otherOdd: f32, %high: i1, %isOdd: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  // strict_device_quantize_tokens of `Convrot.Rotated` of 32 or 16 bit source numbers (bits) without writing the rotated
  // ones: a block of width / 8 threads per token (width a multiple of 256 up to 8192), a warp per group of 256 numbers
  // whose lane holds number lane + 32 j (j 0 to 7), so steps 1 and 4 go over lanes, 16 over lanes and j, 64 over j
  func.func private @strict_device_hadamard_tokens_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %c31 = arith.constant 31 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %groupSize = arith.constant 256 : i64
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %lane = arith.andi %thread, %c31 : i32
    %warp = arith.shrui %thread, %c5 : i32
    %thread64 = arith.extui %thread : i32 to i64
    %lane64 = arith.extui %lane : i32 to i64
    %warp64 = arith.extui %warp : i32 to i64
    %token = arith.extui %block : i32 to i64
    %start = arith.muli %token, %width : i64
    %groupOffset = arith.muli %warp64, %groupSize : i64
    %groupStart = arith.addi %start, %groupOffset : i64
    %at = arith.addi %groupStart, %lane64 : i64
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
    %m0 = llvm.call_intrinsic "llvm.fabs.f32"(%d0) : (f32) -> f32
    %m1 = llvm.call_intrinsic "llvm.fabs.f32"(%d1) : (f32) -> f32
    %m2 = llvm.call_intrinsic "llvm.fabs.f32"(%d2) : (f32) -> f32
    %m3 = llvm.call_intrinsic "llvm.fabs.f32"(%d3) : (f32) -> f32
    %m4 = llvm.call_intrinsic "llvm.fabs.f32"(%d4) : (f32) -> f32
    %m5 = llvm.call_intrinsic "llvm.fabs.f32"(%d5) : (f32) -> f32
    %m6 = llvm.call_intrinsic "llvm.fabs.f32"(%d6) : (f32) -> f32
    %m7 = llvm.call_intrinsic "llvm.fabs.f32"(%d7) : (f32) -> f32
    %largest01 = arith.maxnumf %m0, %m1 : f32
    %largest23 = arith.maxnumf %m2, %m3 : f32
    %largest45 = arith.maxnumf %m4, %m5 : f32
    %largest67 = arith.maxnumf %m6, %m7 : f32
    %largestLow = arith.maxnumf %largest01, %largest23 : f32
    %largestHigh = arith.maxnumf %largest45, %largest67 : f32
    %largest = arith.maxnumf %largestLow, %largestHigh : f32
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
    %warps = arith.divui %width, %groupSize : i64
    cf.br ^gather(%none, %zero : i64, f32)
  ^gather(%other: i64, %gathered: f32):
    %moreOthers = arith.cmpi slt, %other, %warps : i64
    cf.cond_br %moreOthers, ^gatherBody, ^scaled
  ^gatherBody:
    %otherSlot = llvm.getelementptr %shared[%other] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %otherLargest = llvm.load %otherSlot : !llvm.ptr<3> -> f32
    %gatheredNext = arith.maxnumf %gathered, %otherLargest : f32
    %otherNext = arith.addi %other, %one : i64
    cf.br ^gather(%otherNext, %gatheredNext : i64, f32)
  ^scaled:
    %scale = arith.divf %gathered, %most : f32
    %isEmpty = arith.cmpf oeq, %gathered, %zero : f32
    %inverse = arith.divf %most, %gathered : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    %bytes = llvm.getelementptr %target[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaled0 = arith.mulf %d0, %factor : f32
    %rounded0 = func.call @strict_device_rounded(%scaled0) : (f32) -> i32
    %byte0 = arith.trunci %rounded0 : i32 to i8
    %byteSlot0 = llvm.getelementptr %bytes[0] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte0, %byteSlot0 : i8, !llvm.ptr
    %scaled1 = arith.mulf %d1, %factor : f32
    %rounded1 = func.call @strict_device_rounded(%scaled1) : (f32) -> i32
    %byte1 = arith.trunci %rounded1 : i32 to i8
    %byteSlot1 = llvm.getelementptr %bytes[32] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte1, %byteSlot1 : i8, !llvm.ptr
    %scaled2 = arith.mulf %d2, %factor : f32
    %rounded2 = func.call @strict_device_rounded(%scaled2) : (f32) -> i32
    %byte2 = arith.trunci %rounded2 : i32 to i8
    %byteSlot2 = llvm.getelementptr %bytes[64] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte2, %byteSlot2 : i8, !llvm.ptr
    %scaled3 = arith.mulf %d3, %factor : f32
    %rounded3 = func.call @strict_device_rounded(%scaled3) : (f32) -> i32
    %byte3 = arith.trunci %rounded3 : i32 to i8
    %byteSlot3 = llvm.getelementptr %bytes[96] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte3, %byteSlot3 : i8, !llvm.ptr
    %scaled4 = arith.mulf %d4, %factor : f32
    %rounded4 = func.call @strict_device_rounded(%scaled4) : (f32) -> i32
    %byte4 = arith.trunci %rounded4 : i32 to i8
    %byteSlot4 = llvm.getelementptr %bytes[128] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte4, %byteSlot4 : i8, !llvm.ptr
    %scaled5 = arith.mulf %d5, %factor : f32
    %rounded5 = func.call @strict_device_rounded(%scaled5) : (f32) -> i32
    %byte5 = arith.trunci %rounded5 : i32 to i8
    %byteSlot5 = llvm.getelementptr %bytes[160] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte5, %byteSlot5 : i8, !llvm.ptr
    %scaled6 = arith.mulf %d6, %factor : f32
    %rounded6 = func.call @strict_device_rounded(%scaled6) : (f32) -> i32
    %byte6 = arith.trunci %rounded6 : i32 to i8
    %byteSlot6 = llvm.getelementptr %bytes[192] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte6, %byteSlot6 : i8, !llvm.ptr
    %scaled7 = arith.mulf %d7, %factor : f32
    %rounded7 = func.call @strict_device_rounded(%scaled7) : (f32) -> i32
    %byte7 = arith.trunci %rounded7 : i32 to i8
    %byteSlot7 = llvm.getelementptr %bytes[224] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %byte7, %byteSlot7 : i8, !llvm.ptr
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
