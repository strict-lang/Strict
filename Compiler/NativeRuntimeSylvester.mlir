// GPU kernel of `Sylvester.Rotated` (NeuralNetworks), only in programs that call strict_gpu_sylvester
  // the 1024 numbers of a block of strict_device_sylvester_run, 16 unused after every 128 (no bank conflicts)
  llvm.mlir.global internal @strict_device_sylvester_numbers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1152 x f32>
  // a block of 256 threads per 1024 numbers (groups of size, a power of two from 32 to 1024), each thread holds 4 of
  // them 256 apart: the butterflies of bits 8 and 9 in registers and of bits 0 to 4 across the warp, one exchange
  // through shared memory, bits 5 and 6 in registers and 7 across the warp, scaled by 1 / sqrt(size) and written as
  // doubles when wide (to the host), else in the format results
  func.func private @strict_device_sylvester_run(%count: i64, %result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %flips: !llvm.ptr, %flipsCount: i64, %flipsBits: i64, %size: f64, %results: i64) {
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    %c5 = arith.constant 5 : i64
    %c7 = arith.constant 7 : i64
    %c8 = arith.constant 8 : i64
    %c10 = arith.constant 10 : i64
    %c15 = arith.constant 15 : i64
    %c31 = arith.constant 31 : i64
    %c32 = arith.constant 32 : i64
    %c64 = arith.constant 64 : i64
    %c96 = arith.constant 96 : i64
    %c128 = arith.constant 128 : i64
    %c256 = arith.constant 256 : i64
    %c512 = arith.constant 512 : i64
    %c768 = arith.constant 768 : i64
    %first32 = arith.constant 1 : i32
    %warp32 = arith.constant 32 : i32
    %high32 = arith.constant 16 : i32
    %always = arith.constant true
    %unit = arith.constant 1.0 : f32
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %start = arith.shli %block, %c10 : i64
    %groupSize = arith.fptosi %size : f64 to i64
    %lane = arith.andi %thread, %c31 : i64
    %lane32 = arith.trunci %lane : i64 to i32
    %warp = arith.shrui %thread, %c5 : i64
    %at0 = arith.addi %start, %thread : i64
    %at1 = arith.addi %at0, %c256 : i64
    %at2 = arith.addi %at0, %c512 : i64
    %at3 = arith.addi %at0, %c768 : i64
    %a0 = func.call @strict_device_sylvester_flipped(%values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %count, %at0) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    %a1 = func.call @strict_device_sylvester_flipped(%values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %count, %at1) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    %a2 = func.call @strict_device_sylvester_flipped(%values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %count, %at2) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    %a3 = func.call @strict_device_sylvester_flipped(%values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %count, %at3) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    %has256 = arith.cmpi slt, %c256, %groupSize : i64
    %b0, %b1 = func.call @strict_device_sylvester_pair(%a0, %a1, %has256) : (f32, f32, i1) -> (f32, f32)
    %b2, %b3 = func.call @strict_device_sylvester_pair(%a2, %a3, %has256) : (f32, f32, i1) -> (f32, f32)
    %has512 = arith.cmpi slt, %c512, %groupSize : i64
    %d0, %d2 = func.call @strict_device_sylvester_pair(%b0, %b2, %has512) : (f32, f32, i1) -> (f32, f32)
    %d1, %d3 = func.call @strict_device_sylvester_pair(%b1, %b3, %has512) : (f32, f32, i1) -> (f32, f32)
    cf.br ^lanes(%first32, %d0, %d1, %d2, %d3 : i32, f32, f32, f32, f32)
  ^lanes(%offset: i32, %l0: f32, %l1: f32, %l2: f32, %l3: f32):
    %laning = arith.cmpi ult, %offset, %warp32 : i32
    cf.cond_br %laning, ^lane, ^exchange
  ^lane:
    %m0 = func.call @strict_device_sylvester_swap(%l0, %lane32, %offset, %always) : (f32, i32, i32, i1) -> f32
    %m1 = func.call @strict_device_sylvester_swap(%l1, %lane32, %offset, %always) : (f32, i32, i32, i1) -> f32
    %m2 = func.call @strict_device_sylvester_swap(%l2, %lane32, %offset, %always) : (f32, i32, i32, i1) -> f32
    %m3 = func.call @strict_device_sylvester_swap(%l3, %lane32, %offset, %always) : (f32, i32, i32, i1) -> f32
    %nextOffset = arith.shli %offset, %first32 : i32
    cf.br ^lanes(%nextOffset, %m0, %m1, %m2, %m3 : i32, f32, f32, f32, f32)
  ^exchange:
    %shared = llvm.mlir.addressof @strict_device_sylvester_numbers : !llvm.ptr<3>
    %p1 = arith.addi %thread, %c256 : i64
    %p2 = arith.addi %thread, %c512 : i64
    %p3 = arith.addi %thread, %c768 : i64
    func.call @strict_device_sylvester_store(%shared, %thread, %l0) : (!llvm.ptr<3>, i64, f32) -> ()
    func.call @strict_device_sylvester_store(%shared, %p1, %l1) : (!llvm.ptr<3>, i64, f32) -> ()
    func.call @strict_device_sylvester_store(%shared, %p2, %l2) : (!llvm.ptr<3>, i64, f32) -> ()
    func.call @strict_device_sylvester_store(%shared, %p3, %l3) : (!llvm.ptr<3>, i64, f32) -> ()
    func.call @strict_device_barrier() : () -> ()
    %laneLow = arith.andi %lane, %c15 : i64
    %laneHigh = arith.shrui %lane, %c4 : i64
    %laneHighOffset = arith.shli %laneHigh, %c7 : i64
    %warpLow = arith.andi %warp, %c1 : i64
    %warpLowOffset = arith.shli %warpLow, %c4 : i64
    %warpHigh = arith.shrui %warp, %c1 : i64
    %warpHighOffset = arith.shli %warpHigh, %c8 : i64
    %lanesOffset = arith.addi %laneLow, %laneHighOffset : i64
    %warpsOffset = arith.addi %warpLowOffset, %warpHighOffset : i64
    %q0 = arith.addi %lanesOffset, %warpsOffset : i64
    %q1 = arith.addi %q0, %c32 : i64
    %q2 = arith.addi %q0, %c64 : i64
    %q3 = arith.addi %q0, %c96 : i64
    %u0 = func.call @strict_device_sylvester_load(%shared, %q0) : (!llvm.ptr<3>, i64) -> f32
    %u1 = func.call @strict_device_sylvester_load(%shared, %q1) : (!llvm.ptr<3>, i64) -> f32
    %u2 = func.call @strict_device_sylvester_load(%shared, %q2) : (!llvm.ptr<3>, i64) -> f32
    %u3 = func.call @strict_device_sylvester_load(%shared, %q3) : (!llvm.ptr<3>, i64) -> f32
    %has32 = arith.cmpi slt, %c32, %groupSize : i64
    %v0, %v1 = func.call @strict_device_sylvester_pair(%u0, %u1, %has32) : (f32, f32, i1) -> (f32, f32)
    %v2, %v3 = func.call @strict_device_sylvester_pair(%u2, %u3, %has32) : (f32, f32, i1) -> (f32, f32)
    %has64 = arith.cmpi slt, %c64, %groupSize : i64
    %w0, %w2 = func.call @strict_device_sylvester_pair(%v0, %v2, %has64) : (f32, f32, i1) -> (f32, f32)
    %w1, %w3 = func.call @strict_device_sylvester_pair(%v1, %v3, %has64) : (f32, f32, i1) -> (f32, f32)
    %has128 = arith.cmpi slt, %c128, %groupSize : i64
    %x0 = func.call @strict_device_sylvester_swap(%w0, %lane32, %high32, %has128) : (f32, i32, i32, i1) -> f32
    %x1 = func.call @strict_device_sylvester_swap(%w1, %lane32, %high32, %has128) : (f32, i32, i32, i1) -> f32
    %x2 = func.call @strict_device_sylvester_swap(%w2, %lane32, %high32, %has128) : (f32, i32, i32, i1) -> f32
    %x3 = func.call @strict_device_sylvester_swap(%w3, %lane32, %high32, %has128) : (f32, i32, i32, i1) -> f32
    %sizeSingle = arith.truncf %size : f64 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeSingle) : (f32) -> f32
    %factor = arith.divf %unit, %root : f32
    %target0 = arith.addi %start, %q0 : i64
    %target1 = arith.addi %start, %q1 : i64
    %target2 = arith.addi %start, %q2 : i64
    %target3 = arith.addi %start, %q3 : i64
    func.call @strict_device_sylvester_write(%result, %target0, %x0, %factor, %count, %wide, %results) : (!llvm.ptr, i64, f32, f32, i64, i1, i64) -> ()
    func.call @strict_device_sylvester_write(%result, %target1, %x1, %factor, %count, %wide, %results) : (!llvm.ptr, i64, f32, f32, i64, i1, i64) -> ()
    func.call @strict_device_sylvester_write(%result, %target2, %x2, %factor, %count, %wide, %results) : (!llvm.ptr, i64, f32, f32, i64, i1, i64) -> ()
    func.call @strict_device_sylvester_write(%result, %target3, %x3, %factor, %count, %wide, %results) : (!llvm.ptr, i64, f32, f32, i64, i1, i64) -> ()
    return
  }
  // the flipped number at, 0 after count
  func.func private @strict_device_sylvester_flipped(%values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %flips: !llvm.ptr, %flipsCount: i64, %flipsBits: i64, %count: i64, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0.0 : f32
    %inside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %inside, ^load, ^outside
  ^load:
    %value = func.call @strict_device_elements(%values, %valuesBits, %at, %valuesCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %at32 = arith.trunci %at : i64 to i32
    %flipsCount32 = arith.trunci %flipsCount : i64 to i32
    %flipAt32 = arith.remui %at32, %flipsCount32 : i32
    %flipAt = arith.extui %flipAt32 : i32 to i64
    %flip = func.call @strict_device_elements(%flips, %flipsBits, %flipAt, %flipsCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %flipped = arith.mulf %value, %flip : f32
    return %flipped : f32
  ^outside:
    return %zero : f32
  }
  // the sum and difference of a butterfly when active, else both unchanged
  func.func private @strict_device_sylvester_pair(%low: f32, %high: f32, %active: i1) -> (f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %sum = arith.addf %low, %high : f32
    %difference = arith.subf %low, %high : f32
    %first = arith.select %active, %sum, %low : f32
    %second = arith.select %active, %difference, %high : f32
    return %first, %second : f32, f32
  }
  // the butterfly with the lane differing in the bit offset when active (the sum in the lower lane, the difference in
  // the higher), every lane of the warp calls it
  func.func private @strict_device_sylvester_swap(%value: f32, %lane: i32, %offset: i32, %active: i1) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0 : i32
    %bits = arith.bitcast %value : f32 to i32
    %otherBits = func.call @strict_device_swap(%bits, %offset) : (i32, i32) -> i32
    %other = arith.bitcast %otherBits : i32 to f32
    %bit = arith.andi %lane, %offset : i32
    %isHigh = arith.cmpi ne, %bit, %zero : i32
    %sum = arith.addf %value, %other : f32
    %difference = arith.subf %other, %value : f32
    %paired = arith.select %isHigh, %difference, %sum : f32
    %chosen = arith.select %active, %paired, %value : f32
    return %chosen : f32
  }
  // the shared slot of position, 16 numbers skipped after every 128
  func.func private @strict_device_sylvester_slot(%shared: !llvm.ptr<3>, %position: i64) -> !llvm.ptr<3> attributes {llvm.passthrough = ["alwaysinline"]} {
    %seven = arith.constant 7 : i64
    %four = arith.constant 4 : i64
    %row = arith.shrui %position, %seven : i64
    %skipped = arith.shli %row, %four : i64
    %padded = arith.addi %position, %skipped : i64
    %slot = llvm.getelementptr %shared[%padded] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    return %slot : !llvm.ptr<3>
  }
  func.func private @strict_device_sylvester_store(%shared: !llvm.ptr<3>, %position: i64, %value: f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = func.call @strict_device_sylvester_slot(%shared, %position) : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>
    llvm.store %value, %slot : f32, !llvm.ptr<3>
    return
  }
  func.func private @strict_device_sylvester_load(%shared: !llvm.ptr<3>, %position: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = func.call @strict_device_sylvester_slot(%shared, %position) : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>
    %value = llvm.load %slot : !llvm.ptr<3> -> f32
    return %value : f32
  }
  // the scaled number at target unless it is after count
  func.func private @strict_device_sylvester_write(%result: !llvm.ptr, %target: i64, %value: f32, %factor: f32, %count: i64, %wide: i1, %results: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %inside = arith.cmpi slt, %target, %count : i64
    cf.cond_br %inside, ^write, ^done
  ^write:
    %scaled = arith.mulf %value, %factor : f32
    func.call @strict_device_result(%result, %target, %scaled, %wide, %results) : (!llvm.ptr, i64, f32, i1, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_sylvester(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %flips: !llvm.ptr, %flipsCount: i64, %flipsBits: i64, %size: f64, %results: i64, %sizeWhole: i64, %sizeMagic: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_sylvester_run(%count, %result, %wide, %values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %size, %results) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_sylvesters(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %flips: !llvm.ptr, %flipsCount: i64, %flipsBits: i64, %size: f64, %results: i64, %sizeWhole: i64, %sizeMagic: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_sylvester_run(%count, %result, %wide, %values, %valuesCount, %valuesBits, %flips, %flipsCount, %flipsBits, %size, %results) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, i64) -> ()
    llvm.return
  }
