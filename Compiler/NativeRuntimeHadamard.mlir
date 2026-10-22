  // the 256 numbers of the group of a block for strict_device_hadamard_run
  llvm.mlir.global internal @strict_device_hadamard_numbers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<256 x f32>
  // one radix-4 butterfly of the 256 numbers of a block at step (1, 4, 16, 64) for its thread: the sum of the 4 numbers
  // of its quad halved, the one at 3 - digit negated, like `Convrot.Butterfly`
  func.func private @strict_device_hadamard_stage(%value: f32, %thread: i64, %step: i64) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c3 = arith.constant 3 : i64
    %half = arith.constant 0.5 : f32
    %negativeHalf = arith.constant -0.5 : f32
    %shared = llvm.mlir.addressof @strict_device_hadamard_numbers : !llvm.ptr<3>
    %own = llvm.getelementptr %shared[%thread] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %value, %own : f32, !llvm.ptr<3>
    func.call @strict_device_barrier() : () -> ()
    %quotient = arith.divui %thread, %step : i64
    %digit = arith.andi %quotient, %c3 : i64
    %offset = arith.muli %digit, %step : i64
    %first = arith.subi %thread, %offset : i64
    %partner = arith.subi %c3, %digit : i64
    %sum = func.call @strict_device_hadamard_quad(%shared, %first, %step, %partner, %half, %negativeHalf) : (!llvm.ptr<3>, i64, i64, i64, f32, f32) -> f32
    func.call @strict_device_barrier() : () -> ()
    return %sum : f32
  }
  // ((x0 s0 + x1 s1) + x2 s2) + x3 s3 of the quad from first, s is -0.5 at partner and 0.5 elsewhere
  func.func private @strict_device_hadamard_quad(%shared: !llvm.ptr<3>, %first: i64, %step: i64, %partner: i64, %half: f32, %negativeHalf: f32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c3 = arith.constant 3 : i64
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
  // through the 4 radix-4 butterflies in shared memory, written as doubles when wide (to the host) or 32 bit floats
  func.func private @strict_device_hadamard_run(%count: i64, %result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64) {
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
    %singleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %fourth, %singleSlot : f32, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_hadamard(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_hadamard_run(%count, %result, %wide, %values, %valuesCount, %valuesBits) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_hadamards(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_hadamard_run(%count, %result, %wide, %values, %valuesCount, %valuesBits) : (i64, !llvm.ptr, i1, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
