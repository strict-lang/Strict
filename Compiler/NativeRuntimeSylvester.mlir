// GPU kernel of `Sylvester.Rotated` (NeuralNetworks), only in programs that call strict_gpu_sylvester
  // the 1024 numbers of a block of strict_device_sylvester_run
  llvm.mlir.global internal @strict_device_sylvester_numbers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x f32>
  // a block of 256 threads per 1024 numbers (groups of size, a power of two from 32 to 1024, each thread 4 numbers 256
  // apart): the flipped numbers into shared memory, a radix-2 butterfly per bit of size between block barriers (2 per
  // thread), then scaled by 1 / sqrt(size), written as doubles when wide (to the host), else in the format results
  func.func private @strict_device_sylvester_run(%count: i64, %result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %flips: !llvm.ptr, %flipsCount: i64, %flipsBits: i64, %size: f64, %results: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    %c10 = arith.constant 10 : i64
    %c256 = arith.constant 256 : i64
    %c512 = arith.constant 512 : i64
    %unit = arith.constant 1.0 : f32
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %start = arith.shli %block, %c10 : i64
    %groupSize = arith.fptosi %size : f64 to i64
    %flipsCount32 = arith.trunci %flipsCount : i64 to i32
    %shared = llvm.mlir.addressof @strict_device_sylvester_numbers : !llvm.ptr<3>
    cf.br ^load(%c0 : i64)
  ^load(%loadPart: i64):
    %loading = arith.cmpi slt, %loadPart, %c4 : i64
    cf.cond_br %loading, ^loadOne, ^loaded
  ^loadOne:
    %loadOffset = arith.muli %loadPart, %c256 : i64
    %loadIndex = arith.addi %loadOffset, %thread : i64
    %at = arith.addi %start, %loadIndex : i64
    %loadInside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %loadInside, ^loadValue, ^loadNext
  ^loadValue:
    %value = func.call @strict_device_elements(%values, %valuesBits, %at, %valuesCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %at32 = arith.trunci %at : i64 to i32
    %flipAt32 = arith.remui %at32, %flipsCount32 : i32
    %flipAt = arith.extui %flipAt32 : i32 to i64
    %flip = func.call @strict_device_elements(%flips, %flipsBits, %flipAt, %flipsCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %flipped = arith.mulf %value, %flip : f32
    %storeSlot = llvm.getelementptr %shared[%loadIndex] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %flipped, %storeSlot : f32, !llvm.ptr<3>
    cf.br ^loadNext
  ^loadNext:
    %loadFollowing = arith.addi %loadPart, %c1 : i64
    cf.br ^load(%loadFollowing : i64)
  ^loaded:
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%c1, %c0 : i64, i64)
  ^stage(%step: i64, %shift: i64):
    %staging = arith.cmpi slt, %step, %groupSize : i64
    cf.cond_br %staging, ^pairs(%thread : i64), ^scale
  ^pairs(%pair: i64):
    %pairing = arith.cmpi slt, %pair, %c512 : i64
    cf.cond_br %pairing, ^butterfly, ^staged
  ^butterfly:
    %outer = arith.shrui %pair, %shift : i64
    %span = arith.addi %shift, %c1 : i64
    %outerStart = arith.shli %outer, %span : i64
    %stepMask = arith.subi %step, %c1 : i64
    %inner = arith.andi %pair, %stepMask : i64
    %low = arith.addi %outerStart, %inner : i64
    %high = arith.addi %low, %step : i64
    %lowSlot = llvm.getelementptr %shared[%low] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %highSlot = llvm.getelementptr %shared[%high] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %lowValue = llvm.load %lowSlot : !llvm.ptr<3> -> f32
    %highValue = llvm.load %highSlot : !llvm.ptr<3> -> f32
    %sum = arith.addf %lowValue, %highValue : f32
    %difference = arith.subf %lowValue, %highValue : f32
    llvm.store %sum, %lowSlot : f32, !llvm.ptr<3>
    llvm.store %difference, %highSlot : f32, !llvm.ptr<3>
    %pairFollowing = arith.addi %pair, %c256 : i64
    cf.br ^pairs(%pairFollowing : i64)
  ^staged:
    func.call @strict_device_barrier() : () -> ()
    %nextStep = arith.shli %step, %c1 : i64
    %nextShift = arith.addi %shift, %c1 : i64
    cf.br ^stage(%nextStep, %nextShift : i64, i64)
  ^scale:
    %sizeSingle = arith.truncf %size : f64 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeSingle) : (f32) -> f32
    %factor = arith.divf %unit, %root : f32
    cf.br ^write(%c0 : i64)
  ^write(%writePart: i64):
    %writing = arith.cmpi slt, %writePart, %c4 : i64
    cf.cond_br %writing, ^writeOne, ^done
  ^writeOne:
    %writeOffset = arith.muli %writePart, %c256 : i64
    %writeIndex = arith.addi %writeOffset, %thread : i64
    %target = arith.addi %start, %writeIndex : i64
    %writeInside = arith.cmpi slt, %target, %count : i64
    cf.cond_br %writeInside, ^writeValue, ^writeNext
  ^writeValue:
    %readSlot = llvm.getelementptr %shared[%writeIndex] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %rotated = llvm.load %readSlot : !llvm.ptr<3> -> f32
    %scaled = arith.mulf %rotated, %factor : f32
    cf.cond_br %wide, ^double, ^single
  ^double:
    %double = arith.extf %scaled : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%target] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    cf.br ^writeNext
  ^single:
    func.call @strict_device_stored(%result, %target, %scaled, %results) : (!llvm.ptr, i64, f32, i64) -> ()
    cf.br ^writeNext
  ^writeNext:
    %writeFollowing = arith.addi %writePart, %c1 : i64
    cf.br ^write(%writeFollowing : i64)
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
