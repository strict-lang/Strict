  // the input of `Convolved.Summed` normed with the scales and shifts of affine (swish when asked) or the weights as they
  // are (normed 0), 4 numbers per thread (one vector load of bfloat16 or 32 bit floats when aligned), stored as bfloat16
  // for strict_device_convolution, every channels numbers followed by zeros up to padded (a multiple of 32)
  func.func private @strict_device_convolution_normed_run(%count: i64, %result: !llvm.ptr, %results: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %channels: f64, %swish: f64, %normed: f64, %padded: f64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %brainBits = arith.constant 17 : i64
    %singleBits = arith.constant 32 : i64
    %doubleBits = arith.constant 64 : i64
    %perBlock = arith.constant 256 : i64
    %unit = arith.constant 1.0 : f64
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %blockStart = arith.muli %block, %perBlock : i64
    %quad = arith.addi %blockStart, %thread : i64
    %first = arith.muli %quad, %four : i64
    %inside = arith.cmpi slt, %first, %count : i64
    cf.cond_br %inside, ^read, ^done
  ^read:
    %end = arith.addi %first, %four : i64
    %whole = arith.cmpi sle, %end, %count : i64
    %channelCount = arith.fptosi %channels : f64 to i64
    %paddedCount = arith.fptosi %padded : f64 to i64
    %isPadded = arith.cmpi ne, %paddedCount, %channelCount : i64
    cf.cond_br %isPadded, ^padded, ^exact
  ^padded:
    %row = arith.divui %first, %paddedCount : i64
    %rowStart = arith.muli %row, %channelCount : i64
    %column = arith.remui %first, %paddedCount : i64
    %source = arith.addi %rowStart, %column : i64
    %rowEnd = arith.addi %rowStart, %channelCount : i64
    %paddedQuad = func.call @strict_device_convolution_four(%values, %valuesCount, %valuesBits, %source, %rowEnd) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    cf.br ^numbers(%paddedQuad : vector<4xf32>)
  ^exact:
    %address = llvm.ptrtoint %values : !llvm.ptr to i64
    %isBrain = arith.cmpi eq, %valuesBits, %brainBits : i64
    %isSingle = arith.cmpi eq, %valuesBits, %singleBits : i64
    %alignMask = arith.select %isBrain, %seven, %fifteen : i64
    %misaligned = arith.andi %address, %alignMask : i64
    %aligned = arith.cmpi eq, %misaligned, %none : i64
    %vectorFormat = arith.ori %isBrain, %isSingle : i1
    %vectorShape = arith.andi %whole, %aligned : i1
    %vectored = arith.andi %vectorFormat, %vectorShape : i1
    cf.cond_br %vectored, ^vector, ^singles
  ^vector:
    cf.cond_br %isBrain, ^brains, ^floats
  ^brains:
    %brainSlot = llvm.getelementptr %values[%first] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brainQuad = llvm.load %brainSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fromBrains = arith.extf %brainQuad : vector<4xbf16> to vector<4xf32>
    cf.br ^numbers(%fromBrains : vector<4xf32>)
  ^floats:
    %floatSlot = llvm.getelementptr %values[%first] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %floatQuad = llvm.load %floatSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    cf.br ^numbers(%floatQuad : vector<4xf32>)
  ^singles:
    %singleQuad = func.call @strict_device_convolution_four(%values, %valuesCount, %valuesBits, %first, %count) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    cf.br ^numbers(%singleQuad : vector<4xf32>)
  ^numbers(%numbers: vector<4xf32>):
    %isNormed = arith.cmpf oeq, %normed, %unit : f64
    cf.cond_br %isNormed, ^norm, ^store(%numbers : vector<4xf32>)
  ^norm:
    %channel = arith.remui %first, %paddedCount : i64
    %shiftAt = arith.addi %channel, %channelCount : i64
    %scales = func.call @strict_device_convolution_four(%affine, %affineCount, %affineBits, %channel, %channelCount) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %shiftEnd = arith.addi %channelCount, %channelCount : i64
    %shifts = func.call @strict_device_convolution_four(%affine, %affineCount, %affineBits, %shiftAt, %shiftEnd) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %scaled = arith.mulf %numbers, %scales : vector<4xf32>
    %shifted = arith.addf %scaled, %shifts : vector<4xf32>
    %isSwish = arith.cmpf oeq, %swish, %unit : f64
    cf.cond_br %isSwish, ^swished, ^store(%shifted : vector<4xf32>)
  ^swished:
    %units = arith.constant dense<1.0> : vector<4xf32>
    %factors = arith.constant dense<-1.44269504> : vector<4xf32>
    %exponents = arith.mulf %shifted, %factors : vector<4xf32>
    %powers = llvm.call_intrinsic "llvm.exp2.v4f32"(%exponents) : (vector<4xf32>) -> vector<4xf32>
    %below = arith.addf %units, %powers : vector<4xf32>
    %activated = arith.divf %shifted, %below fastmath<afn, arcp> : vector<4xf32>
    cf.br ^store(%activated : vector<4xf32>)
  ^store(%stored: vector<4xf32>):
    %brainResult = arith.cmpi eq, %results, %brainBits : i64
    %resultAddress = llvm.ptrtoint %result : !llvm.ptr to i64
    %resultRest = arith.andi %resultAddress, %seven : i64
    %resultAligned = arith.cmpi eq, %resultRest, %none : i64
    %resultFits = arith.andi %brainResult, %resultAligned : i1
    %storesQuad = arith.andi %resultFits, %whole : i1
    cf.cond_br %storesQuad, ^quadStore, ^oneByOne(%none : i64)
  ^quadStore:
    %brainsOut = arith.truncf %stored : vector<4xf32> to vector<4xbf16>
    %outSlot = llvm.getelementptr %result[%first] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %brainsOut, %outSlot {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    cf.br ^done
  ^oneByOne(%index: i64):
    %more = arith.cmpi slt, %index, %four : i64
    %at = arith.addi %first, %index : i64
    %atInside = arith.cmpi slt, %at, %count : i64
    %storing = arith.andi %more, %atInside : i1
    cf.cond_br %storing, ^storeOne, ^done
  ^storeOne:
    %slot = arith.trunci %index : i64 to i32
    %number = llvm.extractelement %stored[%slot : i32] : vector<4xf32>
    %wide = arith.cmpi eq, %results, %doubleBits : i64
    func.call @strict_device_result(%result, %at, %number, %wide, %results) : (!llvm.ptr, i64, f32, i1, i64) -> ()
    %next = arith.addi %index, %one : i64
    cf.br ^oneByOne(%next : i64)
  ^done:
    return
  }
  // 4 numbers of data from first in any format, zeros from end on
  func.func private @strict_device_convolution_four(%data: !llvm.ptr, %count: i64, %bits: i64, %first: i64, %end: i64) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %empty = llvm.mlir.undef : vector<4xf32>
    cf.br ^number(%none, %empty : i64, vector<4xf32>)
  ^number(%index: i64, %numbers: vector<4xf32>):
    %more = arith.cmpi slt, %index, %four : i64
    cf.cond_br %more, ^read, ^done
  ^read:
    %at = arith.addi %first, %index : i64
    %outside = arith.cmpi sge, %at, %end : i64
    %number = func.call @strict_device_products_number(%data, %count, %bits, %at, %outside) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %slot = arith.trunci %index : i64 to i32
    %withNumber = llvm.insertelement %number, %numbers[%slot : i32] : vector<4xf32>
    %next = arith.addi %index, %one : i64
    llvm.br ^number(%next, %withNumber : i64, vector<4xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %numbers : vector<4xf32>
  }
  llvm.func @strict_device_convolution_normed(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %channels: f64, %swish: f64, %normed: f64, %padded: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %results = llvm.mlir.constant(64 : i64) : i64
    func.call @strict_device_convolution_normed_run(%count, %result, %results, %values, %valuesCount, %valuesBits, %affine, %affineCount, %affineBits, %channels, %swish, %normed, %padded) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, f64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_convolution_normeds(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %channels: f64, %swish: f64, %normed: f64, %padded: f64, %results: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    func.call @strict_device_convolution_normed_run(%count, %result, %results, %values, %valuesCount, %valuesBits, %affine, %affineCount, %affineBits, %channels, %swish, %normed, %padded) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, f64, f64, f64) -> ()
    llvm.return
  }
  // 8 numbers of data from at into 16 bytes of shared memory as bfloat16 (zeros when not valid): copied without waiting
  // from a bfloat16 list, read and converted from other formats
  func.func private @strict_device_convolution_eight(%shared: !llvm.ptr<3>, %data: !llvm.ptr, %count: i64, %bits: i64, %at: i64, %valid: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^copy, ^convert
  ^copy:
    %first = arith.constant 0 : i64
    %safeAt = arith.select %valid, %at, %first : i64
    %global = llvm.getelementptr %data[%safeAt] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    func.call @strict_device_copy_filled(%shared, %global, %valid) : (!llvm.ptr<3>, !llvm.ptr, i1) -> ()
    return
  ^convert:
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %true = arith.constant true
    %outside = arith.xori %valid, %true : i1
    %empty = llvm.mlir.undef : vector<8xf32>
    cf.br ^number(%none, %empty : i64, vector<8xf32>)
  ^number(%index: i64, %numbers: vector<8xf32>):
    %more = arith.cmpi slt, %index, %eight : i64
    cf.cond_br %more, ^read, ^store
  ^read:
    %numberAt = arith.addi %at, %index : i64
    %number = func.call @strict_device_products_number(%data, %count, %bits, %numberAt, %outside) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %slot = arith.trunci %index : i64 to i32
    %withNumber = llvm.insertelement %number, %numbers[%slot : i32] : vector<8xf32>
    %next = arith.addi %index, %one : i64
    llvm.br ^number(%next, %withNumber : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^store:
    %brains = arith.truncf %numbers : vector<8xf32> to vector<8xbf16>
    llvm.store %brains, %shared {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    return
  }
  // copies the 32 numbers of stage along the side x side taps of channels into buffer (0 or 1) of lefts and rights, 16
  // bytes at a time without waiting: 8 channels of the 2 pixels of a thread at the pixel the tap shifts to (row and column
  // - 1 to + 1, for the 2 x 2 taps of a doubled image - 1 to 0 plus the parity of the output, zeros outside of the image)
  // and 8 numbers of the weight rows of its 2 outputs (zeros past the last), 80 bytes per row like strict_device_product_put
  func.func private @strict_device_convolution_copies(%values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %buffer: i32, %stage: i64, %channels: i64, %side: i32, %rowParity: i32, %columnParity: i32, %height: i32, %width: i32, %pixelRows: vector<2xi32>, %pixelColumns: vector<2xi32>, %weightRows: vector<2xi64>, %weightInside: vector<2xi1>, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %five = arith.constant 5 : i64
    %lefts = llvm.mlir.addressof @strict_device_product_lefts : !llvm.ptr<3>
    %rights = llvm.mlir.addressof @strict_device_product_rights : !llvm.ptr<3>
    %inner = arith.shli %stage, %five : i64
    %tap64 = arith.divui %inner, %channels : i64
    %tapStart = arith.muli %tap64, %channels : i64
    %channelFirst = arith.subi %inner, %tapStart : i64
    %tap = arith.trunci %tap64 : i64 to i32
    %tapRow = arith.divui %tap, %side : i32
    %tapRowStart = arith.muli %tapRow, %side : i32
    %tapColumn = arith.subi %tap, %tapRowStart : i32
    %tapRowParity = arith.addi %tapRow, %rowParity : i32
    %tapColumnParity = arith.addi %tapColumn, %columnParity : i32
    %rowShift = arith.subi %tapRowParity, %c1 : i32
    %columnShift = arith.subi %tapColumnParity, %c1 : i32
    %quarter = arith.andi %thread, %c3 : i32
    %quarterNumbers = arith.shli %quarter, %c3 : i32
    %quarter64 = arith.extui %quarterNumbers : i32 to i64
    %channel = arith.addi %channelFirst, %quarter64 : i64
    %bufferStart = arith.muli %buffer, %c10240 : i32
    %quarterBytes = arith.shli %quarter, %c4 : i32
    %localRow = arith.shrui %thread, %c2 : i32
    cf.br ^part(%c0 : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c2 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partRows = arith.muli %part, %c64 : i32
    %row = arith.addi %localRow, %partRows : i32
    %rowStart = arith.muli %row, %c80 : i32
    %rowAt = arith.addi %rowStart, %quarterBytes : i32
    %offset = arith.addi %bufferStart, %rowAt : i32
    %offset64 = arith.extui %offset : i32 to i64
    %leftSlot = llvm.getelementptr %lefts[%offset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %rightSlot = llvm.getelementptr %rights[%offset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %pixelRow = llvm.extractelement %pixelRows[%part : i32] : vector<2xi32>
    %pixelColumn = llvm.extractelement %pixelColumns[%part : i32] : vector<2xi32>
    %sourceRow = arith.addi %pixelRow, %rowShift : i32
    %sourceColumn = arith.addi %pixelColumn, %columnShift : i32
    %rowFrom = arith.cmpi sge, %sourceRow, %c0 : i32
    %rowTo = arith.cmpi slt, %sourceRow, %height : i32
    %columnFrom = arith.cmpi sge, %sourceColumn, %c0 : i32
    %columnTo = arith.cmpi slt, %sourceColumn, %width : i32
    %rowInside = arith.andi %rowFrom, %rowTo : i1
    %columnInside = arith.andi %columnFrom, %columnTo : i1
    %inside = arith.andi %rowInside, %columnInside : i1
    %sourceRowStart = arith.muli %sourceRow, %width : i32
    %source = arith.addi %sourceRowStart, %sourceColumn : i32
    %source64 = arith.extui %source : i32 to i64
    %sourceStart = arith.muli %source64, %channels : i64
    %at = arith.addi %sourceStart, %channel : i64
    func.call @strict_device_convolution_eight(%leftSlot, %values, %valuesCount, %valuesBits, %at, %inside) : (!llvm.ptr<3>, !llvm.ptr, i64, i64, i64, i1) -> ()
    %weightRow = llvm.extractelement %weightRows[%part : i32] : vector<2xi64>
    %weightValid = llvm.extractelement %weightInside[%part : i32] : vector<2xi1>
    %weightRowInner = arith.addi %weightRow, %inner : i64
    %weightAt = arith.addi %weightRowInner, %quarter64 : i64
    func.call @strict_device_convolution_eight(%rightSlot, %weights, %weightsCount, %weightsBits, %weightAt, %weightValid) : (!llvm.ptr<3>, !llvm.ptr, i64, i64, i64, i1) -> ()
    %nextPart = arith.addi %part, %c1 : i32
    llvm.br ^part(%nextPart : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    func.call @strict_device_copies_commit() : () -> ()
    return
  }
  // `Convolved.Summed` of channels a multiple of 32 from the input of strict_device_convolution_normed and its weights
  // (bfloat16 on the GPU): a block of 8 warps takes 128 pixels by 128 outputs over the taps in stages of 32 numbers (the stage of a tap
  // never crosses into the next), copied a stage ahead into the other buffer while the bfloat16 tensor cores multiply
  // this one; the output tiles of the same pixels lie next to each other
  func.func private @strict_device_convolution_run(%result: !llvm.ptr, %results: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: i64, %width: i64, %channels: i64, %outputs: i64, %factor: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %far = llvm.mlir.constant(-2 : i32) : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %hundredTwentySeven = arith.constant 127 : i64
    %true = arith.constant true
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %block = arith.extui %block32 : i32 to i64
    %pixels = arith.muli %height, %width : i64
    %gridHeight = arith.divui %height, %factor : i64
    %gridWidth = arith.divui %width, %factor : i64
    %gridPixels = arith.muli %gridHeight, %gridWidth : i64
    %parities = arith.muli %factor, %factor : i64
    %sideNumber = arith.subi %four, %factor : i64
    %tapCount = arith.muli %sideNumber, %sideNumber : i64
    %outputsUp = arith.addi %outputs, %hundredTwentySeven : i64
    %outputTiles = arith.shrui %outputsUp, %seven : i64
    %pixelsUp = arith.addi %gridPixels, %hundredTwentySeven : i64
    %pixelTiles = arith.shrui %pixelsUp, %seven : i64
    %parityTiles = arith.muli %pixelTiles, %outputTiles : i64
    %tiles = arith.muli %parities, %parityTiles : i64
    %working = arith.cmpi ult, %block, %tiles : i64
    cf.cond_br %working, ^tile, ^exit
  ^tile:
    %parity = arith.divui %block, %parityTiles : i64
    %withinParity = arith.remui %block, %parityTiles : i64
    %pixelTile = arith.divui %withinParity, %outputTiles : i64
    %outputTile = arith.remui %withinParity, %outputTiles : i64
    %pixelFirst = arith.shli %pixelTile, %seven : i64
    %outputFirst = arith.shli %outputTile, %seven : i64
    %taps = arith.muli %channels, %tapCount : i64
    %parityRows = arith.muli %parity, %outputs : i64
    %parityStart = arith.muli %parityRows, %taps : i64
    %rowParity64 = arith.shrui %parity, %one : i64
    %columnParity64 = arith.andi %parity, %one : i64
    %rowParity = arith.trunci %rowParity64 : i64 to i32
    %columnParity = arith.trunci %columnParity64 : i64 to i32
    %side = arith.trunci %sideNumber : i64 to i32
    %stages = arith.shrui %taps, %five : i64
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpColumnIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.shli %warpRowIndex, %c6 : i32
    %warpColumn = arith.shli %warpColumnIndex, %c5 : i32
    %height32 = arith.trunci %gridHeight : i64 to i32
    %width32 = arith.trunci %gridWidth : i64 to i32
    %firstLocal = arith.shrui %thread, %c2 : i32
    %secondLocal = arith.addi %firstLocal, %c64 : i32
    %firstLocal64 = arith.extui %firstLocal : i32 to i64
    %secondLocal64 = arith.extui %secondLocal : i32 to i64
    %firstPixel = arith.addi %pixelFirst, %firstLocal64 : i64
    %secondPixel = arith.addi %pixelFirst, %secondLocal64 : i64
    %firstIsPixel = arith.cmpi slt, %firstPixel, %gridPixels : i64
    %secondIsPixel = arith.cmpi slt, %secondPixel, %gridPixels : i64
    %firstPixel32 = arith.trunci %firstPixel : i64 to i32
    %secondPixel32 = arith.trunci %secondPixel : i64 to i32
    %firstRowAt = arith.divui %firstPixel32, %width32 : i32
    %secondRowAt = arith.divui %secondPixel32, %width32 : i32
    %firstRowStart = arith.muli %firstRowAt, %width32 : i32
    %secondRowStart = arith.muli %secondRowAt, %width32 : i32
    %firstColumn = arith.subi %firstPixel32, %firstRowStart : i32
    %secondColumn = arith.subi %secondPixel32, %secondRowStart : i32
    %firstRow = arith.select %firstIsPixel, %firstRowAt, %far : i32
    %secondRow = arith.select %secondIsPixel, %secondRowAt, %far : i32
    %noRows = llvm.mlir.undef : vector<2xi32>
    %firstRows = llvm.insertelement %firstRow, %noRows[%c0 : i32] : vector<2xi32>
    %pixelRows = llvm.insertelement %secondRow, %firstRows[%c1 : i32] : vector<2xi32>
    %firstColumns = llvm.insertelement %firstColumn, %noRows[%c0 : i32] : vector<2xi32>
    %pixelColumns = llvm.insertelement %secondColumn, %firstColumns[%c1 : i32] : vector<2xi32>
    %firstOutput = arith.addi %outputFirst, %firstLocal64 : i64
    %secondOutput = arith.addi %outputFirst, %secondLocal64 : i64
    %firstOutputInside = arith.cmpi slt, %firstOutput, %outputs : i64
    %secondOutputInside = arith.cmpi slt, %secondOutput, %outputs : i64
    %firstOutputRow = arith.muli %firstOutput, %taps : i64
    %secondOutputRow = arith.muli %secondOutput, %taps : i64
    %firstWeightRow = arith.addi %parityStart, %firstOutputRow : i64
    %secondWeightRow = arith.addi %parityStart, %secondOutputRow : i64
    %noWeightRows = llvm.mlir.undef : vector<2xi64>
    %firstWeightRows = llvm.insertelement %firstWeightRow, %noWeightRows[%c0 : i32] : vector<2xi64>
    %weightRows = llvm.insertelement %secondWeightRow, %firstWeightRows[%c1 : i32] : vector<2xi64>
    %noInside = llvm.mlir.undef : vector<2xi1>
    %firstInside = llvm.insertelement %firstOutputInside, %noInside[%c0 : i32] : vector<2xi1>
    %weightInside = llvm.insertelement %secondOutputInside, %firstInside[%c1 : i32] : vector<2xi1>
    func.call @strict_device_convolution_copies(%values, %valuesCount, %valuesBits, %weights, %weightsCount, %weightsBits, %c0, %none, %channels, %side, %rowParity, %columnParity, %height32, %width32, %pixelRows, %pixelColumns, %weightRows, %weightInside, %thread) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i32, i64, i64, i32, i32, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi64>, vector<2xi1>, i32) -> ()
    cf.br ^stage(%none, %zeros : i64, vector<64xf32>)
  ^stage(%stage: i64, %sums: vector<64xf32>):
    %moreStages = arith.cmpi slt, %stage, %stages : i64
    cf.cond_br %moreStages, ^compute, ^written
  ^compute:
    %nextStage = arith.addi %stage, %one : i64
    %stageParity = arith.andi %stage, %one : i64
    %buffer = arith.trunci %stageParity : i64 to i32
    %otherBuffer = arith.xori %buffer, %c1 : i32
    %hasNext = arith.cmpi slt, %nextStage, %stages : i64
    cf.cond_br %hasNext, ^ahead, ^last
  ^ahead:
    func.call @strict_device_convolution_copies(%values, %valuesCount, %valuesBits, %weights, %weightsCount, %weightsBits, %otherBuffer, %nextStage, %channels, %side, %rowParity, %columnParity, %height32, %width32, %pixelRows, %pixelColumns, %weightRows, %weightInside, %thread) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i32, i64, i64, i32, i32, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi64>, vector<2xi1>, i32) -> ()
    func.call @strict_device_copies_wait() : () -> ()
    cf.br ^ready
  ^last:
    func.call @strict_device_copies_waited() : () -> ()
    cf.br ^ready
  ^ready:
    func.call @strict_device_barrier() : () -> ()
    %firstHalf = func.call @strict_device_product_half(%sums, %buffer, %c0, %warpRow, %warpColumn, %lane, %true) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %bothHalves = func.call @strict_device_product_half(%firstHalf, %buffer, %c1, %warpRow, %warpColumn, %lane, %true) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%nextStage, %bothHalves : i64, vector<64xf32>)
  ^written:
    %group = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %pairColumn = arith.shli %quad, %c1 : i32
    cf.br ^write(%c0 : i32)
  ^write(%slot: i32):
    %moreSlots = arith.cmpi slt, %slot, %c64 : i32
    cf.cond_br %moreSlots, ^writeBody, ^exit
  ^writeBody:
    %rowTileIndex = arith.shrui %slot, %c4 : i32
    %slotPair = arith.shrui %slot, %c2 : i32
    %columnTileIndex = arith.andi %slotPair, %c3 : i32
    %element = arith.andi %slot, %c3 : i32
    %lowerHalf = arith.shrui %element, %c1 : i32
    %oddColumn = arith.andi %element, %c1 : i32
    %tileRows = arith.shli %rowTileIndex, %c4 : i32
    %lowerRows = arith.shli %lowerHalf, %c3 : i32
    %warpTileRow = arith.addi %warpRow, %tileRows : i32
    %groupRow = arith.addi %warpTileRow, %group : i32
    %rowLocal = arith.addi %groupRow, %lowerRows : i32
    %tileColumns = arith.shli %columnTileIndex, %c3 : i32
    %warpTileColumn = arith.addi %warpColumn, %tileColumns : i32
    %pairColumnLocal = arith.addi %warpTileColumn, %pairColumn : i32
    %columnLocal = arith.addi %pairColumnLocal, %oddColumn : i32
    %rowLocal64 = arith.extui %rowLocal : i32 to i64
    %columnLocal64 = arith.extui %columnLocal : i32 to i64
    %gridPixel = arith.addi %pixelFirst, %rowLocal64 : i64
    %gridRow = arith.divui %gridPixel, %gridWidth : i64
    %gridColumn = arith.remui %gridPixel, %gridWidth : i64
    %outputRowScaled = arith.muli %gridRow, %factor : i64
    %outputRow = arith.addi %outputRowScaled, %rowParity64 : i64
    %outputColumnScaled = arith.muli %gridColumn, %factor : i64
    %outputColumn = arith.addi %outputColumnScaled, %columnParity64 : i64
    %outputRowStart = arith.muli %outputRow, %width : i64
    %outputPixel = arith.addi %outputRowStart, %outputColumn : i64
    %isGridPixel = arith.cmpi ult, %gridPixel, %gridPixels : i64
    %pixel = arith.select %isGridPixel, %outputPixel, %pixels : i64
    %output = arith.addi %outputFirst, %columnLocal64 : i64
    %value = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    func.call @strict_device_products_written(%result, %results, %value, %none, %pixel, %output, %pixels, %outputs) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  llvm.func @strict_device_convolution(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: f64, %width: f64, %channels: f64, %outputs: f64, %factor: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    %results = llvm.mlir.constant(64 : i64) : i64
    func.call @strict_device_convolution_shaped(%result, %results, %values, %valuesCount, %valuesBits, %weights, %weightsCount, %weightsBits, %height, %width, %channels, %outputs, %factor) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, f64, f64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_convolutions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: f64, %width: f64, %channels: f64, %outputs: f64, %factor: f64, %results: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    func.call @strict_device_convolution_shaped(%result, %results, %values, %valuesCount, %valuesBits, %weights, %weightsCount, %weightsBits, %height, %width, %channels, %outputs, %factor) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, f64, f64, f64, f64, f64) -> ()
    llvm.return
  }
  func.func private @strict_device_convolution_shaped(%result: !llvm.ptr, %results: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: f64, %width: f64, %channels: f64, %outputs: f64, %factor: f64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %heightNumber = arith.fptosi %height : f64 to i64
    %widthNumber = arith.fptosi %width : f64 to i64
    %channelsNumber = arith.fptosi %channels : f64 to i64
    %outputsNumber = arith.fptosi %outputs : f64 to i64
    %factorNumber = arith.fptosi %factor : f64 to i64
    func.call @strict_device_convolution_run(%result, %results, %values, %valuesCount, %valuesBits, %weights, %weightsCount, %weightsBits, %heightNumber, %widthNumber, %channelsNumber, %outputsNumber, %factorNumber) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    return
  }
