  // the 4 numbers of a pixel from channel (zeros when outside) normed with the scales and shifts of affine in shared
  // memory (swish when asked), one vector load of 32 bit floats or bfloat16 when fast
  func.func private @strict_device_convolution_quad(%values: !llvm.ptr, %count: i64, %bits: i64, %affined: !llvm.ptr<3>, %at: i64, %channel: i64, %channels: i64, %inside: i1, %fast: i1, %swish: i1) -> vector<4xf16> attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %zeros = arith.constant dense<0.0> : vector<4xf16>
    %vectored = arith.andi %inside, %fast : i1
    cf.cond_br %inside, ^read, ^zero
  ^zero:
    return %zeros : vector<4xf16>
  ^read:
    cf.cond_br %vectored, ^vector, ^single
  ^vector:
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brainVector, ^singleVector
  ^brainVector:
    %brainSlot = llvm.getelementptr %values[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fromBrains = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    cf.br ^normed(%fromBrains : vector<4xf32>)
  ^singleVector:
    %slot = llvm.getelementptr %values[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %loaded = llvm.load %slot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    cf.br ^normed(%loaded : vector<4xf32>)
  ^single:
    %false = arith.constant false
    %secondAt = arith.addi %at, %one : i64
    %thirdAt = arith.addi %at, %two : i64
    %fourthAt = arith.addi %at, %three : i64
    %first = func.call @strict_device_products_number(%values, %count, %bits, %at, %false) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %second = func.call @strict_device_products_number(%values, %count, %bits, %secondAt, %false) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %third = func.call @strict_device_products_number(%values, %count, %bits, %thirdAt, %false) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %fourth = func.call @strict_device_products_number(%values, %count, %bits, %fourthAt, %false) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %singles = func.call @strict_device_convolution_vector(%first, %second, %third, %fourth) : (f32, f32, f32, f32) -> vector<4xf32>
    cf.br ^normed(%singles : vector<4xf32>)
  ^normed(%numbers: vector<4xf32>):
    %channel32 = arith.trunci %channel : i64 to i32
    %channels32 = arith.trunci %channels : i64 to i32
    %shiftAt = arith.addi %channel32, %channels32 : i32
    %scaleAt64 = arith.extui %channel32 : i32 to i64
    %shiftAt64 = arith.extui %shiftAt : i32 to i64
    %scaleSlot = llvm.getelementptr %affined[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %shiftSlot = llvm.getelementptr %affined[%shiftAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %scales = llvm.load %scaleSlot {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %shifts = llvm.load %shiftSlot {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %scaled = arith.mulf %numbers, %scales : vector<4xf32>
    %shifted = arith.addf %scaled, %shifts : vector<4xf32>
    cf.cond_br %swish, ^swished, ^done(%shifted : vector<4xf32>)
  ^swished:
    %units = arith.constant dense<1.0> : vector<4xf32>
    %factor = arith.constant dense<-1.44269504> : vector<4xf32>
    %exponent = arith.mulf %shifted, %factor : vector<4xf32>
    %power = llvm.call_intrinsic "llvm.exp2.v4f32"(%exponent) : (vector<4xf32>) -> vector<4xf32>
    %below = arith.addf %units, %power : vector<4xf32>
    %swished = arith.divf %shifted, %below fastmath<afn, arcp> : vector<4xf32>
    cf.br ^done(%swished : vector<4xf32>)
  ^done(%activated: vector<4xf32>):
    %halves = arith.truncf %activated : vector<4xf32> to vector<4xf16>
    return %halves : vector<4xf16>
  }
  func.func private @strict_device_convolution_vector(%first: f32, %second: f32, %third: f32, %fourth: f32) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    %with0 = llvm.insertelement %first, %empty[%c0 : i32] : vector<4xf32>
    %with1 = llvm.insertelement %second, %with0[%c1 : i32] : vector<4xf32>
    %with2 = llvm.insertelement %third, %with1[%c2 : i32] : vector<4xf32>
    %with3 = llvm.insertelement %fourth, %with2[%c3 : i32] : vector<4xf32>
    return %with3 : vector<4xf32>
  }
  // the 16 numbers a thread copies of its 4 pixels (rows and columns, the row is -2 past the last pixel) by the 32
  // numbers of stage along the side x side taps of channels: 4 channels of the pixel the tap shifts to (row and column
  // - 1 to + 1, for the 2 x 2 taps of a doubled image - 1 to 0 plus the parity of the output), zeros outside of the image
  func.func private @strict_device_convolution_fetch(%values: !llvm.ptr, %count: i64, %bits: i64, %affined: !llvm.ptr<3>, %height: i32, %width: i32, %channels: i32, %side: i32, %rowParity: i32, %columnParity: i32, %rows: vector<4xi32>, %columns: vector<4xi32>, %stage: i32, %thread: i32, %fast: i1, %swish: i1) -> vector<16xf16> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %inner = arith.shli %stage, %c5 : i32
    %tap = arith.divui %inner, %channels : i32
    %tapStart = arith.muli %tap, %channels : i32
    %channelFirst = arith.subi %inner, %tapStart : i32
    %tapRow = arith.divui %tap, %side : i32
    %tapRowStart = arith.muli %tapRow, %side : i32
    %tapColumn = arith.subi %tap, %tapRowStart : i32
    %tapRowParity = arith.addi %tapRow, %rowParity : i32
    %tapColumnParity = arith.addi %tapColumn, %columnParity : i32
    %rowShift = arith.subi %tapRowParity, %c1 : i32
    %columnShift = arith.subi %tapColumnParity, %c1 : i32
    %quad = arith.andi %thread, %c7 : i32
    %quadChannels = arith.shli %quad, %c2 : i32
    %channel = arith.addi %channelFirst, %quadChannels : i32
    %channel64 = arith.extsi %channel : i32 to i64
    %channels64 = arith.extsi %channels : i32 to i64
    %empty = llvm.mlir.undef : vector<16xf16>
    cf.br ^part(%c0, %empty : i32, vector<16xf16>)
  ^part(%part: i32, %halves: vector<16xf16>):
    %more = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %row = llvm.extractelement %rows[%part : i32] : vector<4xi32>
    %column = llvm.extractelement %columns[%part : i32] : vector<4xi32>
    %sourceRow = arith.addi %row, %rowShift : i32
    %sourceColumn = arith.addi %column, %columnShift : i32
    %rowFrom = arith.cmpi sge, %sourceRow, %c0 : i32
    %rowTo = arith.cmpi slt, %sourceRow, %height : i32
    %columnFrom = arith.cmpi sge, %sourceColumn, %c0 : i32
    %columnTo = arith.cmpi slt, %sourceColumn, %width : i32
    %rowInside = arith.andi %rowFrom, %rowTo : i1
    %columnInside = arith.andi %columnFrom, %columnTo : i1
    %inside = arith.andi %rowInside, %columnInside : i1
    %sourceRowStart = arith.muli %sourceRow, %width : i32
    %source = arith.addi %sourceRowStart, %sourceColumn : i32
    %source64 = arith.extsi %source : i32 to i64
    %sourceStart = arith.muli %source64, %channels64 : i64
    %at = arith.addi %sourceStart, %channel64 : i64
    %quadHalves = func.call @strict_device_convolution_quad(%values, %count, %bits, %affined, %at, %channel64, %channels64, %inside, %fast, %swish) : (!llvm.ptr, i64, i64, !llvm.ptr<3>, i64, i64, i64, i1, i1, i1) -> vector<4xf16>
    %slot0 = arith.shli %part, %c2 : i32
    %slot1 = arith.addi %slot0, %c1 : i32
    %slot2 = arith.addi %slot0, %c2 : i32
    %slot3 = arith.addi %slot0, %c3 : i32
    %half0 = llvm.extractelement %quadHalves[%c0 : i32] : vector<4xf16>
    %half1 = llvm.extractelement %quadHalves[%c1 : i32] : vector<4xf16>
    %half2 = llvm.extractelement %quadHalves[%c2 : i32] : vector<4xf16>
    %half3 = llvm.extractelement %quadHalves[%c3 : i32] : vector<4xf16>
    %with0 = llvm.insertelement %half0, %halves[%slot0 : i32] : vector<16xf16>
    %with1 = llvm.insertelement %half1, %with0[%slot1 : i32] : vector<16xf16>
    %with2 = llvm.insertelement %half2, %with1[%slot2 : i32] : vector<16xf16>
    %with3 = llvm.insertelement %half3, %with2[%slot3 : i32] : vector<16xf16>
    %nextPart = arith.addi %part, %c1 : i32
    llvm.br ^part(%nextPart, %with3 : i32, vector<16xf16>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %halves : vector<16xf16>
  }
  // the rows and columns of the 4 pixels a thread copies (part p is pixel first + (thread + 256 p) / 8), a row of -2
  // past the last pixel so every tap lies outside
  func.func private @strict_device_convolution_pixels(%pixelFirst: i64, %pixels: i64, %width: i32, %thread: i32) -> (vector<4xi32>, vector<4xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %far = llvm.mlir.constant(-2 : i32) : i32
    %empty = llvm.mlir.undef : vector<4xi32>
    cf.br ^part(%c0, %empty, %empty : i32, vector<4xi32>, vector<4xi32>)
  ^part(%part: i32, %rows: vector<4xi32>, %columns: vector<4xi32>):
    %more = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %c8 : i32
    %linear = arith.addi %thread, %partStart : i32
    %local = arith.shrui %linear, %c3 : i32
    %local64 = arith.extui %local : i32 to i64
    %pixel64 = arith.addi %pixelFirst, %local64 : i64
    %isPixel = arith.cmpi slt, %pixel64, %pixels : i64
    %pixel = arith.trunci %pixel64 : i64 to i32
    %row = arith.divui %pixel, %width : i32
    %rowStart = arith.muli %row, %width : i32
    %column = arith.subi %pixel, %rowStart : i32
    %usedRow = arith.select %isPixel, %row, %far : i32
    %withRow = llvm.insertelement %usedRow, %rows[%part : i32] : vector<4xi32>
    %withColumn = llvm.insertelement %column, %columns[%part : i32] : vector<4xi32>
    %nextPart = arith.addi %part, %c1 : i32
    llvm.br ^part(%nextPart, %withRow, %withColumn : i32, vector<4xi32>, vector<4xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %rows, %columns : vector<4xi32>, vector<4xi32>
  }
  // the scales and then the shifts of affine as 32 bit floats in shared memory (2 numbers per channel, at most 512
  // channels), copied by all threads of the block
  llvm.mlir.global internal @strict_device_convolution_affine() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x f32>
  func.func private @strict_device_convolution_staged(%affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %thread: i32) -> !llvm.ptr<3> attributes {llvm.passthrough = ["alwaysinline"]} {
    %false = arith.constant false
    %step = arith.constant 256 : i64
    %affined = llvm.mlir.addressof @strict_device_convolution_affine : !llvm.ptr<3>
    %first = arith.extui %thread : i32 to i64
    cf.br ^copy(%first : i64)
  ^copy(%index: i64):
    %more = arith.cmpi slt, %index, %affineCount : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %number = func.call @strict_device_products_number(%affine, %affineCount, %affineBits, %index, %false) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %slot = llvm.getelementptr %affined[%index] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %number, %slot : f32, !llvm.ptr<3>
    %next = arith.addi %index, %step : i64
    cf.br ^copy(%next : i64)
  ^done:
    return %affined : !llvm.ptr<3>
  }
  // `Convolved.Summed` of channels a multiple of 32: a block of 8 warps takes 128 pixels by 128 outputs over the 9 taps
  // in stages of 32 numbers (the stage of a tap never crosses into the next), the weights like the right of
  // strict_device_product_tiles; the output tiles of the same pixels lie next to each other
  func.func private @strict_device_convolution_run(%result: !llvm.ptr, %wide: i1, %values: !llvm.ptr, %count: i64, %bits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %shape: vector<6xf64>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %nine = arith.constant 9 : i64
    %fifteen = arith.constant 15 : i64
    %thirtyTwo = arith.constant 32 : i64
    %hundredTwentySeven = arith.constant 127 : i64
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %shapeInts = arith.fptosi %shape : vector<6xf64> to vector<6xi64>
    %height = llvm.extractelement %shapeInts[%c0 : i32] : vector<6xi64>
    %width = llvm.extractelement %shapeInts[%c1 : i32] : vector<6xi64>
    %channels = llvm.extractelement %shapeInts[%c2 : i32] : vector<6xi64>
    %outputs = llvm.extractelement %shapeInts[%c3 : i32] : vector<6xi64>
    %swishFlag = llvm.extractelement %shapeInts[%c4 : i32] : vector<6xi64>
    %factor = llvm.extractelement %shapeInts[%c5 : i32] : vector<6xi64>
    %swish = arith.cmpi eq, %swishFlag, %one : i64
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %block = arith.extui %block32 : i32 to i64
    %pixels = arith.muli %height, %width : i64
    %gridHeight = arith.divui %height, %factor : i64
    %gridWidth = arith.divui %width, %factor : i64
    %gridPixels = arith.muli %gridHeight, %gridWidth : i64
    %parities = arith.muli %factor, %factor : i64
    %four = arith.constant 4 : i64
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
    %lastStage = arith.subi %stages, %one : i64
    %brainFormat = arith.constant 17 : i64
    %isFloats = arith.cmpi eq, %bits, %thirtyTwo : i64
    %isBrainFloats = arith.cmpi eq, %bits, %brainFormat : i64
    %isSingle = arith.ori %isFloats, %isBrainFloats : i1
    %address = llvm.ptrtoint %values : !llvm.ptr to i64
    %addressRest = arith.andi %address, %fifteen : i64
    %aligned = arith.cmpi eq, %addressRest, %none : i64
    %fast = arith.andi %isSingle, %aligned : i1
    %false = arith.constant false
    %rightFast = func.call @strict_device_product_fast(%weights, %weightsBits, %parityStart, %taps, %one, %false) : (!llvm.ptr, i64, i64, i64, i64, i1) -> i1
    %lefts = llvm.mlir.addressof @strict_device_product_lefts : !llvm.ptr<3>
    %rights = llvm.mlir.addressof @strict_device_product_rights : !llvm.ptr<3>
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpColumnIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.shli %warpRowIndex, %c6 : i32
    %warpColumn = arith.shli %warpColumnIndex, %c5 : i32
    %affined = func.call @strict_device_convolution_staged(%affine, %affineCount, %affineBits, %thread) : (!llvm.ptr, i64, i64, i32) -> !llvm.ptr<3>
    %height32 = arith.trunci %gridHeight : i64 to i32
    %width32 = arith.trunci %gridWidth : i64 to i32
    %channels32 = arith.trunci %channels : i64 to i32
    %pixelRows, %pixelColumns = func.call @strict_device_convolution_pixels(%pixelFirst, %gridPixels, %width32, %thread) : (i64, i64, i32, i32) -> (vector<4xi32>, vector<4xi32>)
    func.call @strict_device_barrier() : () -> ()
    %firstLefts = func.call @strict_device_convolution_fetch(%values, %count, %bits, %affined, %height32, %width32, %channels32, %side, %rowParity, %columnParity, %pixelRows, %pixelColumns, %c0, %thread, %fast, %swish) : (!llvm.ptr, i64, i64, !llvm.ptr<3>, i32, i32, i32, i32, i32, i32, vector<4xi32>, vector<4xi32>, i32, i32, i1, i1) -> vector<16xf16>
    %firstRights = func.call @strict_device_product_fetch(%weights, %weightsCount, %weightsBits, %parityStart, %taps, %one, %outputs, %outputFirst, %taps, %none, %thread, %false, %rightFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    func.call @strict_device_product_put(%firstLefts, %lefts, %c0, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%firstRights, %rights, %c0, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%none, %zeros : i64, vector<64xf32>)
  ^stage(%stage: i64, %sums: vector<64xf32>):
    %moreStages = arith.cmpi slt, %stage, %stages : i64
    cf.cond_br %moreStages, ^compute, ^written
  ^compute:
    %nextStage = arith.addi %stage, %one : i64
    %fetchedStage = arith.minsi %nextStage, %lastStage : i64
    %fetchedStage32 = arith.trunci %fetchedStage : i64 to i32
    %nextLefts = func.call @strict_device_convolution_fetch(%values, %count, %bits, %affined, %height32, %width32, %channels32, %side, %rowParity, %columnParity, %pixelRows, %pixelColumns, %fetchedStage32, %thread, %fast, %swish) : (!llvm.ptr, i64, i64, !llvm.ptr<3>, i32, i32, i32, i32, i32, i32, vector<4xi32>, vector<4xi32>, i32, i32, i1, i1) -> vector<16xf16>
    %nextRights = func.call @strict_device_product_fetch(%weights, %weightsCount, %weightsBits, %parityStart, %taps, %one, %outputs, %outputFirst, %taps, %fetchedStage, %thread, %false, %rightFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    %stageParity = arith.andi %stage, %one : i64
    %buffer = arith.trunci %stageParity : i64 to i32
    %firstHalf = func.call @strict_device_product_half(%sums, %buffer, %c0, %warpRow, %warpColumn, %lane) : (vector<64xf32>, i32, i32, i32, i32, i32) -> vector<64xf32>
    %bothHalves = func.call @strict_device_product_half(%firstHalf, %buffer, %c1, %warpRow, %warpColumn, %lane) : (vector<64xf32>, i32, i32, i32, i32, i32) -> vector<64xf32>
    %hasNext = arith.cmpi slt, %nextStage, %stages : i64
    cf.cond_br %hasNext, ^put, ^stage(%nextStage, %bothHalves : i64, vector<64xf32>)
  ^put:
    %otherBuffer = arith.xori %buffer, %c1 : i32
    func.call @strict_device_product_put(%nextLefts, %lefts, %otherBuffer, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%nextRights, %rights, %otherBuffer, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
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
    func.call @strict_device_products_written(%result, %wide, %value, %none, %pixel, %output, %pixels, %outputs) : (!llvm.ptr, i1, f32, i64, i64, i64, i64, i64) -> ()
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  llvm.func @strict_device_convolution(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: f64, %width: f64, %channels: f64, %outputs: f64, %swish: f64, %factor: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    %shape = func.call @strict_device_convolution_shape(%height, %width, %channels, %outputs, %swish, %factor) : (f64, f64, f64, f64, f64, f64) -> vector<6xf64>
    func.call @strict_device_convolution_run(%result, %wide, %values, %valuesCount, %valuesBits, %affine, %affineCount, %affineBits, %weights, %weightsCount, %weightsBits, %shape) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<6xf64>) -> ()
    llvm.return
  }
  llvm.func @strict_device_convolutions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %affine: !llvm.ptr, %affineCount: i64, %affineBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %height: f64, %width: f64, %channels: f64, %outputs: f64, %swish: f64, %factor: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    %shape = func.call @strict_device_convolution_shape(%height, %width, %channels, %outputs, %swish, %factor) : (f64, f64, f64, f64, f64, f64) -> vector<6xf64>
    func.call @strict_device_convolution_run(%result, %wide, %values, %valuesCount, %valuesBits, %affine, %affineCount, %affineBits, %weights, %weightsCount, %weightsBits, %shape) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<6xf64>) -> ()
    llvm.return
  }
  func.func private @strict_device_convolution_shape(%height: f64, %width: f64, %channels: f64, %outputs: f64, %swish: f64, %factor: f64) -> vector<6xf64> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %empty = llvm.mlir.undef : vector<6xf64>
    %with0 = llvm.insertelement %height, %empty[%c0 : i32] : vector<6xf64>
    %with1 = llvm.insertelement %width, %with0[%c1 : i32] : vector<6xf64>
    %with2 = llvm.insertelement %channels, %with1[%c2 : i32] : vector<6xf64>
    %with3 = llvm.insertelement %outputs, %with2[%c3 : i32] : vector<6xf64>
    %with4 = llvm.insertelement %swish, %with3[%c4 : i32] : vector<6xf64>
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %with5 = llvm.insertelement %factor, %with4[%c5 : i32] : vector<6xf64>
    return %with5 : vector<6xf64>
  }
