// GPU kernels of `Batch.Sums` over a prompt or several sequences (strict_device_tokens), only in programs that call
// strict_gpu_tokens, they use the dot product functions of NativeRuntimeDevice.mlir
  // the 8 bit numbers and scales of token index (at most last) of tokens whose numbers lie stride apart
  func.func private @strict_device_token_inputs(%bytes: !llvm.ptr, %scales: !llvm.ptr, %stride: i64, %index: i64, %last: i64) -> (!llvm.ptr, !llvm.ptr) attributes {passthrough = ["alwaysinline"]} {
    %five = arith.constant 5 : i64
    %token = arith.minsi %index, %last : i64
    %offset = arith.muli %token, %stride : i64
    %tokenBytes = llvm.getelementptr %bytes[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleOffset = arith.shrui %offset, %five : i64
    %tokenScales = llvm.getelementptr %scales[%scaleOffset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    return %tokenBytes, %tokenScales : !llvm.ptr, !llvm.ptr
  }
  // dots8 for up to 8 tokens whose 8 bit numbers lie stride apart (a prompt in `Batch.Sums`): each part of the row is
  // decoded once for all of them, tokens after last are left out, the sums come back added up over the warp
  func.func private @strict_device_dots8_tokens(%row: !llvm.ptr, %rowBits: i64, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %stride: i64, %last: i64, %lane: i64, %lanes: i64) -> vector<8xf32> attributes {passthrough = ["noinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    cf.switch %rowBits : i64, [
      default: ^bytes(%lane, %zeros : i64, vector<8xf32>),
      4: ^four(%lane, %zeros : i64, vector<8xf32>),
      6: ^six(%lane, %zeros : i64, vector<8xf32>)
    ]
  ^four(%unit: i64, %fourSums: vector<8xf32>):
    %units = arith.shrui %count, %five : i64
    %fourMore = arith.cmpi slt, %unit, %units : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%none, %fourSums : i64, vector<8xf32>)
  ^fourBody:
    %fourWeights:10 = func.call @strict_device_four_weights(%row, %unit) : (!llvm.ptr, i64) -> (vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64)
    cf.br ^fourToken(%none, %fourSums : i64, vector<8xf32>)
  ^fourToken(%fourIndex: i64, %fourTotals: vector<8xf32>):
    %fourTokens = arith.cmpi slt, %fourIndex, %eight : i64
    cf.cond_br %fourTokens, ^fourUsed, ^fourNext
  ^fourUsed:
    %fourUsed = arith.cmpi sle, %fourIndex, %last : i64
    cf.cond_br %fourUsed, ^fourAdd, ^fourJoined(%fourTotals : vector<8xf32>)
  ^fourAdd:
    %fourInputs:2 = func.call @strict_device_token_inputs(%bytes, %scales, %stride, %fourIndex, %last) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> (!llvm.ptr, !llvm.ptr)
    %fourOld = llvm.extractelement %fourTotals[%fourIndex : i64] : vector<8xf32>
    %fourNew = func.call @strict_device_four_token(%fourWeights#0, %fourWeights#1, %fourWeights#2, %fourWeights#3, %fourWeights#4, %fourWeights#5, %fourWeights#6, %fourWeights#7, %fourWeights#8, %fourWeights#9, %fourInputs#0, %fourInputs#1, %fourOld) : (vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %fourUpdated = llvm.insertelement %fourNew, %fourTotals[%fourIndex : i64] : vector<8xf32>
    cf.br ^fourJoined(%fourUpdated : vector<8xf32>)
  ^fourJoined(%fourKept: vector<8xf32>):
    %fourAfter = arith.addi %fourIndex, %one : i64
    llvm.br ^fourToken(%fourAfter, %fourKept : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^fourNext:
    %fourFollowing = arith.addi %unit, %lanes : i64
    cf.br ^four(%fourFollowing, %fourTotals : i64, vector<8xf32>)
  ^six(%chunk: i64, %sixSums: vector<8xf32>):
    %chunks = arith.shrui %count, %four : i64
    %sixMore = arith.cmpi slt, %chunk, %chunks : i64
    cf.cond_br %sixMore, ^sixBody, ^reduce(%none, %sixSums : i64, vector<8xf32>)
  ^sixBody:
    %sixWeights:8 = func.call @strict_device_six_weights(%row, %count, %chunk) : (!llvm.ptr, i64, i64) -> (vector<4xi32>, f32, f32, f32, f32, f32, i64, i64)
    cf.br ^sixToken(%none, %sixSums : i64, vector<8xf32>)
  ^sixToken(%sixIndex: i64, %sixTotals: vector<8xf32>):
    %sixTokens = arith.cmpi slt, %sixIndex, %eight : i64
    cf.cond_br %sixTokens, ^sixUsed, ^sixNext
  ^sixUsed:
    %sixUsed = arith.cmpi sle, %sixIndex, %last : i64
    cf.cond_br %sixUsed, ^sixAdd, ^sixJoined(%sixTotals : vector<8xf32>)
  ^sixAdd:
    %sixInputs:2 = func.call @strict_device_token_inputs(%bytes, %scales, %stride, %sixIndex, %last) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> (!llvm.ptr, !llvm.ptr)
    %sixOld = llvm.extractelement %sixTotals[%sixIndex : i64] : vector<8xf32>
    %sixNew = func.call @strict_device_six_token(%sixWeights#0, %sixWeights#1, %sixWeights#2, %sixWeights#3, %sixWeights#4, %sixWeights#5, %sixWeights#6, %sixWeights#7, %sixInputs#0, %sixInputs#1, %sixOld) : (vector<4xi32>, f32, f32, f32, f32, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %sixUpdated = llvm.insertelement %sixNew, %sixTotals[%sixIndex : i64] : vector<8xf32>
    cf.br ^sixJoined(%sixUpdated : vector<8xf32>)
  ^sixJoined(%sixKept: vector<8xf32>):
    %sixAfter = arith.addi %sixIndex, %one : i64
    llvm.br ^sixToken(%sixAfter, %sixKept : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^sixNext:
    %sixFollowing = arith.addi %chunk, %lanes : i64
    cf.br ^six(%sixFollowing, %sixTotals : i64, vector<8xf32>)
  ^bytes(%byteUnit: i64, %byteSums: vector<8xf32>):
    %byteUnits = arith.shrui %count, %four : i64
    %byteMore = arith.cmpi slt, %byteUnit, %byteUnits : i64
    cf.cond_br %byteMore, ^bytesBody, ^reduce(%none, %byteSums : i64, vector<8xf32>)
  ^bytesBody:
    %byteWeights:4 = func.call @strict_device_byte_weights(%row, %count, %byteUnit) : (!llvm.ptr, i64, i64) -> (vector<4xi32>, f32, i64, i64)
    cf.br ^byteToken(%none, %byteSums : i64, vector<8xf32>)
  ^byteToken(%byteIndex: i64, %byteTotals: vector<8xf32>):
    %byteTokens = arith.cmpi slt, %byteIndex, %eight : i64
    cf.cond_br %byteTokens, ^byteUsed, ^byteNext
  ^byteUsed:
    %byteUsed = arith.cmpi sle, %byteIndex, %last : i64
    cf.cond_br %byteUsed, ^byteAdd, ^byteJoined(%byteTotals : vector<8xf32>)
  ^byteAdd:
    %byteInputs:2 = func.call @strict_device_token_inputs(%bytes, %scales, %stride, %byteIndex, %last) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> (!llvm.ptr, !llvm.ptr)
    %byteOld = llvm.extractelement %byteTotals[%byteIndex : i64] : vector<8xf32>
    %byteNew = func.call @strict_device_byte_token(%byteWeights#0, %byteWeights#1, %byteWeights#2, %byteWeights#3, %byteInputs#0, %byteInputs#1, %byteOld) : (vector<4xi32>, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %byteUpdated = llvm.insertelement %byteNew, %byteTotals[%byteIndex : i64] : vector<8xf32>
    cf.br ^byteJoined(%byteUpdated : vector<8xf32>)
  ^byteJoined(%byteKept: vector<8xf32>):
    %byteAfter = arith.addi %byteIndex, %one : i64
    llvm.br ^byteToken(%byteAfter, %byteKept : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^byteNext:
    %byteFollowing = arith.addi %byteUnit, %lanes : i64
    cf.br ^bytes(%byteFollowing, %byteTotals : i64, vector<8xf32>)
  ^reduce(%reduceIndex: i64, %totals: vector<8xf32>):
    %reducing = arith.cmpi slt, %reduceIndex, %eight : i64
    cf.cond_br %reducing, ^reduceOne, ^reduced
  ^reduceOne:
    %partial = llvm.extractelement %totals[%reduceIndex : i64] : vector<8xf32>
    %reducedOne = func.call @strict_device_lanes_sums(%partial, %lane, %lanes) : (f32, i64, i64) -> f32
    %replaced = llvm.insertelement %reducedOne, %totals[%reduceIndex : i64] : vector<8xf32>
    %reduceNext = arith.addi %reduceIndex, %one : i64
    llvm.br ^reduce(%reduceNext, %replaced : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^reduced:
    return %totals : vector<8xf32>
  }
  // the 8 sums of a warp of strict_device_tokens: one row of columns numbers times the numbers of the tokens from first
  // to last (those of a token start at token * width), quantized ones through strict_device_dots8_tokens, others one by one
  func.func private @strict_device_tokens_sums(%failed: !llvm.ptr, %rows: !llvm.ptr, %row: i64, %rowCount: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: i64, %first: i64, %last: i64, %lane: i64) -> vector<8xf32> {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %thirtyTwo = arith.constant 32 : i64
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %rowData, %rowBits, %rowTotal = func.call @strict_device_row(%rows, %row) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %start = arith.muli %first, %width : i64
    %relativeLast = arith.subi %last, %first : i64
    %quantized = arith.cmpi eq, %inputBits, %eight : i64
    %widthRest = arith.andi %width, %thirtyOne : i64
    %widthAligned = arith.cmpi eq, %widthRest, %none : i64
    %blocked = arith.andi %quantized, %widthAligned : i1
    %bytes = llvm.getelementptr %input[%start] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.cond_br %blocked, ^check, ^plain
  ^check:
    %fits = func.call @strict_device_fits8(%rowData, %rowBits, %rowCount, %bytes) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %fits, ^eights, ^plain
  ^eights:
    %data = func.call @strict_device_data(%inputCount, %eight) : (i64, i64) -> i64
    %allScales = llvm.getelementptr %input[%data] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleStart = arith.shrui %start, %five : i64
    %scales = llvm.getelementptr %allScales[%scaleStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sums = func.call @strict_device_dots8_tokens(%rowData, %rowBits, %rowCount, %bytes, %scales, %width, %relativeLast, %lane, %thirtyTwo) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> vector<8xf32>
    return %sums : vector<8xf32>
  ^plain:
    %plain = func.call @strict_device_tokens_plain(%failed, %rowData, %rowBits, %rowCount, %input, %inputCount, %inputBits, %width, %first, %relativeLast, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    return %plain : vector<8xf32>
  }
  // the 8 sums of strict_device_tokens_sums for numbers that are not 8 bit, one dot product per token, kept out of line
  // so the 8 bit path keeps its registers
  func.func private @strict_device_tokens_plain(%failed: !llvm.ptr, %rowData: !llvm.ptr, %rowBits: i64, %rowCount: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: i64, %first: i64, %relativeLast: i64, %lane: i64) -> vector<8xf32> attributes {passthrough = ["noinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %thirtyTwo = arith.constant 32 : i64
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    cf.br ^plain(%none, %zeros : i64, vector<8xf32>)
  ^plain(%index: i64, %plainSums: vector<8xf32>):
    %more = arith.cmpi slt, %index, %eight : i64
    cf.cond_br %more, ^plainOne, ^plainDone
  ^plainOne:
    %relative = arith.minsi %index, %relativeLast : i64
    %token = arith.addi %first, %relative : i64
    %tokenStart = arith.muli %token, %width : i64
    %dot = func.call @strict_device_offset_dots(%failed, %rowData, %rowBits, %rowCount, %none, %input, %inputBits, %inputCount, %tokenStart, %rowCount, %lane, %thirtyTwo) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> f32
    %updated = llvm.insertelement %dot, %plainSums[%index : i64] : vector<8xf32>
    %following = arith.addi %index, %one : i64
    llvm.br ^plain(%following, %updated : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<disable = true>>}
  ^plainDone:
    return %plainSums : vector<8xf32>
  }
  // `Batch.Sums` of a single token (or numbers that are not 8 bit): a warp takes one row and 8 tokens, the dot
  // of row r and token t becomes number t * rows + r of the result (doubles when wide)
  func.func private @strict_device_tokens_rows(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    %blockWidth = nvvm.read.ptx.sreg.ntid.x : i32
    %blockStart = arith.muli %block, %blockWidth : i32
    %threadSum = arith.addi %blockStart, %thread : i32
    %global = arith.extui %threadSum : i32 to i64
    %warp = arith.shrui %global, %five : i64
    %lane = arith.andi %global, %thirtyOne : i64
    %rowsGiven = arith.fptosi %rowsNumber : f64 to i64
    %rows = arith.maxsi %rowsGiven, %one : i64
    %width = arith.fptosi %widthNumber : f64 to i64
    %tokens = arith.divui %count, %rows : i64
    %chunk = arith.divui %warp, %rows : i64
    %row = arith.remui %warp, %rows : i64
    %first = arith.muli %chunk, %eight : i64
    %inside = arith.cmpi slt, %first, %tokens : i64
    cf.cond_br %inside, ^body, ^exit
  ^body:
    %chunkLast = arith.addi %first, %seven : i64
    %lastToken = arith.subi %tokens, %one : i64
    %last = arith.minsi %chunkLast, %lastToken : i64
    %sums = func.call @strict_device_tokens_sums(%failed, %rowsData, %row, %columns, %input, %inputCount, %inputBits, %width, %first, %last, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %isFirstLane = arith.cmpi eq, %lane, %none : i64
    cf.cond_br %isFirstLane, ^write(%none : i64), ^exit
  ^write(%index: i64):
    %writes = arith.cmpi slt, %index, %eight : i64
    cf.cond_br %writes, ^token, ^exit
  ^token:
    %token = arith.addi %first, %index : i64
    %writing = arith.cmpi sle, %token, %last : i64
    %value = llvm.extractelement %sums[%index : i64] : vector<8xf32>
    %tokenStart = arith.muli %token, %rows : i64
    %element = arith.addi %tokenStart, %row : i64
    cf.cond_br %writing, ^store, ^stored
  ^store:
    func.call @strict_device_tokens_store(%result, %wide, %element, %value) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^stored
  ^stored:
    %next = arith.addi %index, %one : i64
    llvm.br ^write(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  // the sums of the 8 warps of a block of strict_device_tokens_blocks: 8 values of 32 lanes each
  llvm.mlir.global internal @strict_device_block_sums() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<2048 x f32>
  // where the numbers of token first + offset (at most last) start, place is first, last, the width of a token and the
  // lane, in 32 bits like all positions of strict_device_block_four (fewer registers)
  func.func private @strict_device_block_token(%place: vector<4xi32>, %offset: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %first = llvm.extractelement %place[%c0 : i32] : vector<4xi32>
    %last = llvm.extractelement %place[%c1 : i32] : vector<4xi32>
    %width = llvm.extractelement %place[%c2 : i32] : vector<4xi32>
    %own = arith.addi %first, %offset : i32
    %token = arith.minsi %own, %last : i32
    %start = arith.muli %token, %width : i32
    return %start : i32
  }
  // the sums of a lane over one Q4_K block (256 numbers from column) for row lane / 4 of 8 rows and the tokens
  // 2 * (lane % 4) and the next of up to 4 groups of 8 tokens (slot 2 * group and the next): a lane reads bytes
  // 8 * (lane % 4) to 8 * (lane % 4) + 7 of each part of 32 numbers of the row and of its token (lane / 4 of each
  // group), the same order in both mma operands, so 2 mma add up a part in integers
  func.func private @strict_device_block_four(%sums: vector<8xf32>, %block: !llvm.ptr, %bytes: !llvm.ptr, %scales: !llvm.ptr, %place: vector<4xi32>, %column: i32, %groups: i32) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %nothing = arith.constant 0 : i32
    %ones = arith.constant 16843009 : i32
    %mask = arith.constant 252645135 : i32
    %lane = llvm.extractelement %place[%c3 : i32] : vector<4xi32>
    %quad = arith.andi %lane, %c3 : i32
    %laneRow = arith.shrui %lane, %c2 : i32
    %header = llvm.load %block invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %headerWord = llvm.extractelement %header[%c0 : i32] : vector<4xi32>
    %halves = llvm.bitcast %headerWord : i32 to vector<2xf16>
    %halfScale = llvm.extractelement %halves[%c0 : i32] : vector<2xf16>
    %halfMinimum = llvm.extractelement %halves[%c1 : i32] : vector<2xf16>
    %blockScale = arith.extf %halfScale : f16 to f32
    %blockMinimum = arith.extf %halfMinimum : f16 to f32
    %quadBytes = arith.shli %quad, %c3 : i32
    %columnScales = arith.shrui %column, %c5 : i32
    cf.br ^pair(%c0, %sums : i32, vector<8xf32>)
  ^pair(%pair: i32, %total: vector<8xf32>):
    %morePairs = arith.cmpi slt, %pair, %c4 : i32
    cf.cond_br %morePairs, ^pairBody, ^done
  ^pairBody:
    %pairBytes = arith.shli %pair, %c5 : i32
    %numbersAt = arith.addi %pairBytes, %c16 : i32
    %weightAt = arith.addi %numbersAt, %quadBytes : i32
    %weightOffset = arith.extui %weightAt : i32 to i64
    %weightSlot = llvm.getelementptr %block[%weightOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %weights = llvm.load %weightSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %first = llvm.extractelement %weights[%c0 : i32] : vector<2xi32>
    %second = llvm.extractelement %weights[%c1 : i32] : vector<2xi32>
    %lowA = arith.andi %first, %mask : i32
    %lowB = arith.andi %second, %mask : i32
    %firstShifted = arith.shrui %first, %c4 : i32
    %secondShifted = arith.shrui %second, %c4 : i32
    %highA = arith.andi %firstShifted, %mask : i32
    %highB = arith.andi %secondShifted, %mask : i32
    %lowPart = arith.shli %pair, %c1 : i32
    %highPart = arith.addi %lowPart, %c1 : i32
    %lowScaleBits, %lowMinimumBits = func.call @strict_device_four_scale(%header, %lowPart) : (vector<4xi32>, i32) -> (i32, i32)
    %highScaleBits, %highMinimumBits = func.call @strict_device_four_scale(%header, %highPart) : (vector<4xi32>, i32) -> (i32, i32)
    %lowScaleFloat = arith.uitofp %lowScaleBits : i32 to f32
    %lowMinimumFloat = arith.uitofp %lowMinimumBits : i32 to f32
    %highScaleFloat = arith.uitofp %highScaleBits : i32 to f32
    %highMinimumFloat = arith.uitofp %highMinimumBits : i32 to f32
    %lowScale = arith.mulf %blockScale, %lowScaleFloat : f32
    %lowMinimum = arith.mulf %blockMinimum, %lowMinimumFloat : f32
    %highScale = arith.mulf %blockScale, %highScaleFloat : f32
    %highMinimum = arith.mulf %blockMinimum, %highMinimumFloat : f32
    %lowPartStart = arith.shli %pair, %c6 : i32
    %lowColumn = arith.addi %column, %lowPartStart : i32
    %lowOwn = arith.addi %lowColumn, %quadBytes : i32
    %lowScaleColumn = arith.addi %columnScales, %lowPart : i32
    cf.br ^group(%c0, %total : i32, vector<8xf32>)
  ^group(%group: i32, %groupTotal: vector<8xf32>):
    %moreGroups = arith.cmpi slt, %group, %c4 : i32
    cf.cond_br %moreGroups, ^groupCheck, ^pairNext
  ^groupCheck:
    %used = arith.cmpi slt, %group, %groups : i32
    cf.cond_br %used, ^groupBody, ^groupNext(%groupTotal : vector<8xf32>)
  ^groupBody:
    %groupStart = arith.shli %group, %c3 : i32
    %inputOffset = arith.addi %groupStart, %laneRow : i32
    %inputRow = func.call @strict_device_block_token(%place, %inputOffset) : (vector<4xi32>, i32) -> i32
    %lowInput = arith.addi %inputRow, %lowOwn : i32
    %lowInputOffset = arith.extui %lowInput : i32 to i64
    %lowSlot = llvm.getelementptr %bytes[%lowInputOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowInputs = llvm.load %lowSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %highSlot = llvm.getelementptr %lowSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %highInputs = llvm.load %highSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %lowFirstInput = llvm.extractelement %lowInputs[%c0 : i32] : vector<2xi32>
    %lowSecondInput = llvm.extractelement %lowInputs[%c1 : i32] : vector<2xi32>
    %highFirstInput = llvm.extractelement %highInputs[%c0 : i32] : vector<2xi32>
    %highSecondInput = llvm.extractelement %highInputs[%c1 : i32] : vector<2xi32>
    %lowHalf = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%lowA, %lowFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowHalf0 = llvm.extractvalue %lowHalf[0] : !llvm.struct<(i32, i32)>
    %lowHalf1 = llvm.extractvalue %lowHalf[1] : !llvm.struct<(i32, i32)>
    %lowDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%lowB, %lowSecondInput, %lowHalf0, %lowHalf1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highHalf = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%highA, %highFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highHalf0 = llvm.extractvalue %highHalf[0] : !llvm.struct<(i32, i32)>
    %highHalf1 = llvm.extractvalue %highHalf[1] : !llvm.struct<(i32, i32)>
    %highDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%highB, %highSecondInput, %highHalf0, %highHalf1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowCount = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%ones, %lowFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowCount0 = llvm.extractvalue %lowCount[0] : !llvm.struct<(i32, i32)>
    %lowCount1 = llvm.extractvalue %lowCount[1] : !llvm.struct<(i32, i32)>
    %lowTotal = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%ones, %lowSecondInput, %lowCount0, %lowCount1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highCount = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%ones, %highFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highCount0 = llvm.extractvalue %highCount[0] : !llvm.struct<(i32, i32)>
    %highCount1 = llvm.extractvalue %highCount[1] : !llvm.struct<(i32, i32)>
    %highTotal = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%ones, %highSecondInput, %highCount0, %highCount1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %firstAdded = func.call @strict_device_block_added(%groupTotal, %group, %c0, %lowDot, %highDot, %lowTotal, %highTotal, %lowScale, %lowMinimum, %highScale, %highMinimum, %scales, %place, %lowScaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    %bothAdded = func.call @strict_device_block_added(%firstAdded, %group, %c1, %lowDot, %highDot, %lowTotal, %highTotal, %lowScale, %lowMinimum, %highScale, %highMinimum, %scales, %place, %lowScaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    cf.br ^groupNext(%bothAdded : vector<8xf32>)
  ^groupNext(%nextTotal: vector<8xf32>):
    %groupFollowing = arith.addi %group, %c1 : i32
    llvm.br ^group(%groupFollowing, %nextTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^pairNext:
    %pairFollowing = arith.addi %pair, %c1 : i32
    llvm.br ^pair(%pairFollowing, %groupTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %total : vector<8xf32>
  }
  // adds result element (0 or 1) of the two parts of 32 numbers in strict_device_block_four to its slot: the integer
  // dots times the row scales, minus the row minimums times the token sums, times the scales of the token
  func.func private @strict_device_block_added(%sums: vector<8xf32>, %group: i32, %element: i32, %lowDot: !llvm.struct<(i32, i32)>, %highDot: !llvm.struct<(i32, i32)>, %lowTotal: !llvm.struct<(i32, i32)>, %highTotal: !llvm.struct<(i32, i32)>, %lowScale: f32, %lowMinimum: f32, %highScale: f32, %highMinimum: f32, %scales: !llvm.ptr, %place: vector<4xi32>, %lowScaleColumn: i32) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %isFirst = arith.cmpi eq, %element, %c0 : i32
    %lowDotFirst = llvm.extractvalue %lowDot[0] : !llvm.struct<(i32, i32)>
    %lowDotSecond = llvm.extractvalue %lowDot[1] : !llvm.struct<(i32, i32)>
    %highDotFirst = llvm.extractvalue %highDot[0] : !llvm.struct<(i32, i32)>
    %highDotSecond = llvm.extractvalue %highDot[1] : !llvm.struct<(i32, i32)>
    %lowTotalFirst = llvm.extractvalue %lowTotal[0] : !llvm.struct<(i32, i32)>
    %lowTotalSecond = llvm.extractvalue %lowTotal[1] : !llvm.struct<(i32, i32)>
    %highTotalFirst = llvm.extractvalue %highTotal[0] : !llvm.struct<(i32, i32)>
    %highTotalSecond = llvm.extractvalue %highTotal[1] : !llvm.struct<(i32, i32)>
    %lowDotInt = arith.select %isFirst, %lowDotFirst, %lowDotSecond : i32
    %highDotInt = arith.select %isFirst, %highDotFirst, %highDotSecond : i32
    %lowTotalInt = arith.select %isFirst, %lowTotalFirst, %lowTotalSecond : i32
    %highTotalInt = arith.select %isFirst, %highTotalFirst, %highTotalSecond : i32
    %scaleRow = func.call @strict_device_block_scale_row(%place, %group, %element) : (vector<4xi32>, i32, i32) -> i32
    %lowScaleIndex = arith.addi %scaleRow, %lowScaleColumn : i32
    %lowScaleOffset = arith.extui %lowScaleIndex : i32 to i64
    %lowTokenScaleSlot = llvm.getelementptr %scales[%lowScaleOffset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lowTokenScale = llvm.load %lowTokenScaleSlot invariant : !llvm.ptr -> f32
    %highTokenScaleSlot = llvm.getelementptr %lowTokenScaleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %highTokenScale = llvm.load %highTokenScaleSlot invariant : !llvm.ptr -> f32
    %lowDotFloat = arith.sitofp %lowDotInt : i32 to f32
    %highDotFloat = arith.sitofp %highDotInt : i32 to f32
    %lowTotalFloat = arith.sitofp %lowTotalInt : i32 to f32
    %highTotalFloat = arith.sitofp %highTotalInt : i32 to f32
    %lowScaled = arith.mulf %lowScale, %lowDotFloat : f32
    %lowShift = arith.mulf %lowMinimum, %lowTotalFloat : f32
    %lowPart = arith.subf %lowScaled, %lowShift : f32
    %highScaled = arith.mulf %highScale, %highDotFloat : f32
    %highShift = arith.mulf %highMinimum, %highTotalFloat : f32
    %highPart = arith.subf %highScaled, %highShift : f32
    %lowContribution = arith.mulf %lowPart, %lowTokenScale : f32
    %highContribution = arith.mulf %highPart, %highTokenScale : f32
    %contribution = arith.addf %lowContribution, %highContribution : f32
    %slotStart = arith.shli %group, %c1 : i32
    %slot = arith.addi %slotStart, %element : i32
    %old = llvm.extractelement %sums[%slot : i32] : vector<8xf32>
    %new = arith.addf %old, %contribution : f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<8xf32>
    return %updated : vector<8xf32>
  }
  // where the scales of the token of result element (0 or 1) of group start: token 2 * (lane % 4) + element
  func.func private @strict_device_block_scale_row(%place: vector<4xi32>, %group: i32, %element: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %lane = llvm.extractelement %place[%c3 : i32] : vector<4xi32>
    %quad = arith.andi %lane, %c3 : i32
    %pairToken = arith.shli %quad, %c1 : i32
    %groupStart = arith.shli %group, %c3 : i32
    %inGroup = arith.addi %pairToken, %element : i32
    %offset = arith.addi %groupStart, %inGroup : i32
    %start = func.call @strict_device_block_token(%place, %offset) : (vector<4xi32>, i32) -> i32
    %row = arith.shrui %start, %c5 : i32
    return %row : i32
  }
  // strict_device_block_four for a Q6_K block (208 bytes, blockScale is its 16 bit scale): an mma takes 16 numbers
  // that share one scale, a lane their numbers 4 * (lane % 4) to 4 * (lane % 4) + 3 (centered 6 bit numbers); a step
  // reads the low and high bits of 64 numbers once for the quarters odd and odd + 2 of a half
  func.func private @strict_device_block_six(%sums: vector<8xf32>, %block: !llvm.ptr, %blockScale: f32, %bytes: !llvm.ptr, %scales: !llvm.ptr, %place: vector<4xi32>, %column: i32, %groups: i32) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %highStart = llvm.mlir.constant(128 : i32) : i32
    %scaleStart = llvm.mlir.constant(192 : i32) : i32
    %lane = llvm.extractelement %place[%c3 : i32] : vector<4xi32>
    %quad = arith.andi %lane, %c3 : i32
    %laneRow = arith.shrui %lane, %c2 : i32
    %quadBytes = arith.shli %quad, %c2 : i32
    cf.br ^step(%c0, %sums : i32, vector<8xf32>)
  ^step(%step: i32, %total: vector<8xf32>):
    %moreSteps = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %moreSteps, ^stepBody, ^done
  ^stepBody:
    %half = arith.shrui %step, %c1 : i32
    %odd = arith.andi %step, %c1 : i32
    %lowHalf = arith.shli %half, %c6 : i32
    %oddBytes = arith.shli %odd, %c5 : i32
    %lowBase = arith.addi %lowHalf, %oddBytes : i32
    %lowAt = arith.addi %lowBase, %quadBytes : i32
    %lowOffset = arith.extui %lowAt : i32 to i64
    %lowSlot = llvm.getelementptr %block[%lowOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowFirst = llvm.load %lowSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %lowNextSlot = llvm.getelementptr %lowSlot[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %lowSecond = llvm.load %lowNextSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %highHalf = arith.shli %half, %c5 : i32
    %highBase = arith.addi %highStart, %highHalf : i32
    %highAt = arith.addi %highBase, %quadBytes : i32
    %highOffset = arith.extui %highAt : i32 to i64
    %highSlot = llvm.getelementptr %block[%highOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %highFirst = llvm.load %highSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %highNextSlot = llvm.getelementptr %highSlot[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %highSecond = llvm.load %highNextSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %upper = arith.addi %odd, %c2 : i32
    %lowerA = func.call @strict_device_six_numbers(%lowFirst, %highFirst, %odd) : (i32, i32, i32) -> i32
    %lowerB = func.call @strict_device_six_numbers(%lowSecond, %highSecond, %odd) : (i32, i32, i32) -> i32
    %upperA = func.call @strict_device_six_numbers(%lowFirst, %highFirst, %upper) : (i32, i32, i32) -> i32
    %upperB = func.call @strict_device_six_numbers(%lowSecond, %highSecond, %upper) : (i32, i32, i32) -> i32
    %halfScales = arith.shli %half, %c3 : i32
    %lowerScales = arith.shli %odd, %c1 : i32
    %upperScales = arith.shli %upper, %c1 : i32
    %lowerIndex = arith.addi %halfScales, %lowerScales : i32
    %upperIndex = arith.addi %halfScales, %upperScales : i32
    %lowerScale = func.call @strict_device_six_scales(%block, %scaleStart, %lowerIndex, %blockScale) : (!llvm.ptr, i32, i32, f32) -> vector<2xf32>
    %upperScale = func.call @strict_device_six_scales(%block, %scaleStart, %upperIndex, %blockScale) : (!llvm.ptr, i32, i32, f32) -> vector<2xf32>
    %quarterOffset = arith.shli %lowerIndex, %c4 : i32
    %lowerColumn = arith.addi %column, %quarterOffset : i32
    %upperColumn = arith.addi %lowerColumn, %c64 : i32
    cf.br ^group(%c0, %total : i32, vector<8xf32>)
  ^group(%group: i32, %groupTotal: vector<8xf32>):
    %moreGroups = arith.cmpi slt, %group, %c4 : i32
    cf.cond_br %moreGroups, ^groupCheck, ^stepNext
  ^groupCheck:
    %used = arith.cmpi slt, %group, %groups : i32
    cf.cond_br %used, ^groupBody, ^groupNext(%groupTotal : vector<8xf32>)
  ^groupBody:
    %groupStart = arith.shli %group, %c3 : i32
    %inputOffset = arith.addi %groupStart, %laneRow : i32
    %inputRow = func.call @strict_device_block_token(%place, %inputOffset) : (vector<4xi32>, i32) -> i32
    %lowerAdded = func.call @strict_device_six_quarter(%groupTotal, %group, %lowerA, %lowerB, %lowerScale, %bytes, %inputRow, %lowerColumn, %quadBytes, %scales, %place) : (vector<8xf32>, i32, i32, i32, vector<2xf32>, !llvm.ptr, i32, i32, i32, !llvm.ptr, vector<4xi32>) -> vector<8xf32>
    %upperAdded = func.call @strict_device_six_quarter(%lowerAdded, %group, %upperA, %upperB, %upperScale, %bytes, %inputRow, %upperColumn, %quadBytes, %scales, %place) : (vector<8xf32>, i32, i32, i32, vector<2xf32>, !llvm.ptr, i32, i32, i32, !llvm.ptr, vector<4xi32>) -> vector<8xf32>
    cf.br ^groupNext(%upperAdded : vector<8xf32>)
  ^groupNext(%nextTotal: vector<8xf32>):
    %groupFollowing = arith.addi %group, %c1 : i32
    llvm.br ^group(%groupFollowing, %nextTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^stepNext:
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %groupTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %total : vector<8xf32>
  }
  // the 4 centered 6 bit numbers of quarter (0 to 3) in a word of low bits and a word of high bits of a Q6_K block
  func.func private @strict_device_six_numbers(%low: i32, %high: i32, %quarter: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %lowNibbles = arith.constant 252645135 : i32
    %lowPairs = arith.constant 50529027 : i32
    %signs = arith.constant -2139062144 : i32
    %centers = arith.constant 538976288 : i32
    %upperQuarter = arith.shrui %quarter, %c1 : i32
    %nibbleShift = arith.shli %upperQuarter, %c2 : i32
    %lowMoved = arith.shrui %low, %nibbleShift : i32
    %nibbles = arith.andi %lowMoved, %lowNibbles : i32
    %pairShift = arith.shli %quarter, %c1 : i32
    %highMoved = arith.shrui %high, %pairShift : i32
    %pairs = arith.andi %highMoved, %lowPairs : i32
    %top = arith.shli %pairs, %c4 : i32
    %unsigned = arith.ori %nibbles, %top : i32
    %marked = arith.ori %unsigned, %signs : i32
    %lowered = arith.subi %marked, %centers : i32
    %numbers = arith.xori %lowered, %signs : i32
    return %numbers : i32
  }
  // the two scales of the 16 numbers of a quarter of a Q6_K block (at index of its scale bytes) times the block scale
  func.func private @strict_device_six_scales(%block: !llvm.ptr, %scaleStart: i32, %index: i32, %blockScale: f32) -> vector<2xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %at = arith.addi %scaleStart, %index : i32
    %offset = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %block[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %pair = llvm.load %slot invariant {alignment = 2 : i64} : !llvm.ptr -> vector<2xi8>
    %first = llvm.extractelement %pair[%c0 : i32] : vector<2xi8>
    %second = llvm.extractelement %pair[%c1 : i32] : vector<2xi8>
    %firstFloat = arith.sitofp %first : i8 to f32
    %secondFloat = arith.sitofp %second : i8 to f32
    %firstScale = arith.mulf %blockScale, %firstFloat : f32
    %secondScale = arith.mulf %blockScale, %secondFloat : f32
    %empty = llvm.mlir.undef : vector<2xf32>
    %withFirst = llvm.insertelement %firstScale, %empty[%c0 : i32] : vector<2xf32>
    %both = llvm.insertelement %secondScale, %withFirst[%c1 : i32] : vector<2xf32>
    return %both : vector<2xf32>
  }
  // the two mma of a quarter (32 numbers from column) of a Q6_K block for one group of tokens, each scaled by its row
  // scale, added up and scaled by the token scale of that quarter
  func.func private @strict_device_six_quarter(%sums: vector<8xf32>, %group: i32, %first: i32, %second: i32, %rowScales: vector<2xf32>, %bytes: !llvm.ptr, %inputRow: i32, %column: i32, %quadBytes: i32, %scales: !llvm.ptr, %place: vector<4xi32>) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %nothing = arith.constant 0 : i32
    %ownColumn = arith.addi %column, %quadBytes : i32
    %inputAt = arith.addi %inputRow, %ownColumn : i32
    %inputPosition = arith.extui %inputAt : i32 to i64
    %firstSlot = llvm.getelementptr %bytes[%inputPosition] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %firstInput = llvm.load %firstSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %secondSlot = llvm.getelementptr %firstSlot[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondInput = llvm.load %secondSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %firstDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%first, %firstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %secondDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%second, %secondInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %firstScale = llvm.extractelement %rowScales[%c0 : i32] : vector<2xf32>
    %secondScale = llvm.extractelement %rowScales[%c1 : i32] : vector<2xf32>
    %scaleColumn = arith.shrui %column, %c5 : i32
    %firstAdded = func.call @strict_device_six_added(%sums, %group, %c0, %firstDot, %secondDot, %firstScale, %secondScale, %scales, %place, %scaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    %bothAdded = func.call @strict_device_six_added(%firstAdded, %group, %c1, %firstDot, %secondDot, %firstScale, %secondScale, %scales, %place, %scaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    return %bothAdded : vector<8xf32>
  }
  // adds result element (0 or 1) of the two mma of strict_device_six_quarter to its slot
  func.func private @strict_device_six_added(%sums: vector<8xf32>, %group: i32, %element: i32, %firstDot: !llvm.struct<(i32, i32)>, %secondDot: !llvm.struct<(i32, i32)>, %firstScale: f32, %secondScale: f32, %scales: !llvm.ptr, %place: vector<4xi32>, %scaleColumn: i32) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %isFirst = arith.cmpi eq, %element, %c0 : i32
    %firstOne = llvm.extractvalue %firstDot[0] : !llvm.struct<(i32, i32)>
    %firstOther = llvm.extractvalue %firstDot[1] : !llvm.struct<(i32, i32)>
    %secondOne = llvm.extractvalue %secondDot[0] : !llvm.struct<(i32, i32)>
    %secondOther = llvm.extractvalue %secondDot[1] : !llvm.struct<(i32, i32)>
    %firstInt = arith.select %isFirst, %firstOne, %firstOther : i32
    %secondInt = arith.select %isFirst, %secondOne, %secondOther : i32
    %scaleRow = func.call @strict_device_block_scale_row(%place, %group, %element) : (vector<4xi32>, i32, i32) -> i32
    %scaleIndex = arith.addi %scaleRow, %scaleColumn : i32
    %scaleOffset = arith.extui %scaleIndex : i32 to i64
    %tokenScaleSlot = llvm.getelementptr %scales[%scaleOffset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %tokenScale = llvm.load %tokenScaleSlot invariant : !llvm.ptr -> f32
    %firstFloat = arith.sitofp %firstInt : i32 to f32
    %secondFloat = arith.sitofp %secondInt : i32 to f32
    %firstScaled = arith.mulf %firstFloat, %firstScale : f32
    %secondScaled = arith.mulf %secondFloat, %secondScale : f32
    %both = arith.addf %firstScaled, %secondScaled : f32
    %contribution = arith.mulf %both, %tokenScale : f32
    %slotStart = arith.shli %group, %c1 : i32
    %slot = arith.addi %slotStart, %element : i32
    %old = llvm.extractelement %sums[%slot : i32] : vector<8xf32>
    %new = arith.addf %old, %contribution : f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<8xf32>
    return %updated : vector<8xf32>
  }
  // strict_device_block_four for 256 numbers of an 8 bit row (rowScales has their 8 scales, one per 32): a lane reads
  // bytes 8 * (lane % 4) to 8 * (lane % 4) + 7 of each part of 32 numbers like for Q4_K
  func.func private @strict_device_block_eight(%sums: vector<8xf32>, %block: !llvm.ptr, %rowScales: !llvm.ptr, %bytes: !llvm.ptr, %scales: !llvm.ptr, %place: vector<4xi32>, %column: i32, %groups: i32) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %nothing = arith.constant 0 : i32
    %lane = llvm.extractelement %place[%c3 : i32] : vector<4xi32>
    %quad = arith.andi %lane, %c3 : i32
    %laneRow = arith.shrui %lane, %c2 : i32
    %quadBytes = arith.shli %quad, %c3 : i32
    %columnScales = arith.shrui %column, %c5 : i32
    cf.br ^part(%c0, %sums : i32, vector<8xf32>)
  ^part(%part: i32, %total: vector<8xf32>):
    %moreParts = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %moreParts, ^partBody, ^done
  ^partBody:
    %partBytes = arith.shli %part, %c5 : i32
    %weightAt = arith.addi %partBytes, %quadBytes : i32
    %weightOffset = arith.extui %weightAt : i32 to i64
    %weightSlot = llvm.getelementptr %block[%weightOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %weights = llvm.load %weightSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %first = llvm.extractelement %weights[%c0 : i32] : vector<2xi32>
    %second = llvm.extractelement %weights[%c1 : i32] : vector<2xi32>
    %partOffset = arith.extui %part : i32 to i64
    %rowScaleSlot = llvm.getelementptr %rowScales[%partOffset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %rowScale = llvm.load %rowScaleSlot invariant : !llvm.ptr -> f32
    %partColumn = arith.addi %column, %partBytes : i32
    %ownColumn = arith.addi %partColumn, %quadBytes : i32
    %scaleColumn = arith.addi %columnScales, %part : i32
    cf.br ^group(%c0, %total : i32, vector<8xf32>)
  ^group(%group: i32, %groupTotal: vector<8xf32>):
    %moreGroups = arith.cmpi slt, %group, %c4 : i32
    cf.cond_br %moreGroups, ^groupCheck, ^partNext
  ^groupCheck:
    %used = arith.cmpi slt, %group, %groups : i32
    cf.cond_br %used, ^groupBody, ^groupNext(%groupTotal : vector<8xf32>)
  ^groupBody:
    %groupStart = arith.shli %group, %c3 : i32
    %inputOffset = arith.addi %groupStart, %laneRow : i32
    %inputRow = func.call @strict_device_block_token(%place, %inputOffset) : (vector<4xi32>, i32) -> i32
    %inputAt = arith.addi %inputRow, %ownColumn : i32
    %inputPosition = arith.extui %inputAt : i32 to i64
    %inputSlot = llvm.getelementptr %bytes[%inputPosition] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %inputs = llvm.load %inputSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %firstInput = llvm.extractelement %inputs[%c0 : i32] : vector<2xi32>
    %secondInput = llvm.extractelement %inputs[%c1 : i32] : vector<2xi32>
    %firstDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%first, %firstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %secondDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%second, %secondInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %firstAdded = func.call @strict_device_six_added(%groupTotal, %group, %c0, %firstDot, %secondDot, %rowScale, %rowScale, %scales, %place, %scaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    %bothAdded = func.call @strict_device_six_added(%firstAdded, %group, %c1, %firstDot, %secondDot, %rowScale, %rowScale, %scales, %place, %scaleColumn) : (vector<8xf32>, i32, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, f32, f32, !llvm.ptr, vector<4xi32>, i32) -> vector<8xf32>
    cf.br ^groupNext(%bothAdded : vector<8xf32>)
  ^groupNext(%nextTotal: vector<8xf32>):
    %groupFollowing = arith.addi %group, %c1 : i32
    llvm.br ^group(%groupFollowing, %nextTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^partNext:
    %partFollowing = arith.addi %part, %c1 : i32
    llvm.br ^part(%partFollowing, %groupTotal : i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %total : vector<8xf32>
  }
  // `Batch.Sums` of 2 tokens or more: a block takes 8 rows and up to 32 tokens, each of its 8 warps an eighth of the
  // parts of 256 numbers of Q4_K, Q6_K or 8 bit rows on the 8 bit tensor cores, added up in shared memory (other rows go
  // row by row); the dot of row r and token t becomes number t * rows + r of the result
  func.func private @strict_device_tokens_blocks(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %blockBits = arith.constant 255 : i64
    %fourBlockBytes = arith.constant 144 : i64
    %sixBlockBytes = arith.constant 208 : i64
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %zeroSum = arith.constant 0.0 : f32
    %thread32 = nvvm.read.ptx.sreg.tid.x : i32
    %block32 = nvvm.read.ptx.sreg.ctaid.x : i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %warp = arith.shrui %thread, %five : i64
    %lane = arith.andi %thread, %thirtyOne : i64
    %quad = arith.andi %lane, %three : i64
    %laneRow = arith.shrui %lane, %two : i64
    %rowsGiven = arith.fptosi %rowsNumber : f64 to i64
    %rows = arith.maxsi %rowsGiven, %one : i64
    %width = arith.fptosi %widthNumber : f64 to i64
    %tokens = arith.divui %count, %rows : i64
    %tokensRounded = arith.addi %tokens, %seven : i64
    %groupsNeeded = arith.shrui %tokensRounded, %three : i64
    %groups = arith.minsi %groupsNeeded, %four : i64
    %blockTokens = arith.shli %groups, %three : i64
    %spareTokens = arith.subi %blockTokens, %one : i64
    %tokensUp = arith.addi %tokens, %spareTokens : i64
    %tokenBlocks = arith.divui %tokensUp, %blockTokens : i64
    %tokenBlock = arith.remui %block, %tokenBlocks : i64
    %rowBlock = arith.divui %block, %tokenBlocks : i64
    %thirtyTwo = arith.constant 32 : i64
    %prompt = arith.cmpi sgt, %tokens, %thirtyTwo : i64
    %rowGroupShift = arith.select %prompt, %three, %none : i64
    %splitShift = arith.subi %three, %rowGroupShift : i64
    %rowShift = arith.addi %rowGroupShift, %three : i64
    %blockRows = arith.shli %one, %rowShift : i64
    %rowFirst = arith.shli %rowBlock, %rowShift : i64
    %tokenFirst = arith.muli %tokenBlock, %blockTokens : i64
    %rowLast = arith.subi %rows, %one : i64
    %tokenLast = arith.subi %tokens, %one : i64
    %inside = arith.cmpi sle, %rowFirst, %rowLast : i64
    cf.cond_br %inside, ^check, ^exit
  ^check:
    %blockEnd = arith.addi %rowFirst, %blockRows : i64
    %blockLast = arith.subi %blockEnd, %one : i64
    %lastRow = arith.minsi %blockLast, %rowLast : i64
    %firstData, %firstBits, %firstTotal = func.call @strict_device_row(%rowsData, %rowFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %lastData, %lastBits, %lastTotal = func.call @strict_device_row(%rowsData, %lastRow) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %six = arith.constant 6 : i64
    %firstFour = arith.cmpi eq, %firstBits, %four : i64
    %firstSix = arith.cmpi eq, %firstBits, %six : i64
    %firstEight = arith.cmpi eq, %firstBits, %eight : i64
    %blockFirst = arith.ori %firstFour, %firstSix : i1
    %packedFirst = arith.ori %blockFirst, %firstEight : i1
    %lastSame = arith.cmpi eq, %lastBits, %firstBits : i64
    %columnsRest = arith.andi %columns, %blockBits : i64
    %wholeBlocks = arith.cmpi eq, %columnsRest, %none : i64
    %quantized = arith.cmpi eq, %inputBits, %eight : i64
    %widthRest = arith.andi %width, %thirtyOne : i64
    %widthAligned = arith.cmpi eq, %widthRest, %none : i64
    %bothPacked = arith.andi %packedFirst, %lastSame : i1
    %blocked = arith.andi %bothPacked, %wholeBlocks : i1
    %eightBits = arith.andi %quantized, %widthAligned : i1
    %tensors = arith.andi %blocked, %eightBits : i1
    cf.cond_br %tensors, ^start, ^pairs(%warp : i64)
  ^start:
    %scalesOffset = func.call @strict_device_data(%inputCount, %eight) : (i64, i64) -> i64
    %inputScales = llvm.getelementptr %input[%scalesOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowGroups = arith.shrui %blockRows, %three : i64
    %rowGroupCount = arith.subi %rowGroups, %one : i64
    %rowGroup = arith.andi %warp, %rowGroupCount : i64
    %split = arith.shrui %warp, %rowGroupShift : i64
    %splits = arith.shli %one, %splitShift : i64
    %rowGroupStart = arith.shli %rowGroup, %three : i64
    %groupRow = arith.addi %rowFirst, %rowGroupStart : i64
    %ownRow = arith.addi %groupRow, %laneRow : i64
    %row = arith.minsi %ownRow, %rowLast : i64
    %rowData, %rowBits, %rowTotal = func.call @strict_device_row(%rowsData, %row) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %pairToken = arith.shli %quad, %one : i64
    %rowDataBytes = func.call @strict_device_data(%columns, %rowBits) : (i64, i64) -> i64
    %blockScales = llvm.getelementptr %rowData[%rowDataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %tokenFirst32 = arith.trunci %tokenFirst : i64 to i32
    %tokenLast32 = arith.trunci %tokenLast : i64 to i32
    %width32 = arith.trunci %width : i64 to i32
    %lane32 = arith.trunci %lane : i64 to i32
    %groups32 = arith.trunci %groups : i64 to i32
    %place = func.call @strict_device_words(%tokenFirst32, %tokenLast32, %width32, %lane32) : (i32, i32, i32, i32) -> vector<4xi32>
    %blocksPerRow = arith.shrui %columns, %eight : i64
    %splitSpare = arith.subi %splits, %one : i64
    %blocksSpare = arith.addi %blocksPerRow, %splitSpare : i64
    %warpBlocks = arith.shrui %blocksSpare, %splitShift : i64
    %firstBlock = arith.muli %split, %warpBlocks : i64
    %blocksEnd = arith.addi %firstBlock, %warpBlocks : i64
    %lastBlock = arith.minsi %blocksEnd, %blocksPerRow : i64
    cf.br ^blockLoop(%firstBlock, %zeros : i64, vector<8xf32>)
  ^blockLoop(%index: i64, %sums: vector<8xf32>):
    %moreBlocks = arith.cmpi slt, %index, %lastBlock : i64
    cf.cond_br %moreBlocks, ^blockBody, ^finish(%sums : vector<8xf32>)
  ^blockBody:
    %column = arith.shli %index, %eight : i64
    %column32 = arith.trunci %column : i64 to i32
    cf.switch %rowBits : i64, [
      default: ^eightBlock,
      4: ^fourBlock,
      6: ^sixBlock
    ]
  ^eightBlock:
    %eightStart = llvm.getelementptr %rowData[%column] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %eightScaleStart = arith.shli %index, %three : i64
    %eightScales = llvm.getelementptr %blockScales[%eightScaleStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %eightAdded = func.call @strict_device_block_eight(%sums, %eightStart, %eightScales, %input, %inputScales, %place, %column32, %groups32) : (vector<8xf32>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, vector<4xi32>, i32, i32) -> vector<8xf32>
    cf.br ^blockNext(%eightAdded : vector<8xf32>)
  ^fourBlock:
    %blockOffset = arith.muli %index, %fourBlockBytes : i64
    %blockStart = llvm.getelementptr %rowData[%blockOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %fourAdded = func.call @strict_device_block_four(%sums, %blockStart, %input, %inputScales, %place, %column32, %groups32) : (vector<8xf32>, !llvm.ptr, !llvm.ptr, !llvm.ptr, vector<4xi32>, i32, i32) -> vector<8xf32>
    cf.br ^blockNext(%fourAdded : vector<8xf32>)
  ^sixBlock:
    %sixOffset = arith.muli %index, %sixBlockBytes : i64
    %sixStart = llvm.getelementptr %rowData[%sixOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halfSlot = llvm.getelementptr %blockScales[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfScale = llvm.load %halfSlot invariant : !llvm.ptr -> f16
    %sixScale = arith.extf %halfScale : f16 to f32
    %sixAdded = func.call @strict_device_block_six(%sums, %sixStart, %sixScale, %input, %inputScales, %place, %column32, %groups32) : (vector<8xf32>, !llvm.ptr, f32, !llvm.ptr, !llvm.ptr, vector<4xi32>, i32, i32) -> vector<8xf32>
    cf.br ^blockNext(%sixAdded : vector<8xf32>)
  ^blockNext(%added: vector<8xf32>):
    %nextIndex = arith.addi %index, %one : i64
    cf.br ^blockLoop(%nextIndex, %added : i64, vector<8xf32>)
  ^finish(%own: vector<8xf32>):
    %partials = llvm.mlir.addressof @strict_device_block_sums : !llvm.ptr<3>
    %warpStart = arith.shli %warp, %three : i64
    cf.br ^keep(%none : i64)
  ^keep(%keptSlot: i64):
    %moreKept = arith.cmpi slt, %keptSlot, %eight : i64
    cf.cond_br %moreKept, ^keepBody, ^kept
  ^keepBody:
    %keptValue = llvm.extractelement %own[%keptSlot : i64] : vector<8xf32>
    %keptRow = arith.addi %warpStart, %keptSlot : i64
    %keptStart = arith.shli %keptRow, %five : i64
    %keptAt = arith.addi %keptStart, %lane : i64
    %keptPointer = llvm.getelementptr %partials[0, %keptAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<2048 x f32>
    llvm.store %keptValue, %keptPointer : f32, !llvm.ptr<3>
    %keptNext = arith.addi %keptSlot, %one : i64
    llvm.br ^keep(%keptNext : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^kept:
    nvvm.barrier0
    cf.br ^slot(%split : i64)
  ^slot(%slot: i64):
    %moreSlots = arith.cmpi slt, %slot, %eight : i64
    cf.cond_br %moreSlots, ^gather(%none, %zeroSum : i64, f32), ^exit
  ^gather(%other: i64, %gathered: f32):
    %moreOthers = arith.cmpi slt, %other, %splits : i64
    cf.cond_br %moreOthers, ^gatherAdd, ^write
  ^gatherAdd:
    %otherSplit = arith.shli %other, %rowGroupShift : i64
    %otherWarp = arith.addi %otherSplit, %rowGroup : i64
    %otherStart = arith.shli %otherWarp, %three : i64
    %otherRow = arith.addi %otherStart, %slot : i64
    %otherRowStart = arith.shli %otherRow, %five : i64
    %otherAt = arith.addi %otherRowStart, %lane : i64
    %otherPointer = llvm.getelementptr %partials[0, %otherAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<2048 x f32>
    %otherValue = llvm.load %otherPointer : !llvm.ptr<3> -> f32
    %gatheredNext = arith.addf %gathered, %otherValue : f32
    %otherFollowing = arith.addi %other, %one : i64
    cf.br ^gather(%otherFollowing, %gatheredNext : i64, f32)
  ^write:
    %writeGroup = arith.shrui %slot, %one : i64
    %writeOdd = arith.andi %slot, %one : i64
    %writeGroupStart = arith.shli %writeGroup, %three : i64
    %inBlock = arith.addi %writeGroupStart, %pairToken : i64
    %withOdd = arith.addi %inBlock, %writeOdd : i64
    %writeToken = arith.addi %tokenFirst, %withOdd : i64
    %groupUsed = arith.cmpi slt, %writeGroup, %groups : i64
    %tokenUsed = arith.cmpi sle, %writeToken, %tokenLast : i64
    %rowUsed = arith.cmpi sle, %ownRow, %rowLast : i64
    %groupToken = arith.andi %groupUsed, %tokenUsed : i1
    %writing = arith.andi %groupToken, %rowUsed : i1
    cf.cond_br %writing, ^writeStore, ^slotNext
  ^writeStore:
    %writeTokenStart = arith.muli %writeToken, %rows : i64
    %writeElement = arith.addi %writeTokenStart, %ownRow : i64
    func.call @strict_device_tokens_store(%result, %wide, %writeElement, %gathered) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^slotNext
  ^slotNext:
    %slotFollowing = arith.addi %slot, %splits : i64
    cf.br ^slot(%slotFollowing : i64)
  ^pairs(%pair: i64):
    %pairCount = arith.muli %blockRows, %groups : i64
    %morePairs = arith.cmpi slt, %pair, %pairCount : i64
    cf.cond_br %morePairs, ^pairBody, ^exit
  ^pairBody:
    %pairRow = arith.divui %pair, %groups : i64
    %pairGroup = arith.remui %pair, %groups : i64
    %fallbackRow = arith.addi %rowFirst, %pairRow : i64
    %pairStart = arith.shli %pairGroup, %three : i64
    %first = arith.addi %tokenFirst, %pairStart : i64
    %rowValid = arith.cmpi sle, %fallbackRow, %rowLast : i64
    %firstValid = arith.cmpi sle, %first, %tokenLast : i64
    %valid = arith.andi %rowValid, %firstValid : i1
    cf.cond_br %valid, ^pairSums, ^pairNext
  ^pairSums:
    %chunkEnd = arith.addi %first, %seven : i64
    %last = arith.minsi %chunkEnd, %tokenLast : i64
    %pairResults = func.call @strict_device_tokens_sums(%failed, %rowsData, %fallbackRow, %columns, %input, %inputCount, %inputBits, %width, %first, %last, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %isFirstLane = arith.cmpi eq, %lane, %none : i64
    cf.cond_br %isFirstLane, ^pairWrite(%none : i64), ^pairNext
  ^pairWrite(%writeIndex: i64):
    %writes = arith.cmpi slt, %writeIndex, %eight : i64
    cf.cond_br %writes, ^pairToken, ^pairNext
  ^pairToken:
    %token = arith.addi %first, %writeIndex : i64
    %tokenWriting = arith.cmpi sle, %token, %last : i64
    %value = llvm.extractelement %pairResults[%writeIndex : i64] : vector<8xf32>
    %tokenStart = arith.muli %token, %rows : i64
    %element = arith.addi %tokenStart, %fallbackRow : i64
    cf.cond_br %tokenWriting, ^pairStore, ^pairStored
  ^pairStore:
    func.call @strict_device_tokens_store(%result, %wide, %element, %value) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^pairStored
  ^pairStored:
    %nextWrite = arith.addi %writeIndex, %one : i64
    llvm.br ^pairWrite(%nextWrite : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^pairNext:
    %pairFollowing = arith.addi %pair, %eight : i64
    cf.br ^pairs(%pairFollowing : i64)
  ^exit:
    return
  }
  func.func private @strict_device_tokens_store(%result: !llvm.ptr, %wide: i1, %element: i64, %value: f32) attributes {passthrough = ["alwaysinline"]} {
    cf.cond_br %wide, ^double, ^single
  ^double:
    %wideValue = arith.extf %value : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%element] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %wideValue, %doubleSlot : f64, !llvm.ptr
    return
  ^single:
    %singleSlot = llvm.getelementptr %result[%element] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %singleSlot : f32, !llvm.ptr
    return
  }
  // a tile of `Batch.Sums` for a prompt in shared memory: the Q4_K or Q6_K blocks of 64 rows (144 or 208 bytes each),
  // their scales (per 32 or 16 numbers) and minimums (Q4_K per 32) and the 16 bit scale of each Q6_K block, the 8 bit
  // numbers of 64 tokens for the same 256 columns (272 bytes apart so the lanes of an mma operand read different banks),
  // their scales and scaled sums per 32 numbers
  llvm.mlir.global internal @strict_device_tile_weights() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<13312 x i8>
  llvm.mlir.global internal @strict_device_tile_tokens() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<17408 x i8>
  llvm.mlir.global internal @strict_device_tile_row_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x f32>
  llvm.mlir.global internal @strict_device_tile_row_blocks() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<64 x f32>
  llvm.mlir.global internal @strict_device_tile_row_minimums() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  llvm.mlir.global internal @strict_device_tile_token_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  llvm.mlir.global internal @strict_device_tile_token_sums() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  // 16 bytes of shared memory at offset of a tile array
  func.func private @strict_device_tile_at(%array: !llvm.ptr<3>, %offset: i32) -> !llvm.ptr<3> attributes {passthrough = ["alwaysinline"]} {
    %offset64 = arith.extui %offset : i32 to i64
    %slot = llvm.getelementptr %array[%offset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    return %slot : !llvm.ptr<3>
  }
  // the parts of 16 bytes of block number superblock of 64 rows and of 64 tokens a thread copies into the tile (rows and
  // tokens after the last repeat the last): up to 4 parts of the rows, 4 of the tokens, 2 token scales and the 16 bit
  // scale of a Q6_K row, loaded into registers so the loads of the next block wait while the tile computes
  func.func private @strict_device_tile_fetch(%rowsData: !llvm.ptr, %rowFirst: i64, %rowLast: i64, %superblock: i64, %input: !llvm.ptr, %inputScales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %thread: i32, %rowBits: i64, %columns: i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %five = arith.constant 5 : i64
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    %fourBlockBytes = arith.constant 144 : i64
    %sixBlockBytes = arith.constant 208 : i64
    %blockBytes = arith.select %isFour, %fourBlockBytes, %sixBlockBytes : i64
    %c9 = llvm.mlir.constant(9 : i32) : i32
    %c13 = llvm.mlir.constant(13 : i32) : i32
    %c576 = llvm.mlir.constant(576 : i32) : i32
    %c832 = llvm.mlir.constant(832 : i32) : i32
    %rowParts = arith.select %isFour, %c9, %c13 : i32
    %weightParts = arith.select %isFour, %c576, %c832 : i32
    %blockNumbers = arith.constant 256 : i64
    %blockOffset = arith.muli %superblock, %blockBytes : i64
    %columnOffset = arith.muli %superblock, %blockNumbers : i64
    %zeroWords = arith.constant dense<0> : vector<16xi32>
    %zeroScales = arith.constant dense<0.0> : vector<4xf32>
    cf.br ^weight(%c0, %zeroWords : i32, vector<16xi32>)
  ^weight(%step: i32, %weights: vector<16xi32>):
    %moreSteps = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %moreSteps, ^weightCheck, ^tokens
  ^weightCheck:
    %stepStart = arith.muli %step, %c256 : i32
    %weightPart = arith.addi %thread, %stepStart : i32
    %usedPart = arith.cmpi slt, %weightPart, %weightParts : i32
    cf.cond_br %usedPart, ^weightBody, ^weightNext(%weights : vector<16xi32>)
  ^weightBody:
    %row = arith.divui %weightPart, %rowParts : i32
    %rowPart = arith.remui %weightPart, %rowParts : i32
    %row64 = arith.extui %row : i32 to i64
    %ownRow = arith.addi %rowFirst, %row64 : i64
    %rowIndex = arith.minsi %ownRow, %rowLast : i64
    %rowData, %ownBits, %rowTotal = func.call @strict_device_row(%rowsData, %rowIndex) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %rowPartBytes = arith.muli %rowPart, %c16 : i32
    %rowPartBytes64 = arith.extui %rowPartBytes : i32 to i64
    %sourceOffset = arith.addi %blockOffset, %rowPartBytes64 : i64
    %source = llvm.getelementptr %rowData[%sourceOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %words = llvm.load %source invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %withWords = func.call @strict_device_tile_placed(%weights, %step, %words) : (vector<16xi32>, i32, vector<4xi32>) -> vector<16xi32>
    cf.br ^weightNext(%withWords : vector<16xi32>)
  ^weightNext(%nextWeights: vector<16xi32>):
    %stepNext = arith.addi %step, %c1 : i32
    llvm.br ^weight(%stepNext, %nextWeights : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tokens:
    cf.br ^token(%c0, %zeroWords : i32, vector<16xi32>)
  ^token(%tokenStep: i32, %tokenWords: vector<16xi32>):
    %moreTokens = arith.cmpi slt, %tokenStep, %c4 : i32
    cf.cond_br %moreTokens, ^tokenBody, ^scales
  ^tokenBody:
    %tokenStepStart = arith.muli %tokenStep, %c256 : i32
    %tokenPart = arith.addi %thread, %tokenStepStart : i32
    %token = arith.shrui %tokenPart, %c4 : i32
    %tokenChunk = arith.andi %tokenPart, %c15 : i32
    %token64 = arith.extui %token : i32 to i64
    %ownToken = arith.addi %tokenFirst, %token64 : i64
    %tokenIndex = arith.minsi %ownToken, %tokenLast : i64
    %tokenStart = arith.muli %tokenIndex, %width : i64
    %tokenColumn = arith.addi %tokenStart, %columnOffset : i64
    %tokenChunkBytes = arith.muli %tokenChunk, %c16 : i32
    %tokenChunkBytes64 = arith.extui %tokenChunkBytes : i32 to i64
    %tokenOffset = arith.addi %tokenColumn, %tokenChunkBytes64 : i64
    %tokenSource = llvm.getelementptr %input[%tokenOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %numbers = llvm.load %tokenSource invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %withNumbers = func.call @strict_device_tile_placed(%tokenWords, %tokenStep, %numbers) : (vector<16xi32>, i32, vector<4xi32>) -> vector<16xi32>
    %tokenStepNext = arith.addi %tokenStep, %c1 : i32
    llvm.br ^token(%tokenStepNext, %withNumbers : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scales:
    cf.br ^scale(%c0, %zeroScales : i32, vector<4xf32>)
  ^scale(%scaleStep: i32, %scales: vector<4xf32>):
    %moreScales = arith.cmpi slt, %scaleStep, %c2 : i32
    cf.cond_br %moreScales, ^scaleBody, ^blockScale
  ^scaleBody:
    %scaleStepStart = arith.muli %scaleStep, %c256 : i32
    %scalePart = arith.addi %thread, %scaleStepStart : i32
    %scaleSubFull = arith.remui %scalePart, %c8 : i32
    %scaleToken = arith.divui %scalePart, %c8 : i32
    %scaleToken64 = arith.extui %scaleToken : i32 to i64
    %scaleOwnToken = arith.addi %tokenFirst, %scaleToken64 : i64
    %scaleTokenIndex = arith.minsi %scaleOwnToken, %tokenLast : i64
    %scaleTokenStart = arith.muli %scaleTokenIndex, %width : i64
    %scaleColumn = arith.addi %scaleTokenStart, %columnOffset : i64
    %scaleBlock = arith.shrui %scaleColumn, %five : i64
    %scaleSub64 = arith.extui %scaleSubFull : i32 to i64
    %scaleIndex = arith.addi %scaleBlock, %scaleSub64 : i64
    %scaleSource = llvm.getelementptr %inputScales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %scaleValue = llvm.load %scaleSource invariant : !llvm.ptr -> f32
    %withScale = llvm.insertelement %scaleValue, %scales[%scaleStep : i32] : vector<4xf32>
    %scaleStepNext = arith.addi %scaleStep, %c1 : i32
    llvm.br ^scale(%scaleStepNext, %withScale : i32, vector<4xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^blockScale:
    %scaleRow = arith.cmpi slt, %thread, %c64 : i32
    %isSix = arith.cmpi ne, %rowBits, %fourBits : i64
    %sixScale = arith.andi %scaleRow, %isSix : i1
    cf.cond_br %sixScale, ^blockScaleBody, ^done(%scales : vector<4xf32>)
  ^blockScaleBody:
    %threadRow = arith.extui %thread : i32 to i64
    %ownScaleRow = arith.addi %rowFirst, %threadRow : i64
    %scaleRowIndex = arith.minsi %ownScaleRow, %rowLast : i64
    %scaleData, %scaleBits, %scaleTotal = func.call @strict_device_row(%rowsData, %scaleRowIndex) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %scaleDataBytes = func.call @strict_device_data(%columns, %rowBits) : (i64, i64) -> i64
    %halves = llvm.getelementptr %scaleData[%scaleDataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halfSlot = llvm.getelementptr %halves[%superblock] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %half = llvm.load %halfSlot invariant : !llvm.ptr -> f16
    %blockScaleValue = arith.extf %half : f16 to f32
    %withBlockScale = llvm.insertelement %blockScaleValue, %scales[%c2 : i32] : vector<4xf32>
    cf.br ^done(%withBlockScale : vector<4xf32>)
  ^done(%fetchedScales: vector<4xf32>):
    return %weights, %tokenWords, %fetchedScales : vector<16xi32>, vector<16xi32>, vector<4xf32>
  }
  // words with the 4 words of part number step (0 to 3)
  func.func private @strict_device_tile_placed(%words: vector<16xi32>, %step: i32, %part: vector<4xi32>) -> vector<16xi32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %first = arith.shli %step, %c2 : i32
    %second = arith.addi %first, %c1 : i32
    %third = arith.addi %first, %c2 : i32
    %fourth = arith.addi %first, %c3 : i32
    %word0 = llvm.extractelement %part[%c0 : i32] : vector<4xi32>
    %word1 = llvm.extractelement %part[%c1 : i32] : vector<4xi32>
    %word2 = llvm.extractelement %part[%c2 : i32] : vector<4xi32>
    %word3 = llvm.extractelement %part[%c3 : i32] : vector<4xi32>
    %with0 = llvm.insertelement %word0, %words[%first : i32] : vector<16xi32>
    %with1 = llvm.insertelement %word1, %with0[%second : i32] : vector<16xi32>
    %with2 = llvm.insertelement %word2, %with1[%third : i32] : vector<16xi32>
    %with3 = llvm.insertelement %word3, %with2[%fourth : i32] : vector<16xi32>
    return %with3 : vector<16xi32>
  }
  // the 4 words of part number step (0 to 3) of words
  func.func private @strict_device_tile_part(%words: vector<16xi32>, %step: i32) -> vector<4xi32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %first = arith.shli %step, %c2 : i32
    %second = arith.addi %first, %c1 : i32
    %third = arith.addi %first, %c2 : i32
    %fourth = arith.addi %first, %c3 : i32
    %word0 = llvm.extractelement %words[%first : i32] : vector<16xi32>
    %word1 = llvm.extractelement %words[%second : i32] : vector<16xi32>
    %word2 = llvm.extractelement %words[%third : i32] : vector<16xi32>
    %word3 = llvm.extractelement %words[%fourth : i32] : vector<16xi32>
    %empty = llvm.mlir.undef : vector<4xi32>
    %with0 = llvm.insertelement %word0, %empty[%c0 : i32] : vector<4xi32>
    %with1 = llvm.insertelement %word1, %with0[%c1 : i32] : vector<4xi32>
    %with2 = llvm.insertelement %word2, %with1[%c2 : i32] : vector<4xi32>
    %with3 = llvm.insertelement %word3, %with2[%c3 : i32] : vector<4xi32>
    return %with3 : vector<4xi32>
  }
  // the parts strict_device_tile_fetch loaded into the tile
  func.func private @strict_device_tile_put(%weights: vector<16xi32>, %tokenWords: vector<16xi32>, %scales: vector<4xf32>, %thread: i32, %rowBits: i64) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c9 = llvm.mlir.constant(9 : i32) : i32
    %c13 = llvm.mlir.constant(13 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c208 = llvm.mlir.constant(208 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %c576 = llvm.mlir.constant(576 : i32) : i32
    %c832 = llvm.mlir.constant(832 : i32) : i32
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    %rowParts = arith.select %isFour, %c9, %c13 : i32
    %rowBytes = arith.select %isFour, %c144, %c208 : i32
    %weightParts = arith.select %isFour, %c576, %c832 : i32
    %weightTile = llvm.mlir.addressof @strict_device_tile_weights : !llvm.ptr<3>
    %tokenTile = llvm.mlir.addressof @strict_device_tile_tokens : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %rowBlocks = llvm.mlir.addressof @strict_device_tile_row_blocks : !llvm.ptr<3>
    cf.br ^weight(%c0 : i32)
  ^weight(%step: i32):
    %moreSteps = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %moreSteps, ^weightCheck, ^tokens
  ^weightCheck:
    %stepStart = arith.muli %step, %c256 : i32
    %weightPart = arith.addi %thread, %stepStart : i32
    %usedPart = arith.cmpi slt, %weightPart, %weightParts : i32
    cf.cond_br %usedPart, ^weightBody, ^weightNext
  ^weightBody:
    %row = arith.divui %weightPart, %rowParts : i32
    %rowPart = arith.remui %weightPart, %rowParts : i32
    %rowStart = arith.muli %row, %rowBytes : i32
    %rowPartBytes = arith.muli %rowPart, %c16 : i32
    %weightAt = arith.addi %rowStart, %rowPartBytes : i32
    %weightSlot = func.call @strict_device_tile_at(%weightTile, %weightAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %words = func.call @strict_device_tile_part(%weights, %step) : (vector<16xi32>, i32) -> vector<4xi32>
    llvm.store %words, %weightSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.br ^weightNext
  ^weightNext:
    %stepNext = arith.addi %step, %c1 : i32
    llvm.br ^weight(%stepNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tokens:
    cf.br ^token(%c0 : i32)
  ^token(%tokenStep: i32):
    %moreTokens = arith.cmpi slt, %tokenStep, %c4 : i32
    cf.cond_br %moreTokens, ^tokenBody, ^scales
  ^tokenBody:
    %tokenStepStart = arith.muli %tokenStep, %c256 : i32
    %tokenPart = arith.addi %thread, %tokenStepStart : i32
    %token = arith.shrui %tokenPart, %c4 : i32
    %tokenChunk = arith.andi %tokenPart, %c15 : i32
    %tokenRowStart = arith.muli %token, %c272 : i32
    %tokenChunkBytes = arith.muli %tokenChunk, %c16 : i32
    %tokenAt = arith.addi %tokenRowStart, %tokenChunkBytes : i32
    %tokenSlot = func.call @strict_device_tile_at(%tokenTile, %tokenAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %numbers = func.call @strict_device_tile_part(%tokenWords, %tokenStep) : (vector<16xi32>, i32) -> vector<4xi32>
    llvm.store %numbers, %tokenSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %tokenStepNext = arith.addi %tokenStep, %c1 : i32
    llvm.br ^token(%tokenStepNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scales:
    %firstScale = llvm.extractelement %scales[%c0 : i32] : vector<4xf32>
    %secondScale = llvm.extractelement %scales[%c1 : i32] : vector<4xf32>
    %first64 = arith.extui %thread : i32 to i64
    %secondPart = arith.addi %thread, %c256 : i32
    %second64 = arith.extui %secondPart : i32 to i64
    %firstSlot = llvm.getelementptr %tokenScales[%first64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %firstScale, %firstSlot : f32, !llvm.ptr<3>
    %secondSlot = llvm.getelementptr %tokenScales[%second64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %secondScale, %secondSlot : f32, !llvm.ptr<3>
    %scaleRow = arith.cmpi slt, %thread, %c64 : i32
    %isSix = arith.cmpi ne, %rowBits, %fourBits : i64
    %sixScale = arith.andi %scaleRow, %isSix : i1
    cf.cond_br %sixScale, ^blockScale, ^done
  ^blockScale:
    %blockScaleValue = llvm.extractelement %scales[%c2 : i32] : vector<4xf32>
    %blockSlot = llvm.getelementptr %rowBlocks[%first64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %blockScaleValue, %blockSlot : f32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // the scale and minimum of each 32 numbers of the 64 rows (from the Q4_K headers in the tile) and the scaled sum of each
  // 32 numbers of the 64 tokens, after the tile is staged
  func.func private @strict_device_tile_prepare(%thread: i32, %rowBits: i64) attributes {passthrough = ["alwaysinline"]} {
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    cf.cond_br %isFour, ^four, ^six
  ^six:
    func.call @strict_device_tile_six_scales(%thread) : (i32) -> ()
    return
  ^four:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %ones = arith.constant 16843009 : i32
    %weights = llvm.mlir.addressof @strict_device_tile_weights : !llvm.ptr<3>
    %tokens = llvm.mlir.addressof @strict_device_tile_tokens : !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowMinimums = llvm.mlir.addressof @strict_device_tile_row_minimums : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %tokenSums = llvm.mlir.addressof @strict_device_tile_token_sums : !llvm.ptr<3>
    cf.br ^part(%thread : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c512 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %owner = arith.divui %part, %c8 : i32
    %sub = arith.remui %part, %c8 : i32
    %part64 = arith.extui %part : i32 to i64
    %headerAt = arith.muli %owner, %c144 : i32
    %headerSlot = func.call @strict_device_tile_at(%weights, %headerAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %header = llvm.load %headerSlot {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xi32>
    %headerWord = llvm.extractelement %header[%c0 : i32] : vector<4xi32>
    %halves = llvm.bitcast %headerWord : i32 to vector<2xf16>
    %halfScale = llvm.extractelement %halves[%c0 : i32] : vector<2xf16>
    %halfMinimum = llvm.extractelement %halves[%c1 : i32] : vector<2xf16>
    %blockScale = arith.extf %halfScale : f16 to f32
    %blockMinimum = arith.extf %halfMinimum : f16 to f32
    %scaleBits, %minimumBits = func.call @strict_device_four_scale(%header, %sub) : (vector<4xi32>, i32) -> (i32, i32)
    %scaleFloat = arith.uitofp %scaleBits : i32 to f32
    %minimumFloat = arith.uitofp %minimumBits : i32 to f32
    %rowScale = arith.mulf %blockScale, %scaleFloat : f32
    %rowMinimum = arith.mulf %blockMinimum, %minimumFloat : f32
    %rowScaleSlot = llvm.getelementptr %rowScales[%part64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %rowScale, %rowScaleSlot : f32, !llvm.ptr<3>
    %rowMinimumSlot = llvm.getelementptr %rowMinimums[0, %part64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<512 x f32>
    llvm.store %rowMinimum, %rowMinimumSlot : f32, !llvm.ptr<3>
    %tokenRow = arith.muli %owner, %c272 : i32
    %subBytes = arith.muli %sub, %c32 : i32
    %tokenAt = arith.addi %tokenRow, %subBytes : i32
    %tokenSlot = func.call @strict_device_tile_at(%tokens, %tokenAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %tokenWords = llvm.load %tokenSlot {alignment = 16 : i64} : !llvm.ptr<3> -> vector<8xi32>
    cf.br ^word(%c0, %c0 : i32, i32)
  ^word(%word: i32, %sum: i32):
    %moreWords = arith.cmpi slt, %word, %c8 : i32
    cf.cond_br %moreWords, ^wordBody, ^summed
  ^wordBody:
    %numbers = llvm.extractelement %tokenWords[%word : i32] : vector<8xi32>
    %added = llvm.call_intrinsic "llvm.nvvm.idp4a.s.s"(%numbers, %ones, %sum) : (i32, i32, i32) -> i32
    %wordNext = arith.addi %word, %c1 : i32
    llvm.br ^word(%wordNext, %added : i32, i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^summed:
    %sumFloat = arith.sitofp %sum : i32 to f32
    %tokenScaleSlot = llvm.getelementptr %tokenScales[0, %part64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<512 x f32>
    %tokenScale = llvm.load %tokenScaleSlot : !llvm.ptr<3> -> f32
    %tokenSum = arith.mulf %tokenScale, %sumFloat : f32
    %tokenSumSlot = llvm.getelementptr %tokenSums[0, %part64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<512 x f32>
    llvm.store %tokenSum, %tokenSumSlot : f32, !llvm.ptr<3>
    %partNext = arith.addi %part, %c256 : i32
    cf.br ^part(%partNext : i32)
  ^done:
    return
  }
  // the scale of each 16 numbers of the 64 rows of a Q6_K tile (its 8 bit scales times the 16 bit scale of the block)
  func.func private @strict_device_tile_six_scales(%thread: i32) attributes {passthrough = ["alwaysinline"]} {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %c192 = llvm.mlir.constant(192 : i32) : i32
    %c208 = llvm.mlir.constant(208 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %weights = llvm.mlir.addressof @strict_device_tile_weights : !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowBlocks = llvm.mlir.addressof @strict_device_tile_row_blocks : !llvm.ptr<3>
    cf.br ^part(%thread : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c1024 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %row = arith.shrui %part, %c4 : i32
    %index = arith.andi %part, %c15 : i32
    %rowStart = arith.muli %row, %c208 : i32
    %scalesStart = arith.addi %rowStart, %c192 : i32
    %scaleAt = arith.addi %scalesStart, %index : i32
    %scaleSlot = func.call @strict_device_tile_at(%weights, %scaleAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %scaleByte = llvm.load %scaleSlot : !llvm.ptr<3> -> i8
    %scaleNumber = arith.sitofp %scaleByte : i8 to f32
    %row64 = arith.extui %row : i32 to i64
    %blockSlot = llvm.getelementptr %rowBlocks[%row64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %blockScale = llvm.load %blockSlot : !llvm.ptr<3> -> f32
    %scale = arith.mulf %blockScale, %scaleNumber : f32
    %part64 = arith.extui %part : i32 to i64
    %rowScaleSlot = llvm.getelementptr %rowScales[%part64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %scale, %rowScaleSlot : f32, !llvm.ptr<3>
    %partNext = arith.addi %part, %c256 : i32
    cf.br ^part(%partNext : i32)
  ^done:
    return
  }
  // the 4 centered 6 bit numbers of row (in a Q6_K tile) for the numbers 16 * half + at to + 3 of the 32 numbers sub
  func.func private @strict_device_tile_six(%row: i32, %sub: i32, %half: i32, %at: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %c208 = llvm.mlir.constant(208 : i32) : i32
    %weights = llvm.mlir.addressof @strict_device_tile_weights : !llvm.ptr<3>
    %blockHalf = arith.shrui %sub, %c2 : i32
    %quarter = arith.andi %sub, %c3 : i32
    %odd = arith.andi %quarter, %c1 : i32
    %halfNumbers = arith.shli %half, %c4 : i32
    %offset = arith.addi %halfNumbers, %at : i32
    %rowStart = arith.muli %row, %c208 : i32
    %lowHalf = arith.shli %blockHalf, %c6 : i32
    %lowOdd = arith.shli %odd, %c5 : i32
    %lowStart = arith.addi %lowHalf, %lowOdd : i32
    %lowInBlock = arith.addi %lowStart, %offset : i32
    %lowAt = arith.addi %rowStart, %lowInBlock : i32
    %highHalf = arith.shli %blockHalf, %c5 : i32
    %highStart = arith.addi %highHalf, %c128 : i32
    %highInBlock = arith.addi %highStart, %offset : i32
    %highAt = arith.addi %rowStart, %highInBlock : i32
    %lowSlot = func.call @strict_device_tile_at(%weights, %lowAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %highSlot = func.call @strict_device_tile_at(%weights, %highAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %low = llvm.load %lowSlot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    %high = llvm.load %highSlot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    %numbers = func.call @strict_device_six_numbers(%low, %high, %quarter) : (i32, i32, i32) -> i32
    return %numbers : i32
  }
  // the scale of the 16 numbers half of the 32 numbers sub of row in a Q6_K tile
  func.func private @strict_device_tile_six_scale(%row: i32, %sub: i32, %half: i32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowStart = arith.shli %row, %c4 : i32
    %subStart = arith.shli %sub, %c1 : i32
    %inRow = arith.addi %subStart, %half : i32
    %index = arith.addi %rowStart, %inRow : i32
    %index64 = arith.extui %index : i32 to i64
    %slot = llvm.getelementptr %rowScales[%index64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %scale = llvm.load %slot : !llvm.ptr<3> -> f32
    return %scale : f32
  }
  // strict_device_tile_sub for a Q6_K tile: a tile of 8 rows by 8 tokens is an mma for each 16 numbers with its own
  // scale (the numbers are centered, no minimums)
  func.func private @strict_device_tile_six_sub(%sums: vector<16xf32>, %sub: i32, %warpRow: i32, %warpToken: i32, %laneRow: i32, %quad: i32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %nothing = arith.constant 0 : i32
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %quadBytes = arith.shli %quad, %c2 : i32
    %upperBytes = arith.addi %quadBytes, %c16 : i32
    %pairToken = arith.shli %quad, %c1 : i32
    cf.br ^tile(%c0, %sums : i32, vector<16xf32>)
  ^tile(%tile: i32, %current: vector<16xf32>):
    %more = arith.cmpi slt, %tile, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %rowTile = arith.shrui %tile, %c1 : i32
    %tokenTile = arith.andi %tile, %c1 : i32
    %rowTileStart = arith.muli %rowTile, %c8 : i32
    %rowBase = arith.addi %warpRow, %rowTileStart : i32
    %row = arith.addi %rowBase, %laneRow : i32
    %tokenTileStart = arith.muli %tokenTile, %c8 : i32
    %tokenBase = arith.addi %warpToken, %tokenTileStart : i32
    %operandToken = arith.addi %tokenBase, %laneRow : i32
    %firstToken = arith.addi %tokenBase, %pairToken : i32
    %secondToken = arith.addi %firstToken, %c1 : i32
    %lowWeights = func.call @strict_device_tile_six(%row, %sub, %c0, %quadBytes) : (i32, i32, i32, i32) -> i32
    %highWeights = func.call @strict_device_tile_six(%row, %sub, %c1, %quadBytes) : (i32, i32, i32, i32) -> i32
    %lowTokens = func.call @strict_device_tile_token(%operandToken, %sub, %quadBytes) : (i32, i32, i32) -> i32
    %highTokens = func.call @strict_device_tile_token(%operandToken, %sub, %upperBytes) : (i32, i32, i32) -> i32
    %lowDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%lowWeights, %lowTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highDot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%highWeights, %highTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowScale = func.call @strict_device_tile_six_scale(%row, %sub, %c0) : (i32, i32, i32) -> f32
    %highScale = func.call @strict_device_tile_six_scale(%row, %sub, %c1) : (i32, i32, i32) -> f32
    %firstScale = func.call @strict_device_tile_float(%tokenScales, %firstToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %secondScale = func.call @strict_device_tile_float(%tokenScales, %secondToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %slot = arith.shli %tile, %c1 : i32
    %with0 = func.call @strict_device_tile_six_added(%current, %slot, %lowDot, %highDot, %c0, %lowScale, %highScale, %firstScale) : (vector<16xf32>, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, i32, f32, f32, f32) -> vector<16xf32>
    %slot1 = arith.addi %slot, %c1 : i32
    %with1 = func.call @strict_device_tile_six_added(%with0, %slot1, %lowDot, %highDot, %c1, %lowScale, %highScale, %secondScale) : (vector<16xf32>, i32, !llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>, i32, f32, f32, f32) -> vector<16xf32>
    %tileNext = arith.addi %tile, %c1 : i32
    llvm.br ^tile(%tileNext, %with1 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %current : vector<16xf32>
  }
  // sums at slot plus the token scale times the two dots of element (0 or 1) times their row scales
  func.func private @strict_device_tile_six_added(%sums: vector<16xf32>, %slot: i32, %lowDot: !llvm.struct<(i32, i32)>, %highDot: !llvm.struct<(i32, i32)>, %element: i32, %lowScale: f32, %highScale: f32, %tokenScale: f32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %isFirst = arith.cmpi eq, %element, %c0 : i32
    %lowFirst = llvm.extractvalue %lowDot[0] : !llvm.struct<(i32, i32)>
    %lowSecond = llvm.extractvalue %lowDot[1] : !llvm.struct<(i32, i32)>
    %highFirst = llvm.extractvalue %highDot[0] : !llvm.struct<(i32, i32)>
    %highSecond = llvm.extractvalue %highDot[1] : !llvm.struct<(i32, i32)>
    %low = arith.select %isFirst, %lowFirst, %lowSecond : i32
    %high = arith.select %isFirst, %highFirst, %highSecond : i32
    %lowFloat = arith.sitofp %low : i32 to f32
    %highFloat = arith.sitofp %high : i32 to f32
    %lowPart = arith.mulf %lowFloat, %lowScale : f32
    %both = llvm.call_intrinsic "llvm.fma.f32"(%highFloat, %highScale, %lowPart) : (f32, f32, f32) -> f32
    %old = llvm.extractelement %sums[%slot : i32] : vector<16xf32>
    %new = llvm.call_intrinsic "llvm.fma.f32"(%both, %tokenScale, %old) : (f32, f32, f32) -> f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<16xf32>
    return %updated : vector<16xf32>
  }
  // the 4 bit numbers of row (in the tile) for 4 numbers from at of the 32 numbers sub as 8 bit numbers
  func.func private @strict_device_tile_four(%row: i32, %sub: i32, %at: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %mask = arith.constant 252645135 : i32
    %weights = llvm.mlir.addressof @strict_device_tile_weights : !llvm.ptr<3>
    %pair = arith.shrui %sub, %c1 : i32
    %pairBytes = arith.shli %pair, %c5 : i32
    %rowStart = arith.muli %row, %c144 : i32
    %numbersStart = arith.addi %rowStart, %c16 : i32
    %pairStart = arith.addi %numbersStart, %pairBytes : i32
    %wordAt = arith.addi %pairStart, %at : i32
    %slot = func.call @strict_device_tile_at(%weights, %wordAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %word = llvm.load %slot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    %high = arith.andi %sub, %c1 : i32
    %shift = arith.shli %high, %c2 : i32
    %shifted = arith.shrui %word, %shift : i32
    %numbers = arith.andi %shifted, %mask : i32
    return %numbers : i32
  }
  // 4 token numbers from at of the 32 numbers sub of token (in the tile)
  func.func private @strict_device_tile_token(%token: i32, %sub: i32, %at: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %tokens = llvm.mlir.addressof @strict_device_tile_tokens : !llvm.ptr<3>
    %tokenStart = arith.muli %token, %c272 : i32
    %subBytes = arith.shli %sub, %c5 : i32
    %subStart = arith.addi %tokenStart, %subBytes : i32
    %wordAt = arith.addi %subStart, %at : i32
    %slot = func.call @strict_device_tile_at(%tokens, %wordAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %word = llvm.load %slot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    return %word : i32
  }
  // a number of a float tile array (8 per row or token)
  func.func private @strict_device_tile_float(%array: !llvm.ptr<3>, %owner: i32, %sub: i32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %ownerStart = arith.shli %owner, %c3 : i32
    %index = arith.addi %ownerStart, %sub : i32
    %index64 = arith.extui %index : i32 to i64
    %slot = llvm.getelementptr %array[%index64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %value = llvm.load %slot : !llvm.ptr<3> -> f32
    return %value : f32
  }
  // the 32 numbers sub of the tile for the 32 rows and 16 tokens of a warp: 4 x 2 tiles of 8 rows by 8 tokens, each two
  // mma of 16 numbers (mma.m8n8k16), the integer sums scaled by row and token, minus the row minimums times the scaled
  // token sums; a lane has row lane / 4 and tokens 2 * (lane % 4) and the next of each tile
  func.func private @strict_device_tile_sub(%sums: vector<16xf32>, %sub: i32, %warpRow: i32, %warpToken: i32, %laneRow: i32, %quad: i32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %nothing = arith.constant 0 : i32
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowMinimums = llvm.mlir.addressof @strict_device_tile_row_minimums : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %tokenSums = llvm.mlir.addressof @strict_device_tile_token_sums : !llvm.ptr<3>
    %quadBytes = arith.shli %quad, %c2 : i32
    %upperBytes = arith.addi %quadBytes, %c16 : i32
    %pairToken = arith.shli %quad, %c1 : i32
    cf.br ^tile(%c0, %sums : i32, vector<16xf32>)
  ^tile(%tile: i32, %current: vector<16xf32>):
    %more = arith.cmpi slt, %tile, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %rowTile = arith.shrui %tile, %c1 : i32
    %tokenTile = arith.andi %tile, %c1 : i32
    %rowTileStart = arith.muli %rowTile, %c8 : i32
    %rowBase = arith.addi %warpRow, %rowTileStart : i32
    %row = arith.addi %rowBase, %laneRow : i32
    %tokenTileStart = arith.muli %tokenTile, %c8 : i32
    %tokenBase = arith.addi %warpToken, %tokenTileStart : i32
    %operandToken = arith.addi %tokenBase, %laneRow : i32
    %firstToken = arith.addi %tokenBase, %pairToken : i32
    %secondToken = arith.addi %firstToken, %c1 : i32
    %lowWeights = func.call @strict_device_tile_four(%row, %sub, %quadBytes) : (i32, i32, i32) -> i32
    %highWeights = func.call @strict_device_tile_four(%row, %sub, %upperBytes) : (i32, i32, i32) -> i32
    %lowTokens = func.call @strict_device_tile_token(%operandToken, %sub, %quadBytes) : (i32, i32, i32) -> i32
    %highTokens = func.call @strict_device_tile_token(%operandToken, %sub, %upperBytes) : (i32, i32, i32) -> i32
    %half = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%lowWeights, %lowTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %half0 = llvm.extractvalue %half[0] : !llvm.struct<(i32, i32)>
    %half1 = llvm.extractvalue %half[1] : !llvm.struct<(i32, i32)>
    %dot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%highWeights, %highTokens, %half0, %half1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %rowScale = func.call @strict_device_tile_float(%rowScales, %row, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %rowMinimum = func.call @strict_device_tile_float(%rowMinimums, %row, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %firstScale = func.call @strict_device_tile_float(%tokenScales, %firstToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %secondScale = func.call @strict_device_tile_float(%tokenScales, %secondToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %firstSum = func.call @strict_device_tile_float(%tokenSums, %firstToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %secondSum = func.call @strict_device_tile_float(%tokenSums, %secondToken, %sub) : (!llvm.ptr<3>, i32, i32) -> f32
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(i32, i32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(i32, i32)>
    %slot = arith.shli %tile, %c1 : i32
    %with0 = func.call @strict_device_tile_added(%current, %slot, %dot0, %rowScale, %firstScale, %rowMinimum, %firstSum) : (vector<16xf32>, i32, i32, f32, f32, f32, f32) -> vector<16xf32>
    %slot1 = arith.addi %slot, %c1 : i32
    %with1 = func.call @strict_device_tile_added(%with0, %slot1, %dot1, %rowScale, %secondScale, %rowMinimum, %secondSum) : (vector<16xf32>, i32, i32, f32, f32, f32, f32) -> vector<16xf32>
    %tileNext = arith.addi %tile, %c1 : i32
    llvm.br ^tile(%tileNext, %with1 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %current : vector<16xf32>
  }
  // sums at slot plus the integer dot times row and token scale minus the row minimum times the scaled token sum
  func.func private @strict_device_tile_added(%sums: vector<16xf32>, %slot: i32, %dot: i32, %rowScale: f32, %tokenScale: f32, %rowMinimum: f32, %tokenSum: f32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %dotFloat = arith.sitofp %dot : i32 to f32
    %scale = arith.mulf %rowScale, %tokenScale : f32
    %shift = arith.mulf %rowMinimum, %tokenSum : f32
    %old = llvm.extractelement %sums[%slot : i32] : vector<16xf32>
    %scaled = llvm.call_intrinsic "llvm.fma.f32"(%dotFloat, %scale, %old) : (f32, f32, f32) -> f32
    %new = arith.subf %scaled, %shift : f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<16xf32>
    return %updated : vector<16xf32>
  }
  // `Batch.Sums` of 64 tokens or more with Q4_K or Q6_K rows: a block of 8 warps takes 64 rows and 64 tokens, stages one block
  // of 256 columns of all of them in shared memory at a time and each warp computes 32 rows by 16 tokens of it on the 8
  // bit tensor cores; the dot of row r and token t becomes number t * rows + r of the result
  func.func private @strict_device_tiles_run(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %sixtyThree = arith.constant 63 : i64
    %sixtyFour = arith.constant 64 : i64
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    %block32 = nvvm.read.ptx.sreg.ctaid.x : i32
    %block = arith.extui %block32 : i32 to i64
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %quad = arith.andi %lane, %c3 : i32
    %laneRow = arith.shrui %lane, %c2 : i32
    %rowsGiven = arith.fptosi %rowsNumber : f64 to i64
    %rows = arith.maxsi %rowsGiven, %one : i64
    %width = arith.fptosi %widthNumber : f64 to i64
    %tokens = arith.divui %count, %rows : i64
    %tokensUp = arith.addi %tokens, %sixtyThree : i64
    %six = arith.constant 6 : i64
    %tokenTileCount = arith.shrui %tokensUp, %six : i64
    %tokenTile = arith.remui %block, %tokenTileCount : i64
    %rowTile = arith.divui %block, %tokenTileCount : i64
    %rowFirst = arith.muli %rowTile, %sixtyFour : i64
    %tokenFirst = arith.muli %tokenTile, %sixtyFour : i64
    %rowLast = arith.subi %rows, %one : i64
    %tokenLast = arith.subi %tokens, %one : i64
    %isQuantized = arith.cmpi eq, %inputBits, %eight : i64
    %thirtyOne = arith.constant 31 : i64
    %widthRest = arith.andi %width, %thirtyOne : i64
    %widthAligned = arith.cmpi eq, %widthRest, %none : i64
    %usable = arith.andi %isQuantized, %widthAligned : i1
    cf.cond_br %usable, ^start, ^unusable
  ^unusable:
    %isFirst = arith.cmpi eq, %thread, %c0 : i32
    cf.cond_br %isFirst, ^markFailed, ^exit
  ^markFailed:
    llvm.store %one, %failed : i64, !llvm.ptr
    cf.br ^exit
  ^start:
    %scalesOffset = func.call @strict_device_data(%inputCount, %eight) : (i64, i64) -> i64
    %inputScales = llvm.getelementptr %input[%scalesOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpTokenIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.muli %warpRowIndex, %c32 : i32
    %warpToken = arith.muli %warpTokenIndex, %c16 : i32
    %blocks = arith.shrui %columns, %eight : i64
    %firstData, %rowBits, %firstTotal = func.call @strict_device_row(%rowsData, %rowFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    %firstWeights, %firstTokens, %firstScales = func.call @strict_device_tile_fetch(%rowsData, %rowFirst, %rowLast, %none, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %rowBits, %columns) : (!llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i64, i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>)
    func.call @strict_device_tile_put(%firstWeights, %firstTokens, %firstScales, %thread, %rowBits) : (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i64) -> ()
    nvvm.barrier0
    func.call @strict_device_tile_prepare(%thread, %rowBits) : (i32, i64) -> ()
    nvvm.barrier0
    %lastBlock = arith.subi %blocks, %one : i64
    cf.br ^superblock(%none, %zeros : i64, vector<16xf32>)
  ^superblock(%superblock: i64, %sums: vector<16xf32>):
    %moreBlocks = arith.cmpi slt, %superblock, %blocks : i64
    cf.cond_br %moreBlocks, ^stage, ^write(%c0 : i32)
  ^stage:
    %nextBlock = arith.addi %superblock, %one : i64
    %fetchedBlock = arith.minsi %nextBlock, %lastBlock : i64
    %nextWeights, %nextTokens, %nextScales = func.call @strict_device_tile_fetch(%rowsData, %rowFirst, %rowLast, %fetchedBlock, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %rowBits, %columns) : (!llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i64, i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>)
    cf.br ^sub(%c0, %sums : i32, vector<16xf32>)
  ^sub(%sub: i32, %subSums: vector<16xf32>):
    %moreSubs = arith.cmpi slt, %sub, %c8 : i32
    cf.cond_br %moreSubs, ^subBody, ^superblockNext
  ^subBody:
    cf.cond_br %isFour, ^subFour, ^subSix
  ^subFour:
    %fourAdded = func.call @strict_device_tile_sub(%subSums, %sub, %warpRow, %warpToken, %laneRow, %quad) : (vector<16xf32>, i32, i32, i32, i32, i32) -> vector<16xf32>
    cf.br ^subNext(%fourAdded : vector<16xf32>)
  ^subSix:
    %sixAdded = func.call @strict_device_tile_six_sub(%subSums, %sub, %warpRow, %warpToken, %laneRow, %quad) : (vector<16xf32>, i32, i32, i32, i32, i32) -> vector<16xf32>
    cf.br ^subNext(%sixAdded : vector<16xf32>)
  ^subNext(%added: vector<16xf32>):
    %subFollowing = arith.addi %sub, %c1 : i32
    llvm.br ^sub(%subFollowing, %added : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^superblockNext:
    %hasNext = arith.cmpi slt, %nextBlock, %blocks : i64
    cf.cond_br %hasNext, ^put, ^superblock(%nextBlock, %subSums : i64, vector<16xf32>)
  ^put:
    nvvm.barrier0
    func.call @strict_device_tile_put(%nextWeights, %nextTokens, %nextScales, %thread, %rowBits) : (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i64) -> ()
    nvvm.barrier0
    func.call @strict_device_tile_prepare(%thread, %rowBits) : (i32, i64) -> ()
    nvvm.barrier0
    cf.br ^superblock(%nextBlock, %subSums : i64, vector<16xf32>)
  ^write(%slot: i32):
    %moreSlots = arith.cmpi slt, %slot, %c16 : i32
    cf.cond_br %moreSlots, ^writeBody, ^exit
  ^writeBody:
    %tile = arith.shrui %slot, %c1 : i32
    %odd = arith.andi %slot, %c1 : i32
    %rowTileIndex = arith.shrui %tile, %c1 : i32
    %tokenTileIndex = arith.andi %tile, %c1 : i32
    %rowTileStart = arith.muli %rowTileIndex, %c8 : i32
    %rowInTile = arith.addi %warpRow, %rowTileStart : i32
    %rowLocal = arith.addi %rowInTile, %laneRow : i32
    %tokenTileStart = arith.muli %tokenTileIndex, %c8 : i32
    %tokenInTile = arith.addi %warpToken, %tokenTileStart : i32
    %pairToken = arith.shli %quad, %c1 : i32
    %tokenWithPair = arith.addi %tokenInTile, %pairToken : i32
    %tokenLocal = arith.addi %tokenWithPair, %odd : i32
    %rowLocal64 = arith.extui %rowLocal : i32 to i64
    %tokenLocal64 = arith.extui %tokenLocal : i32 to i64
    %row = arith.addi %rowFirst, %rowLocal64 : i64
    %token = arith.addi %tokenFirst, %tokenLocal64 : i64
    %rowUsed = arith.cmpi sle, %row, %rowLast : i64
    %tokenUsed = arith.cmpi sle, %token, %tokenLast : i64
    %writing = arith.andi %rowUsed, %tokenUsed : i1
    cf.cond_br %writing, ^store, ^writeNext
  ^store:
    %value = llvm.extractelement %sums[%slot : i32] : vector<16xf32>
    %tokenStart = arith.muli %token, %rows : i64
    %resultIndex = arith.addi %tokenStart, %row : i64
    func.call @strict_device_tokens_store(%result, %wide, %resultIndex, %value) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^writeNext
  ^writeNext:
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  llvm.func @strict_device_tiles(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tiles_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_tiless(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tiles_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_tokens(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tokens_rows(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_blocks(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tokens_blocks(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_blockss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tokens_blocks(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_tokenss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tokens_rows(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    llvm.return
  }
