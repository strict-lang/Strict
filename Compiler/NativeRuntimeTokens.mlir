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
  // the first row of the token's rows: bases(token / 64), the chunks of tokens of all three kernels lie in one such
  // tile (8, 32 or 64 tokens from a multiple of them, 24 only below 64 tokens), 0 without bases
  func.func private @strict_device_token_base(%bases: !llvm.ptr, %count: i64, %bits: i64, %token: i64) -> i64 {
    %none = arith.constant 0 : i64
    %six = arith.constant 6 : i64
    %tile = arith.shrui %token, %six : i64
    %hasBases = arith.cmpi sgt, %count, %none : i64
    cf.cond_br %hasBases, ^read, ^none
  ^read:
    %value = func.call @strict_device_elements(%bases, %bits, %tile, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    %base = arith.fptosi %value : f32 to i64
    return %base : i64
  ^none:
    return %none : i64
  }
  func.func private @strict_device_tokens_rows(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %blockWidth = func.call @strict_device_block_size() : () -> i32
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
    %base = func.call @strict_device_token_base(%bases, %basesCount, %basesBits, %first) : (!llvm.ptr, i64, i64, i64) -> i64
    %basedRow = arith.addi %row, %base : i64
    %sums = func.call @strict_device_tokens_sums(%failed, %rowsData, %basedRow, %columns, %input, %inputCount, %inputBits, %width, %first, %last, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
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
    func.call @strict_device_tokens_store(%result, %wide, %element, %value, %row, %rows, %biases, %biasesCount, %biasesBits) : (!llvm.ptr, i1, i64, f32, i64, i64, !llvm.ptr, i64, i64) -> ()
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
    %lowHalf = func.call @strict_device_mma8(%lowA, %lowFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowHalf0 = llvm.extractvalue %lowHalf[0] : !llvm.struct<(i32, i32)>
    %lowHalf1 = llvm.extractvalue %lowHalf[1] : !llvm.struct<(i32, i32)>
    %lowDot = func.call @strict_device_mma8(%lowB, %lowSecondInput, %lowHalf0, %lowHalf1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highHalf = func.call @strict_device_mma8(%highA, %highFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highHalf0 = llvm.extractvalue %highHalf[0] : !llvm.struct<(i32, i32)>
    %highHalf1 = llvm.extractvalue %highHalf[1] : !llvm.struct<(i32, i32)>
    %highDot = func.call @strict_device_mma8(%highB, %highSecondInput, %highHalf0, %highHalf1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowCount = func.call @strict_device_mma8(%ones, %lowFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowCount0 = llvm.extractvalue %lowCount[0] : !llvm.struct<(i32, i32)>
    %lowCount1 = llvm.extractvalue %lowCount[1] : !llvm.struct<(i32, i32)>
    %lowTotal = func.call @strict_device_mma8(%ones, %lowSecondInput, %lowCount0, %lowCount1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highCount = func.call @strict_device_mma8(%ones, %highFirstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highCount0 = llvm.extractvalue %highCount[0] : !llvm.struct<(i32, i32)>
    %highCount1 = llvm.extractvalue %highCount[1] : !llvm.struct<(i32, i32)>
    %highTotal = func.call @strict_device_mma8(%ones, %highSecondInput, %highCount0, %highCount1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
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
    %firstDot = func.call @strict_device_mma8(%first, %firstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %secondDot = func.call @strict_device_mma8(%second, %secondInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
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
    %firstDot = func.call @strict_device_mma8(%first, %firstInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %secondDot = func.call @strict_device_mma8(%second, %secondInput, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
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
  func.func private @strict_device_tokens_blocks(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) {
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
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
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
    %base = func.call @strict_device_token_base(%bases, %basesCount, %basesBits, %tokenFirst) : (!llvm.ptr, i64, i64, i64) -> i64
    %basedFirst = arith.addi %rowFirst, %base : i64
    %basedLast = arith.addi %lastRow, %base : i64
    %firstData, %firstBits, %firstTotal = func.call @strict_device_row(%rowsData, %basedFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %lastData, %lastBits, %lastTotal = func.call @strict_device_row(%rowsData, %basedLast) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
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
    %basedRow = arith.addi %row, %base : i64
    %rowData, %rowBits, %rowTotal = func.call @strict_device_row(%rowsData, %basedRow) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
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
    func.call @strict_device_barrier() : () -> ()
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
    func.call @strict_device_tokens_store(%result, %wide, %writeElement, %gathered, %ownRow, %rows, %biases, %biasesCount, %biasesBits) : (!llvm.ptr, i1, i64, f32, i64, i64, !llvm.ptr, i64, i64) -> ()
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
    %basedFallback = arith.addi %fallbackRow, %base : i64
    %pairResults = func.call @strict_device_tokens_sums(%failed, %rowsData, %basedFallback, %columns, %input, %inputCount, %inputBits, %width, %first, %last, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
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
    func.call @strict_device_tokens_store(%result, %wide, %element, %value, %fallbackRow, %rows, %biases, %biasesCount, %biasesBits) : (!llvm.ptr, i1, i64, f32, i64, i64, !llvm.ptr, i64, i64) -> ()
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
  // number element of the result (row of rows), plus the bias of the row when there is one per row, the bias at element
  // when there are as many as results, otherwise at element modulo their count (`Batch.Sums` adds biases(index % length))
  func.func private @strict_device_tokens_store(%result: !llvm.ptr, %wide: i1, %element: i64, %sum: f32, %row: i64, %rows: i64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64) attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %hasBiases = arith.cmpi sgt, %biasesCount, %none : i64
    cf.cond_br %hasBiases, ^biased, ^stored(%sum : f32)
  ^biased:
    %perRow = arith.cmpi eq, %biasesCount, %rows : i64
    cf.cond_br %perRow, ^read(%row : i64), ^element
  ^element:
    %inside = arith.cmpi slt, %element, %biasesCount : i64
    cf.cond_br %inside, ^read(%element : i64), ^wrapped
  ^wrapped:
    %wrapped = arith.remui %element, %biasesCount : i64
    cf.br ^read(%wrapped : i64)
  ^read(%biasIndex: i64):
    %bias = func.call @strict_device_elements(%biases, %biasesBits, %biasIndex, %biasesCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %biasedSum = arith.addf %sum, %bias : f32
    cf.br ^stored(%biasedSum : f32)
  ^stored(%value: f32):
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
  // a tile of `Batch.Sums` for a prompt in shared memory: the Q4_K or Q6_K blocks of 128 rows (144 or 208 bytes each),
  // their scales (per 32 or 16 numbers) and minimums (Q4_K per 32) and the 16 bit scale of each Q6_K block, the 8 bit
  // numbers of 64 tokens for the same 256 columns (272 bytes apart so the lanes of an mma operand read different banks),
  // their scales and sums per 32 numbers
  // Q4_K rows unpacked to 8 bit numbers, 272 bytes per row like the tokens, Q6_K rows as they are (208 bytes)
  // the rows (128 of 272 bytes) and then the tokens (64 of 272 bytes) lie in the dynamic shared memory of a block
  // (strict_gpu_launch gives blocks of 512 threads 52224 bytes, more than the 48 KB of static shared memory)
  llvm.mlir.global external @strict_device_tile_dynamic() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  func.func private @strict_device_tile_weights() -> !llvm.ptr<3> attributes {passthrough = ["alwaysinline"]} {
    %dynamic = llvm.mlir.addressof @strict_device_tile_dynamic : !llvm.ptr<3>
    return %dynamic : !llvm.ptr<3>
  }
  func.func private @strict_device_tile_tokens() -> !llvm.ptr<3> attributes {passthrough = ["alwaysinline"]} {
    %dynamic = llvm.mlir.addressof @strict_device_tile_dynamic : !llvm.ptr<3>
    %rowsBytes = llvm.mlir.constant(34816 : i32) : i32
    %tokens = llvm.getelementptr %dynamic[%rowsBytes] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    return %tokens : !llvm.ptr<3>
  }
  // the 16 header bytes of each Q4_K row (16 bit scale and minimum, 6 bit scales and minimums of its 8 subs)
  llvm.mlir.global internal @strict_device_tile_headers() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<2048 x i8>
  llvm.mlir.global internal @strict_device_tile_row_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<2048 x f32>
  llvm.mlir.global internal @strict_device_tile_row_blocks() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<128 x f32>
  llvm.mlir.global internal @strict_device_tile_row_minimums() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x f32>
  llvm.mlir.global internal @strict_device_tile_token_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  // the 16 sums of each thread of the splits of a tile when its columns are split over several blocks (fewer tiles than
  // multiprocessors), the last block of a tile adds them up
  llvm.mlir.global internal @strict_device_tile_partials() {addr_space = 1 : i32, alignment = 16 : i64} : !llvm.array<2097152 x f32>
  llvm.mlir.global internal @strict_device_tile_counters(dense<0> : tensor<256xi32>) {addr_space = 1 : i32} : !llvm.array<256 x i32>
  llvm.mlir.global internal @strict_device_tile_last() {addr_space = 3 : i32} : i32
  // the 16 sums of thread in the partials of split number part
  func.func private @strict_device_tile_partial(%part: i32, %thread: i32) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %threads = llvm.mlir.constant(512 : i32) : i32
    %numbers = llvm.mlir.constant(16 : i32) : i32
    %partials = llvm.mlir.addressof @strict_device_tile_partials : !llvm.ptr<1>
    %start = llvm.addrspacecast %partials : !llvm.ptr<1> to !llvm.ptr
    %partStart = arith.muli %part, %threads : i32
    %threadIndex = arith.addi %partStart, %thread : i32
    %offset = arith.muli %threadIndex, %numbers : i32
    %offset64 = arith.extui %offset : i32 to i64
    %slot = llvm.getelementptr %start[%offset64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    return %slot : !llvm.ptr
  }
  llvm.mlir.global internal @strict_device_tile_token_sums() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  // 16 bytes of shared memory at offset of a tile array
  func.func private @strict_device_tile_at(%array: !llvm.ptr<3>, %offset: i32) -> !llvm.ptr<3> attributes {passthrough = ["alwaysinline"]} {
    %offset64 = arith.extui %offset : i32 to i64
    %slot = llvm.getelementptr %array[%offset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    return %slot : !llvm.ptr<3>
  }
  // the parts of 16 bytes of block number superblock of 128 rows and of 64 tokens a thread copies into the tile (rows and
  // tokens after the last repeat the last): up to 4 parts of the rows, 2 of the tokens, a token scale and the 16 bit
  // scale of a Q6_K row, loaded into registers so the loads of the next block wait while the tile computes; the rows
  // lie stride bytes apart from firstData (row rowFirst)
  func.func private @strict_device_tile_fetch(%firstData: !llvm.ptr, %stride: i64, %rowFirst: i64, %rowLast: i64, %superblock: i64, %input: !llvm.ptr, %inputScales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %thread: i32, %rowBits: i64, %columns: i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>) attributes {passthrough = ["alwaysinline"]} {
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
    %c1152 = llvm.mlir.constant(1152 : i32) : i32
    %c1664 = llvm.mlir.constant(1664 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %rowParts = arith.select %isFour, %c9, %c13 : i32
    %weightParts = arith.select %isFour, %c1152, %c1664 : i32
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
    %stepStart = arith.muli %step, %c512 : i32
    %weightPart = arith.addi %thread, %stepStart : i32
    %usedPart = arith.cmpi slt, %weightPart, %weightParts : i32
    cf.cond_br %usedPart, ^weightBody, ^weightNext(%weights : vector<16xi32>)
  ^weightBody:
    %row, %rowPart = func.call @strict_device_tile_row_part(%weightPart, %isFour) : (i32, i1) -> (i32, i32)
    %row64 = arith.extui %row : i32 to i64
    %ownRow = arith.addi %rowFirst, %row64 : i64
    %rowIndex = arith.minsi %ownRow, %rowLast : i64
    %rowNumber = arith.subi %rowIndex, %rowFirst : i64
    %rowOffset = arith.muli %rowNumber, %stride : i64
    %rowPartBytes = arith.muli %rowPart, %c16 : i32
    %rowPartBytes64 = arith.extui %rowPartBytes : i32 to i64
    %partOffset = arith.addi %blockOffset, %rowPartBytes64 : i64
    %sourceOffset = arith.addi %partOffset, %rowOffset : i64
    %source = llvm.getelementptr %firstData[%sourceOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %words = llvm.load %source invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %withWords = func.call @strict_device_tile_placed(%weights, %step, %words) : (vector<16xi32>, i32, vector<4xi32>) -> vector<16xi32>
    cf.br ^weightNext(%withWords : vector<16xi32>)
  ^weightNext(%nextWeights: vector<16xi32>):
    %stepNext = arith.addi %step, %c1 : i32
    llvm.br ^weight(%stepNext, %nextWeights : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tokens:
    cf.br ^token(%c0, %zeroWords : i32, vector<16xi32>)
  ^token(%tokenStep: i32, %tokenWords: vector<16xi32>):
    %moreTokens = arith.cmpi slt, %tokenStep, %c2 : i32
    cf.cond_br %moreTokens, ^tokenBody, ^scales
  ^tokenBody:
    %tokenStepStart = arith.muli %tokenStep, %c512 : i32
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
    %moreScales = arith.cmpi slt, %scaleStep, %c1 : i32
    cf.cond_br %moreScales, ^scaleBody, ^blockScale
  ^scaleBody:
    %scaleStepStart = arith.muli %scaleStep, %c512 : i32
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
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %scaleRow = arith.cmpi slt, %thread, %c128 : i32
    %isSix = arith.cmpi ne, %rowBits, %fourBits : i64
    %sixScale = arith.andi %scaleRow, %isSix : i1
    cf.cond_br %sixScale, ^blockScaleBody, ^done(%scales : vector<4xf32>)
  ^blockScaleBody:
    %threadRow = arith.extui %thread : i32 to i64
    %ownScaleRow = arith.addi %rowFirst, %threadRow : i64
    %scaleRowIndex = arith.minsi %ownScaleRow, %rowLast : i64
    %scaleRowNumber = arith.subi %scaleRowIndex, %rowFirst : i64
    %scaleRowOffset = arith.muli %scaleRowNumber, %stride : i64
    %scaleDataBytes = func.call @strict_device_data(%columns, %rowBits) : (i64, i64) -> i64
    %halvesOffset = arith.addi %scaleRowOffset, %scaleDataBytes : i64
    %halves = llvm.getelementptr %firstData[%halvesOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halfSlot = llvm.getelementptr %halves[%superblock] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %half = llvm.load %halfSlot invariant : !llvm.ptr -> f16
    %blockScaleValue = arith.extf %half : f16 to f32
    %withBlockScale = llvm.insertelement %blockScaleValue, %scales[%c2 : i32] : vector<4xf32>
    cf.br ^done(%withBlockScale : vector<4xf32>)
  ^done(%fetchedScales: vector<4xf32>):
    return %weights, %tokenWords, %fetchedScales : vector<16xi32>, vector<16xi32>, vector<4xf32>
  }
  // the row and the part of 16 bytes in it of part number part of a Q4_K tile (9 parts per row) or Q6_K tile (13), the
  // divisions by constants become multiplications
  func.func private @strict_device_tile_row_part(%part: i32, %isFour: i1) -> (i32, i32) attributes {passthrough = ["alwaysinline"]} {
    %c9 = llvm.mlir.constant(9 : i32) : i32
    %c13 = llvm.mlir.constant(13 : i32) : i32
    %fourRow = arith.divui %part, %c9 : i32
    %sixRow = arith.divui %part, %c13 : i32
    %row = arith.select %isFour, %fourRow, %sixRow : i32
    %rowParts = arith.select %isFour, %c9, %c13 : i32
    %rowStart = arith.muli %row, %rowParts : i32
    %rowPart = arith.subi %part, %rowStart : i32
    return %row, %rowPart : i32, i32
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
  // part rowPart of a Q4_K row into the tile: part 0 is the header, part 1 + 2 * g + h holds the bytes 16 * h to + 15 of
  // the 64 numbers group g, their low 4 bits are the numbers from 64 * g + 16 * h, their high bits those 32 later
  func.func private @strict_device_tile_unpacked(%words: vector<4xi32>, %row: i32, %rowPart: i32) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %lowBits = arith.constant dense<252645135> : vector<4xi32>
    %shift = arith.constant dense<4> : vector<4xi32>
    %weightTile = func.call @strict_device_tile_weights() : () -> !llvm.ptr<3>
    %headers = llvm.mlir.addressof @strict_device_tile_headers : !llvm.ptr<3>
    %isHeader = arith.cmpi eq, %rowPart, %c0 : i32
    cf.cond_br %isHeader, ^header, ^numbers
  ^header:
    %headerAt = arith.muli %row, %c16 : i32
    %headerSlot = func.call @strict_device_tile_at(%headers, %headerAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %words, %headerSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.br ^done
  ^numbers:
    %chunk = arith.subi %rowPart, %c1 : i32
    %group = arith.shrui %chunk, %c1 : i32
    %chunkHalf = arith.andi %chunk, %c1 : i32
    %groupBytes = arith.shli %group, %c6 : i32
    %halfBytes = arith.shli %chunkHalf, %c4 : i32
    %rowStart = arith.muli %row, %c272 : i32
    %groupStart = arith.addi %rowStart, %groupBytes : i32
    %lowAt = arith.addi %groupStart, %halfBytes : i32
    %highAt = arith.addi %lowAt, %c32 : i32
    %low = arith.andi %words, %lowBits : vector<4xi32>
    %shifted = arith.shrui %words, %shift : vector<4xi32>
    %high = arith.andi %shifted, %lowBits : vector<4xi32>
    %lowSlot = func.call @strict_device_tile_at(%weightTile, %lowAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %low, %lowSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %highSlot = func.call @strict_device_tile_at(%weightTile, %highAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %high, %highSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // the sum of the 8 bit numbers of a token for the 32 numbers of the part of 16 bytes chunk (lanes with the even and
  // the odd chunk of the same 32 numbers add up their halves), stored by the lane of the even chunk; the lanes scale it
  func.func private @strict_device_tile_token_sum(%numbers: vector<4xi32>, %token: i32, %chunk: i32) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %all = llvm.mlir.constant(-1 : i32) : i32
    %ones = llvm.mlir.constant(dense<16843009> : vector<4xi32>) : vector<4xi32>
    %tokenSums = llvm.mlir.addressof @strict_device_tile_token_sums : !llvm.ptr<3>
    %half = func.call @strict_device_dp4as(%numbers, %ones) : (vector<4xi32>, vector<4xi32>) -> i32
    %otherHalf = func.call @strict_device_swap(%half, %c1) : (i32, i32) -> i32
    %sum = arith.addi %half, %otherHalf : i32
    %odd = arith.andi %chunk, %c1 : i32
    %isEven = arith.cmpi eq, %odd, %c0 : i32
    cf.cond_br %isEven, ^store, ^done
  ^store:
    %sub = arith.shrui %chunk, %c1 : i32
    %slot = func.call @strict_device_tile_token_slot(%token, %sub) : (i32, i32) -> i32
    %slot64 = arith.extui %slot : i32 to i64
    %sumSlot = llvm.getelementptr %tokenSums[%slot64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %sumFloat = arith.sitofp %sum : i32 to f32
    llvm.store %sumFloat, %sumSlot : f32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
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
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %c1152 = llvm.mlir.constant(1152 : i32) : i32
    %c1664 = llvm.mlir.constant(1664 : i32) : i32
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    %rowParts = arith.select %isFour, %c9, %c13 : i32
    %rowBytes = arith.select %isFour, %c144, %c208 : i32
    %weightParts = arith.select %isFour, %c1152, %c1664 : i32
    %weightTile = func.call @strict_device_tile_weights() : () -> !llvm.ptr<3>
    %tokenTile = func.call @strict_device_tile_tokens() : () -> !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %rowBlocks = llvm.mlir.addressof @strict_device_tile_row_blocks : !llvm.ptr<3>
    cf.br ^weight(%c0 : i32)
  ^weight(%step: i32):
    %moreSteps = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %moreSteps, ^weightCheck, ^tokens
  ^weightCheck:
    %stepStart = arith.muli %step, %c512 : i32
    %weightPart = arith.addi %thread, %stepStart : i32
    %usedPart = arith.cmpi slt, %weightPart, %weightParts : i32
    cf.cond_br %usedPart, ^weightBody, ^weightNext
  ^weightBody:
    %row, %rowPart = func.call @strict_device_tile_row_part(%weightPart, %isFour) : (i32, i1) -> (i32, i32)
    %words = func.call @strict_device_tile_part(%weights, %step) : (vector<16xi32>, i32) -> vector<4xi32>
    cf.cond_br %isFour, ^fourPart, ^rawPart
  ^rawPart:
    %rowStart = arith.muli %row, %rowBytes : i32
    %rowPartBytes = arith.muli %rowPart, %c16 : i32
    %weightAt = arith.addi %rowStart, %rowPartBytes : i32
    %weightSlot = func.call @strict_device_tile_at(%weightTile, %weightAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %words, %weightSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.br ^weightNext
  ^fourPart:
    func.call @strict_device_tile_unpacked(%words, %row, %rowPart) : (vector<4xi32>, i32, i32) -> ()
    cf.br ^weightNext
  ^weightNext:
    %stepNext = arith.addi %step, %c1 : i32
    llvm.br ^weight(%stepNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tokens:
    cf.br ^token(%c0 : i32)
  ^token(%tokenStep: i32):
    %moreTokens = arith.cmpi slt, %tokenStep, %c2 : i32
    cf.cond_br %moreTokens, ^tokenBody, ^scales
  ^tokenBody:
    %tokenStepStart = arith.muli %tokenStep, %c512 : i32
    %tokenPart = arith.addi %thread, %tokenStepStart : i32
    %token = arith.shrui %tokenPart, %c4 : i32
    %tokenChunk = arith.andi %tokenPart, %c15 : i32
    %tokenRowStart = arith.muli %token, %c272 : i32
    %tokenChunkBytes = arith.muli %tokenChunk, %c16 : i32
    %tokenAt = arith.addi %tokenRowStart, %tokenChunkBytes : i32
    %tokenSlot = func.call @strict_device_tile_at(%tokenTile, %tokenAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %numbers = func.call @strict_device_tile_part(%tokenWords, %tokenStep) : (vector<16xi32>, i32) -> vector<4xi32>
    llvm.store %numbers, %tokenSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.cond_br %isFour, ^tokenSum, ^tokenNext
  ^tokenSum:
    func.call @strict_device_tile_token_sum(%numbers, %token, %tokenChunk) : (vector<4xi32>, i32, i32) -> ()
    cf.br ^tokenNext
  ^tokenNext:
    %tokenStepNext = arith.addi %tokenStep, %c1 : i32
    llvm.br ^token(%tokenStepNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scales:
    %firstScale = llvm.extractelement %scales[%c0 : i32] : vector<4xf32>
    %first64 = arith.extui %thread : i32 to i64
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %scaleSub = arith.andi %thread, %c7 : i32
    %firstToken = arith.shrui %thread, %c3 : i32
    %firstIndex = func.call @strict_device_tile_token_slot(%firstToken, %scaleSub) : (i32, i32) -> i32
    %firstIndex64 = arith.extui %firstIndex : i32 to i64
    %firstSlot = llvm.getelementptr %tokenScales[%firstIndex64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %firstScale, %firstSlot : f32, !llvm.ptr<3>
    %scaleRow = arith.cmpi slt, %thread, %c128 : i32
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
  // the scale and minimum of each 32 numbers of the 128 rows from the Q4_K headers in the tile, after it is staged (the
  // token sums are added up when the tokens are put)
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
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowMinimums = llvm.mlir.addressof @strict_device_tile_row_minimums : !llvm.ptr<3>
    cf.br ^part(%thread : i32)
  ^part(%part: i32):
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %more = arith.cmpi slt, %part, %c1024 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %owner = arith.divui %part, %c8 : i32
    %sub = arith.remui %part, %c8 : i32
    %headers = llvm.mlir.addressof @strict_device_tile_headers : !llvm.ptr<3>
    %headerBytes = llvm.mlir.constant(16 : i32) : i32
    %headerAt = arith.muli %owner, %headerBytes : i32
    %headerSlot = func.call @strict_device_tile_at(%headers, %headerAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
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
    %rowIndex = func.call @strict_device_tile_row_slot(%owner, %sub) : (i32, i32) -> i32
    %rowIndex64 = arith.extui %rowIndex : i32 to i64
    %rowScaleSlot = llvm.getelementptr %rowScales[%rowIndex64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %rowScale, %rowScaleSlot : f32, !llvm.ptr<3>
    %rowMinimumSlot = llvm.getelementptr %rowMinimums[0, %rowIndex64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
    llvm.store %rowMinimum, %rowMinimumSlot : f32, !llvm.ptr<3>
    cf.br ^partNext
  ^partNext:
    %partFollowing = arith.addi %part, %c512 : i32
    cf.br ^part(%partFollowing : i32)
  ^done:
    return
  }
  // the scale of each 16 numbers of the 128 rows of a Q6_K tile (its 8 bit scales times the 16 bit scale of the block)
  func.func private @strict_device_tile_six_scales(%thread: i32) attributes {passthrough = ["alwaysinline"]} {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %c192 = llvm.mlir.constant(192 : i32) : i32
    %c208 = llvm.mlir.constant(208 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %c2048 = llvm.mlir.constant(2048 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %weights = func.call @strict_device_tile_weights() : () -> !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowBlocks = llvm.mlir.addressof @strict_device_tile_row_blocks : !llvm.ptr<3>
    cf.br ^part(%thread : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c2048 : i32
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
    %partNext = arith.addi %part, %c512 : i32
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
    %weights = func.call @strict_device_tile_weights() : () -> !llvm.ptr<3>
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
    %lowDot = func.call @strict_device_mma8(%lowWeights, %lowTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %highDot = func.call @strict_device_mma8(%highWeights, %highTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %lowScale = func.call @strict_device_tile_six_scale(%row, %sub, %c0) : (i32, i32, i32) -> f32
    %highScale = func.call @strict_device_tile_six_scale(%row, %sub, %c1) : (i32, i32, i32) -> f32
    %firstIndex = func.call @strict_device_tile_token_slot(%firstToken, %sub) : (i32, i32) -> i32
    %secondIndex = func.call @strict_device_tile_token_slot(%secondToken, %sub) : (i32, i32) -> i32
    %firstScale = func.call @strict_device_tile_float(%tokenScales, %firstIndex) : (!llvm.ptr<3>, i32) -> f32
    %secondScale = func.call @strict_device_tile_float(%tokenScales, %secondIndex) : (!llvm.ptr<3>, i32) -> f32
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
  // 4 token numbers from at of the 32 numbers sub of token (in the tile)
  func.func private @strict_device_tile_token(%token: i32, %sub: i32, %at: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %tokens = func.call @strict_device_tile_tokens() : () -> !llvm.ptr<3>
    %tokenStart = arith.muli %token, %c272 : i32
    %subBytes = arith.shli %sub, %c5 : i32
    %subStart = arith.addi %tokenStart, %subBytes : i32
    %wordAt = arith.addi %subStart, %at : i32
    %slot = func.call @strict_device_tile_at(%tokens, %wordAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %word = llvm.load %slot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    return %word : i32
  }
  // the number at index of a float tile array
  func.func private @strict_device_tile_float(%array: !llvm.ptr<3>, %index: i32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %index64 = arith.extui %index : i32 to i64
    %slot = llvm.getelementptr %array[%index64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %value = llvm.load %slot : !llvm.ptr<3> -> f32
    return %value : f32
  }
  // the 4 numbers from index (a multiple of 4) of a float tile array
  func.func private @strict_device_tile_floats(%array: !llvm.ptr<3>, %index: i32) -> vector<4xf32> attributes {passthrough = ["alwaysinline"]} {
    %index64 = arith.extui %index : i32 to i64
    %slot = llvm.getelementptr %array[%index64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %values = llvm.load %slot {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
    return %values : vector<4xf32>
  }
  // where the scale or minimum of row (0 to 127) for the 32 numbers sub lies: 128 per sub, the 4 rows of a lane (row,
  // row + 8, row + 16 and row + 24 of its warp) next to each other
  func.func private @strict_device_tile_row_slot(%row: i32, %sub: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c96 = llvm.mlir.constant(96 : i32) : i32
    %subStart = arith.shli %sub, %c7 : i32
    %warpRows = arith.andi %row, %c96 : i32
    %laneRow = arith.andi %row, %c7 : i32
    %laneStart = arith.shli %laneRow, %c2 : i32
    %tile = arith.shrui %row, %c3 : i32
    %tileInWarp = arith.andi %tile, %c3 : i32
    %warpStart = arith.addi %subStart, %warpRows : i32
    %laneSlot = arith.addi %warpStart, %laneStart : i32
    %slot = arith.addi %laneSlot, %tileInWarp : i32
    return %slot : i32
  }
  // where the scale or scaled sum of token (0 to 63) for the 32 numbers sub lies: 64 per sub, the 4 tokens of a lane
  // (token, token + 1, token + 8 and token + 9 of its warp) next to each other
  func.func private @strict_device_tile_token_slot(%token: i32, %sub: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c48 = llvm.mlir.constant(48 : i32) : i32
    %subStart = arith.shli %sub, %c6 : i32
    %warpTokens = arith.andi %token, %c48 : i32
    %pair = arith.andi %token, %c6 : i32
    %pairStart = arith.shli %pair, %c1 : i32
    %tileBits = arith.shrui %token, %c2 : i32
    %tileStart = arith.andi %tileBits, %c2 : i32
    %odd = arith.andi %token, %c1 : i32
    %warpStart = arith.addi %subStart, %warpTokens : i32
    %pairSlot = arith.addi %warpStart, %pairStart : i32
    %tileSlot = arith.addi %pairSlot, %tileStart : i32
    %slot = arith.addi %tileSlot, %odd : i32
    return %slot : i32
  }
  // the 32 numbers sub of the tile for the 32 rows and 16 tokens of a warp: 4 x 2 tiles of 8 rows by 8 tokens, each two
  // mma of 16 numbers (mma.m8n8k16) on the 8 bit numbers the Q4_K rows were unpacked to, the integer sums scaled by row
  // and token, minus the row minimums times the scaled token sums; a lane has row lane / 4 and tokens 2 * (lane % 4) and
  // the next of each tile, three ldmatrix load the operands of all tiles (lanes give the rows of 4 matrices of 8 x 16);
  // from sm_80 on one mma.m16n8k32 takes two of those tiles with both halves of the 32 numbers, its row operands are
  // loaded in the order of the mma (rows 0 to 7, then 8 to 15 of the first 16 numbers, then of the second 16)
  func.func private @strict_device_tile_sub(%sums: vector<16xf32>, %sub: i32, %warpRow: i32, %warpToken: i32, %laneRow: i32, %quad: i32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %nothing = arith.constant 0 : i32
    %weights = func.call @strict_device_tile_weights() : () -> !llvm.ptr<3>
    %tokens = func.call @strict_device_tile_tokens() : () -> !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowMinimums = llvm.mlir.addressof @strict_device_tile_row_minimums : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %tokenSums = llvm.mlir.addressof @strict_device_tile_token_sums : !llvm.ptr<3>
    %laneRowStart = arith.shli %laneRow, %c2 : i32
    %lane = arith.addi %laneRowStart, %quad : i32
    %matrix = arith.shrui %lane, %c3 : i32
    %matrixRow = arith.andi %lane, %c7 : i32
    %matrixPair = arith.shrui %matrix, %c1 : i32
    %pairStart = arith.muli %matrixPair, %c8 : i32
    %pairRow = arith.addi %pairStart, %matrixRow : i32
    %matrixHalf = arith.andi %matrix, %c1 : i32
    %halfBytes = arith.shli %matrixHalf, %c4 : i32
    %subBytes = arith.shli %sub, %c5 : i32
    %column = arith.addi %subBytes, %halfBytes : i32
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = llvm.mlir.constant(800 : i32) : i32
    %isWide = arith.cmpi sge, %arch, %c800 : i32
    %wideRowStart = arith.shli %matrixHalf, %c3 : i32
    %wideMatrixRow = arith.addi %wideRowStart, %matrixRow : i32
    %wideBytes = arith.shli %matrixPair, %c4 : i32
    %wideColumn = arith.addi %subBytes, %wideBytes : i32
    %weightRow = arith.select %isWide, %wideMatrixRow, %pairRow : i32
    %weightColumn = arith.select %isWide, %wideColumn, %column : i32
    %firstRow = arith.addi %warpRow, %weightRow : i32
    %tokenRow = arith.addi %warpToken, %pairRow : i32
    %first = func.call @strict_device_tile_matrices(%weights, %firstRow, %weightColumn) : (!llvm.ptr<3>, i32, i32) -> vector<4xi32>
    %c4352 = llvm.mlir.constant(4352 : i32) : i32
    %secondWeights = llvm.getelementptr %weights[%c4352] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %second = func.call @strict_device_tile_matrices(%secondWeights, %firstRow, %weightColumn) : (!llvm.ptr<3>, i32, i32) -> vector<4xi32>
    %rowOperands = llvm.shufflevector %first, %second [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xi32>
    %tokenOperands = func.call @strict_device_tile_matrices(%tokens, %tokenRow, %column) : (!llvm.ptr<3>, i32, i32) -> vector<4xi32>
    %pairToken = arith.shli %quad, %c1 : i32
    cf.cond_br %isWide, ^wideStart, ^tile(%c0, %sums : i32, vector<16xf32>)
  ^wideStart:
    %laneFirstRow = arith.addi %warpRow, %laneRow : i32
    %laneFirstToken = arith.addi %warpToken, %pairToken : i32
    %laneRowSlot = func.call @strict_device_tile_row_slot(%laneFirstRow, %sub) : (i32, i32) -> i32
    %laneTokenSlot = func.call @strict_device_tile_token_slot(%laneFirstToken, %sub) : (i32, i32) -> i32
    %laneRowScales = func.call @strict_device_tile_floats(%rowScales, %laneRowSlot) : (!llvm.ptr<3>, i32) -> vector<4xf32>
    %laneRowMinimums = func.call @strict_device_tile_floats(%rowMinimums, %laneRowSlot) : (!llvm.ptr<3>, i32) -> vector<4xf32>
    %laneTokenScales = func.call @strict_device_tile_floats(%tokenScales, %laneTokenSlot) : (!llvm.ptr<3>, i32) -> vector<4xf32>
    %laneTokenTotals = func.call @strict_device_tile_floats(%tokenSums, %laneTokenSlot) : (!llvm.ptr<3>, i32) -> vector<4xf32>
    %laneTokenSums = arith.mulf %laneTokenTotals, %laneTokenScales : vector<4xf32>
    cf.br ^wide(%c0, %sums : i32, vector<16xf32>)
  ^wide(%wideTile: i32, %wideCurrent: vector<16xf32>):
    %wideMore = arith.cmpi slt, %wideTile, %c4 : i32
    cf.cond_br %wideMore, ^wideBody, ^done(%wideCurrent : vector<16xf32>)
  ^wideBody:
    %rowPair = arith.shrui %wideTile, %c1 : i32
    %wideTokenTile = arith.andi %wideTile, %c1 : i32
    %operand0 = arith.shli %rowPair, %c2 : i32
    %operand1 = arith.addi %operand0, %c1 : i32
    %operand2 = arith.addi %operand0, %c2 : i32
    %operand3 = arith.addi %operand0, %c3 : i32
    %rows0 = llvm.extractelement %rowOperands[%operand0 : i32] : vector<8xi32>
    %rows1 = llvm.extractelement %rowOperands[%operand1 : i32] : vector<8xi32>
    %rows2 = llvm.extractelement %rowOperands[%operand2 : i32] : vector<8xi32>
    %rows3 = llvm.extractelement %rowOperands[%operand3 : i32] : vector<8xi32>
    %tokenLowIndex = arith.shli %wideTokenTile, %c1 : i32
    %tokenHighIndex = arith.addi %tokenLowIndex, %c1 : i32
    %tokensLow = llvm.extractelement %tokenOperands[%tokenLowIndex : i32] : vector<4xi32>
    %tokensHigh = llvm.extractelement %tokenOperands[%tokenHighIndex : i32] : vector<4xi32>
    %wideDot = func.call @strict_device_mma16(%rows0, %rows1, %rows2, %rows3, %tokensLow, %tokensHigh, %nothing, %nothing, %nothing, %nothing) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %upperRow = arith.shli %rowPair, %c1 : i32
    %lowerRow = arith.addi %upperRow, %c1 : i32
    %rowPairSlots = arith.shli %rowPair, %c2 : i32
    %wideTileSlot = arith.addi %rowPairSlots, %wideTokenTile : i32
    %wideSlot = arith.shli %wideTileSlot, %c1 : i32
    %wideSlot1 = arith.addi %wideSlot, %c1 : i32
    %wideSlot4 = arith.addi %wideSlot, %c4 : i32
    %wideSlot5 = arith.addi %wideSlot4, %c1 : i32
    %wideDot0 = llvm.extractvalue %wideDot[0] : !llvm.struct<(i32, i32, i32, i32)>
    %wideDot1 = llvm.extractvalue %wideDot[1] : !llvm.struct<(i32, i32, i32, i32)>
    %wideDot2 = llvm.extractvalue %wideDot[2] : !llvm.struct<(i32, i32, i32, i32)>
    %wideDot3 = llvm.extractvalue %wideDot[3] : !llvm.struct<(i32, i32, i32, i32)>
    %wide0 = func.call @strict_device_tile_lane_added(%wideCurrent, %wideSlot, %wideDot0, %upperRow, %tokenLowIndex, %laneRowScales, %laneRowMinimums, %laneTokenScales, %laneTokenSums) : (vector<16xf32>, i32, i32, i32, i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<16xf32>
    %wide1 = func.call @strict_device_tile_lane_added(%wide0, %wideSlot1, %wideDot1, %upperRow, %tokenHighIndex, %laneRowScales, %laneRowMinimums, %laneTokenScales, %laneTokenSums) : (vector<16xf32>, i32, i32, i32, i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<16xf32>
    %wide2 = func.call @strict_device_tile_lane_added(%wide1, %wideSlot4, %wideDot2, %lowerRow, %tokenLowIndex, %laneRowScales, %laneRowMinimums, %laneTokenScales, %laneTokenSums) : (vector<16xf32>, i32, i32, i32, i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<16xf32>
    %wide3 = func.call @strict_device_tile_lane_added(%wide2, %wideSlot5, %wideDot3, %lowerRow, %tokenHighIndex, %laneRowScales, %laneRowMinimums, %laneTokenScales, %laneTokenSums) : (vector<16xf32>, i32, i32, i32, i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<16xf32>
    %wideNext = arith.addi %wideTile, %c1 : i32
    llvm.br ^wide(%wideNext, %wide3 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tile(%tile: i32, %current: vector<16xf32>):
    %more = arith.cmpi slt, %tile, %c8 : i32
    cf.cond_br %more, ^body, ^done(%current : vector<16xf32>)
  ^body:
    %rowTile = arith.shrui %tile, %c1 : i32
    %tokenTile = arith.andi %tile, %c1 : i32
    %rowLow = arith.shli %rowTile, %c1 : i32
    %rowHigh = arith.addi %rowLow, %c1 : i32
    %tokenLow = arith.shli %tokenTile, %c1 : i32
    %tokenHigh = arith.addi %tokenLow, %c1 : i32
    %lowWeights = llvm.extractelement %rowOperands[%rowLow : i32] : vector<8xi32>
    %highWeights = llvm.extractelement %rowOperands[%rowHigh : i32] : vector<8xi32>
    %lowTokens = llvm.extractelement %tokenOperands[%tokenLow : i32] : vector<4xi32>
    %highTokens = llvm.extractelement %tokenOperands[%tokenHigh : i32] : vector<4xi32>
    %rowTileStart = arith.muli %rowTile, %c8 : i32
    %rowBase = arith.addi %warpRow, %rowTileStart : i32
    %row = arith.addi %rowBase, %laneRow : i32
    %tokenTileStart = arith.muli %tokenTile, %c8 : i32
    %tokenBase = arith.addi %warpToken, %tokenTileStart : i32
    %firstToken = arith.addi %tokenBase, %pairToken : i32
    %secondToken = arith.addi %firstToken, %c1 : i32
    %half = func.call @strict_device_mma8(%lowWeights, %lowTokens, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %half0 = llvm.extractvalue %half[0] : !llvm.struct<(i32, i32)>
    %half1 = llvm.extractvalue %half[1] : !llvm.struct<(i32, i32)>
    %dot = func.call @strict_device_mma8(%highWeights, %highTokens, %half0, %half1) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(i32, i32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(i32, i32)>
    %slot = arith.shli %tile, %c1 : i32
    %with0 = func.call @strict_device_tile_scaled(%current, %slot, %dot0, %row, %firstToken, %sub) : (vector<16xf32>, i32, i32, i32, i32, i32) -> vector<16xf32>
    %slot1 = arith.addi %slot, %c1 : i32
    %with1 = func.call @strict_device_tile_scaled(%with0, %slot1, %dot1, %row, %secondToken, %sub) : (vector<16xf32>, i32, i32, i32, i32, i32) -> vector<16xf32>
    %tileNext = arith.addi %tile, %c1 : i32
    llvm.br ^tile(%tileNext, %with1 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done(%final: vector<16xf32>):
    return %final : vector<16xf32>
  }
  // sums at slot plus the integer dot of row and token for the 32 numbers sub, scaled by strict_device_tile_added
  func.func private @strict_device_tile_scaled(%sums: vector<16xf32>, %slot: i32, %dot: i32, %row: i32, %token: i32, %sub: i32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %rowScales = llvm.mlir.addressof @strict_device_tile_row_scales : !llvm.ptr<3>
    %rowMinimums = llvm.mlir.addressof @strict_device_tile_row_minimums : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_tile_token_scales : !llvm.ptr<3>
    %tokenSums = llvm.mlir.addressof @strict_device_tile_token_sums : !llvm.ptr<3>
    %rowSlot = func.call @strict_device_tile_row_slot(%row, %sub) : (i32, i32) -> i32
    %tokenSlot = func.call @strict_device_tile_token_slot(%token, %sub) : (i32, i32) -> i32
    %rowScale = func.call @strict_device_tile_float(%rowScales, %rowSlot) : (!llvm.ptr<3>, i32) -> f32
    %rowMinimum = func.call @strict_device_tile_float(%rowMinimums, %rowSlot) : (!llvm.ptr<3>, i32) -> f32
    %tokenScale = func.call @strict_device_tile_float(%tokenScales, %tokenSlot) : (!llvm.ptr<3>, i32) -> f32
    %tokenTotal = func.call @strict_device_tile_float(%tokenSums, %tokenSlot) : (!llvm.ptr<3>, i32) -> f32
    %tokenSum = arith.mulf %tokenScale, %tokenTotal : f32
    %updated = func.call @strict_device_tile_added(%sums, %slot, %dot, %rowScale, %tokenScale, %rowMinimum, %tokenSum) : (vector<16xf32>, i32, i32, f32, f32, f32, f32) -> vector<16xf32>
    return %updated : vector<16xf32>
  }
  // sums at slot plus the integer dot scaled like strict_device_tile_added with the scales of row and token (0 to 3) of
  // the 4 rows and 4 tokens of a lane
  func.func private @strict_device_tile_lane_added(%sums: vector<16xf32>, %slot: i32, %dot: i32, %row: i32, %token: i32, %rowScales: vector<4xf32>, %rowMinimums: vector<4xf32>, %tokenScales: vector<4xf32>, %tokenSums: vector<4xf32>) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %rowScale = llvm.extractelement %rowScales[%row : i32] : vector<4xf32>
    %rowMinimum = llvm.extractelement %rowMinimums[%row : i32] : vector<4xf32>
    %tokenScale = llvm.extractelement %tokenScales[%token : i32] : vector<4xf32>
    %tokenSum = llvm.extractelement %tokenSums[%token : i32] : vector<4xf32>
    %updated = func.call @strict_device_tile_added(%sums, %slot, %dot, %rowScale, %tokenScale, %rowMinimum, %tokenSum) : (vector<16xf32>, i32, i32, f32, f32, f32, f32) -> vector<16xf32>
    return %updated : vector<16xf32>
  }
  // 4 matrices of 8 rows by 16 bytes for mma operands (ldmatrix.x4): lane l gives the address of its row (rows 272 bytes
  // apart from row, at byte column), each lane gets 4 bytes of row lane / 4 of each matrix
  func.func private @strict_device_tile_matrices(%array: !llvm.ptr<3>, %row: i32, %column: i32) -> vector<4xi32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c272 = llvm.mlir.constant(272 : i32) : i32
    %rowStart = arith.muli %row, %c272 : i32
    %at = arith.addi %rowStart, %column : i32
    %slot = func.call @strict_device_tile_at(%array, %at) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %loaded = func.call @strict_device_matrices(%slot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %first = llvm.extractvalue %loaded[0] : !llvm.struct<(i32, i32, i32, i32)>
    %second = llvm.extractvalue %loaded[1] : !llvm.struct<(i32, i32, i32, i32)>
    %third = llvm.extractvalue %loaded[2] : !llvm.struct<(i32, i32, i32, i32)>
    %fourth = llvm.extractvalue %loaded[3] : !llvm.struct<(i32, i32, i32, i32)>
    %empty = llvm.mlir.undef : vector<4xi32>
    %with0 = llvm.insertelement %first, %empty[%c0 : i32] : vector<4xi32>
    %with1 = llvm.insertelement %second, %with0[%c1 : i32] : vector<4xi32>
    %with2 = llvm.insertelement %third, %with1[%c2 : i32] : vector<4xi32>
    %with3 = llvm.insertelement %fourth, %with2[%c3 : i32] : vector<4xi32>
    return %with3 : vector<4xi32>
  }
  // sums at slot plus the integer dot times row and token scale minus the row minimum times the scaled token sum
  func.func private @strict_device_tile_added(%sums: vector<16xf32>, %slot: i32, %dot: i32, %rowScale: f32, %tokenScale: f32, %rowMinimum: f32, %tokenSum: f32) -> vector<16xf32> attributes {passthrough = ["alwaysinline"]} {
    %dotFloat = arith.sitofp %dot : i32 to f32
    %scale = arith.mulf %rowScale, %tokenScale : f32
    %negativeMinimum = arith.negf %rowMinimum : f32
    %old = llvm.extractelement %sums[%slot : i32] : vector<16xf32>
    %scaled = llvm.call_intrinsic "llvm.fma.f32"(%dotFloat, %scale, %old) : (f32, f32, f32) -> f32
    %new = llvm.call_intrinsic "llvm.fma.f32"(%negativeMinimum, %tokenSum, %scaled) : (f32, f32, f32) -> f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<16xf32>
    return %updated : vector<16xf32>
  }
  // `Batch.Sums` of 64 tokens or more with Q4_K or Q6_K rows: a block of 16 warps takes 128 rows and 64 tokens, stages
  // one block of 256 columns of all of them in shared memory at a time and each warp computes 32 rows by 16 tokens of
  // it on the 8 bit tensor cores (with fewer tiles than multiprocessors the blocks of a tile split its columns and the
  // last adds up their sums); the dot of row r and token t becomes number t * rows + r of the result
  func.func private @strict_device_tiles_run(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {passthrough = ["alwaysinline"]} {
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
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
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
    %rowSpare = arith.constant 127 : i64
    %rowsUp = arith.addi %rows, %rowSpare : i64
    %seven = arith.constant 7 : i64
    %rowTileCount = arith.shrui %rowsUp, %seven : i64
    %tileCount = arith.muli %rowTileCount, %tokenTileCount : i64
    %grid32 = func.call @strict_device_block_count() : () -> i32
    %grid = arith.extui %grid32 : i32 to i64
    %splits = arith.divui %grid, %tileCount : i64
    %tile = arith.divui %block, %splits : i64
    %split = arith.remui %block, %splits : i64
    %tokenTile = arith.remui %tile, %tokenTileCount : i64
    %rowTile = arith.divui %tile, %tokenTileCount : i64
    %oneHundredTwentyEight = arith.constant 128 : i64
    %rowFirst = arith.muli %rowTile, %oneHundredTwentyEight : i64
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
    %warpRowIndex = arith.andi %warp, %c3 : i32
    %warpTokenIndex = arith.shrui %warp, %c2 : i32
    %warpRow = arith.muli %warpRowIndex, %c32 : i32
    %warpToken = arith.muli %warpTokenIndex, %c16 : i32
    %allBlocks = arith.shrui %columns, %eight : i64
    %splitStart = arith.muli %split, %allBlocks : i64
    %firstBlock = arith.divui %splitStart, %splits : i64
    %splitNext = arith.addi %split, %one : i64
    %splitEnd = arith.muli %splitNext, %allBlocks : i64
    %blocks = arith.divui %splitEnd, %splits : i64
    %base = func.call @strict_device_token_base(%bases, %basesCount, %basesBits, %tokenFirst) : (!llvm.ptr, i64, i64, i64) -> i64
    %basedFirst = arith.addi %rowFirst, %base : i64
    %firstData, %rowBits, %firstTotal = func.call @strict_device_row(%rowsData, %basedFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %rowsWord = llvm.load %rowsData : !llvm.ptr -> i64
    %lowHalf = arith.constant 2147483647 : i64
    %stride = arith.andi %rowsWord, %lowHalf : i64
    %fourBits = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %rowBits, %fourBits : i64
    %firstWeights, %firstTokens, %firstScales = func.call @strict_device_tile_fetch(%firstData, %stride, %rowFirst, %rowLast, %firstBlock, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %rowBits, %columns) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i64, i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>)
    func.call @strict_device_tile_put(%firstWeights, %firstTokens, %firstScales, %thread, %rowBits) : (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i64) -> ()
    func.call @strict_device_barrier() : () -> ()
    func.call @strict_device_tile_prepare(%thread, %rowBits) : (i32, i64) -> ()
    func.call @strict_device_barrier() : () -> ()
    %lastBlock = arith.subi %blocks, %one : i64
    cf.br ^superblock(%firstBlock, %zeros : i64, vector<16xf32>)
  ^superblock(%superblock: i64, %sums: vector<16xf32>):
    %moreBlocks = arith.cmpi slt, %superblock, %blocks : i64
    cf.cond_br %moreBlocks, ^stage, ^combine
  ^combine:
    %single = arith.cmpi eq, %splits, %one : i64
    cf.cond_br %single, ^write(%c0, %sums : i32, vector<16xf32>), ^share
  ^share:
    %tile32 = arith.trunci %tile : i64 to i32
    %splits32 = arith.trunci %splits : i64 to i32
    %split32 = arith.trunci %split : i64 to i32
    %tileParts = arith.muli %tile32, %splits32 : i32
    %ownPart = arith.addi %tileParts, %split32 : i32
    %ownSlot = func.call @strict_device_tile_partial(%ownPart, %thread) : (i32, i32) -> !llvm.ptr
    llvm.store %sums, %ownSlot {alignment = 16 : i64} : vector<16xf32>, !llvm.ptr
    func.call @strict_device_fence() : () -> ()
    func.call @strict_device_barrier() : () -> ()
    %lastSlot = llvm.mlir.addressof @strict_device_tile_last : !llvm.ptr<3>
    %isFirstThread = arith.cmpi eq, %thread, %c0 : i32
    cf.cond_br %isFirstThread, ^count, ^counted
  ^count:
    %counters = llvm.mlir.addressof @strict_device_tile_counters : !llvm.ptr<1>
    %counter = llvm.getelementptr %counters[0, %tile] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<256 x i32>
    %done = llvm.atomicrmw add %counter, %c1 acq_rel : !llvm.ptr<1>, i32
    %lastSplit = arith.subi %splits32, %c1 : i32
    %isLast = arith.cmpi eq, %done, %lastSplit : i32
    %lastFlag = arith.extui %isLast : i1 to i32
    llvm.store %lastFlag, %lastSlot : i32, !llvm.ptr<3>
    cf.cond_br %isLast, ^reset, ^counted
  ^reset:
    llvm.store %c0, %counter : i32, !llvm.ptr<1>
    cf.br ^counted
  ^counted:
    func.call @strict_device_barrier() : () -> ()
    %flag = llvm.load %lastSlot : !llvm.ptr<3> -> i32
    %isLastBlock = arith.cmpi ne, %flag, %c0 : i32
    cf.cond_br %isLastBlock, ^collect, ^exit
  ^collect:
    func.call @strict_device_fence() : () -> ()
    cf.br ^gather(%c0, %sums : i32, vector<16xf32>)
  ^gather(%part: i32, %gathered: vector<16xf32>):
    %moreParts = arith.cmpi slt, %part, %splits32 : i32
    cf.cond_br %moreParts, ^gatherBody, ^write(%c0, %gathered : i32, vector<16xf32>)
  ^gatherBody:
    %isOwn = arith.cmpi eq, %part, %split32 : i32
    cf.cond_br %isOwn, ^gatherNext(%gathered : vector<16xf32>), ^gatherAdd
  ^gatherAdd:
    %partAt = arith.addi %tileParts, %part : i32
    %partSlot = func.call @strict_device_tile_partial(%partAt, %thread) : (i32, i32) -> !llvm.ptr
    %partSums = llvm.load volatile %partSlot {alignment = 16 : i64} : !llvm.ptr -> vector<16xf32>
    %addedParts = arith.addf %gathered, %partSums : vector<16xf32>
    cf.br ^gatherNext(%addedParts : vector<16xf32>)
  ^gatherNext(%gatheredNext: vector<16xf32>):
    %partFollowing = arith.addi %part, %c1 : i32
    cf.br ^gather(%partFollowing, %gatheredNext : i32, vector<16xf32>)
  ^stage:
    %nextBlock = arith.addi %superblock, %one : i64
    %fetchedBlock = arith.minsi %nextBlock, %lastBlock : i64
    %nextWeights, %nextTokens, %nextScales = func.call @strict_device_tile_fetch(%firstData, %stride, %rowFirst, %rowLast, %fetchedBlock, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %rowBits, %columns) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i64, i64) -> (vector<16xi32>, vector<16xi32>, vector<4xf32>)
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
    func.call @strict_device_barrier() : () -> ()
    func.call @strict_device_tile_put(%nextWeights, %nextTokens, %nextScales, %thread, %rowBits) : (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i64) -> ()
    func.call @strict_device_barrier() : () -> ()
    func.call @strict_device_tile_prepare(%thread, %rowBits) : (i32, i64) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^superblock(%nextBlock, %subSums : i64, vector<16xf32>)
  ^write(%slot: i32, %written: vector<16xf32>):
    %moreSlots = arith.cmpi slt, %slot, %c16 : i32
    cf.cond_br %moreSlots, ^writeBody, ^exit
  ^writeBody:
    %pair = arith.shrui %slot, %c1 : i32
    %odd = arith.andi %slot, %c1 : i32
    %rowTileIndex = arith.shrui %pair, %c1 : i32
    %tokenTileIndex = arith.andi %pair, %c1 : i32
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
    %value = llvm.extractelement %written[%slot : i32] : vector<16xf32>
    %tokenStart = arith.muli %token, %rows : i64
    %resultIndex = arith.addi %tokenStart, %row : i64
    func.call @strict_device_tokens_store(%result, %wide, %resultIndex, %value, %row, %rows, %biases, %biasesCount, %biasesBits) : (!llvm.ptr, i1, i64, f32, i64, i64, !llvm.ptr, i64, i64) -> ()
    cf.br ^writeNext
  ^writeNext:
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext, %written : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  // `Batch.Sums` of 64 tokens or more with 8 bit rows: a block of 8 warps takes 128 rows and 128 tokens, 64 columns of
  // both at a time in shared memory (80 bytes per row or token so the rows of an ldmatrix lie in distinct banks, two
  // stages in turn), with the scales of their 32 numbers
  llvm.mlir.global internal @strict_device_eights_rows() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<20480 x i8>
  llvm.mlir.global internal @strict_device_eights_tokens() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<20480 x i8>
  llvm.mlir.global internal @strict_device_eights_row_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  llvm.mlir.global internal @strict_device_eights_token_scales() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<512 x f32>
  // the 2 parts of 16 bytes of the rows and of the tokens and the row and token scale a thread copies for the 64 columns
  // of stage (rows and tokens after the last repeat the last), loaded into registers while the other stage computes
  func.func private @strict_device_eights_fetch(%firstData: !llvm.ptr, %stride: i64, %rowFirst: i64, %rowLast: i64, %columns: i64, %input: !llvm.ptr, %inputScales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %stage: i64, %thread: i32, %whole: i1) -> (vector<16xi32>, vector<2xf32>) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %sixtyFour = arith.constant 64 : i64
    %zeroWords = arith.constant dense<0> : vector<16xi32>
    %thread64 = arith.extui %thread : i32 to i64
    %column = arith.shli %stage, %six : i64
    %piece = arith.andi %thread64, %three : i64
    %pieceBytes = arith.shli %piece, %four : i64
    %pieceColumn = arith.addi %column, %pieceBytes : i64
    %firstLocal = arith.shrui %thread64, %two : i64
    cf.br ^part(%c0, %firstLocal, %zeroWords : i32, i64, vector<16xi32>)
  ^part(%step: i32, %local: i64, %words: vector<16xi32>):
    %moreParts = arith.cmpi slt, %step, %c2 : i32
    cf.cond_br %moreParts, ^partBody, ^scaling
  ^partBody:
    %ownRow = arith.addi %rowFirst, %local : i64
    %row = arith.minsi %ownRow, %rowLast : i64
    %rowNumber = arith.subi %row, %rowFirst : i64
    %rowStart = arith.muli %rowNumber, %stride : i64
    %rowOffset = arith.addi %rowStart, %pieceColumn : i64
    %rowSource = llvm.getelementptr %firstData[%rowOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowWords = llvm.load %rowSource invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %ownToken = arith.addi %tokenFirst, %local : i64
    %token = arith.minsi %ownToken, %tokenLast : i64
    %tokenStart = arith.muli %token, %width : i64
    %tokenOffset = arith.addi %tokenStart, %pieceColumn : i64
    %tokenSource = llvm.getelementptr %input[%tokenOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %tokenWords = llvm.load %tokenSource invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %withRow = func.call @strict_device_tile_placed(%words, %step, %rowWords) : (vector<16xi32>, i32, vector<4xi32>) -> vector<16xi32>
    %tokenStep = arith.addi %step, %c2 : i32
    %withToken = func.call @strict_device_tile_placed(%withRow, %tokenStep, %tokenWords) : (vector<16xi32>, i32, vector<4xi32>) -> vector<16xi32>
    %stepNext = arith.addi %step, %c1 : i32
    %localNext = arith.addi %local, %sixtyFour : i64
    llvm.br ^part(%stepNext, %localNext, %withToken : i32, i64, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scaling:
    %unscaled = arith.constant dense<0.0> : vector<2xf32>
    cf.cond_br %whole, ^fetched(%unscaled : vector<2xf32>), ^scales
  ^scales:
    %scaleLocal = arith.shrui %thread64, %one : i64
    %sub = arith.andi %thread64, %one : i64
    %columnScale = arith.shrui %column, %five : i64
    %scaleIndex = arith.addi %columnScale, %sub : i64
    %scaleBytes = arith.shli %scaleIndex, %two : i64
    %ownScaleRow = arith.addi %rowFirst, %scaleLocal : i64
    %scaleRow = arith.minsi %ownScaleRow, %rowLast : i64
    %scaleRowNumber = arith.subi %scaleRow, %rowFirst : i64
    %scaleRowStart = arith.muli %scaleRowNumber, %stride : i64
    %scaleRowData = arith.addi %scaleRowStart, %columns : i64
    %rowScaleOffset = arith.addi %scaleRowData, %scaleBytes : i64
    %rowScaleSource = llvm.getelementptr %firstData[%rowScaleOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowScale = llvm.load %rowScaleSource invariant : !llvm.ptr -> f32
    %ownScaleToken = arith.addi %tokenFirst, %scaleLocal : i64
    %scaleToken = arith.minsi %ownScaleToken, %tokenLast : i64
    %scaleTokenStart = arith.muli %scaleToken, %width : i64
    %scaleTokenBlock = arith.shrui %scaleTokenStart, %five : i64
    %tokenScaleIndex = arith.addi %scaleTokenBlock, %scaleIndex : i64
    %tokenScaleSource = llvm.getelementptr %inputScales[%tokenScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %tokenScale = llvm.load %tokenScaleSource invariant : !llvm.ptr -> f32
    %noScales = llvm.mlir.undef : vector<2xf32>
    %withRowScale = llvm.insertelement %rowScale, %noScales[%c0 : i32] : vector<2xf32>
    %scales = llvm.insertelement %tokenScale, %withRowScale[%c1 : i32] : vector<2xf32>
    cf.br ^fetched(%scales : vector<2xf32>)
  ^fetched(%fetchedScales: vector<2xf32>):
    return %words, %fetchedScales : vector<16xi32>, vector<2xf32>
  }
  // the parts strict_device_eights_fetch loaded into stage buffer (0 or 1) of the shared memory
  func.func private @strict_device_eights_put(%words: vector<16xi32>, %scales: vector<2xf32>, %buffer: i32, %thread: i32, %whole: i1) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c5120 = llvm.mlir.constant(5120 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %rows = llvm.mlir.addressof @strict_device_eights_rows : !llvm.ptr<3>
    %tokens = llvm.mlir.addressof @strict_device_eights_tokens : !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_eights_row_scales : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_eights_token_scales : !llvm.ptr<3>
    %bufferStart = arith.muli %buffer, %c10240 : i32
    %firstLocal = arith.shrui %thread, %c2 : i32
    %piece = arith.andi %thread, %c3 : i32
    %pieceBytes = arith.shli %piece, %c4 : i32
    %firstStart = arith.muli %firstLocal, %c80 : i32
    %firstOffset = arith.addi %firstStart, %pieceBytes : i32
    %firstAt = arith.addi %bufferStart, %firstOffset : i32
    %secondAt = arith.addi %firstAt, %c5120 : i32
    %firstRows = func.call @strict_device_tile_part(%words, %c0) : (vector<16xi32>, i32) -> vector<4xi32>
    %secondRows = func.call @strict_device_tile_part(%words, %c1) : (vector<16xi32>, i32) -> vector<4xi32>
    %firstTokens = func.call @strict_device_tile_part(%words, %c2) : (vector<16xi32>, i32) -> vector<4xi32>
    %secondTokens = func.call @strict_device_tile_part(%words, %c3) : (vector<16xi32>, i32) -> vector<4xi32>
    %firstRowSlot = func.call @strict_device_tile_at(%rows, %firstAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %firstRows, %firstRowSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %secondRowSlot = func.call @strict_device_tile_at(%rows, %secondAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %secondRows, %secondRowSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %firstTokenSlot = func.call @strict_device_tile_at(%tokens, %firstAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %firstTokens, %firstTokenSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %secondTokenSlot = func.call @strict_device_tile_at(%tokens, %secondAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    llvm.store %secondTokens, %secondTokenSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    cf.cond_br %whole, ^done, ^scales
  ^scales:
    %scaleBuffer = arith.shli %buffer, %c8 : i32
    %sub = arith.andi %thread, %c1 : i32
    %subStart = arith.shli %sub, %c7 : i32
    %scaleLocal = arith.shrui %thread, %c1 : i32
    %scaleStart = arith.addi %scaleBuffer, %subStart : i32
    %scaleAt = arith.addi %scaleStart, %scaleLocal : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %rowScale = llvm.extractelement %scales[%c0 : i32] : vector<2xf32>
    %tokenScale = llvm.extractelement %scales[%c1 : i32] : vector<2xf32>
    %rowScaleSlot = llvm.getelementptr %rowScales[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %rowScale, %rowScaleSlot : f32, !llvm.ptr<3>
    %tokenScaleSlot = llvm.getelementptr %tokenScales[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %tokenScale, %tokenScaleSlot : f32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // the 4 integer sums from slot (as bits of floats) plus the mma.m16n8k32 of a and b, rows with one scale add up whole
  func.func private @strict_device_eights_whole(%sums: vector<64xf32>, %slot: i32, %a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32) -> vector<64xf32> attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %old0 = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    %old1 = llvm.extractelement %sums[%slot1 : i32] : vector<64xf32>
    %old2 = llvm.extractelement %sums[%slot2 : i32] : vector<64xf32>
    %old3 = llvm.extractelement %sums[%slot3 : i32] : vector<64xf32>
    %bits0 = arith.bitcast %old0 : f32 to i32
    %bits1 = arith.bitcast %old1 : f32 to i32
    %bits2 = arith.bitcast %old2 : f32 to i32
    %bits3 = arith.bitcast %old3 : f32 to i32
    %dot = func.call @strict_device_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %bits0, %bits1, %bits2, %bits3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(i32, i32, i32, i32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(i32, i32, i32, i32)>
    %dot2 = llvm.extractvalue %dot[2] : !llvm.struct<(i32, i32, i32, i32)>
    %dot3 = llvm.extractvalue %dot[3] : !llvm.struct<(i32, i32, i32, i32)>
    %new0 = arith.bitcast %dot0 : i32 to f32
    %new1 = arith.bitcast %dot1 : i32 to f32
    %new2 = arith.bitcast %dot2 : i32 to f32
    %new3 = arith.bitcast %dot3 : i32 to f32
    %with0 = llvm.insertelement %new0, %sums[%slot : i32] : vector<64xf32>
    %with1 = llvm.insertelement %new1, %with0[%slot1 : i32] : vector<64xf32>
    %with2 = llvm.insertelement %new2, %with1[%slot2 : i32] : vector<64xf32>
    %with3 = llvm.insertelement %new3, %with2[%slot3 : i32] : vector<64xf32>
    return %with3 : vector<64xf32>
  }
  // sums at slot plus the integer dot times scale
  func.func private @strict_device_eights_added(%sums: vector<64xf32>, %slot: i32, %dot: i32, %scale: f32) -> vector<64xf32> attributes {passthrough = ["alwaysinline"]} {
    %dotFloat = arith.sitofp %dot : i32 to f32
    %old = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    %new = llvm.call_intrinsic "llvm.fma.f32"(%dotFloat, %scale, %old) : (f32, f32, f32) -> f32
    %updated = llvm.insertelement %new, %sums[%slot : i32] : vector<64xf32>
    return %updated : vector<64xf32>
  }
  // the 32 numbers sub (0 or 1) of stage buffer for the 64 rows and 32 tokens of a warp: an ldmatrix.x4 per 16 rows and
  // per 16 tokens, 16 mma.m16n8k32 scaled by row and token; sums holds the 4 results (rows g and g + 8 of the lane by
  // tokens 2 * (lane % 4) and the next) of the 4 x 4 tiles of 16 rows by 8 tokens, row tile first, whole adds up integers
  func.func private @strict_device_eights_sub(%sums: vector<64xf32>, %buffer: i32, %sub: i32, %warpRow: i32, %warpToken: i32, %lane: i32, %whole: i1) -> vector<64xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c1280 = llvm.mlir.constant(1280 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %nothing = arith.constant 0 : i32
    %rows = llvm.mlir.addressof @strict_device_eights_rows : !llvm.ptr<3>
    %tokens = llvm.mlir.addressof @strict_device_eights_tokens : !llvm.ptr<3>
    %rowScales = llvm.mlir.addressof @strict_device_eights_row_scales : !llvm.ptr<3>
    %tokenScales = llvm.mlir.addressof @strict_device_eights_token_scales : !llvm.ptr<3>
    %bufferStart = arith.muli %buffer, %c10240 : i32
    %scaleBuffer = arith.shli %buffer, %c8 : i32
    %subScales = arith.shli %sub, %c7 : i32
    %scaleStart = arith.addi %scaleBuffer, %subScales : i32
    %matrix = arith.shrui %lane, %c3 : i32
    %matrixRow = arith.andi %lane, %c7 : i32
    %matrixHalf = arith.andi %matrix, %c1 : i32
    %matrixPair = arith.shrui %matrix, %c1 : i32
    %subBytes = arith.shli %sub, %c5 : i32
    %halfRows = arith.shli %matrixHalf, %c3 : i32
    %rowInTile = arith.addi %halfRows, %matrixRow : i32
    %pairBytes = arith.shli %matrixPair, %c4 : i32
    %rowColumn = arith.addi %pairBytes, %subBytes : i32
    %pairRows = arith.shli %matrixPair, %c3 : i32
    %tokenInTile = arith.addi %pairRows, %matrixRow : i32
    %halfBytes = arith.shli %matrixHalf, %c4 : i32
    %tokenColumn = arith.addi %halfBytes, %subBytes : i32
    %group = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %pairToken = arith.shli %quad, %c1 : i32
    %laneToken = arith.addi %warpToken, %tokenInTile : i32
    %tokenRowStart = arith.muli %laneToken, %c80 : i32
    %tokenRowAt = arith.addi %tokenRowStart, %tokenColumn : i32
    %firstTokenAt = arith.addi %bufferStart, %tokenRowAt : i32
    %secondTokenAt = arith.addi %firstTokenAt, %c1280 : i32
    %firstTokenSlot = func.call @strict_device_tile_at(%tokens, %firstTokenAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %secondTokenSlot = func.call @strict_device_tile_at(%tokens, %secondTokenAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %firstTokens = func.call @strict_device_matrices(%firstTokenSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %secondTokens = func.call @strict_device_matrices(%secondTokenSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %t0 = llvm.extractvalue %firstTokens[0] : !llvm.struct<(i32, i32, i32, i32)>
    %t1 = llvm.extractvalue %firstTokens[1] : !llvm.struct<(i32, i32, i32, i32)>
    %t2 = llvm.extractvalue %firstTokens[2] : !llvm.struct<(i32, i32, i32, i32)>
    %t3 = llvm.extractvalue %firstTokens[3] : !llvm.struct<(i32, i32, i32, i32)>
    %t4 = llvm.extractvalue %secondTokens[0] : !llvm.struct<(i32, i32, i32, i32)>
    %t5 = llvm.extractvalue %secondTokens[1] : !llvm.struct<(i32, i32, i32, i32)>
    %t6 = llvm.extractvalue %secondTokens[2] : !llvm.struct<(i32, i32, i32, i32)>
    %t7 = llvm.extractvalue %secondTokens[3] : !llvm.struct<(i32, i32, i32, i32)>
    %firstHalf = func.call @strict_device_words(%t0, %t1, %t2, %t3) : (i32, i32, i32, i32) -> vector<4xi32>
    %secondHalf = func.call @strict_device_words(%t4, %t5, %t6, %t7) : (i32, i32, i32, i32) -> vector<4xi32>
    %tokenOperands = llvm.shufflevector %firstHalf, %secondHalf [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xi32>
    %laneTokenScales = arith.addi %scaleStart, %warpToken : i32
    %firstScaleAt = arith.addi %laneTokenScales, %pairToken : i32
    %noScales = arith.constant dense<0.0> : vector<8xf32>
    cf.cond_br %whole, ^rowTiles(%noScales : vector<8xf32>), ^scale(%c0, %firstScaleAt, %noScales : i32, i32, vector<8xf32>)
  ^scale(%scaleTile: i32, %scaleAt: i32, %tokenScaleValues: vector<8xf32>):
    %moreScales = arith.cmpi slt, %scaleTile, %c4 : i32
    cf.cond_br %moreScales, ^scaleBody, ^rowTiles(%tokenScaleValues : vector<8xf32>)
  ^scaleBody:
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleSlot = llvm.getelementptr %tokenScales[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %pairScales = llvm.load %scaleSlot {alignment = 8 : i64} : !llvm.ptr<3> -> vector<2xf32>
    %lowScale = llvm.extractelement %pairScales[%c0 : i32] : vector<2xf32>
    %highScale = llvm.extractelement %pairScales[%c1 : i32] : vector<2xf32>
    %lowIndex = arith.shli %scaleTile, %c1 : i32
    %highIndex = arith.addi %lowIndex, %c1 : i32
    %withLow = llvm.insertelement %lowScale, %tokenScaleValues[%lowIndex : i32] : vector<8xf32>
    %withHigh = llvm.insertelement %highScale, %withLow[%highIndex : i32] : vector<8xf32>
    %scaleTileNext = arith.addi %scaleTile, %c1 : i32
    %scaleAtNext = arith.addi %scaleAt, %c8 : i32
    llvm.br ^scale(%scaleTileNext, %scaleAtNext, %withHigh : i32, i32, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^rowTiles(%tileScales: vector<8xf32>):
    cf.br ^rowTile(%c0, %sums : i32, vector<64xf32>)
  ^rowTile(%rowTileIndex: i32, %rowSums: vector<64xf32>):
    %moreRowTiles = arith.cmpi slt, %rowTileIndex, %c4 : i32
    cf.cond_br %moreRowTiles, ^rowTileBody, ^done
  ^rowTileBody:
    %tileRows = arith.shli %rowTileIndex, %c4 : i32
    %tileRow = arith.addi %warpRow, %tileRows : i32
    %laneRow = arith.addi %tileRow, %rowInTile : i32
    %rowStart = arith.muli %laneRow, %c80 : i32
    %rowAt = arith.addi %rowStart, %rowColumn : i32
    %rowBufferAt = arith.addi %bufferStart, %rowAt : i32
    %rowSlot = func.call @strict_device_tile_at(%rows, %rowBufferAt) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %rowOperands = func.call @strict_device_matrices(%rowSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %a0 = llvm.extractvalue %rowOperands[0] : !llvm.struct<(i32, i32, i32, i32)>
    %a1 = llvm.extractvalue %rowOperands[1] : !llvm.struct<(i32, i32, i32, i32)>
    %a2 = llvm.extractvalue %rowOperands[2] : !llvm.struct<(i32, i32, i32, i32)>
    %a3 = llvm.extractvalue %rowOperands[3] : !llvm.struct<(i32, i32, i32, i32)>
    %groupRow = arith.addi %tileRow, %group : i32
    %upperScaleAt = arith.addi %scaleStart, %groupRow : i32
    %lowerScaleAt = arith.addi %upperScaleAt, %c8 : i32
    %upperScale = func.call @strict_device_tile_float(%rowScales, %upperScaleAt) : (!llvm.ptr<3>, i32) -> f32
    %lowerScale = func.call @strict_device_tile_float(%rowScales, %lowerScaleAt) : (!llvm.ptr<3>, i32) -> f32
    %tileSlots = arith.shli %rowTileIndex, %c4 : i32
    cf.br ^tokenTile(%c0, %rowSums : i32, vector<64xf32>)
  ^tokenTile(%tokenTileIndex: i32, %tokenSums: vector<64xf32>):
    %moreTokenTiles = arith.cmpi slt, %tokenTileIndex, %c4 : i32
    cf.cond_br %moreTokenTiles, ^tokenTileBody, ^rowTileNext
  ^tokenTileBody:
    %lowOperand = arith.shli %tokenTileIndex, %c1 : i32
    %highOperand = arith.addi %lowOperand, %c1 : i32
    %b0 = llvm.extractelement %tokenOperands[%lowOperand : i32] : vector<8xi32>
    %b1 = llvm.extractelement %tokenOperands[%highOperand : i32] : vector<8xi32>
    %tokenSlots = arith.shli %tokenTileIndex, %c2 : i32
    %slot0 = arith.addi %tileSlots, %tokenSlots : i32
    cf.cond_br %whole, ^wholeTile, ^scaledTile
  ^wholeTile:
    %wholeSums = func.call @strict_device_eights_whole(%tokenSums, %slot0, %a0, %a1, %a2, %a3, %b0, %b1) : (vector<64xf32>, i32, i32, i32, i32, i32, i32, i32) -> vector<64xf32>
    cf.br ^tokenTileNext(%wholeSums : vector<64xf32>)
  ^scaledTile:
    %dot = func.call @strict_device_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %nothing, %nothing, %nothing, %nothing) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(i32, i32, i32, i32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(i32, i32, i32, i32)>
    %dot2 = llvm.extractvalue %dot[2] : !llvm.struct<(i32, i32, i32, i32)>
    %dot3 = llvm.extractvalue %dot[3] : !llvm.struct<(i32, i32, i32, i32)>
    %lowTokenScale = llvm.extractelement %tileScales[%lowOperand : i32] : vector<8xf32>
    %highTokenScale = llvm.extractelement %tileScales[%highOperand : i32] : vector<8xf32>
    %scale0 = arith.mulf %upperScale, %lowTokenScale : f32
    %scale1 = arith.mulf %upperScale, %highTokenScale : f32
    %scale2 = arith.mulf %lowerScale, %lowTokenScale : f32
    %scale3 = arith.mulf %lowerScale, %highTokenScale : f32
    %slot1 = arith.addi %slot0, %c1 : i32
    %slot2 = arith.addi %slot0, %c2 : i32
    %slot3 = arith.addi %slot0, %c3 : i32
    %with0 = func.call @strict_device_eights_added(%tokenSums, %slot0, %dot0, %scale0) : (vector<64xf32>, i32, i32, f32) -> vector<64xf32>
    %with1 = func.call @strict_device_eights_added(%with0, %slot1, %dot1, %scale1) : (vector<64xf32>, i32, i32, f32) -> vector<64xf32>
    %with2 = func.call @strict_device_eights_added(%with1, %slot2, %dot2, %scale2) : (vector<64xf32>, i32, i32, f32) -> vector<64xf32>
    %with3 = func.call @strict_device_eights_added(%with2, %slot3, %dot3, %scale3) : (vector<64xf32>, i32, i32, f32) -> vector<64xf32>
    cf.br ^tokenTileNext(%with3 : vector<64xf32>)
  ^tokenTileNext(%nextSums: vector<64xf32>):
    %tokenTileNext = arith.addi %tokenTileIndex, %c1 : i32
    llvm.br ^tokenTile(%tokenTileNext, %nextSums : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^rowTileNext:
    %rowTileFollowing = arith.addi %rowTileIndex, %c1 : i32
    llvm.br ^rowTile(%rowTileFollowing, %tokenSums : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %rowSums : vector<64xf32>
  }
  // the sums of the stages of 64 columns of a tile (two shared memory buffers in turn, the next stage loaded into
  // registers while one computes), whole for rows with one scale
  func.func private @strict_device_eights_stages(%firstData: !llvm.ptr, %stride: i64, %rowFirst: i64, %rowLast: i64, %columns: i64, %input: !llvm.ptr, %inputScales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %thread: i32, %warpRow: i32, %warpToken: i32, %lane: i32, %whole: i1) -> vector<64xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %six = arith.constant 6 : i64
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %stages = arith.shrui %columns, %six : i64
    %lastStage = arith.subi %stages, %one : i64
    %firstWords, %firstScales = func.call @strict_device_eights_fetch(%firstData, %stride, %rowFirst, %rowLast, %columns, %input, %inputScales, %width, %tokenFirst, %tokenLast, %none, %thread, %whole) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i32, i1) -> (vector<16xi32>, vector<2xf32>)
    func.call @strict_device_eights_put(%firstWords, %firstScales, %c0, %thread, %whole) : (vector<16xi32>, vector<2xf32>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%none, %zeros : i64, vector<64xf32>)
  ^stage(%stage: i64, %sums: vector<64xf32>):
    %moreStages = arith.cmpi slt, %stage, %stages : i64
    cf.cond_br %moreStages, ^compute, ^done
  ^compute:
    %nextStage = arith.addi %stage, %one : i64
    %fetchedStage = arith.minsi %nextStage, %lastStage : i64
    %nextWords, %nextScales = func.call @strict_device_eights_fetch(%firstData, %stride, %rowFirst, %rowLast, %columns, %input, %inputScales, %width, %tokenFirst, %tokenLast, %fetchedStage, %thread, %whole) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i32, i1) -> (vector<16xi32>, vector<2xf32>)
    %stageParity = arith.andi %stage, %one : i64
    %buffer = arith.trunci %stageParity : i64 to i32
    %firstSub = func.call @strict_device_eights_sub(%sums, %buffer, %c0, %warpRow, %warpToken, %lane, %whole) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %bothSubs = func.call @strict_device_eights_sub(%firstSub, %buffer, %c1, %warpRow, %warpToken, %lane, %whole) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %hasNext = arith.cmpi slt, %nextStage, %stages : i64
    cf.cond_br %hasNext, ^put, ^stage(%nextStage, %bothSubs : i64, vector<64xf32>)
  ^put:
    %otherBuffer = arith.xori %buffer, %c1 : i32
    func.call @strict_device_eights_put(%nextWords, %nextScales, %otherBuffer, %thread, %whole) : (vector<16xi32>, vector<2xf32>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%nextStage, %bothSubs : i64, vector<64xf32>)
  ^done:
    return %sums : vector<64xf32>
  }
  // each warp computes 64 rows by 32 tokens of the tile, the dot of row r and token t becomes number t * rows + r of the
  // result
  func.func private @strict_device_eights_run(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64) attributes {passthrough = ["alwaysinline"]} {
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
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %sixtyThree = arith.constant 63 : i64
    %oneHundredTwentySeven = arith.constant 127 : i64
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %block = arith.extui %block32 : i32 to i64
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %rowsGiven = arith.fptosi %rowsNumber : f64 to i64
    %rows = arith.maxsi %rowsGiven, %one : i64
    %width = arith.fptosi %widthNumber : f64 to i64
    %tokens = arith.divui %count, %rows : i64
    %tokensUp = arith.addi %tokens, %oneHundredTwentySeven : i64
    %tokenTiles = arith.shrui %tokensUp, %seven : i64
    %tokenTile = arith.remui %block, %tokenTiles : i64
    %rowTile = arith.divui %block, %tokenTiles : i64
    %rowFirst = arith.shli %rowTile, %seven : i64
    %tokenFirst = arith.shli %tokenTile, %seven : i64
    %rowLast = arith.subi %rows, %one : i64
    %tokenLast = arith.subi %tokens, %one : i64
    %isQuantized = arith.cmpi eq, %inputBits, %eight : i64
    %widthRest = arith.andi %width, %sixtyThree : i64
    %widthAligned = arith.cmpi eq, %widthRest, %none : i64
    %sameWidth = arith.cmpi eq, %columns, %width : i64
    %alignedSame = arith.andi %widthAligned, %sameWidth : i1
    %usable = arith.andi %isQuantized, %alignedSame : i1
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
    %firstData, %rowBits, %rowTotal = func.call @strict_device_row(%rowsData, %rowFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %rowsWord = llvm.load %rowsData : !llvm.ptr -> i64
    %lowHalf = arith.constant 2147483647 : i64
    %stride = arith.andi %rowsWord, %lowHalf : i64
    %stages = arith.shrui %columns, %six : i64
    %lastStage = arith.subi %stages, %one : i64
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpTokenIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.shli %warpRowIndex, %c6 : i32
    %warpToken = arith.shli %warpTokenIndex, %c5 : i32
    %wholeBit = arith.constant 31 : i64
    %wholeShifted = arith.shrui %rowsWord, %wholeBit : i64
    %wholeFlag = arith.andi %wholeShifted, %one : i64
    %whole = arith.cmpi ne, %wholeFlag, %none : i64
    %true = arith.constant true
    %false = arith.constant false
    cf.cond_br %whole, ^wholeStages, ^scaledStages
  ^wholeStages:
    %wholeSums = func.call @strict_device_eights_stages(%firstData, %stride, %rowFirst, %rowLast, %columns, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %warpRow, %warpToken, %lane, %true) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, i1) -> vector<64xf32>
    cf.br ^written(%wholeSums : vector<64xf32>)
  ^scaledStages:
    %scaledSums = func.call @strict_device_eights_stages(%firstData, %stride, %rowFirst, %rowLast, %columns, %input, %inputScales, %width, %tokenFirst, %tokenLast, %thread, %warpRow, %warpToken, %lane, %false) : (!llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, i1) -> vector<64xf32>
    cf.br ^written(%scaledSums : vector<64xf32>)
  ^written(%sums: vector<64xf32>):
    %group = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %pairToken = arith.shli %quad, %c1 : i32
    cf.br ^write(%c0 : i32)
  ^write(%slot: i32):
    %moreSlots = arith.cmpi slt, %slot, %c64 : i32
    cf.cond_br %moreSlots, ^writeBody, ^exit
  ^writeBody:
    %rowTileIndex = arith.shrui %slot, %c4 : i32
    %slotPair = arith.shrui %slot, %c2 : i32
    %tokenTileIndex = arith.andi %slotPair, %c3 : i32
    %element = arith.andi %slot, %c3 : i32
    %lowerHalf = arith.shrui %element, %c1 : i32
    %oddToken = arith.andi %element, %c1 : i32
    %tileRows = arith.shli %rowTileIndex, %c4 : i32
    %lowerRows = arith.shli %lowerHalf, %c3 : i32
    %warpTileRow = arith.addi %warpRow, %tileRows : i32
    %groupRow = arith.addi %warpTileRow, %group : i32
    %rowLocal = arith.addi %groupRow, %lowerRows : i32
    %tileTokens = arith.shli %tokenTileIndex, %c3 : i32
    %warpTileToken = arith.addi %warpToken, %tileTokens : i32
    %pairTokenLocal = arith.addi %warpTileToken, %pairToken : i32
    %tokenLocal = arith.addi %pairTokenLocal, %oddToken : i32
    %rowLocal64 = arith.extui %rowLocal : i32 to i64
    %tokenLocal64 = arith.extui %tokenLocal : i32 to i64
    %row = arith.addi %rowFirst, %rowLocal64 : i64
    %token = arith.addi %tokenFirst, %tokenLocal64 : i64
    %rowUsed = arith.cmpi sle, %row, %rowLast : i64
    %tokenUsed = arith.cmpi sle, %token, %tokenLast : i64
    %writing = arith.andi %rowUsed, %tokenUsed : i1
    cf.cond_br %writing, ^store, ^writeNext
  ^store:
    %value = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    cf.cond_br %whole, ^wholeValue, ^stored(%value : f32)
  ^wholeValue:
    %integer = arith.bitcast %value : f32 to i32
    %integerFloat = arith.sitofp %integer : i32 to f32
    %scaleRowNumber = arith.subi %row, %rowFirst : i64
    %scaleRowStart = arith.muli %scaleRowNumber, %stride : i64
    %scaleRowAt = arith.addi %scaleRowStart, %columns : i64
    %rowScaleSource = llvm.getelementptr %firstData[%scaleRowAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowScale = llvm.load %rowScaleSource invariant : !llvm.ptr -> f32
    %scaleTokenStart = arith.muli %token, %width : i64
    %five = arith.constant 5 : i64
    %scaleTokenIndex = arith.shrui %scaleTokenStart, %five : i64
    %tokenScaleSource = llvm.getelementptr %inputScales[%scaleTokenIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %tokenScale = llvm.load %tokenScaleSource invariant : !llvm.ptr -> f32
    %bothScales = arith.mulf %rowScale, %tokenScale : f32
    %wholeValue = arith.mulf %integerFloat, %bothScales : f32
    cf.br ^stored(%wholeValue : f32)
  ^stored(%final: f32):
    %tokenStart = arith.muli %token, %rows : i64
    %resultIndex = arith.addi %tokenStart, %row : i64
    func.call @strict_device_tokens_store(%result, %wide, %resultIndex, %final, %row, %rows, %biases, %biasesCount, %biasesBits) : (!llvm.ptr, i1, i64, f32, i64, i64, !llvm.ptr, i64, i64) -> ()
    cf.br ^writeNext
  ^writeNext:
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^exit:
    return
  }
  llvm.func @strict_device_eights(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_eights_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_eightss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_eights_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_tiles(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tiles_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_tiless(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tiles_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_tokens(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tokens_rows(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_blocks(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tokens_blocks(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_blockss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tokens_blocks(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
  llvm.func @strict_device_tokenss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %biases: !llvm.ptr, %biasesCount: i64, %biasesBits: i64, %bases: !llvm.ptr, %basesCount: i64, %basesBits: i64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tokens_rows(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %wide, %biases, %biasesCount, %biasesBits, %bases, %basesCount, %basesBits) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
