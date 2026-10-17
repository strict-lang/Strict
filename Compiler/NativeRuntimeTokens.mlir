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
    cf.cond_br %blocked, ^check, ^plain(%none, %zeros : i64, vector<8xf32>)
  ^check:
    %fits = func.call @strict_device_fits8(%rowData, %rowBits, %rowCount, %bytes) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %fits, ^eights, ^plain(%none, %zeros : i64, vector<8xf32>)
  ^eights:
    %data = func.call @strict_device_data(%inputCount, %eight) : (i64, i64) -> i64
    %allScales = llvm.getelementptr %input[%data] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleStart = arith.shrui %start, %five : i64
    %scales = llvm.getelementptr %allScales[%scaleStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sums = func.call @strict_device_dots8_tokens(%rowData, %rowBits, %rowCount, %bytes, %scales, %width, %relativeLast, %lane, %thirtyTwo) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> vector<8xf32>
    return %sums : vector<8xf32>
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
    llvm.br ^plain(%following, %updated : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^plainDone:
    return %plainSums : vector<8xf32>
  }
  // one step of 16 numbers of an 8 by 32 tile with the int8 tensor cores: the 4 bytes a of this lane's row (row lane / 4,
  // numbers 4 * (lane % 4) on) times the 8 bit numbers of 4 groups of 8 tokens, scaled and added to the 8 sums of the lane
  // (tokens 2 * (lane % 4) and the next of each group), minimum times the token sums is subtracted (Q4_K)
  func.func private @strict_device_tile_step(%sums: vector<8xf32>, %a: i32, %scale: f32, %minimum: f32, %bytes: !llvm.ptr, %scales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %group: i64, %part: i64, %column: i64) -> vector<8xf32> attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %nothing = arith.constant 0 : i32
    %ones = arith.constant 16843009 : i32
    %offset = arith.muli %part, %four : i64
    %own = arith.addi %column, %offset : i64
    %shifted = arith.shrui %column, %five : i64
    cf.br ^group(%none, %sums : i64, vector<8xf32>)
  ^group(%index: i64, %total: vector<8xf32>):
    %more = arith.cmpi slt, %index, %four : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %groupStart = arith.muli %index, %eight : i64
    %first = arith.addi %tokenFirst, %groupStart : i64
    %tokenAt = arith.addi %first, %group : i64
    %token = arith.minsi %tokenAt, %tokenLast : i64
    %tokenStart = arith.muli %token, %width : i64
    %inputAt = arith.addi %tokenStart, %own : i64
    %inputSlot = llvm.getelementptr %bytes[%inputAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %b = llvm.load %inputSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %dot = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%a, %b, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %summed = llvm.call_intrinsic "llvm.nvvm.mma.m8n8k16.row.col.s8"(%ones, %b, %nothing, %nothing) : (i32, i32, i32, i32) -> !llvm.struct<(i32, i32)>
    %pairStart = arith.muli %part, %two : i64
    %outFirst = arith.addi %first, %pairStart : i64
    %outSecond = arith.addi %outFirst, %one : i64
    %firstToken = arith.minsi %outFirst, %tokenLast : i64
    %secondToken = arith.minsi %outSecond, %tokenLast : i64
    %firstScaleRow = arith.muli %firstToken, %width : i64
    %secondScaleRow = arith.muli %secondToken, %width : i64
    %firstScaleStart = arith.shrui %firstScaleRow, %five : i64
    %secondScaleStart = arith.shrui %secondScaleRow, %five : i64
    %firstScaleAt = arith.addi %firstScaleStart, %shifted : i64
    %secondScaleAt = arith.addi %secondScaleStart, %shifted : i64
    %firstScaleSlot = llvm.getelementptr %scales[%firstScaleAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %firstInputScale = llvm.load %firstScaleSlot invariant : !llvm.ptr -> f32
    %secondScaleSlot = llvm.getelementptr %scales[%secondScaleAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %secondInputScale = llvm.load %secondScaleSlot invariant : !llvm.ptr -> f32
    %firstDot = llvm.extractvalue %dot[0] : !llvm.struct<(i32, i32)>
    %secondDot = llvm.extractvalue %dot[1] : !llvm.struct<(i32, i32)>
    %firstTotal = llvm.extractvalue %summed[0] : !llvm.struct<(i32, i32)>
    %secondTotal = llvm.extractvalue %summed[1] : !llvm.struct<(i32, i32)>
    %firstDotFloat = arith.sitofp %firstDot : i32 to f32
    %secondDotFloat = arith.sitofp %secondDot : i32 to f32
    %firstTotalFloat = arith.sitofp %firstTotal : i32 to f32
    %secondTotalFloat = arith.sitofp %secondTotal : i32 to f32
    %firstScaled = arith.mulf %scale, %firstDotFloat : f32
    %firstShifted = arith.mulf %minimum, %firstTotalFloat : f32
    %firstPart = arith.subf %firstScaled, %firstShifted : f32
    %firstContribution = arith.mulf %firstPart, %firstInputScale : f32
    %secondScaled = arith.mulf %scale, %secondDotFloat : f32
    %secondShifted = arith.mulf %minimum, %secondTotalFloat : f32
    %secondPart = arith.subf %secondScaled, %secondShifted : f32
    %secondContribution = arith.mulf %secondPart, %secondInputScale : f32
    %firstSlot = arith.muli %index, %two : i64
    %secondSlot = arith.addi %firstSlot, %one : i64
    %firstOld = llvm.extractelement %total[%firstSlot : i64] : vector<8xf32>
    %firstNew = arith.addf %firstOld, %firstContribution : f32
    %withFirst = llvm.insertelement %firstNew, %total[%firstSlot : i64] : vector<8xf32>
    %secondOld = llvm.extractelement %withFirst[%secondSlot : i64] : vector<8xf32>
    %secondNew = arith.addf %secondOld, %secondContribution : f32
    %withSecond = llvm.insertelement %secondNew, %withFirst[%secondSlot : i64] : vector<8xf32>
    %following = arith.addi %index, %one : i64
    llvm.br ^group(%following, %withSecond : i64, vector<8xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %total : vector<8xf32>
  }
  // the sums of a lane in an 8 by 32 tile of a row of Q4_K blocks (Q4_K numbers fit the 8 bit tensor cores as they are)
  func.func private @strict_device_four_tile(%row: !llvm.ptr, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %group: i64, %part: i64) -> vector<8xf32> attributes {passthrough = ["noinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %mask = arith.constant 252645135 : i32
    %blockBytes = arith.constant 144 : i64
    %sixteen = arith.constant 16 : i64
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %steps = arith.shrui %count, %four : i64
    %partBytes = arith.muli %part, %four : i64
    cf.br ^step(%none, %zeros : i64, vector<8xf32>)
  ^step(%step: i64, %sums: vector<8xf32>):
    %more = arith.cmpi slt, %step, %steps : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %block = arith.shrui %step, %four : i64
    %fifteen = arith.constant 15 : i64
    %inBlock = arith.andi %step, %fifteen : i64
    %sub = arith.shrui %inBlock, %one : i64
    %half = arith.andi %inBlock, %one : i64
    %pair = arith.shrui %sub, %one : i64
    %high = arith.andi %sub, %one : i64
    %blockOffset = arith.muli %block, %blockBytes : i64
    %blockStart = llvm.getelementptr %row[%blockOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %header = llvm.load %blockStart invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %pairBytes = arith.shli %pair, %five : i64
    %halfBytes = arith.shli %half, %four : i64
    %inPair = arith.addi %pairBytes, %halfBytes : i64
    %withPart = arith.addi %inPair, %partBytes : i64
    %wordAt = arith.addi %withPart, %sixteen : i64
    %wordSlot = llvm.getelementptr %blockStart[%wordAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %word = llvm.load %wordSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %high32 = arith.trunci %high : i64 to i32
    %shift = arith.shli %high32, %c1 : i32
    %shiftTwice = arith.shli %shift, %c1 : i32
    %moved = arith.shrui %word, %shiftTwice : i32
    %a = arith.andi %moved, %mask : i32
    %headerWord = llvm.extractelement %header[%c0 : i32] : vector<4xi32>
    %halves = llvm.bitcast %headerWord : i32 to vector<2xf16>
    %halfScale = llvm.extractelement %halves[%c0 : i32] : vector<2xf16>
    %halfMinimum = llvm.extractelement %halves[%c1 : i32] : vector<2xf16>
    %blockScale = arith.extf %halfScale : f16 to f32
    %blockMinimum = arith.extf %halfMinimum : f16 to f32
    %sub32 = arith.trunci %sub : i64 to i32
    %scaleBits, %minimumBits = func.call @strict_device_four_scale(%header, %sub32) : (vector<4xi32>, i32) -> (i32, i32)
    %scaleFloat = arith.uitofp %scaleBits : i32 to f32
    %minimumFloat = arith.uitofp %minimumBits : i32 to f32
    %scale = arith.mulf %blockScale, %scaleFloat : f32
    %minimum = arith.mulf %blockMinimum, %minimumFloat : f32
    %column = arith.shli %step, %four : i64
    %next = func.call @strict_device_tile_step(%sums, %a, %scale, %minimum, %bytes, %scales, %width, %tokenFirst, %tokenLast, %group, %part, %column) : (vector<8xf32>, i32, f32, f32, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %following = arith.addi %step, %one : i64
    cf.br ^step(%following, %next : i64, vector<8xf32>)
  ^done:
    return %sums : vector<8xf32>
  }
  // the sums of a lane in an 8 by 32 tile of a row of Q6_K blocks: 16 numbers of a step share one scale
  func.func private @strict_device_six_tile(%row: !llvm.ptr, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %group: i64, %part: i64) -> vector<8xf32> attributes {passthrough = ["noinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %lowNibbles = arith.constant 252645135 : i32
    %lowPairs = arith.constant 50529027 : i32
    %signs = arith.constant -2139062144 : i32
    %centers = arith.constant 538976288 : i32
    %blockBytes = arith.constant 208 : i64
    %highStart = arith.constant 128 : i64
    %scaleStart = arith.constant 192 : i64
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %steps = arith.shrui %count, %four : i64
    %partBytes = arith.muli %part, %four : i64
    %dataBytes = func.call @strict_device_data(%count, %six) : (i64, i64) -> i64
    %blockScales = llvm.getelementptr %row[%dataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.br ^step(%none, %zeros : i64, vector<8xf32>)
  ^step(%step: i64, %sums: vector<8xf32>):
    %more = arith.cmpi slt, %step, %steps : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %block = arith.shrui %step, %four : i64
    %inBlock = arith.andi %step, %fifteen : i64
    %half = arith.shrui %inBlock, %three : i64
    %quarters = arith.shrui %inBlock, %one : i64
    %quarter = arith.andi %quarters, %three : i64
    %stepHalf = arith.andi %inBlock, %one : i64
    %stepBytes = arith.shli %stepHalf, %four : i64
    %inQuarter = arith.addi %stepBytes, %partBytes : i64
    %blockOffset = arith.muli %block, %blockBytes : i64
    %blockStart = llvm.getelementptr %row[%blockOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowHalf = arith.shli %half, %six : i64
    %oddQuarter = arith.andi %quarter, %one : i64
    %oddBytes = arith.shli %oddQuarter, %five : i64
    %lowBase = arith.addi %lowHalf, %oddBytes : i64
    %lowAt = arith.addi %lowBase, %inQuarter : i64
    %lowSlot = llvm.getelementptr %blockStart[%lowAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %low = llvm.load %lowSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %highHalf = arith.shli %half, %five : i64
    %highBase = arith.addi %highStart, %highHalf : i64
    %highAt = arith.addi %highBase, %inQuarter : i64
    %highSlot = llvm.getelementptr %blockStart[%highAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %highBits = llvm.load %highSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %upperQuarter = arith.shrui %quarter, %one : i64
    %upper32 = arith.trunci %upperQuarter : i64 to i32
    %nibbleShiftHalf = arith.shli %upper32, %c1 : i32
    %nibbleShift = arith.shli %nibbleShiftHalf, %c1 : i32
    %lowMoved = arith.shrui %low, %nibbleShift : i32
    %nibbles = arith.andi %lowMoved, %lowNibbles : i32
    %quarter32 = arith.trunci %quarter : i64 to i32
    %pairShift = arith.shli %quarter32, %c1 : i32
    %highMoved = arith.shrui %highBits, %pairShift : i32
    %pairs = arith.andi %highMoved, %lowPairs : i32
    %fourShift = arith.constant 4 : i32
    %top = arith.shli %pairs, %fourShift : i32
    %unsigned = arith.ori %nibbles, %top : i32
    %marked = arith.ori %unsigned, %signs : i32
    %lowered = arith.subi %marked, %centers : i32
    %a = arith.xori %lowered, %signs : i32
    %halfScales = arith.shli %half, %three : i64
    %quarterScales = arith.shli %quarter, %one : i64
    %scaleBase = arith.addi %halfScales, %quarterScales : i64
    %scaleIndex = arith.addi %scaleBase, %stepHalf : i64
    %scaleAt = arith.addi %scaleStart, %scaleIndex : i64
    %scaleSlot = llvm.getelementptr %blockStart[%scaleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleByte = llvm.load %scaleSlot invariant : !llvm.ptr -> i8
    %scaleFloat = arith.sitofp %scaleByte : i8 to f32
    %halfSlot = llvm.getelementptr %blockScales[%block] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfScale = llvm.load %halfSlot invariant : !llvm.ptr -> f16
    %blockScale = arith.extf %halfScale : f16 to f32
    %scale = arith.mulf %blockScale, %scaleFloat : f32
    %column = arith.shli %step, %four : i64
    %next = func.call @strict_device_tile_step(%sums, %a, %scale, %zero, %bytes, %scales, %width, %tokenFirst, %tokenLast, %group, %part, %column) : (vector<8xf32>, i32, f32, f32, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %following = arith.addi %step, %one : i64
    cf.br ^step(%following, %next : i64, vector<8xf32>)
  ^done:
    return %sums : vector<8xf32>
  }
  // the sums of a lane in an 8 by 32 tile of an 8 bit row (a scale per 32 numbers after them)
  func.func private @strict_device_byte_tile(%row: !llvm.ptr, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %width: i64, %tokenFirst: i64, %tokenLast: i64, %group: i64, %part: i64) -> vector<8xf32> attributes {passthrough = ["noinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %steps = arith.shrui %count, %four : i64
    %partBytes = arith.muli %part, %four : i64
    %dataBytes = func.call @strict_device_data(%count, %eight) : (i64, i64) -> i64
    %rowScales = llvm.getelementptr %row[%dataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.br ^step(%none, %zeros : i64, vector<8xf32>)
  ^step(%step: i64, %sums: vector<8xf32>):
    %more = arith.cmpi slt, %step, %steps : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %column = arith.shli %step, %four : i64
    %own = arith.addi %column, %partBytes : i64
    %wordSlot = llvm.getelementptr %row[%own] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %a = llvm.load %wordSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %scaleIndex = arith.shrui %column, %five : i64
    %scaleSlot = llvm.getelementptr %rowScales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %scale = llvm.load %scaleSlot invariant : !llvm.ptr -> f32
    %next = func.call @strict_device_tile_step(%sums, %a, %scale, %zero, %bytes, %scales, %width, %tokenFirst, %tokenLast, %group, %part, %column) : (vector<8xf32>, i32, f32, f32, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %following = arith.addi %step, %one : i64
    cf.br ^step(%following, %next : i64, vector<8xf32>)
  ^done:
    return %sums : vector<8xf32>
  }
  // `Batch.Sums` over a prompt of 32 tokens or more: a warp takes a tile of 8 rows and 32 tokens, through the tensor
  // cores when its rows share a packed format and the token numbers are 8 bit, otherwise row by row with 8 tokens per
  // step; the dot of row r and token t becomes number t * rows + r of the result (doubles when wide)
  func.func private @strict_device_tokens_tiles(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %wide: i1) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %thirtyTwo = arith.constant 32 : i64
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
    %rowsRounded = arith.addi %rows, %seven : i64
    %rowTiles = arith.shrui %rowsRounded, %three : i64
    %rowTile = arith.remui %warp, %rowTiles : i64
    %tokenTile = arith.divui %warp, %rowTiles : i64
    %rowFirst = arith.shli %rowTile, %three : i64
    %tokenFirst = arith.shli %tokenTile, %five : i64
    %inside = arith.cmpi slt, %tokenFirst, %tokens : i64
    cf.cond_br %inside, ^body, ^exit
  ^body:
    %rowLast = arith.subi %rows, %one : i64
    %tokenLast = arith.subi %tokens, %one : i64
    %group = arith.shrui %lane, %two : i64
    %part = arith.andi %lane, %three : i64
    %ownRow = arith.addi %rowFirst, %group : i64
    %row = arith.minsi %ownRow, %rowLast : i64
    %tileEnd = arith.addi %rowFirst, %seven : i64
    %tileLast = arith.minsi %tileEnd, %rowLast : i64
    %rowData, %rowBits, %rowTotal = func.call @strict_device_row(%rowsData, %row) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %firstData, %firstBits, %firstTotal = func.call @strict_device_row(%rowsData, %rowFirst) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %lastData, %lastBits, %lastTotal = func.call @strict_device_row(%rowsData, %tileLast) : (!llvm.ptr, i64) -> (!llvm.ptr, i64, i64)
    %sameBits = arith.cmpi eq, %firstBits, %lastBits : i64
    %quantized = arith.cmpi eq, %inputBits, %eight : i64
    %widthRest = arith.andi %width, %thirtyOne : i64
    %widthAligned = arith.cmpi eq, %widthRest, %none : i64
    %fits = func.call @strict_device_tile_fits(%firstBits, %columns) : (i64, i64) -> i1
    %uniform = arith.andi %sameBits, %fits : i1
    %eightBits = arith.andi %quantized, %widthAligned : i1
    %tiled = arith.andi %uniform, %eightBits : i1
    cf.cond_br %tiled, ^tile, ^rows(%none : i64)
  ^tile:
    %data = func.call @strict_device_data(%inputCount, %eight) : (i64, i64) -> i64
    %scales = llvm.getelementptr %input[%data] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.switch %rowBits : i64, [
      default: ^bytes,
      4: ^four,
      6: ^six
    ]
  ^four:
    %fourSums = func.call @strict_device_four_tile(%rowData, %columns, %input, %scales, %width, %tokenFirst, %tokenLast, %group, %part) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> vector<8xf32>
    cf.br ^tiled(%fourSums : vector<8xf32>)
  ^six:
    %sixSums = func.call @strict_device_six_tile(%rowData, %columns, %input, %scales, %width, %tokenFirst, %tokenLast, %group, %part) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> vector<8xf32>
    cf.br ^tiled(%sixSums : vector<8xf32>)
  ^bytes:
    %byteSums = func.call @strict_device_byte_tile(%rowData, %columns, %input, %scales, %width, %tokenFirst, %tokenLast, %group, %part) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> vector<8xf32>
    cf.br ^tiled(%byteSums : vector<8xf32>)
  ^tiled(%tileSums: vector<8xf32>):
    %ownsRow = arith.cmpi sle, %ownRow, %rowLast : i64
    cf.cond_br %ownsRow, ^tileWrite(%none : i64), ^exit
  ^tileWrite(%slot: i64):
    %slots = arith.cmpi slt, %slot, %eight : i64
    cf.cond_br %slots, ^tileToken, ^exit
  ^tileToken:
    %slotGroup = arith.shrui %slot, %one : i64
    %slotOdd = arith.andi %slot, %one : i64
    %groupStart = arith.shli %slotGroup, %three : i64
    %pairStart = arith.shli %part, %one : i64
    %inTile = arith.addi %groupStart, %pairStart : i64
    %withOdd = arith.addi %inTile, %slotOdd : i64
    %tileToken = arith.addi %tokenFirst, %withOdd : i64
    %tokenInside = arith.cmpi sle, %tileToken, %tokenLast : i64
    %tileValue = llvm.extractelement %tileSums[%slot : i64] : vector<8xf32>
    %tileTokenStart = arith.muli %tileToken, %rows : i64
    %tileElement = arith.addi %tileTokenStart, %ownRow : i64
    cf.cond_br %tokenInside, ^tileStore, ^tileNext
  ^tileStore:
    func.call @strict_device_tokens_store(%result, %wide, %tileElement, %tileValue) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^tileNext
  ^tileNext:
    %slotNext = arith.addi %slot, %one : i64
    llvm.br ^tileWrite(%slotNext : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^rows(%rowIndex: i64):
    %rowCurrent = arith.addi %rowFirst, %rowIndex : i64
    %rowInside = arith.cmpi sle, %rowCurrent, %rowLast : i64
    %rowSlots = arith.cmpi slt, %rowIndex, %eight : i64
    %rowMore = arith.andi %rowInside, %rowSlots : i1
    cf.cond_br %rowMore, ^chunks(%none : i64), ^exit
  ^chunks(%chunk: i64):
    %chunkStart = arith.shli %chunk, %three : i64
    %first = arith.addi %tokenFirst, %chunkStart : i64
    %chunkInside = arith.cmpi sle, %first, %tokenLast : i64
    %chunkSlots = arith.cmpi slt, %chunk, %four : i64
    %chunkMore = arith.andi %chunkInside, %chunkSlots : i1
    cf.cond_br %chunkMore, ^chunk, ^rowNext
  ^chunk:
    %chunkEnd = arith.addi %first, %seven : i64
    %last = arith.minsi %chunkEnd, %tokenLast : i64
    %sums = func.call @strict_device_tokens_sums(%failed, %rowsData, %rowCurrent, %columns, %input, %inputCount, %inputBits, %width, %first, %last, %lane) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> vector<8xf32>
    %isFirstLane = arith.cmpi eq, %lane, %none : i64
    cf.cond_br %isFirstLane, ^write(%none : i64), ^chunkNext
  ^write(%index: i64):
    %writes = arith.cmpi slt, %index, %eight : i64
    cf.cond_br %writes, ^token, ^chunkNext
  ^token:
    %token = arith.addi %first, %index : i64
    %writing = arith.cmpi sle, %token, %last : i64
    %value = llvm.extractelement %sums[%index : i64] : vector<8xf32>
    %tokenStart = arith.muli %token, %rows : i64
    %element = arith.addi %tokenStart, %rowCurrent : i64
    cf.cond_br %writing, ^store, ^stored
  ^store:
    func.call @strict_device_tokens_store(%result, %wide, %element, %value) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^stored
  ^stored:
    %next = arith.addi %index, %one : i64
    llvm.br ^write(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^chunkNext:
    %chunkFollowing = arith.addi %chunk, %one : i64
    cf.br ^chunks(%chunkFollowing : i64)
  ^rowNext:
    %rowFollowing = arith.addi %rowIndex, %one : i64
    cf.br ^rows(%rowFollowing : i64)
  ^exit:
    return
  }
  // `Batch.Sums` over fewer than 32 tokens: a warp takes one row and 8 tokens (more warps than tiles of 8 rows), the dot
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
  // a prompt of 32 tokens or more goes in tiles of 8 rows and 32 tokens, a shorter one row by row
  func.func private @strict_device_tokens_run(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %rowsData: !llvm.ptr, %columns: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %widthNumber: f64, %rowsNumber: f64, %tileTokens: f64, %wide: i1) {
    %sixteen = arith.constant 16.0 : f64
    %tiled = arith.cmpf ogt, %tileTokens, %sixteen : f64
    cf.cond_br %tiled, ^tiles, ^rows
  ^tiles:
    func.call @strict_device_tokens_tiles(%count, %result, %failed, %rowsData, %columns, %input, %inputCount, %inputBits, %widthNumber, %rowsNumber, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    return
  ^rows:
    func.call @strict_device_tokens_rows(%count, %result, %failed, %rowsData, %columns, %input, %inputCount, %inputBits, %widthNumber, %rowsNumber, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, i1) -> ()
    return
  }
  // a row format the tensor cores take: whole Q4_K or Q6_K blocks or 8 bit numbers in groups of 32
  func.func private @strict_device_tile_fits(%bits: i64, %count: i64) -> i1 attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %blockMask = arith.constant 255 : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %isEight = arith.cmpi eq, %bits, %eight : i64
    %isBlocked = arith.ori %isFour, %isSix : i1
    %blockRest = arith.andi %count, %blockMask : i64
    %wholeBlocks = arith.cmpi eq, %blockRest, %none : i64
    %wordRest = arith.andi %count, %thirtyOne : i64
    %wholeWords = arith.cmpi eq, %wordRest, %none : i64
    %blockedFits = arith.andi %isBlocked, %wholeBlocks : i1
    %eightFits = arith.andi %isEight, %wholeWords : i1
    %fits = arith.ori %blockedFits, %eightFits : i1
    return %fits : i1
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
  llvm.func @strict_device_tokens(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %tileTokens: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_tokens_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %tileTokens, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_tokenss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %rowsData: !llvm.ptr, %rowsCount: i64, %rowsBits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %width: f64, %rows: f64, %tileTokens: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_tokens_run(%count, %result, %failed, %rowsData, %rowsCount, %input, %inputCount, %inputBits, %width, %rows, %tileTokens, %wide) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, f64, f64, f64, i1) -> ()
    llvm.return
  }
