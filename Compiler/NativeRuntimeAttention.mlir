  // two tiles of 32 positions of keys and three of values in 16 bit floats, 136 apart per position (padded so the rows
  // of an ldmatrix lie in different banks)
  llvm.mlir.global internal @strict_device_attention_keys() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8704 x f16>
  llvm.mlir.global internal @strict_device_attention_values() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<13056 x f16>
  // 4 numbers from at of a list of doubles, 32 or 16 bit floats, bfloat16 (bits 17) or 8 bit floats (bits 7) as 32 bit
  // floats
  func.func private @strict_device_attention_read(%data: !llvm.ptr, %bits: i64, %at: i64) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    cf.switch %bits : i64, [
      default: ^doubles,
      32: ^singles,
      16: ^halves,
      17: ^brains,
      7: ^bytes
    ]
  ^brains:
    %brainSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %brainSingles = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    return %brainSingles : vector<4xf32>
  ^doubles:
    %doubleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubles = llvm.load %doubleSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xf64>
    %narrowed = arith.truncf %doubles : vector<4xf64> to vector<4xf32>
    return %narrowed : vector<4xf32>
  ^singles:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singles = llvm.load %singleSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %singles : vector<4xf32>
  ^halves:
    %halfSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halves = llvm.load %halfSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xf16>
    %widened = arith.extf %halves : vector<4xf16> to vector<4xf32>
    return %widened : vector<4xf32>
  ^bytes:
    %byteSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %bytes = llvm.load %byteSlot invariant {alignment = 4 : i64} : !llvm.ptr -> vector<4xi8>
    %byte0 = llvm.extractelement %bytes[%c0 : i32] : vector<4xi8>
    %byte1 = llvm.extractelement %bytes[%c1 : i32] : vector<4xi8>
    %byte2 = llvm.extractelement %bytes[%c2 : i32] : vector<4xi8>
    %byte3 = llvm.extractelement %bytes[%c3 : i32] : vector<4xi8>
    %number0 = func.call @strict_device_float8_number(%byte0) : (i8) -> f32
    %number1 = func.call @strict_device_float8_number(%byte1) : (i8) -> f32
    %number2 = func.call @strict_device_float8_number(%byte2) : (i8) -> f32
    %number3 = func.call @strict_device_float8_number(%byte3) : (i8) -> f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %with0 = llvm.insertelement %number0, %empty[%c0 : i32] : vector<4xf32>
    %with1 = llvm.insertelement %number1, %with0[%c1 : i32] : vector<4xf32>
    %with2 = llvm.insertelement %number2, %with1[%c2 : i32] : vector<4xf32>
    %with3 = llvm.insertelement %number3, %with2[%c3 : i32] : vector<4xf32>
    return %with3 : vector<4xf32>
  }
  // two bfloat16 (the first in the low bits) as the 16 bit pair of an mma operand
  func.func private @strict_device_attention_brains(%pair: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %upperMask = llvm.mlir.constant(-65536 : i32) : i32
    %lowBits = arith.shli %pair, %c16 : i32
    %highBits = arith.andi %pair, %upperMask : i32
    %first = llvm.bitcast %lowBits : i32 to f32
    %second = llvm.bitcast %highBits : i32 to f32
    %bits = func.call @strict_device_attention_pair(%first, %second) : (f32, f32) -> i32
    return %bits : i32
  }
  // two 32 bit floats as the 16 bit pair of an mma operand
  func.func private @strict_device_attention_pair(%first: f32, %second: f32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %firstHalf = arith.truncf %first : f32 to f16
    %secondHalf = arith.truncf %second : f32 to f16
    %empty = llvm.mlir.undef : vector<2xf16>
    %withFirst = llvm.insertelement %firstHalf, %empty[%c0 : i32] : vector<2xf16>
    %pair = llvm.insertelement %secondHalf, %withFirst[%c1 : i32] : vector<2xf16>
    %bits = llvm.bitcast %pair : vector<2xf16> to i32
    return %bits : i32
  }
  // the numbers of a tile of 32 positions from first (keys or values) a thread stores into shared memory, loaded
  // without waiting for them: parts thread + 256 * i of 4 numbers of the rows of size rounded up to 16, zeros after the
  // last position and after size, the bits of 32 bit floats or (bits 17) the bfloat16 pairs as loaded: converting them
  // here would wait for the loads before the block computes the current tile
  func.func private @strict_device_attention_fetch(%data: !llvm.ptr, %bits: i64, %first: i32, %positions: i32, %stride: i32, %base: i32, %size: i32, %thread: i32) -> vector<16xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.switch %bits : i64, [
      default: ^doubles,
      32: ^singles,
      16: ^halves,
      17: ^brains,
      8: ^eights,
      7: ^bytes
    ]
  ^eights:
    %eights = func.call @strict_device_attention_fetch_eights(%data, %first, %positions, %stride, %base, %thread) : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %eights : vector<16xi32>
  ^doubles:
    %doubleBits = arith.constant 64 : i64
    %doubles = func.call @strict_device_attention_fetched(%data, %doubleBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %doubles : vector<16xi32>
  ^singles:
    %singleBits = arith.constant 32 : i64
    %singles = func.call @strict_device_attention_fetched(%data, %singleBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %singles : vector<16xi32>
  ^halves:
    %halfBits = arith.constant 16 : i64
    %halves = func.call @strict_device_attention_fetched(%data, %halfBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %halves : vector<16xi32>
  ^brains:
    %brainBits = arith.constant 17 : i64
    %brains = func.call @strict_device_attention_fetched(%data, %brainBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %brains : vector<16xi32>
  ^bytes:
    %byteBits = arith.constant 7 : i64
    %bytes = func.call @strict_device_attention_fetched(%data, %byteBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    return %bytes : vector<16xi32>
  }
  func.func private @strict_device_attention_fetched(%data: !llvm.ptr, %bits: i64, %first: i32, %positions: i32, %stride: i32, %base: i32, %size: i32, %thread: i32) -> vector<16xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %brainBits = arith.constant 17 : i64
    %zeros = arith.constant dense<0> : vector<16xi32>
    %parts = func.call @strict_device_attention_parts(%size) : (i32) -> i32
    cf.br ^part(%c0, %zeros : i32, vector<16xi32>)
  ^part(%index: i32, %numbers: vector<16xi32>):
    %more = arith.cmpi slt, %index, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %offset = arith.muli %index, %c256 : i32
    %part = arith.addi %thread, %offset : i32
    %row = arith.divui %part, %parts : i32
    %partInRow = arith.remui %part, %parts : i32
    %number = arith.shli %partInRow, %c2 : i32
    %position = arith.addi %first, %row : i32
    %inTile = arith.cmpi slt, %row, %c32 : i32
    %inside = arith.cmpi slt, %position, %positions : i32
    %used = arith.cmpi slt, %number, %size : i32
    %present = arith.andi %inTile, %inside : i1
    %reading = arith.andi %present, %used : i1
    cf.cond_br %reading, ^load, ^next(%numbers : vector<16xi32>)
  ^load:
    %positionStart = arith.muli %position, %stride : i32
    %rowStart = arith.addi %positionStart, %base : i32
    %at = arith.addi %rowStart, %number : i32
    %at64 = arith.extui %at : i32 to i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brains, ^floats
  ^brains:
    %brainSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %pairs = llvm.load %brainSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %pairSlot = arith.shli %index, %c1 : i32
    %pairSlot1 = arith.addi %pairSlot, %c1 : i32
    %pair0 = llvm.extractelement %pairs[%c0 : i32] : vector<2xi32>
    %pair1 = llvm.extractelement %pairs[%c1 : i32] : vector<2xi32>
    %withPair0 = llvm.insertelement %pair0, %numbers[%pairSlot : i32] : vector<16xi32>
    %withPair1 = llvm.insertelement %pair1, %withPair0[%pairSlot1 : i32] : vector<16xi32>
    cf.br ^next(%withPair1 : vector<16xi32>)
  ^floats:
    %read = func.call @strict_device_attention_read(%data, %bits, %at64) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %words = llvm.bitcast %read : vector<4xf32> to vector<4xi32>
    %slot = arith.shli %index, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %read0 = llvm.extractelement %words[%c0 : i32] : vector<4xi32>
    %read1 = llvm.extractelement %words[%c1 : i32] : vector<4xi32>
    %read2 = llvm.extractelement %words[%c2 : i32] : vector<4xi32>
    %read3 = llvm.extractelement %words[%c3 : i32] : vector<4xi32>
    %with0 = llvm.insertelement %read0, %numbers[%slot : i32] : vector<16xi32>
    %with1 = llvm.insertelement %read1, %with0[%slot1 : i32] : vector<16xi32>
    %with2 = llvm.insertelement %read2, %with1[%slot2 : i32] : vector<16xi32>
    %with3 = llvm.insertelement %read3, %with2[%slot3 : i32] : vector<16xi32>
    cf.br ^next(%with3 : vector<16xi32>)
  ^next(%nextNumbers: vector<16xi32>):
    %following = arith.addi %index, %c1 : i32
    llvm.br ^part(%following, %nextNumbers : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %numbers : vector<16xi32>
  }
  // parts of 4 numbers per position of a tile: size rounded up to 16 so the last mma step reads zeros
  func.func private @strict_device_attention_parts(%size: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %wholeMask = llvm.mlir.constant(-16 : i32) : i32
    %spare = arith.addi %size, %c15 : i32
    %padded = arith.andi %spare, %wholeMask : i32
    %parts = arith.shrui %padded, %c2 : i32
    return %parts : i32
  }
  // where tile number buffer of 32 positions starts in keys or values
  func.func private @strict_device_attention_buffer(%tiles: !llvm.ptr<3>, %buffer: i32) -> !llvm.ptr<3> attributes {llvm.passthrough = ["alwaysinline"]} {
    %tileHalves = llvm.mlir.constant(4352 : i32) : i32
    %start = arith.muli %buffer, %tileHalves : i32
    %start64 = arith.extui %start : i32 to i64
    %tile = llvm.getelementptr %tiles[%start64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    return %tile : !llvm.ptr<3>
  }
  // the numbers of strict_device_attention_fetch as 16 bit floats into a tile in shared memory (136 numbers per
  // position so the 8 rows of an ldmatrix lie in different banks)
  func.func private @strict_device_attention_stored(%tile: !llvm.ptr<3>, %numbers: vector<16xi32>, %bits: i64, %size: i32, %thread: i32, %brain: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %brainBits = arith.constant 17 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c136 = llvm.mlir.constant(136 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %parts = func.call @strict_device_attention_parts(%size) : (i32) -> i32
    %total = arith.shli %parts, %c5 : i32
    cf.br ^part(%c0 : i32)
  ^part(%index: i32):
    %more = arith.cmpi slt, %index, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %offset = arith.muli %index, %c256 : i32
    %part = arith.addi %thread, %offset : i32
    %inTile = arith.cmpi slt, %part, %total : i32
    cf.cond_br %inTile, ^store, ^next
  ^store:
    %row = arith.divui %part, %parts : i32
    %partInRow = arith.remui %part, %parts : i32
    %number = arith.shli %partInRow, %c2 : i32
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brains, ^floats
  ^brains:
    %pairSlot = arith.shli %index, %c1 : i32
    %pairSlot1 = arith.addi %pairSlot, %c1 : i32
    %pair0 = llvm.extractelement %numbers[%pairSlot : i32] : vector<16xi32>
    %pair1 = llvm.extractelement %numbers[%pairSlot1 : i32] : vector<16xi32>
    cf.cond_br %brain, ^write(%pair0, %pair1 : i32, i32), ^halves
  ^halves:
    %lowBrains = func.call @strict_device_attention_brains(%pair0) : (i32) -> i32
    %highBrains = func.call @strict_device_attention_brains(%pair1) : (i32) -> i32
    cf.br ^write(%lowBrains, %highBrains : i32, i32)
  ^floats:
    %slot = arith.shli %index, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %numberBits0 = llvm.extractelement %numbers[%slot : i32] : vector<16xi32>
    %numberBits1 = llvm.extractelement %numbers[%slot1 : i32] : vector<16xi32>
    %numberBits2 = llvm.extractelement %numbers[%slot2 : i32] : vector<16xi32>
    %numberBits3 = llvm.extractelement %numbers[%slot3 : i32] : vector<16xi32>
    %number0 = llvm.bitcast %numberBits0 : i32 to f32
    %number1 = llvm.bitcast %numberBits1 : i32 to f32
    %number2 = llvm.bitcast %numberBits2 : i32 to f32
    %number3 = llvm.bitcast %numberBits3 : i32 to f32
    %lowFloats = func.call @strict_device_attention_pair(%number0, %number1) : (f32, f32) -> i32
    %highFloats = func.call @strict_device_attention_pair(%number2, %number3) : (f32, f32) -> i32
    cf.br ^write(%lowFloats, %highFloats : i32, i32)
  ^write(%low: i32, %high: i32):
    %empty = llvm.mlir.undef : vector<2xi32>
    %withLow = llvm.insertelement %low, %empty[%c0 : i32] : vector<2xi32>
    %both = llvm.insertelement %high, %withLow[%c1 : i32] : vector<2xi32>
    %rowStart = arith.muli %row, %c136 : i32
    %at = arith.addi %rowStart, %number : i32
    %at64 = arith.extui %at : i32 to i64
    %target = llvm.getelementptr %tile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    llvm.store %both, %target {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<3>
    cf.br ^next
  ^next:
    %following = arith.addi %index, %c1 : i32
    llvm.br ^part(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // the scores of the 16 rows of a warp with the 32 positions of a key tile (4 parts of 8 positions, a lane has 4 of
  // each: row lane / 4 and the one 8 below, positions 2 * (lane % 4) and the next), queries scaled; an m16n8k16 per 16
  // numbers, one ldmatrix.x4 gives the key operands of two parts
  func.func private @strict_device_attention_scores(%queries: vector<32xi32>, %keyTile: !llvm.ptr<3>, %size: i32, %lane: i32, %brain: i1) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c15 = llvm.mlir.constant(15 : i32) : i32
    %c136 = llvm.mlir.constant(136 : i32) : i32
    %c2176 = llvm.mlir.constant(2176 : i32) : i32
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %spare = arith.addi %size, %c15 : i32
    %steps = arith.shrui %spare, %c4 : i32
    %laneHalf = arith.shrui %lane, %c4 : i32
    %halfRows = arith.shli %laneHalf, %c3 : i32
    %laneRow = arith.andi %lane, %c7 : i32
    %matrixRow = arith.addi %halfRows, %laneRow : i32
    %laneEighth = arith.shrui %lane, %c3 : i32
    %laneOdd = arith.andi %laneEighth, %c1 : i32
    %matrixColumn = arith.shli %laneOdd, %c3 : i32
    %rowStart = arith.muli %matrixRow, %c136 : i32
    %laneStart = arith.addi %rowStart, %matrixColumn : i32
    cf.br ^step(%c0, %zeros : i32, vector<16xf32>)
  ^step(%step: i32, %scores: vector<16xf32>):
    %moreSteps = arith.cmpi slt, %step, %c8 : i32
    cf.cond_br %moreSteps, ^stepCheck, ^done
  ^stepCheck:
    %used = arith.cmpi slt, %step, %steps : i32
    cf.cond_br %used, ^stepBody, ^stepNext(%scores : vector<16xf32>)
  ^stepBody:
    %slot = arith.shli %step, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %bits0 = llvm.extractelement %queries[%slot : i32] : vector<32xi32>
    %bits1 = llvm.extractelement %queries[%slot1 : i32] : vector<32xi32>
    %bits2 = llvm.extractelement %queries[%slot2 : i32] : vector<32xi32>
    %bits3 = llvm.extractelement %queries[%slot3 : i32] : vector<32xi32>
    %query0 = llvm.bitcast %bits0 : i32 to vector<2xf16>
    %query1 = llvm.bitcast %bits1 : i32 to vector<2xf16>
    %query2 = llvm.bitcast %bits2 : i32 to vector<2xf16>
    %query3 = llvm.bitcast %bits3 : i32 to vector<2xf16>
    %stepNumbers = arith.shli %step, %c4 : i32
    %stepStart = arith.addi %laneStart, %stepNumbers : i32
    cf.br ^pair(%c0, %scores : i32, vector<16xf32>)
  ^pair(%pair: i32, %pairScores: vector<16xf32>):
    %morePairs = arith.cmpi slt, %pair, %c2 : i32
    cf.cond_br %morePairs, ^pairBody, ^stepNext(%pairScores : vector<16xf32>)
  ^pairBody:
    %pairOffset = arith.muli %pair, %c2176 : i32
    %at = arith.addi %stepStart, %pairOffset : i32
    %at64 = arith.extui %at : i32 to i64
    %keySlot = llvm.getelementptr %keyTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %matrices = func.call @strict_device_matrices(%keySlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %firstScores = arith.shli %pair, %c3 : i32
    %secondScores = arith.addi %firstScores, %c4 : i32
    %matrix0 = llvm.extractvalue %matrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix1 = llvm.extractvalue %matrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix2 = llvm.extractvalue %matrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix3 = llvm.extractvalue %matrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %withFirst = func.call @strict_device_attention_product(%query0, %query1, %query2, %query3, %matrix0, %matrix1, %pairScores, %firstScores, %brain) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<16xf32>, i32, i1) -> vector<16xf32>
    %withSecond = func.call @strict_device_attention_product(%query0, %query1, %query2, %query3, %matrix2, %matrix3, %withFirst, %secondScores, %brain) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<16xf32>, i32, i1) -> vector<16xf32>
    %pairFollowing = arith.addi %pair, %c1 : i32
    llvm.br ^pair(%pairFollowing, %withSecond : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^stepNext(%nextScores: vector<16xf32>):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %nextScores : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %scores : vector<16xf32>
  }
  // an m16n8k16 adding the a operands times the b operands (bits of 16 bit pairs) to the 4 sums at slot of sums
  func.func private @strict_device_attention_product(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %bits0: i32, %bits1: i32, %sums: vector<16xf32>, %slot: i32, %brain: i1) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %b0 = llvm.bitcast %bits0 : i32 to vector<2xf16>
    %b1 = llvm.bitcast %bits1 : i32 to vector<2xf16>
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %sum0 = llvm.extractelement %sums[%slot : i32] : vector<16xf32>
    %sum1 = llvm.extractelement %sums[%slot1 : i32] : vector<16xf32>
    %sum2 = llvm.extractelement %sums[%slot2 : i32] : vector<16xf32>
    %sum3 = llvm.extractelement %sums[%slot3 : i32] : vector<16xf32>
    %product = func.call @strict_device_attention_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %sum0, %sum1, %sum2, %sum3, %brain) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, i1) -> !llvm.struct<(f32, f32, f32, f32)>
    %product0 = llvm.extractvalue %product[0] : !llvm.struct<(f32, f32, f32, f32)>
    %product1 = llvm.extractvalue %product[1] : !llvm.struct<(f32, f32, f32, f32)>
    %product2 = llvm.extractvalue %product[2] : !llvm.struct<(f32, f32, f32, f32)>
    %product3 = llvm.extractvalue %product[3] : !llvm.struct<(f32, f32, f32, f32)>
    %with0 = llvm.insertelement %product0, %sums[%slot : i32] : vector<16xf32>
    %with1 = llvm.insertelement %product1, %with0[%slot1 : i32] : vector<16xf32>
    %with2 = llvm.insertelement %product2, %with1[%slot2 : i32] : vector<16xf32>
    %with3 = llvm.insertelement %product3, %with2[%slot3 : i32] : vector<16xf32>
    return %with3 : vector<16xf32>
  }
  // the largest of the 16 scores of a lane for its row lane / 4 (lower) or the one 8 below, over the 4 lanes of a row,
  // positions after last hidden
  func.func private @strict_device_attention_largest(%scores: vector<16xf32>, %lower: i32, %first: i32, %quad: i32, %last: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %quadPositions = arith.shli %quad, %c1 : i32
    %rowStart = arith.addi %first, %quadPositions : i32
    cf.br ^part(%c0, %hidden : i32, f32)
  ^part(%part: i32, %largest: f32):
    %more = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %c3 : i32
    %position = arith.addi %rowStart, %partStart : i32
    %positionNext = arith.addi %position, %c1 : i32
    %slots = arith.shli %part, %c2 : i32
    %slot = arith.addi %slots, %lower : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %score = llvm.extractelement %scores[%slot : i32] : vector<16xf32>
    %scoreNext = llvm.extractelement %scores[%slotNext : i32] : vector<16xf32>
    %visible = arith.cmpi sle, %position, %last : i32
    %visibleNext = arith.cmpi sle, %positionNext, %last : i32
    %shown = arith.select %visible, %score, %hidden : f32
    %shownNext = arith.select %visibleNext, %scoreNext, %hidden : f32
    %larger = arith.maximumf %largest, %shown : f32
    %largerNext = arith.maximumf %larger, %shownNext : f32
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following, %largerNext : i32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    %fromOne = func.call @strict_device_shuffles(%largest, %c1) : (f32, i32) -> f32
    %pairLargest = arith.maximumf %largest, %fromOne : f32
    %fromTwo = func.call @strict_device_shuffles(%pairLargest, %c2) : (f32, i32) -> f32
    %rowLargest = arith.maximumf %pairLargest, %fromTwo : f32
    return %rowLargest : f32
  }
  // the scores of a tile turned into shares (2 to the power of score minus the new largest score of the row, 0 after
  // the last visible position), the new largest scores and sums of the two rows of a lane and how much the outputs
  // shrink
  func.func private @strict_device_attention_softmax(%scores: vector<16xf32>, %state: vector<4xf32>, %first: i32, %lastLower: i32, %lastUpper: i32, %quad: i32) -> (vector<16xf32>, vector<4xf32>, f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %lowerLargest = func.call @strict_device_attention_largest(%scores, %c0, %first, %quad, %lastLower) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %upperLargest = func.call @strict_device_attention_largest(%scores, %c2, %first, %quad, %lastUpper) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %newLower, %newUpper, %lowerScale, %upperScale = func.call @strict_device_attention_raised(%state, %lowerLargest, %upperLargest) : (vector<4xf32>, f32, f32) -> (f32, f32, f32, f32)
    %shared, %lowerTotal, %upperTotal = func.call @strict_device_attention_tile_shares(%scores, %first, %lastLower, %lastUpper, %newLower, %newUpper, %quad) : (vector<16xf32>, i32, i32, i32, f32, f32, i32) -> (vector<16xf32>, f32, f32)
    %newState = func.call @strict_device_attention_summed(%state, %newLower, %newUpper, %lowerScale, %upperScale, %lowerTotal, %upperTotal) : (vector<4xf32>, f32, f32, f32, f32, f32, f32) -> vector<4xf32>
    return %shared, %newState, %lowerScale, %upperScale : vector<16xf32>, vector<4xf32>, f32, f32
  }
  // the new largest scores of the two rows of a lane with the largest of a tile and how much their outputs shrink
  func.func private @strict_device_attention_raised(%state: vector<4xf32>, %lowerLargest: f32, %upperLargest: f32) -> (f32, f32, f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %oldLower = llvm.extractelement %state[%c0 : i32] : vector<4xf32>
    %oldUpper = llvm.extractelement %state[%c2 : i32] : vector<4xf32>
    %newLower = arith.maximumf %oldLower, %lowerLargest : f32
    %newUpper = arith.maximumf %oldUpper, %upperLargest : f32
    %lowerShift = arith.subf %oldLower, %newLower : f32
    %upperShift = arith.subf %oldUpper, %newUpper : f32
    %lowerScale = func.call @strict_device_binary_power(%lowerShift) : (f32) -> f32
    %upperScale = func.call @strict_device_binary_power(%upperShift) : (f32) -> f32
    return %newLower, %newUpper, %lowerScale, %upperScale : f32, f32, f32, f32
  }
  // the shares of a tile of 32 positions from first (strict_device_attention_shares of both rows) and their sums
  func.func private @strict_device_attention_tile_shares(%scores: vector<16xf32>, %first: i32, %lastLower: i32, %lastUpper: i32, %newLower: f32, %newUpper: f32, %quad: i32) -> (vector<16xf32>, f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %quadPositions = arith.shli %quad, %c1 : i32
    %rowStart = arith.addi %first, %quadPositions : i32
    cf.br ^shares(%c0, %zero, %zero, %scores : i32, f32, f32, vector<16xf32>)
  ^shares(%sharePart: i32, %lowerTotal: f32, %upperTotal: f32, %shared: vector<16xf32>):
    %moreShares = arith.cmpi slt, %sharePart, %c4 : i32
    cf.cond_br %moreShares, ^shareBody, ^sharesDone
  ^shareBody:
    %partStart = arith.shli %sharePart, %c3 : i32
    %position = arith.addi %rowStart, %partStart : i32
    %positionNext = arith.addi %position, %c1 : i32
    %slots = arith.shli %sharePart, %c2 : i32
    %lowerShare, %lowerShareNext, %afterLower = func.call @strict_device_attention_shares(%shared, %slots, %position, %positionNext, %lastLower, %newLower) : (vector<16xf32>, i32, i32, i32, i32, f32) -> (f32, f32, vector<16xf32>)
    %upperSlots = arith.addi %slots, %c2 : i32
    %upperShare, %upperShareNext, %afterUpper = func.call @strict_device_attention_shares(%afterLower, %upperSlots, %position, %positionNext, %lastUpper, %newUpper) : (vector<16xf32>, i32, i32, i32, i32, f32) -> (f32, f32, vector<16xf32>)
    %lowerPair = arith.addf %lowerShare, %lowerShareNext : f32
    %upperPair = arith.addf %upperShare, %upperShareNext : f32
    %lowerMore = arith.addf %lowerTotal, %lowerPair : f32
    %upperMore = arith.addf %upperTotal, %upperPair : f32
    %shareFollowing = arith.addi %sharePart, %c1 : i32
    llvm.br ^shares(%shareFollowing, %lowerMore, %upperMore, %afterUpper : i32, f32, f32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^sharesDone:
    return %shared, %lowerTotal, %upperTotal : vector<16xf32>, f32, f32
  }
  // the state of the two rows of a lane with new largest scores, their old sums shrunk by scale plus totals
  func.func private @strict_device_attention_summed(%state: vector<4xf32>, %newLower: f32, %newUpper: f32, %lowerScale: f32, %upperScale: f32, %lowerTotal: f32, %upperTotal: f32) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %lowerSum = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperSum = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %lowerNewSum = llvm.intr.fma(%lowerSum, %lowerScale, %lowerTotal) : (f32, f32, f32) -> f32
    %upperNewSum = llvm.intr.fma(%upperSum, %upperScale, %upperTotal) : (f32, f32, f32) -> f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %newLower, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %lowerNewSum, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %newUpper, %state1[%c2 : i32] : vector<4xf32>
    %newState = llvm.insertelement %upperNewSum, %state2[%c3 : i32] : vector<4xf32>
    return %newState : vector<4xf32>
  }
  // the shares (2 to the power of score minus largest, 0 after last) of the two scores at slot and the next, written
  // back for the values
  func.func private @strict_device_attention_shares(%scores: vector<16xf32>, %slot: i32, %position: i32, %positionNext: i32, %last: i32, %largest: f32) -> (f32, f32, vector<16xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %slotNext = arith.addi %slot, %c1 : i32
    %score = llvm.extractelement %scores[%slot : i32] : vector<16xf32>
    %scoreNext = llvm.extractelement %scores[%slotNext : i32] : vector<16xf32>
    %shifted = arith.subf %score, %largest : f32
    %shiftedNext = arith.subf %scoreNext, %largest : f32
    %share = func.call @strict_device_binary_power(%shifted) : (f32) -> f32
    %shareNext = func.call @strict_device_binary_power(%shiftedNext) : (f32) -> f32
    %visible = arith.cmpi sle, %position, %last : i32
    %visibleNext = arith.cmpi sle, %positionNext, %last : i32
    %shown = arith.select %visible, %share, %zero : f32
    %shownNext = arith.select %visibleNext, %shareNext, %zero : f32
    %withShare = llvm.insertelement %shown, %scores[%slot : i32] : vector<16xf32>
    %withBoth = llvm.insertelement %shownNext, %withShare[%slotNext : i32] : vector<16xf32>
    return %shown, %shownNext, %withBoth : f32, f32, vector<16xf32>
  }
  // the outputs of the two rows of a lane shrunk by their scales plus the shares times a value tile: an m16n8k16 with
  // 16 bit sums per 16 positions and 8 numbers (a tile of 32 positions adds up in 16 bits, the outputs in 32 bits), one
  // ldmatrix.x4.trans gives the value operands of 16 numbers
  func.func private @strict_device_attention_weighted(%shares: vector<16xf32>, %outputs: vector<64xf32>, %lowerScale: f32, %upperScale: f32, %valueTile: !llvm.ptr<3>, %size: i32, %lane: i32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c136 = llvm.mlir.constant(136 : i32) : i32
    %c2176 = llvm.mlir.constant(2176 : i32) : i32
    %noOperands = arith.constant dense<0> : vector<8xi32>
    %noPartials = arith.constant dense<0> : vector<32xi32>
    %first = func.call @strict_device_attention_shared(%shares, %c0, %noOperands, %c0) : (vector<16xf32>, i32, vector<8xi32>, i32) -> vector<8xi32>
    %packing = func.call @strict_device_attention_shared(%shares, %c8, %first, %c4) : (vector<16xf32>, i32, vector<8xi32>, i32) -> vector<8xi32>
    %columns = arith.shrui %size, %c3 : i32
    %laneEighth = arith.shrui %lane, %c3 : i32
    %laneOdd = arith.andi %laneEighth, %c1 : i32
    %oddRows = arith.shli %laneOdd, %c3 : i32
    %laneRow = arith.andi %lane, %c7 : i32
    %matrixRow = arith.addi %oddRows, %laneRow : i32
    %laneHalf = arith.shrui %lane, %c4 : i32
    %matrixColumn = arith.shli %laneHalf, %c3 : i32
    %rowStart = arith.muli %matrixRow, %c136 : i32
    %laneStart = arith.addi %rowStart, %matrixColumn : i32
    cf.br ^step(%c0, %noPartials : i32, vector<32xi32>)
  ^step(%step: i32, %partials: vector<32xi32>):
    %moreSteps = arith.cmpi slt, %step, %c2 : i32
    cf.cond_br %moreSteps, ^stepBody, ^fold(%c0, %outputs : i32, vector<64xf32>)
  ^stepBody:
    %slot = arith.shli %step, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %bits0 = llvm.extractelement %packing[%slot : i32] : vector<8xi32>
    %bits1 = llvm.extractelement %packing[%slot1 : i32] : vector<8xi32>
    %bits2 = llvm.extractelement %packing[%slot2 : i32] : vector<8xi32>
    %bits3 = llvm.extractelement %packing[%slot3 : i32] : vector<8xi32>
    %share0 = llvm.bitcast %bits0 : i32 to vector<2xf16>
    %share1 = llvm.bitcast %bits1 : i32 to vector<2xf16>
    %share2 = llvm.bitcast %bits2 : i32 to vector<2xf16>
    %share3 = llvm.bitcast %bits3 : i32 to vector<2xf16>
    %stepRows = arith.muli %step, %c2176 : i32
    %stepStart = arith.addi %laneStart, %stepRows : i32
    cf.br ^pair(%c0, %partials : i32, vector<32xi32>)
  ^pair(%pair: i32, %pairPartials: vector<32xi32>):
    %morePairs = arith.cmpi slt, %pair, %c8 : i32
    cf.cond_br %morePairs, ^pairCheck, ^stepNext(%pairPartials : vector<32xi32>)
  ^pairCheck:
    %firstColumn = arith.shli %pair, %c1 : i32
    %usedPair = arith.cmpi slt, %firstColumn, %columns : i32
    cf.cond_br %usedPair, ^pairBody, ^pairNext(%pairPartials : vector<32xi32>)
  ^pairBody:
    %pairNumbers = arith.shli %pair, %c4 : i32
    %at = arith.addi %stepStart, %pairNumbers : i32
    %at64 = arith.extui %at : i32 to i64
    %valueSlot = llvm.getelementptr %valueTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %matrices = func.call @strict_device_matrices_transposed(%valueSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %matrix0 = llvm.extractvalue %matrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix1 = llvm.extractvalue %matrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix2 = llvm.extractvalue %matrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix3 = llvm.extractvalue %matrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %firstSlot = arith.shli %pair, %c2 : i32
    %secondSlot = arith.addi %firstSlot, %c2 : i32
    %withFirst = func.call @strict_device_attention_accumulated(%share0, %share1, %share2, %share3, %matrix0, %matrix1, %pairPartials, %firstSlot) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<32xi32>, i32) -> vector<32xi32>
    %withSecond = func.call @strict_device_attention_accumulated(%share0, %share1, %share2, %share3, %matrix2, %matrix3, %withFirst, %secondSlot) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<32xi32>, i32) -> vector<32xi32>
    cf.br ^pairNext(%withSecond : vector<32xi32>)
  ^pairNext(%nextPairPartials: vector<32xi32>):
    %pairFollowing = arith.addi %pair, %c1 : i32
    llvm.br ^pair(%pairFollowing, %nextPairPartials : i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^stepNext(%nextPartials: vector<32xi32>):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %nextPartials : i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^fold(%column: i32, %sums: vector<64xf32>):
    %moreColumns = arith.cmpi slt, %column, %c16 : i32
    cf.cond_br %moreColumns, ^foldCheck, ^done
  ^foldCheck:
    %usedColumn = arith.cmpi slt, %column, %columns : i32
    cf.cond_br %usedColumn, ^foldBody, ^foldNext(%sums : vector<64xf32>)
  ^foldBody:
    %partialSlot = arith.shli %column, %c1 : i32
    %partialNext = arith.addi %partialSlot, %c1 : i32
    %lowBits = llvm.extractelement %partials[%partialSlot : i32] : vector<32xi32>
    %highBits = llvm.extractelement %partials[%partialNext : i32] : vector<32xi32>
    %low = llvm.bitcast %lowBits : i32 to vector<2xf16>
    %high = llvm.bitcast %highBits : i32 to vector<2xf16>
    %outputSlot = arith.shli %column, %c2 : i32
    %folded = func.call @strict_device_attention_folded(%sums, %outputSlot, %low, %high, %lowerScale, %upperScale) : (vector<64xf32>, i32, vector<2xf16>, vector<2xf16>, f32, f32) -> vector<64xf32>
    cf.br ^foldNext(%folded : vector<64xf32>)
  ^foldNext(%nextFolded: vector<64xf32>):
    %columnFollowing = arith.addi %column, %c1 : i32
    llvm.br ^fold(%columnFollowing, %nextFolded : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %sums : vector<64xf32>
  }
  // an m16n8k16 with 16 bit sums adding shares times values to the 16 bit pairs at slot and the next of partials
  func.func private @strict_device_attention_accumulated(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %bits0: i32, %bits1: i32, %partials: vector<32xi32>, %slot: i32) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %b0 = llvm.bitcast %bits0 : i32 to vector<2xf16>
    %b1 = llvm.bitcast %bits1 : i32 to vector<2xf16>
    %slotNext = arith.addi %slot, %c1 : i32
    %lowBits = llvm.extractelement %partials[%slot : i32] : vector<32xi32>
    %highBits = llvm.extractelement %partials[%slotNext : i32] : vector<32xi32>
    %low = llvm.bitcast %lowBits : i32 to vector<2xf16>
    %high = llvm.bitcast %highBits : i32 to vector<2xf16>
    %sums = func.call @strict_device_halves_mma16_halves(%a0, %a1, %a2, %a3, %b0, %b1, %low, %high) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %newLow = llvm.extractvalue %sums[0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %newHigh = llvm.extractvalue %sums[1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %newLowBits = llvm.bitcast %newLow : vector<2xf16> to i32
    %newHighBits = llvm.bitcast %newHigh : vector<2xf16> to i32
    %withLow = llvm.insertelement %newLowBits, %partials[%slot : i32] : vector<32xi32>
    %withHigh = llvm.insertelement %newHighBits, %withLow[%slotNext : i32] : vector<32xi32>
    return %withHigh : vector<32xi32>
  }
  // the shares of an m16n8k16 operand (slots slot, + 2, + 4 and + 6 and the next of each) as 16 bit pairs into
  // operands from at
  func.func private @strict_device_attention_shared(%shares: vector<16xf32>, %slot: i32, %operands: vector<8xi32>, %at: i32) -> vector<8xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %slot4 = arith.addi %slot, %c4 : i32
    %slot5 = arith.addi %slot, %c5 : i32
    %slot6 = arith.addi %slot, %c6 : i32
    %slot7 = arith.addi %slot, %c7 : i32
    %share0 = llvm.extractelement %shares[%slot : i32] : vector<16xf32>
    %share1 = llvm.extractelement %shares[%slot1 : i32] : vector<16xf32>
    %share2 = llvm.extractelement %shares[%slot2 : i32] : vector<16xf32>
    %share3 = llvm.extractelement %shares[%slot3 : i32] : vector<16xf32>
    %share4 = llvm.extractelement %shares[%slot4 : i32] : vector<16xf32>
    %share5 = llvm.extractelement %shares[%slot5 : i32] : vector<16xf32>
    %share6 = llvm.extractelement %shares[%slot6 : i32] : vector<16xf32>
    %share7 = llvm.extractelement %shares[%slot7 : i32] : vector<16xf32>
    %pair0 = func.call @strict_device_attention_pair(%share0, %share1) : (f32, f32) -> i32
    %pair1 = func.call @strict_device_attention_pair(%share2, %share3) : (f32, f32) -> i32
    %pair2 = func.call @strict_device_attention_pair(%share4, %share5) : (f32, f32) -> i32
    %pair3 = func.call @strict_device_attention_pair(%share6, %share7) : (f32, f32) -> i32
    %at1 = arith.addi %at, %c1 : i32
    %at2 = arith.addi %at, %c2 : i32
    %at3 = arith.addi %at, %c3 : i32
    %with0 = llvm.insertelement %pair0, %operands[%at : i32] : vector<8xi32>
    %with1 = llvm.insertelement %pair1, %with0[%at1 : i32] : vector<8xi32>
    %with2 = llvm.insertelement %pair2, %with1[%at2 : i32] : vector<8xi32>
    %with3 = llvm.insertelement %pair3, %with2[%at3 : i32] : vector<8xi32>
    return %with3 : vector<8xi32>
  }
  // parts of 4 numbers per position of a tile: size rounded up to 16 so the last mma step reads zeros
  // the numbers of strict_device_attention_fetch as 16 bit floats into a tile in shared memory (136 numbers per
  // the scores of the 16 rows of a warp with the 64 positions of the key tile (8 parts of 8 positions, a lane has 4 of
  // an m16n8k16 adding the a operands times the b operands (bits of 16 bit pairs) to the 4 sums at slot of sums
  // the scores of a tile turned into shares (2 to the power of score minus the new largest score of the row, 0 after
  // the outputs of the two rows of a lane shrunk by their scales plus the shares times the value tile: an m16n8k16 with
  // an m16n8k16 with 16 bit sums adding shares times values to the 16 bit pairs at slot and the next of partials
  // the 16 bit sums of a tile (low the two of row lane / 4, high those of the row 8 below) added to the 4 outputs at
  // slot shrunk by the scales of their rows
  func.func private @strict_device_attention_folded(%sums: vector<64xf32>, %slot: i32, %low: vector<2xf16>, %high: vector<2xf16>, %lowerScale: f32, %upperScale: f32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %lowWide = arith.extf %low : vector<2xf16> to vector<2xf32>
    %highWide = arith.extf %high : vector<2xf16> to vector<2xf32>
    %part0 = llvm.extractelement %lowWide[%c0 : i32] : vector<2xf32>
    %part1 = llvm.extractelement %lowWide[%c1 : i32] : vector<2xf32>
    %part2 = llvm.extractelement %highWide[%c0 : i32] : vector<2xf32>
    %part3 = llvm.extractelement %highWide[%c1 : i32] : vector<2xf32>
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %sum0 = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    %sum1 = llvm.extractelement %sums[%slot1 : i32] : vector<64xf32>
    %sum2 = llvm.extractelement %sums[%slot2 : i32] : vector<64xf32>
    %sum3 = llvm.extractelement %sums[%slot3 : i32] : vector<64xf32>
    %new0 = llvm.intr.fma(%sum0, %lowerScale, %part0) : (f32, f32, f32) -> f32
    %new1 = llvm.intr.fma(%sum1, %lowerScale, %part1) : (f32, f32, f32) -> f32
    %new2 = llvm.intr.fma(%sum2, %upperScale, %part2) : (f32, f32, f32) -> f32
    %new3 = llvm.intr.fma(%sum3, %upperScale, %part3) : (f32, f32, f32) -> f32
    %with0 = llvm.insertelement %new0, %sums[%slot : i32] : vector<64xf32>
    %with1 = llvm.insertelement %new1, %with0[%slot1 : i32] : vector<64xf32>
    %with2 = llvm.insertelement %new2, %with1[%slot2 : i32] : vector<64xf32>
    %with3 = llvm.insertelement %new3, %with2[%slot3 : i32] : vector<64xf32>
    return %with3 : vector<64xf32>
  }
  // strict_device_attention_query with the bits known in each branch, so its loads are not separated by branches
  func.func private @strict_device_attention_queries(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %size: i32, %quad: i32, %scale: f32, %brain: i1) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %singles = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    cf.switch %bits : i64, [
      default: ^double,
      32: ^single,
      17: ^brain
    ]
  ^single:
    %singlePairs = func.call @strict_device_attention_query(%queries, %singles, %first, %second, %size, %quad, %scale, %brain) : (!llvm.ptr, i64, i64, i64, i32, i32, f32, i1) -> vector<32xi32>
    return %singlePairs : vector<32xi32>
  ^brain:
    %brainPairs = func.call @strict_device_attention_query(%queries, %brainBits, %first, %second, %size, %quad, %scale, %brain) : (!llvm.ptr, i64, i64, i64, i32, i32, f32, i1) -> vector<32xi32>
    return %brainPairs : vector<32xi32>
  ^double:
    %doubles = arith.constant 64 : i64
    %doublePairs = func.call @strict_device_attention_query(%queries, %doubles, %first, %second, %size, %quad, %scale, %brain) : (!llvm.ptr, i64, i64, i64, i32, i32, f32, i1) -> vector<32xi32>
    return %doublePairs : vector<32xi32>
  }
  // the query numbers of the two rows of a lane (rows at first and second, row lane / 4 and the one 8 below) as the 16
  // bit operands of all 16 steps of 8 numbers, times scale; steps after size / 8 stay zero
  func.func private @strict_device_attention_query(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %size: i32, %quad: i32, %scale: f32, %brain: i1) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %zeros = arith.constant dense<0> : vector<32xi32>
    %steps = arith.shrui %size, %c3 : i32
    %quadNumbers = arith.shli %quad, %c1 : i32
    cf.br ^step(%c0, %zeros : i32, vector<32xi32>)
  ^step(%step: i32, %pairs: vector<32xi32>):
    %more = arith.cmpi slt, %step, %c16 : i32
    cf.cond_br %more, ^check, ^done
  ^check:
    %used = arith.cmpi slt, %step, %steps : i32
    cf.cond_br %used, ^body, ^next(%pairs : vector<32xi32>)
  ^body:
    %stepStart = arith.shli %step, %c3 : i32
    %number = arith.addi %stepStart, %quadNumbers : i32
    %number64 = arith.extui %number : i32 to i64
    %firstAt = arith.addi %first, %number64 : i64
    %secondAt = arith.addi %second, %number64 : i64
    %firstPair = func.call @strict_device_attention_two(%queries, %bits, %firstAt, %scale, %brain) : (!llvm.ptr, i64, i64, f32, i1) -> i32
    %secondPair = func.call @strict_device_attention_two(%queries, %bits, %secondAt, %scale, %brain) : (!llvm.ptr, i64, i64, f32, i1) -> i32
    %slot = arith.shli %step, %c1 : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %withFirst = llvm.insertelement %firstPair, %pairs[%slot : i32] : vector<32xi32>
    %withSecond = llvm.insertelement %secondPair, %withFirst[%slotNext : i32] : vector<32xi32>
    cf.br ^next(%withSecond : vector<32xi32>)
  ^next(%nextPairs: vector<32xi32>):
    %following = arith.addi %step, %c1 : i32
    llvm.br ^step(%following, %nextPairs : i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %pairs : vector<32xi32>
  }
  // two numbers from at times scale as a 16 bit pair
  func.func private @strict_device_attention_two(%data: !llvm.ptr, %bits: i64, %at: i64, %scale: f32, %brain: i1) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.switch %bits : i64, [
      default: ^double,
      32: ^single,
      17: ^brain
    ]
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleFirst = llvm.load %singleSlot invariant : !llvm.ptr -> f32
    %singleNext = llvm.getelementptr %singleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %singleSecond = llvm.load %singleNext invariant : !llvm.ptr -> f32
    cf.br ^scaled(%singleFirst, %singleSecond : f32, f32)
  ^brain:
    %brainSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    cf.cond_br %brain, ^native, ^brainSingles
  ^native:
    %nativePair = llvm.load %brainSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    return %nativePair : i32
  ^brainSingles:
    %brains = llvm.load %brainSlot invariant : !llvm.ptr -> vector<2xbf16>
    %brainSingles = arith.extf %brains : vector<2xbf16> to vector<2xf32>
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %brainFirst = llvm.extractelement %brainSingles[%c0 : i32] : vector<2xf32>
    %brainSecond = llvm.extractelement %brainSingles[%c1 : i32] : vector<2xf32>
    cf.br ^scaled(%brainFirst, %brainSecond : f32, f32)
  ^double:
    %doubleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubleFirst = llvm.load %doubleSlot invariant : !llvm.ptr -> f64
    %doubleNext = llvm.getelementptr %doubleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %doubleSecond = llvm.load %doubleNext invariant : !llvm.ptr -> f64
    %narrowFirst = arith.truncf %doubleFirst : f64 to f32
    %narrowSecond = arith.truncf %doubleSecond : f64 to f32
    cf.br ^scaled(%narrowFirst, %narrowSecond : f32, f32)
  ^scaled(%first: f32, %second: f32):
    %scaledFirst = arith.mulf %first, %scale : f32
    %scaledSecond = arith.mulf %second, %scale : f32
    %pair = func.call @strict_device_attention_pair(%scaledFirst, %scaledSecond) : (f32, f32) -> i32
    return %pair : i32
  }
  // the shares (2 to the power of score minus largest, 0 after last) of the two scores at slot and the next, written
  // `Attention.Merged` like FlashAttention 2 on the 16 bit tensor cores: a block takes a sequence, a group of keys and
  // the rows (token and head) of up to 8 x 16 of its queries, walks the visible positions in tiles of 64 in shared
  // memory and keeps the largest score and sum of every row, so the scores are never stored
  func.func private @strict_device_attention_run(%result: !llvm.ptr, %queries: !llvm.ptr, %queriesCount: i64, %queryBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %headsNumber: f64, %groupsNumber: f64, %sizeNumber: f64, %sequencesNumber: f64, %causalNumber: f64, %wide: i1, %valueOffsetNumber: f64, %valueStrideNumber: f64, %pooled: !llvm.ptr, %pooledBits: i64, %routingNumber: f64, %exactNumber: f64) {
    %zeroNumber = arith.constant 0.0 : f64
    %causal = arith.cmpf une, %causalNumber, %zeroNumber : f64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %heads = arith.fptosi %headsNumber : f64 to i32
    %groups = arith.fptosi %groupsNumber : f64 to i32
    %size = arith.fptosi %sizeNumber : f64 to i32
    %sequences = arith.fptosi %sequencesNumber : f64 to i32
    %perGroup = arith.divui %heads, %groups : i32
    %headsSize = arith.muli %heads, %size : i32
    %sequenceQueries = arith.muli %headsSize, %sequences : i32
    %queries32 = arith.trunci %queriesCount : i64 to i32
    %tokens = arith.divui %queries32, %sequenceQueries : i32
    %groupsSize = arith.muli %groups, %size : i32
    %stride = arith.muli %groupsSize, %sequences : i32
    %valueStart = arith.fptosi %valueOffsetNumber : f64 to i32
    %requestedStep = arith.fptosi %valueStrideNumber : f64 to i32
    %packedValues = arith.cmpi eq, %requestedStep, %c0 : i32
    %valueStep = arith.select %packedValues, %stride, %requestedStep : i32
    %keys32 = arith.trunci %keysCount : i64 to i32
    %positions = arith.divui %keys32, %stride : i32
    %warpTokens = arith.divui %c16, %perGroup : i32
    %decodeTokens = arith.shli %warpTokens, %c2 : i32
    %decoding = arith.cmpi sle, %tokens, %decodeTokens : i32
    cf.cond_br %decoding, ^decode, ^prompt
  ^decode:
    func.call @strict_device_attention_decode(%result, %queries, %queryBits, %keys, %keyBits, %values, %valueBits, %heads, %groups, %size, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32) -> ()
    return
  ^prompt:
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %singleBits = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    %halves = arith.constant false
    %isWhole = arith.cmpi eq, %size, %c128 : i32
    %eightBits = arith.constant 8 : i64
    %singleQueries = arith.cmpi eq, %queryBits, %singleBits : i64
    %brainQueries = arith.cmpi eq, %queryBits, %brainBits : i64
    %knownQueries = arith.ori %singleQueries, %brainQueries : i1
    %eightKeys = arith.cmpi eq, %keyBits, %eightBits : i64
    %singleValues = arith.cmpi eq, %valueBits, %singleBits : i64
    %brainValues = arith.cmpi eq, %valueBits, %brainBits : i64
    %knownValues = arith.ori %singleValues, %brainValues : i1
    %knownInputs = arith.andi %knownQueries, %knownValues : i1
    %eightsInputs = arith.andi %knownInputs, %eightKeys : i1
    %eightsChosen = arith.andi %eightsInputs, %isWhole : i1
    cf.cond_br %eightsChosen, ^eightsPrompt, ^bitsPrompt
  ^eightsPrompt:
    func.call @strict_device_attention_eights_prompt(%result, %queries, %queryBits, %keys, %values, %brainValues, %heads, %groups, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i1, i32, i32, i32, i32, i32, i1, i1, i32, i32) -> ()
    return
  ^bitsPrompt:
    %mixedKeys = arith.cmpi eq, %keyBits, %singleBits : i64
    %mixedValues = arith.cmpi eq, %valueBits, %brainBits : i64
    %mixedBits = arith.andi %mixedKeys, %mixedValues : i1
    %knownMixed = arith.andi %mixedBits, %isWhole : i1
    cf.cond_br %knownMixed, ^mixedPrompt, ^matching
  ^mixedPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %singleBits, %values, %brainBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %halves) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  ^matching:
    %sameBits = arith.cmpi eq, %keyBits, %valueBits : i64
    %known = arith.andi %isWhole, %sameBits : i1
    cf.cond_br %known, ^knownBits, ^anyPrompt
  ^knownBits:
    cf.switch %keyBits : i64, [
      default: ^anyPrompt,
      32: ^knownPrompt,
      17: ^brainPrompt
    ]
  ^knownPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %singleBits, %values, %singleBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %halves) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  ^brainPrompt:
    %matchingQueries = arith.cmpi eq, %queryBits, %brainBits : i64
    cf.cond_br %matchingQueries, ^nativeBrainPrompt, ^mixedBrainPrompt
  ^nativeBrainPrompt:
    func.call @strict_device_attention_brain_prompt(%result, %queries, %keys, %values, %heads, %groups, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i1, i1, i32, i32) -> ()
    return
  ^mixedBrainPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %brainBits, %values, %brainBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %halves) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  ^anyPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %keyBits, %values, %valueBits, %heads, %groups, %size, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %halves) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  }
  // four numbers times factor (at most 127 in magnitude) rounded to 8 bits, the first in the low byte: the low byte of
  // the bits of 1.5 * 2^23 plus a number is the number rounded, no conversion
  func.func private @strict_device_attention_packed(%first: f32, %second: f32, %third: f32, %fourth: f32, %factor: f32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %bias = arith.constant 12582912.0 : f32
    %scaled0 = llvm.intr.fma(%first, %factor, %bias) : (f32, f32, f32) -> f32
    %scaled1 = llvm.intr.fma(%second, %factor, %bias) : (f32, f32, f32) -> f32
    %scaled2 = llvm.intr.fma(%third, %factor, %bias) : (f32, f32, f32) -> f32
    %scaled3 = llvm.intr.fma(%fourth, %factor, %bias) : (f32, f32, f32) -> f32
    %bits0 = arith.bitcast %scaled0 : f32 to i32
    %bits1 = arith.bitcast %scaled1 : f32 to i32
    %bits2 = arith.bitcast %scaled2 : f32 to i32
    %bits3 = arith.bitcast %scaled3 : f32 to i32
    %byte0 = arith.trunci %bits0 : i32 to i8
    %byte1 = arith.trunci %bits1 : i32 to i8
    %byte2 = arith.trunci %bits2 : i32 to i8
    %byte3 = arith.trunci %bits3 : i32 to i8
    %empty = llvm.mlir.undef : vector<4xi8>
    %with0 = llvm.insertelement %byte0, %empty[%c0 : i32] : vector<4xi8>
    %with1 = llvm.insertelement %byte1, %with0[%c1 : i32] : vector<4xi8>
    %with2 = llvm.insertelement %byte2, %with1[%c2 : i32] : vector<4xi8>
    %with3 = llvm.insertelement %byte3, %with2[%c3 : i32] : vector<4xi8>
    %word = llvm.bitcast %with3 : vector<4xi8> to i32
    return %word : i32
  }
  // 127 / largest, 0 for a largest of 0
  func.func private @strict_device_attention_inverse(%largest: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %isEmpty = arith.cmpf oeq, %largest, %zero : f32
    %inverse = arith.divf %most, %largest : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    return %factor : f32
  }
  // the 32 numbers of a query row (32 bit floats or bfloat16, size 128) a lane multiplies, 4 from 4 * quad of each 16,
  // as 8 bytes of 4 numbers scaled to the largest magnitude of the row over its 4 lanes, and the factor of its scores
  func.func private @strict_device_attention_row_bytes(%queries: !llvm.ptr, %bits: i64, %row: i64, %quad: i32, %scale: f32) -> (f32, vector<8xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %noNumbers = arith.constant dense<0.0> : vector<32xf32>
    %noBytes = arith.constant dense<0> : vector<8xi32>
    %quadNumbers = arith.shli %quad, %c2 : i32
    cf.br ^load(%c0, %noNumbers, %zero : i32, vector<32xf32>, f32)
  ^load(%part: i32, %numbers: vector<32xf32>, %largest: f32):
    %more = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %more, ^read, ^reduce
  ^read:
    %partStart = arith.shli %part, %c4 : i32
    %number = arith.addi %partStart, %quadNumbers : i32
    %number64 = arith.extui %number : i32 to i64
    %at = arith.addi %row, %number64 : i64
    %four = func.call @strict_device_attention_read(%queries, %bits, %at) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %slot0 = arith.shli %part, %c2 : i32
    %slot1 = arith.addi %slot0, %c1 : i32
    %slot2 = arith.addi %slot0, %c2 : i32
    %slot3 = arith.addi %slot0, %c3 : i32
    %read0 = llvm.extractelement %four[%c0 : i32] : vector<4xf32>
    %read1 = llvm.extractelement %four[%c1 : i32] : vector<4xf32>
    %read2 = llvm.extractelement %four[%c2 : i32] : vector<4xf32>
    %read3 = llvm.extractelement %four[%c3 : i32] : vector<4xf32>
    %with0 = llvm.insertelement %read0, %numbers[%slot0 : i32] : vector<32xf32>
    %with1 = llvm.insertelement %read1, %with0[%slot1 : i32] : vector<32xf32>
    %with2 = llvm.insertelement %read2, %with1[%slot2 : i32] : vector<32xf32>
    %with3 = llvm.insertelement %read3, %with2[%slot3 : i32] : vector<32xf32>
    %magnitude0 = llvm.call_intrinsic "llvm.fabs.f32"(%read0) : (f32) -> f32
    %magnitude1 = llvm.call_intrinsic "llvm.fabs.f32"(%read1) : (f32) -> f32
    %magnitude2 = llvm.call_intrinsic "llvm.fabs.f32"(%read2) : (f32) -> f32
    %magnitude3 = llvm.call_intrinsic "llvm.fabs.f32"(%read3) : (f32) -> f32
    %larger0 = arith.maxnumf %largest, %magnitude0 : f32
    %larger1 = arith.maxnumf %larger0, %magnitude1 : f32
    %larger2 = arith.maxnumf %larger1, %magnitude2 : f32
    %larger3 = arith.maxnumf %larger2, %magnitude3 : f32
    %following = arith.addi %part, %c1 : i32
    llvm.br ^load(%following, %with3, %larger3 : i32, vector<32xf32>, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^reduce:
    %fromOne = func.call @strict_device_shuffles(%largest, %c1) : (f32, i32) -> f32
    %pairLargest = arith.maxnumf %largest, %fromOne : f32
    %fromTwo = func.call @strict_device_shuffles(%pairLargest, %c2) : (f32, i32) -> f32
    %rowLargest = arith.maxnumf %pairLargest, %fromTwo : f32
    %factor = func.call @strict_device_attention_inverse(%rowLargest) : (f32) -> f32
    %rowScale = arith.divf %rowLargest, %most : f32
    %scoreFactor = arith.mulf %rowScale, %scale : f32
    cf.br ^pack(%c0, %noBytes : i32, vector<8xi32>)
  ^pack(%packPart: i32, %bytes: vector<8xi32>):
    %morePacks = arith.cmpi slt, %packPart, %c8 : i32
    cf.cond_br %morePacks, ^packBody, ^done
  ^packBody:
    %packSlot0 = arith.shli %packPart, %c2 : i32
    %packSlot1 = arith.addi %packSlot0, %c1 : i32
    %packSlot2 = arith.addi %packSlot0, %c2 : i32
    %packSlot3 = arith.addi %packSlot0, %c3 : i32
    %number0 = llvm.extractelement %numbers[%packSlot0 : i32] : vector<32xf32>
    %number1 = llvm.extractelement %numbers[%packSlot1 : i32] : vector<32xf32>
    %number2 = llvm.extractelement %numbers[%packSlot2 : i32] : vector<32xf32>
    %number3 = llvm.extractelement %numbers[%packSlot3 : i32] : vector<32xf32>
    %word = func.call @strict_device_attention_packed(%number0, %number1, %number2, %number3, %factor) : (f32, f32, f32, f32, f32) -> i32
    %withWord = llvm.insertelement %word, %bytes[%packPart : i32] : vector<8xi32>
    %packFollowing = arith.addi %packPart, %c1 : i32
    llvm.br ^pack(%packFollowing, %withWord : i32, vector<8xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %scoreFactor, %bytes : f32, vector<8xi32>
  }
  // the query rows of a lane (at first and second, 32 bit floats, size 128) as the 8 bit operands of the 4 m16n8k32
  // steps of strict_device_attention_scores_eights (slots 4 * step to 4 * step + 3), slots 16 and 17 the bits of the
  // factors of their scores (largest magnitude / 127 times scale)
  func.func private @strict_device_attention_quantized(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %quad: i32, %scale: f32) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c17 = llvm.mlir.constant(17 : i32) : i32
    %zeros = arith.constant dense<0> : vector<32xi32>
    %lowerFactor, %lowerBytes = func.call @strict_device_attention_row_bytes(%queries, %bits, %first, %quad, %scale) : (!llvm.ptr, i64, i64, i32, f32) -> (f32, vector<8xi32>)
    %upperFactor, %upperBytes = func.call @strict_device_attention_row_bytes(%queries, %bits, %second, %quad, %scale) : (!llvm.ptr, i64, i64, i32, f32) -> (f32, vector<8xi32>)
    cf.br ^step(%c0, %zeros : i32, vector<32xi32>)
  ^step(%step: i32, %operands: vector<32xi32>):
    %more = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %low = arith.shli %step, %c1 : i32
    %high = arith.addi %low, %c1 : i32
    %lowerLow = llvm.extractelement %lowerBytes[%low : i32] : vector<8xi32>
    %upperLow = llvm.extractelement %upperBytes[%low : i32] : vector<8xi32>
    %lowerHigh = llvm.extractelement %lowerBytes[%high : i32] : vector<8xi32>
    %upperHigh = llvm.extractelement %upperBytes[%high : i32] : vector<8xi32>
    %slot = arith.shli %step, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %with0 = llvm.insertelement %lowerLow, %operands[%slot : i32] : vector<32xi32>
    %with1 = llvm.insertelement %upperLow, %with0[%slot1 : i32] : vector<32xi32>
    %with2 = llvm.insertelement %lowerHigh, %with1[%slot2 : i32] : vector<32xi32>
    %with3 = llvm.insertelement %upperHigh, %with2[%slot3 : i32] : vector<32xi32>
    %following = arith.addi %step, %c1 : i32
    llvm.br ^step(%following, %with3 : i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    %lowerBits = arith.bitcast %lowerFactor : f32 to i32
    %upperBits = arith.bitcast %upperFactor : f32 to i32
    %withLower = llvm.insertelement %lowerBits, %operands[%c16 : i32] : vector<32xi32>
    %withUpper = llvm.insertelement %upperBits, %withLower[%c17 : i32] : vector<32xi32>
    return %withUpper : vector<32xi32>
  }
  // the 8 bit keys of a tile of 32 positions from first (bits 8: size 128, a scale per 32 numbers after the count of
  // positions times stride) a thread stores: 16 bytes of position thread / 8 in slots 0 to 3 and, for the first 32
  // threads, the scale of position thread in slot 4; zeros after the last position
  func.func private @strict_device_attention_fetch_eights(%keys: !llvm.ptr, %first: i32, %positions: i32, %stride: i32, %base: i32, %thread: i32) -> vector<16xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %zeros = arith.constant dense<0> : vector<16xi32>
    %row = arith.shrui %thread, %c3 : i32
    %chunk = arith.andi %thread, %c7 : i32
    %chunkBytes = arith.shli %chunk, %c4 : i32
    %position = arith.addi %first, %row : i32
    %inside = arith.cmpi slt, %position, %positions : i32
    cf.cond_br %inside, ^bytes, ^scale(%zeros : vector<16xi32>)
  ^bytes:
    %positionStart = arith.muli %position, %stride : i32
    %rowStart = arith.addi %positionStart, %base : i32
    %at = arith.addi %rowStart, %chunkBytes : i32
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %keys[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %words = llvm.load %slot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %word0 = llvm.extractelement %words[%c0 : i32] : vector<4xi32>
    %word1 = llvm.extractelement %words[%c1 : i32] : vector<4xi32>
    %word2 = llvm.extractelement %words[%c2 : i32] : vector<4xi32>
    %word3 = llvm.extractelement %words[%c3 : i32] : vector<4xi32>
    %with0 = llvm.insertelement %word0, %zeros[%c0 : i32] : vector<16xi32>
    %with1 = llvm.insertelement %word1, %with0[%c1 : i32] : vector<16xi32>
    %with2 = llvm.insertelement %word2, %with1[%c2 : i32] : vector<16xi32>
    %with3 = llvm.insertelement %word3, %with2[%c3 : i32] : vector<16xi32>
    cf.br ^scale(%with3 : vector<16xi32>)
  ^scale(%loaded: vector<16xi32>):
    %scaleThread = arith.cmpi slt, %thread, %c32 : i32
    %scalePosition = arith.addi %first, %thread : i32
    %scaleInside = arith.cmpi slt, %scalePosition, %positions : i32
    %scaling = arith.andi %scaleThread, %scaleInside : i1
    cf.cond_br %scaling, ^scaleLoad, ^done(%loaded : vector<16xi32>)
  ^scaleLoad:
    %count = arith.muli %positions, %stride : i32
    %count64 = arith.extui %count : i32 to i64
    %scales = llvm.getelementptr %keys[%count64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleStart = arith.muli %scalePosition, %stride : i32
    %scaleRow = arith.addi %scaleStart, %base : i32
    %scaleIndex = arith.shrui %scaleRow, %c5 : i32
    %scaleIndex64 = arith.extui %scaleIndex : i32 to i64
    %scaleSlot = llvm.getelementptr %scales[%scaleIndex64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %scaleBits = llvm.load %scaleSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %withScale = llvm.insertelement %scaleBits, %loaded[%c4 : i32] : vector<16xi32>
    cf.br ^done(%withScale : vector<16xi32>)
  ^done(%fetched: vector<16xi32>):
    return %fetched : vector<16xi32>
  }
  // the keys of strict_device_attention_fetch_eights into a tile in shared memory, 144 bytes apart per position (the 8
  // rows of an ldmatrix in different banks), the 32 scales after them at byte 4608
  func.func private @strict_device_attention_stored_eights(%tile: !llvm.ptr<3>, %numbers: vector<16xi32>, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c4608 = llvm.mlir.constant(4608 : i32) : i32
    %row = arith.shrui %thread, %c3 : i32
    %chunk = arith.andi %thread, %c7 : i32
    %chunkBytes = arith.shli %chunk, %c4 : i32
    %rowStart = arith.muli %row, %c144 : i32
    %at = arith.addi %rowStart, %chunkBytes : i32
    %at64 = arith.extui %at : i32 to i64
    %target = llvm.getelementptr %tile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %word0 = llvm.extractelement %numbers[%c0 : i32] : vector<16xi32>
    %word1 = llvm.extractelement %numbers[%c1 : i32] : vector<16xi32>
    %word2 = llvm.extractelement %numbers[%c2 : i32] : vector<16xi32>
    %word3 = llvm.extractelement %numbers[%c3 : i32] : vector<16xi32>
    %empty = llvm.mlir.undef : vector<4xi32>
    %with0 = llvm.insertelement %word0, %empty[%c0 : i32] : vector<4xi32>
    %with1 = llvm.insertelement %word1, %with0[%c1 : i32] : vector<4xi32>
    %with2 = llvm.insertelement %word2, %with1[%c2 : i32] : vector<4xi32>
    %words = llvm.insertelement %word3, %with2[%c3 : i32] : vector<4xi32>
    llvm.store %words, %target {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %scaleThread = arith.cmpi slt, %thread, %c32 : i32
    cf.cond_br %scaleThread, ^scale, ^done
  ^scale:
    %scaleBits = llvm.extractelement %numbers[%c4 : i32] : vector<16xi32>
    %scaleBytes = arith.shli %thread, %c2 : i32
    %scaleAt = arith.addi %c4608, %scaleBytes : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleTarget = llvm.getelementptr %tile[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    llvm.store %scaleBits, %scaleTarget {alignment = 4 : i64} : i32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // an m16n8k32 adding the 8 bit a operands times the b operands to the 4 integer sums at slot of sums
  func.func private @strict_device_attention_product8(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %sums: vector<16xi32>, %slot: i32) -> vector<16xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %sum0 = llvm.extractelement %sums[%slot : i32] : vector<16xi32>
    %sum1 = llvm.extractelement %sums[%slot1 : i32] : vector<16xi32>
    %sum2 = llvm.extractelement %sums[%slot2 : i32] : vector<16xi32>
    %sum3 = llvm.extractelement %sums[%slot3 : i32] : vector<16xi32>
    %product = func.call @strict_device_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %sum0, %sum1, %sum2, %sum3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %product0 = llvm.extractvalue %product[0] : !llvm.struct<(i32, i32, i32, i32)>
    %product1 = llvm.extractvalue %product[1] : !llvm.struct<(i32, i32, i32, i32)>
    %product2 = llvm.extractvalue %product[2] : !llvm.struct<(i32, i32, i32, i32)>
    %product3 = llvm.extractvalue %product[3] : !llvm.struct<(i32, i32, i32, i32)>
    %with0 = llvm.insertelement %product0, %sums[%slot : i32] : vector<16xi32>
    %with1 = llvm.insertelement %product1, %with0[%slot1 : i32] : vector<16xi32>
    %with2 = llvm.insertelement %product2, %with1[%slot2 : i32] : vector<16xi32>
    %with3 = llvm.insertelement %product3, %with2[%slot3 : i32] : vector<16xi32>
    return %with3 : vector<16xi32>
  }
  // an integer sum of 8 bit products the tensor cores added to the bits of 1.5 * 2^23 (1262485504, exact below 2^22)
  // as a float: an add instead of a conversion, which runs at a quarter of the rate
  func.func private @strict_device_attention_unbiased(%sum: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %bias = arith.constant 12582912.0 : f32
    %biased = arith.bitcast %sum : i32 to f32
    %number = arith.subf %biased, %bias : f32
    return %number : f32
  }
  // the scores of strict_device_attention_scores from the 8 bit queries of strict_device_attention_quantized and the
  // key tile of strict_device_attention_stored_eights: an m16n8k32 per 8 positions and 32 numbers, one ldmatrix.x4
  // gives the key operands of 16 positions, the integer sums times the factor of the row and the scale of the position
  func.func private @strict_device_attention_scores_eights(%queries: vector<32xi32>, %keyTile: !llvm.ptr<3>, %lane: i32) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c17 = llvm.mlir.constant(17 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c2304 = llvm.mlir.constant(2304 : i32) : i32
    %c4608 = llvm.mlir.constant(4608 : i32) : i32
    %noSums = arith.constant dense<1262485504> : vector<16xi32>
    %noScores = arith.constant dense<0.0> : vector<16xf32>
    %laneHalf = arith.shrui %lane, %c4 : i32
    %halfRows = arith.shli %laneHalf, %c3 : i32
    %laneRow = arith.andi %lane, %c7 : i32
    %matrixRow = arith.addi %halfRows, %laneRow : i32
    %laneEighth = arith.shrui %lane, %c3 : i32
    %laneOdd = arith.andi %laneEighth, %c1 : i32
    %matrixColumn = arith.shli %laneOdd, %c4 : i32
    %rowStart = arith.muli %matrixRow, %c144 : i32
    %laneStart = arith.addi %rowStart, %matrixColumn : i32
    cf.br ^step(%c0, %noSums : i32, vector<16xi32>)
  ^step(%step: i32, %sums: vector<16xi32>):
    %moreSteps = arith.cmpi slt, %step, %c4 : i32
    cf.cond_br %moreSteps, ^stepBody, ^scale(%c0, %noScores : i32, vector<16xf32>)
  ^stepBody:
    %slot = arith.shli %step, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %query0 = llvm.extractelement %queries[%slot : i32] : vector<32xi32>
    %query1 = llvm.extractelement %queries[%slot1 : i32] : vector<32xi32>
    %query2 = llvm.extractelement %queries[%slot2 : i32] : vector<32xi32>
    %query3 = llvm.extractelement %queries[%slot3 : i32] : vector<32xi32>
    %stepBytes = arith.shli %step, %c5 : i32
    %stepStart = arith.addi %laneStart, %stepBytes : i32
    cf.br ^pair(%c0, %sums : i32, vector<16xi32>)
  ^pair(%pair: i32, %pairSums: vector<16xi32>):
    %morePairs = arith.cmpi slt, %pair, %c2 : i32
    cf.cond_br %morePairs, ^pairBody, ^stepNext(%pairSums : vector<16xi32>)
  ^pairBody:
    %pairOffset = arith.muli %pair, %c2304 : i32
    %at = arith.addi %stepStart, %pairOffset : i32
    %at64 = arith.extui %at : i32 to i64
    %keySlot = llvm.getelementptr %keyTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %matrices = func.call @strict_device_matrices(%keySlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %firstSums = arith.shli %pair, %c3 : i32
    %secondSums = arith.addi %firstSums, %c4 : i32
    %matrix0 = llvm.extractvalue %matrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix1 = llvm.extractvalue %matrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix2 = llvm.extractvalue %matrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix3 = llvm.extractvalue %matrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %withFirst = func.call @strict_device_attention_product8(%query0, %query1, %query2, %query3, %matrix0, %matrix1, %pairSums, %firstSums) : (i32, i32, i32, i32, i32, i32, vector<16xi32>, i32) -> vector<16xi32>
    %withSecond = func.call @strict_device_attention_product8(%query0, %query1, %query2, %query3, %matrix2, %matrix3, %withFirst, %secondSums) : (i32, i32, i32, i32, i32, i32, vector<16xi32>, i32) -> vector<16xi32>
    %pairFollowing = arith.addi %pair, %c1 : i32
    llvm.br ^pair(%pairFollowing, %withSecond : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^stepNext(%nextSums: vector<16xi32>):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %nextSums : i32, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scale(%part: i32, %scores: vector<16xf32>):
    %moreParts = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %moreParts, ^scaleBody, ^done(%scores : vector<16xf32>)
  ^scaleBody:
    %lowerBits = llvm.extractelement %queries[%c16 : i32] : vector<32xi32>
    %upperBits = llvm.extractelement %queries[%c17 : i32] : vector<32xi32>
    %lowerFactor = arith.bitcast %lowerBits : i32 to f32
    %upperFactor = arith.bitcast %upperBits : i32 to f32
    %quad = arith.andi %lane, %c3 : i32
    %quadPositions = arith.shli %quad, %c1 : i32
    %partPositions = arith.shli %part, %c3 : i32
    %position = arith.addi %partPositions, %quadPositions : i32
    %positionBytes = arith.shli %position, %c2 : i32
    %scaleAt = arith.addi %c4608, %positionBytes : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleSlot = llvm.getelementptr %keyTile[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %keyScales = llvm.load %scaleSlot {alignment = 8 : i64} : !llvm.ptr<3> -> vector<2xf32>
    %keyScale = llvm.extractelement %keyScales[%c0 : i32] : vector<2xf32>
    %keyScaleNext = llvm.extractelement %keyScales[%c1 : i32] : vector<2xf32>
    %sumSlot = arith.shli %part, %c2 : i32
    %sumSlot1 = arith.addi %sumSlot, %c1 : i32
    %sumSlot2 = arith.addi %sumSlot, %c2 : i32
    %sumSlot3 = arith.addi %sumSlot, %c3 : i32
    %sum0 = llvm.extractelement %sums[%sumSlot : i32] : vector<16xi32>
    %sum1 = llvm.extractelement %sums[%sumSlot1 : i32] : vector<16xi32>
    %sum2 = llvm.extractelement %sums[%sumSlot2 : i32] : vector<16xi32>
    %sum3 = llvm.extractelement %sums[%sumSlot3 : i32] : vector<16xi32>
    %float0 = func.call @strict_device_attention_unbiased(%sum0) : (i32) -> f32
    %float1 = func.call @strict_device_attention_unbiased(%sum1) : (i32) -> f32
    %float2 = func.call @strict_device_attention_unbiased(%sum2) : (i32) -> f32
    %float3 = func.call @strict_device_attention_unbiased(%sum3) : (i32) -> f32
    %lowerKey = arith.mulf %lowerFactor, %keyScale : f32
    %lowerKeyNext = arith.mulf %lowerFactor, %keyScaleNext : f32
    %upperKey = arith.mulf %upperFactor, %keyScale : f32
    %upperKeyNext = arith.mulf %upperFactor, %keyScaleNext : f32
    %score0 = arith.mulf %float0, %lowerKey : f32
    %score1 = arith.mulf %float1, %lowerKeyNext : f32
    %score2 = arith.mulf %float2, %upperKey : f32
    %score3 = arith.mulf %float3, %upperKeyNext : f32
    %with0 = llvm.insertelement %score0, %scores[%sumSlot : i32] : vector<16xf32>
    %with1 = llvm.insertelement %score1, %with0[%sumSlot1 : i32] : vector<16xf32>
    %with2 = llvm.insertelement %score2, %with1[%sumSlot2 : i32] : vector<16xf32>
    %with3 = llvm.insertelement %score3, %with2[%sumSlot3 : i32] : vector<16xf32>
    %partFollowing = arith.addi %part, %c1 : i32
    llvm.br ^scale(%partFollowing, %with3 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done(%scaled: vector<16xf32>):
    return %scaled : vector<16xf32>
  }
  // strict_device_attention_stored, or with eights the 8 bit keys of strict_device_attention_stored_eights
  func.func private @strict_device_attention_keys_stored(%tile: !llvm.ptr<3>, %numbers: vector<16xi32>, %bits: i64, %size: i32, %thread: i32, %brain: i1, %eights: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.cond_br %eights, ^eights, ^halves
  ^eights:
    func.call @strict_device_attention_stored_eights(%tile, %numbers, %thread) : (!llvm.ptr<3>, vector<16xi32>, i32) -> ()
    return
  ^halves:
    func.call @strict_device_attention_stored(%tile, %numbers, %bits, %size, %thread, %brain) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32, i1) -> ()
    return
  }
  // strict_device_attention_scores, or with eights strict_device_attention_scores_eights
  func.func private @strict_device_attention_scored(%queries: vector<32xi32>, %keyTile: !llvm.ptr<3>, %size: i32, %lane: i32, %brain: i1, %eights: i1) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.cond_br %eights, ^eights, ^halves
  ^eights:
    %eightScores = func.call @strict_device_attention_scores_eights(%queries, %keyTile, %lane) : (vector<32xi32>, !llvm.ptr<3>, i32) -> vector<16xf32>
    return %eightScores : vector<16xf32>
  ^halves:
    %halfScores = func.call @strict_device_attention_scores(%queries, %keyTile, %size, %lane, %brain) : (vector<32xi32>, !llvm.ptr<3>, i32, i32, i1) -> vector<16xf32>
    return %halfScores : vector<16xf32>
  }
  // keys of 8 bits and size 128 (strict_gpu_attention quantized them) with 8 bit products, queries and values of 32
  // bits or bfloat16
  func.func private @strict_device_attention_eights_prompt(%result: !llvm.ptr, %queries: !llvm.ptr, %queryBits: i64, %keys: !llvm.ptr, %values: !llvm.ptr, %brainValues: i1, %heads: i32, %groups: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1, %valueStart: i32, %valueStep: i32) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["noinline"]} {
    %singleBits = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    %eightBits = arith.constant 8 : i64
    %c128 = arith.constant 128 : i32
    %eights = arith.constant true
    cf.cond_br %brainValues, ^brainValued, ^singleValued
  ^brainValued:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %eightBits, %values, %brainBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %eights) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  ^singleValued:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %eightBits, %values, %singleBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %eights) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  }
  // a pooled number as a 32 bit float or (wide) a double into the result at at
  func.func private @strict_device_attention_pooled_number(%result: !llvm.ptr, %wide: i1, %at: i32, %number: f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %at64 = arith.extui %at : i32 to i64
    cf.cond_br %wide, ^double, ^single
  ^double:
    %double = arith.extf %number : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    return
  ^single:
    %singleSlot = llvm.getelementptr %result[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %number, %singleSlot : f32, !llvm.ptr
    return
  }
  // the means strict_device_bytes_planned routes with: per sequence and group the mean key, then the mean value of
  // every block of 64 positions, then per sequence and head the mean query of every block of 64 tokens; a block of the
  // attention grid (128 tokens, a group, a sequence, one head per group) pools its two blocks, a thread per number each
  func.func private @strict_device_attention_pooling(%result: !llvm.ptr, %wide: i1, %queries: !llvm.ptr, %queriesCount: i64, %queryBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %headsNumber: f64, %groupsNumber: f64, %sizeNumber: f64, %sequencesNumber: f64, %valueOffsetNumber: f64, %valueStrideNumber: f64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c63 = llvm.mlir.constant(63 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c127 = llvm.mlir.constant(127 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %heads = arith.fptosi %headsNumber : f64 to i32
    %groups = arith.fptosi %groupsNumber : f64 to i32
    %size = arith.fptosi %sizeNumber : f64 to i32
    %sequences = arith.fptosi %sequencesNumber : f64 to i32
    %headsSize = arith.muli %heads, %size : i32
    %sequenceQueries = arith.muli %headsSize, %sequences : i32
    %queries32 = arith.trunci %queriesCount : i64 to i32
    %tokens = arith.divui %queries32, %sequenceQueries : i32
    %groupsSize = arith.muli %groups, %size : i32
    %stride = arith.muli %groupsSize, %sequences : i32
    %keys32 = arith.trunci %keysCount : i64 to i32
    %positions = arith.divui %keys32, %stride : i32
    %tokensUp = arith.addi %tokens, %c127 : i32
    %tokenBlocks = arith.shrui %tokensUp, %c7 : i32
    %tokenBlock = arith.remui %block, %tokenBlocks : i32
    %rest = arith.divui %block, %tokenBlocks : i32
    %group = arith.remui %rest, %groups : i32
    %sequence = arith.divui %rest, %groups : i32
    %half = arith.shrui %thread, %c7 : i32
    %number = arith.andi %thread, %c127 : i32
    %pairStart = arith.shli %tokenBlock, %c1 : i32
    %pooledBlock = arith.addi %pairStart, %half : i32
    %positionsUp = arith.addi %positions, %c63 : i32
    %keyBlocks = arith.shrui %positionsUp, %c6 : i32
    %tokenBlocksUp = arith.addi %tokens, %c63 : i32
    %queryBlocks = arith.shrui %tokenBlocksUp, %c6 : i32
    %requestedStep = arith.fptosi %valueStrideNumber : f64 to i32
    %packedValues = arith.cmpi eq, %requestedStep, %c0 : i32
    %valueStep = arith.select %packedValues, %stride, %requestedStep : i32
    %valueStart = arith.fptosi %valueOffsetNumber : f64 to i32
    %sequenceGroups = arith.muli %sequence, %groups : i32
    %groupStart = arith.addi %sequenceGroups, %group : i32
    %base = arith.muli %groupStart, %size : i32
    %keyBase = arith.addi %base, %number : i32
    %valueBase = arith.addi %keyBase, %valueStart : i32
    %allGroups = arith.muli %sequences, %groups : i32
    %allBlocks = arith.muli %allGroups, %keyBlocks : i32
    %section = arith.muli %allBlocks, %size : i32
    %inKeys = arith.cmpi slt, %pooledBlock, %keyBlocks : i32
    cf.cond_br %inKeys, ^keys, ^queries
  ^keys:
    %firstPosition = arith.shli %pooledBlock, %c6 : i32
    %left = arith.subi %positions, %firstPosition : i32
    %count = arith.minsi %left, %c64 : i32
    cf.br ^position(%c0, %zero, %zero : i32, f32, f32)
  ^position(%index: i32, %keySum: f32, %valueSum: f32):
    %morePositions = arith.cmpi slt, %index, %count : i32
    cf.cond_br %morePositions, ^positionBody, ^keysDone
  ^positionBody:
    %position = arith.addi %firstPosition, %index : i32
    %keyRow = arith.muli %position, %stride : i32
    %keyAt = arith.addi %keyRow, %keyBase : i32
    %keyAt64 = arith.extui %keyAt : i32 to i64
    %key = func.call @strict_device_floats(%keys, %keyBits, %keyAt64) : (!llvm.ptr, i64, i64) -> f32
    %valueRow = arith.muli %position, %valueStep : i32
    %valueAt = arith.addi %valueRow, %valueBase : i32
    %valueAt64 = arith.extui %valueAt : i32 to i64
    %value = func.call @strict_device_floats(%values, %valueBits, %valueAt64) : (!llvm.ptr, i64, i64) -> f32
    %nextKeySum = arith.addf %keySum, %key : f32
    %nextValueSum = arith.addf %valueSum, %value : f32
    %following = arith.addi %index, %c1 : i32
    cf.br ^position(%following, %nextKeySum, %nextValueSum : i32, f32, f32)
  ^keysDone:
    %countFloat = arith.sitofp %count : i32 to f32
    %keyMean = arith.divf %keySum, %countFloat : f32
    %valueMean = arith.divf %valueSum, %countFloat : f32
    %groupBlocks = arith.muli %groupStart, %keyBlocks : i32
    %blockRow = arith.addi %groupBlocks, %pooledBlock : i32
    %blockStart = arith.muli %blockRow, %size : i32
    %meanAt = arith.addi %blockStart, %number : i32
    %valueMeanAt = arith.addi %meanAt, %section : i32
    func.call @strict_device_attention_pooled_number(%result, %wide, %meanAt, %keyMean) : (!llvm.ptr, i1, i32, f32) -> ()
    func.call @strict_device_attention_pooled_number(%result, %wide, %valueMeanAt, %valueMean) : (!llvm.ptr, i1, i32, f32) -> ()
    cf.br ^queries
  ^queries:
    %inQueries = arith.cmpi slt, %pooledBlock, %queryBlocks : i32
    cf.cond_br %inQueries, ^query, ^done
  ^query:
    %firstToken = arith.shli %pooledBlock, %c6 : i32
    %tokensLeft = arith.subi %tokens, %firstToken : i32
    %tokenCount = arith.minsi %tokensLeft, %c64 : i32
    %sequenceTokens = arith.muli %sequence, %tokens : i32
    %blockTokens = arith.addi %sequenceTokens, %firstToken : i32
    cf.br ^token(%c0, %zero : i32, f32)
  ^token(%tokenIndex: i32, %querySum: f32):
    %moreTokens = arith.cmpi slt, %tokenIndex, %tokenCount : i32
    cf.cond_br %moreTokens, ^tokenBody, ^queryDone
  ^tokenBody:
    %tokenRow = arith.addi %blockTokens, %tokenIndex : i32
    %tokenHeads = arith.muli %tokenRow, %heads : i32
    %queryRow = arith.addi %tokenHeads, %group : i32
    %queryStart = arith.muli %queryRow, %size : i32
    %queryAt = arith.addi %queryStart, %number : i32
    %queryAt64 = arith.extui %queryAt : i32 to i64
    %query = func.call @strict_device_floats(%queries, %queryBits, %queryAt64) : (!llvm.ptr, i64, i64) -> f32
    %nextQuerySum = arith.addf %querySum, %query : f32
    %tokenFollowing = arith.addi %tokenIndex, %c1 : i32
    cf.br ^token(%tokenFollowing, %nextQuerySum : i32, f32)
  ^queryDone:
    %tokenCountFloat = arith.sitofp %tokenCount : i32 to f32
    %queryMean = arith.divf %querySum, %tokenCountFloat : f32
    %sequenceHeads = arith.muli %sequence, %heads : i32
    %headRow = arith.addi %sequenceHeads, %group : i32
    %headBlocks = arith.muli %headRow, %queryBlocks : i32
    %queryBlockRow = arith.addi %headBlocks, %pooledBlock : i32
    %queryBlockStart = arith.muli %queryBlockRow, %size : i32
    %queryMeanIn = arith.addi %queryBlockStart, %number : i32
    %twoSections = arith.shli %section, %c1 : i32
    %queryMeanAt = arith.addi %queryMeanIn, %twoSections : i32
    func.call @strict_device_attention_pooled_number(%result, %wide, %queryMeanAt, %queryMean) : (!llvm.ptr, i1, i32, f32) -> ()
    cf.br ^done
  ^done:
    return
  }
  // a sum over the 32 lanes of a warp
  func.func private @strict_device_attention_warp_sum(%part: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %from16 = func.call @strict_device_shuffles(%part, %c16) : (f32, i32) -> f32
    %sum16 = arith.addf %part, %from16 : f32
    %from8 = func.call @strict_device_shuffles(%sum16, %c8) : (f32, i32) -> f32
    %sum8 = arith.addf %sum16, %from8 : f32
    %from4 = func.call @strict_device_shuffles(%sum8, %c4) : (f32, i32) -> f32
    %sum4 = arith.addf %sum8, %from4 : f32
    %from2 = func.call @strict_device_shuffles(%sum4, %c2) : (f32, i32) -> f32
    %sum2 = arith.addf %sum4, %from2 : f32
    %from1 = func.call @strict_device_shuffles(%sum2, %c1) : (f32, i32) -> f32
    %sum = arith.addf %sum2, %from1 : f32
    return %sum : f32
  }
  // the threshold of a query block for its centroid at slot of the scratch (routing standard deviations of the scores of
  // the block's proxy row) at slot 1536 and its product with the mean key at 1538, by one warp
  func.func private @strict_device_attention_threshold(%scratch: !llvm.ptr<3>, %centroid: i32, %lane: i32, %routing: f32, %scale: f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %c1536 = llvm.mlir.constant(1536 : i32) : i32
    %c1538 = llvm.mlir.constant(1538 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %tiny = arith.constant 1.0e-6 : f32
    %laneStart = arith.muli %lane, %c4 : i32
    cf.br ^number(%c0, %zero, %zero : i32, f32, f32)
  ^number(%index: i32, %spread: f32, %meaned: f32):
    %more = arith.cmpi slt, %index, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %number = arith.addi %laneStart, %index : i32
    %centroidAt = arith.addi %centroid, %number : i32
    %varianceAt = arith.addi %number, %c128 : i32
    %centroidNumber = func.call @strict_device_attention_scratch(%scratch, %centroidAt) : (!llvm.ptr<3>, i32) -> f32
    %variance = func.call @strict_device_attention_scratch(%scratch, %varianceAt) : (!llvm.ptr<3>, i32) -> f32
    %mean = func.call @strict_device_attention_scratch(%scratch, %number) : (!llvm.ptr<3>, i32) -> f32
    %squared = arith.mulf %centroidNumber, %centroidNumber : f32
    %nextSpread = llvm.intr.fma(%squared, %variance, %spread) : (f32, f32, f32) -> f32
    %nextMeaned = llvm.intr.fma(%centroidNumber, %mean, %meaned) : (f32, f32, f32) -> f32
    %following = arith.addi %index, %c1 : i32
    llvm.br ^number(%following, %nextSpread, %nextMeaned : i32, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    %spreadSum = func.call @strict_device_attention_warp_sum(%spread) : (f32) -> f32
    %meanedSum = func.call @strict_device_attention_warp_sum(%meaned) : (f32) -> f32
    %scaleSquared = arith.mulf %scale, %scale : f32
    %scaledSpread = llvm.intr.fma(%spreadSum, %scaleSquared, %tiny) : (f32, f32, f32) -> f32
    %deviation = llvm.call_intrinsic "llvm.sqrt.f32"(%scaledSpread) : (f32) -> f32
    %threshold = arith.mulf %routing, %deviation : f32
    %isFirst = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %isFirst, ^store, ^stored
  ^store:
    func.call @strict_device_attention_scratch_put(%scratch, %c1536, %threshold) : (!llvm.ptr<3>, i32, f32) -> ()
    func.call @strict_device_attention_scratch_put(%scratch, %c1538, %meanedSum) : (!llvm.ptr<3>, i32, f32) -> ()
    cf.br ^stored
  ^stored:
    return
  }
  func.func private @strict_device_attention_scratch(%scratch: !llvm.ptr<3>, %at: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %scratch[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %number = llvm.load %slot : !llvm.ptr<3> -> f32
    return %number : f32
  }
  func.func private @strict_device_attention_scratch_put(%scratch: !llvm.ptr<3>, %at: i32, %number: f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %scratch[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %number, %slot : f32, !llvm.ptr<3>
    return
  }
  // a pooled number at at of the 32 bit results of strict_device_attention_pools
  func.func private @strict_device_attention_pooled_at(%pooled: !llvm.ptr, %at: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %pooled[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %number = llvm.load %slot invariant : !llvm.ptr -> f32
    return %number : f32
  }
  // the query row at row (32 bit floats or bfloat16, size 128) times the mean key of strict_device_bytes_planned
  // and scale over the 4 lanes of the row: what the centered scores of its exact positions lack against pooled ones
  func.func private @strict_device_attention_shift(%queries: !llvm.ptr, %bits: i64, %row: i64, %quad: i32, %scale: f32, %means: !llvm.ptr<3>) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %quadNumbers = arith.shli %quad, %c2 : i32
    cf.br ^part(%c0, %zero : i32, f32)
  ^part(%part: i32, %sum: f32):
    %more = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %c4 : i32
    %number = arith.addi %partStart, %quadNumbers : i32
    %number64 = arith.extui %number : i32 to i64
    %at = arith.addi %row, %number64 : i64
    %four = func.call @strict_device_attention_read(%queries, %bits, %at) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %read0 = llvm.extractelement %four[%c0 : i32] : vector<4xf32>
    %read1 = llvm.extractelement %four[%c1 : i32] : vector<4xf32>
    %read2 = llvm.extractelement %four[%c2 : i32] : vector<4xf32>
    %read3 = llvm.extractelement %four[%c3 : i32] : vector<4xf32>
    %number1 = arith.addi %number, %c1 : i32
    %number2 = arith.addi %number, %c2 : i32
    %number3 = arith.addi %number, %c3 : i32
    %mean0 = func.call @strict_device_attention_scratch(%means, %number) : (!llvm.ptr<3>, i32) -> f32
    %mean1 = func.call @strict_device_attention_scratch(%means, %number1) : (!llvm.ptr<3>, i32) -> f32
    %mean2 = func.call @strict_device_attention_scratch(%means, %number2) : (!llvm.ptr<3>, i32) -> f32
    %mean3 = func.call @strict_device_attention_scratch(%means, %number3) : (!llvm.ptr<3>, i32) -> f32
    %sum0 = llvm.intr.fma(%read0, %mean0, %sum) : (f32, f32, f32) -> f32
    %sum1 = llvm.intr.fma(%read1, %mean1, %sum0) : (f32, f32, f32) -> f32
    %sum2 = llvm.intr.fma(%read2, %mean2, %sum1) : (f32, f32, f32) -> f32
    %sum3 = llvm.intr.fma(%read3, %mean3, %sum2) : (f32, f32, f32) -> f32
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following, %sum3 : i32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    %rowSum = func.call @strict_device_attention_quad(%sum) : (f32) -> f32
    %shift = arith.mulf %rowSum, %scale : f32
    return %shift : f32
  }
  // keep operand formats constant here: merging this with the mixed path issues both predicated MMA formats
  func.func private @strict_device_attention_brain_prompt(%result: !llvm.ptr, %queries: !llvm.ptr, %keys: !llvm.ptr, %values: !llvm.ptr, %heads: i32, %groups: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1, %valueStart: i32, %valueStep: i32) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["noinline"]} {
    %brainBits = arith.constant 17 : i64
    %c128 = arith.constant 128 : i32
    %halves = arith.constant false
    func.call @strict_device_attention_prompt(%result, %queries, %brainBits, %keys, %brainBits, %values, %brainBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide, %valueStart, %valueStep, %halves) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32, i1) -> ()
    return
  }
  // `Attention.Merged` of a prompt like FlashAttention 2 on the 16 bit tensor cores: a block takes a sequence, a group
  // of keys and the rows (token and head) of up to 8 x 16 of its queries, walks the visible positions in tiles of 32
  // (two in shared memory, the next one loads while the block computes the current one) and keeps the largest score
  // and sum of every row, so the scores are never stored
  func.func private @strict_device_attention_prompt(%result: !llvm.ptr, %queries: !llvm.ptr, %queryBits: i64, %keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %heads: i32, %groups: i32, %size: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1, %viewOffset: i32, %viewStride: i32, %eights: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %brainBits = arith.constant 17 : i64
    %brainQueries = arith.cmpi eq, %queryBits, %brainBits : i64
    %brainKeys = arith.cmpi eq, %keyBits, %brainBits : i64
    %brainInputs = arith.andi %brainQueries, %brainKeys : i1
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %nativeBrains = arith.cmpi sge, %arch, %c800 : i32
    %brain = arith.andi %brainInputs, %nativeBrains : i1
    %attentionHalves = arith.constant false
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %logarithm = arith.constant 1.44269504 : f32
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %perGroup = arith.divui %heads, %groups : i32
    %warpTokens = arith.divui %c16, %perGroup : i32
    %groupsSize = arith.muli %groups, %size : i32
    %stride = arith.muli %groupsSize, %sequences : i32
    %blockTokens = arith.shli %warpTokens, %c3 : i32
    %spareTokens = arith.subi %blockTokens, %c1 : i32
    %tokensUp = arith.addi %tokens, %spareTokens : i32
    %tokenBlocks = arith.divui %tokensUp, %blockTokens : i32
    %tokenBlock = arith.remui %block, %tokenBlocks : i32
    %rest = arith.divui %block, %tokenBlocks : i32
    %group = arith.remui %rest, %groups : i32
    %sequence = arith.divui %rest, %groups : i32
    %blockFirst = arith.muli %tokenBlock, %blockTokens : i32
    %warpStart = arith.muli %warp, %warpTokens : i32
    %warpFirst = arith.addi %blockFirst, %warpStart : i32
    %tokenLast = arith.subi %tokens, %c1 : i32
    %before, %openLast = func.call @strict_device_attention_before(%positions, %tokens, %causal) : (i32, i32, i1) -> (i32, i32)
    %groupHeads = arith.muli %group, %perGroup : i32
    %sequenceTokens = arith.muli %sequence, %tokens : i32
    %upperRow = arith.addi %row, %c8 : i32
    %lowerToken, %lowerLast, %lowerQuery = func.call @strict_device_attention_row(%row, %perGroup, %warpFirst, %tokenLast, %before, %openLast, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %upperToken, %upperLast, %upperQuery = func.call @strict_device_attention_row(%upperRow, %perGroup, %warpFirst, %tokenLast, %before, %openLast, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %sizeFloat = arith.sitofp %size : i32 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeFloat) : (f32) -> f32
    %scale = arith.divf %logarithm, %root : f32
    cf.cond_br %eights, ^quantizedQueries, ^halfQueries
  ^quantizedQueries:
    %quantizedPairs = func.call @strict_device_attention_quantized(%queries, %queryBits, %lowerQuery, %upperQuery, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, f32) -> vector<32xi32>
    cf.br ^queried(%quantizedPairs : vector<32xi32>)
  ^halfQueries:
    %halfPairs = func.call @strict_device_attention_queries(%queries, %queryBits, %lowerQuery, %upperQuery, %size, %quad, %scale, %brain) : (!llvm.ptr, i64, i64, i64, i32, i32, f32, i1) -> vector<32xi32>
    cf.br ^queried(%halfPairs : vector<32xi32>)
  ^queried(%queryPairs: vector<32xi32>):
    %blockEnd = arith.addi %blockFirst, %spareTokens : i32
    %blockLastToken = arith.minsi %blockEnd, %tokenLast : i32
    %blockOwn = arith.addi %before, %blockLastToken : i32
    %blockLast = arith.minsi %blockOwn, %openLast : i32
    %blockPositions = arith.addi %blockLast, %c32 : i32
    %tiles = arith.shrui %blockPositions, %c5 : i32
    %warpSpare = arith.subi %warpTokens, %c1 : i32
    %warpEnd = arith.addi %warpFirst, %warpSpare : i32
    %warpLastToken = arith.minsi %warpEnd, %tokenLast : i32
    %warpOwn = arith.addi %before, %warpLastToken : i32
    %warpLast = arith.minsi %warpOwn, %openLast : i32
    %warpActive = arith.cmpi sle, %warpFirst, %tokenLast : i32
    %sequenceGroups = arith.muli %sequence, %groups : i32
    %groupStart = arith.addi %sequenceGroups, %group : i32
    %base = arith.muli %groupStart, %size : i32
    %viewBase = arith.addi %base, %viewOffset : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %hidden, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %zero, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %hidden, %state1[%c2 : i32] : vector<4xf32>
    %started = llvm.insertelement %zero, %state2[%c3 : i32] : vector<4xf32>
    %keyTile = llvm.mlir.addressof @strict_device_attention_keys : !llvm.ptr<3>
    %valueTile = llvm.mlir.addressof @strict_device_attention_values : !llvm.ptr<3>
    %firstKeys = func.call @strict_device_attention_fetch(%keys, %keyBits, %c0, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %firstValues = func.call @strict_device_attention_fetch(%values, %valueBits, %c0, %positions, %viewStride, %viewBase, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    func.call @strict_device_attention_keys_stored(%keyTile, %firstKeys, %keyBits, %size, %thread, %brain, %eights) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32, i1, i1) -> ()
    func.call @strict_device_attention_stored(%valueTile, %firstValues, %valueBits, %size, %thread, %attentionHalves) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %late = arith.cmpi uge, %warp, %c4 : i32
    %unit = arith.constant 1.0 : f32
    %noShares = arith.constant dense<0.0> : vector<16xf32>
    %nothing = arith.constant false
    %tileHalves = arith.constant 4352 : i32
    cf.br ^tile(%c0, %zeros, %started, %noShares, %unit, %unit, %nothing : i32, vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1)
  ^tile(%tile: i32, %outputs: vector<64xf32>, %state: vector<4xf32>, %pending: vector<16xf32>, %pendingLower: f32, %pendingUpper: f32, %isPending: i1):
    %moreTiles = arith.cmpi sle, %tile, %tiles : i32
    cf.cond_br %moreTiles, ^load, ^finish
  ^load:
    %first = arith.shli %tile, %c5 : i32
    %tileFollowing = arith.addi %tile, %c1 : i32
    %nextFirst = arith.shli %tileFollowing, %c5 : i32
    %nextKeys = func.call @strict_device_attention_fetch(%keys, %keyBits, %nextFirst, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %nextValues = func.call @strict_device_attention_fetch(%values, %valueBits, %nextFirst, %positions, %viewStride, %viewBase, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %keyBuffer = arith.andi %tile, %c1 : i32
    %nextKeyBuffer = arith.andi %tileFollowing, %c1 : i32
    %valueBuffer = arith.remui %tile, %c3 : i32
    %nextValueBuffer = arith.remui %tileFollowing, %c3 : i32
    %tileBefore = arith.addi %tile, %c2 : i32
    %previousValueBuffer = arith.remui %tileBefore, %c3 : i32
    %keysNow = func.call @strict_device_attention_buffer(%keyTile, %keyBuffer) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %keysNext = func.call @strict_device_attention_buffer(%keyTile, %nextKeyBuffer) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %valuesNow = func.call @strict_device_attention_buffer(%valueTile, %valueBuffer) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %valuesNext = func.call @strict_device_attention_buffer(%valueTile, %nextValueBuffer) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    %valuesBefore = func.call @strict_device_attention_buffer(%valueTile, %previousValueBuffer) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    cf.cond_br %isPending, ^previous, ^previousDone(%outputs : vector<64xf32>)
  ^previous:
    %previousOutputs = func.call @strict_device_attention_weighted(%pending, %outputs, %pendingLower, %pendingUpper, %valuesBefore, %size, %lane) : (vector<16xf32>, vector<64xf32>, f32, f32, !llvm.ptr<3>, i32, i32) -> vector<64xf32>
    cf.br ^previousDone(%previousOutputs : vector<64xf32>)
  ^previousDone(%doneOutputs: vector<64xf32>):
    %inTiles = arith.cmpi slt, %tile, %tiles : i32
    %visible = arith.cmpi sle, %first, %warpLast : i32
    %active = arith.andi %warpActive, %visible : i1
    %computing = arith.andi %active, %inTiles : i1
    cf.cond_br %computing, ^compute, ^computed(%doneOutputs, %state, %noShares, %unit, %unit, %nothing : vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1)
  ^compute:
    %tileScored = func.call @strict_device_attention_scored(%queryPairs, %keysNow, %size, %lane, %brain, %eights) : (vector<32xi32>, !llvm.ptr<3>, i32, i32, i1, i1) -> vector<16xf32>
    cf.br ^scored(%tileScored : vector<16xf32>)
  ^scored(%scores: vector<16xf32>):
    cf.cond_br %brain, ^scaleScores, ^softmax(%scores : vector<16xf32>)
  ^scaleScores:
    %emptyScales = llvm.mlir.undef : vector<16xf32>
    %firstScale = llvm.insertelement %scale, %emptyScales[%c0 : i32] : vector<16xf32>
    %scoreScales = llvm.shufflevector %firstScale, %firstScale [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %scaledScores = arith.mulf %scores, %scoreScales : vector<16xf32>
    cf.br ^softmax(%scaledScores : vector<16xf32>)
  ^softmax(%readyScores: vector<16xf32>):
    %shares, %newState, %lowerScale, %upperScale = func.call @strict_device_attention_softmax(%readyScores, %state, %first, %lowerLast, %upperLast, %quad) : (vector<16xf32>, vector<4xf32>, i32, i32, i32, i32) -> (vector<16xf32>, vector<4xf32>, f32, f32)
    %always = arith.constant true
    cf.cond_br %late, ^computed(%doneOutputs, %newState, %shares, %lowerScale, %upperScale, %always : vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1), ^early
  ^early:
    %weighted = func.call @strict_device_attention_weighted(%shares, %doneOutputs, %lowerScale, %upperScale, %valuesNow, %size, %lane) : (vector<16xf32>, vector<64xf32>, f32, f32, !llvm.ptr<3>, i32, i32) -> vector<64xf32>
    cf.br ^computed(%weighted, %newState, %noShares, %unit, %unit, %nothing : vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1)
  ^computed(%newOutputs: vector<64xf32>, %computedState: vector<4xf32>, %nextPending: vector<16xf32>, %nextLower: f32, %nextUpper: f32, %nextIsPending: i1):
    func.call @strict_device_attention_keys_stored(%keysNext, %nextKeys, %keyBits, %size, %thread, %brain, %eights) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32, i1, i1) -> ()
    func.call @strict_device_attention_stored(%valuesNext, %nextValues, %valueBits, %size, %thread, %attentionHalves) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^tile(%tileFollowing, %newOutputs, %computedState, %nextPending, %nextLower, %nextUpper, %nextIsPending : i32, vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1)
  ^finish:
    func.call @strict_device_attention_finish(%result, %wide, %outputs, %state, %lowerToken, %upperToken, %tokenLast, %lowerQuery, %upperQuery, %size, %quad) : (!llvm.ptr, i1, vector<64xf32>, vector<4xf32>, i32, i32, i32, i64, i64, i32, i32) -> ()
    return
  }
  // the token of a row of a warp (16 rows of tokens times the heads of a group), its last visible position and where
  // its query starts
  // the position before the first token's own (the last visible of every token without causal masking) and the last position
  func.func private @strict_device_attention_before(%positions: i32, %tokens: i32, %causal: i1) -> (i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %positionLast = arith.subi %positions, %c1 : i32
    %causalBefore = arith.subi %positions, %tokens : i32
    %before = arith.select %causal, %causalBefore, %positionLast : i32
    return %before, %positionLast : i32, i32
  }
  func.func private @strict_device_attention_row(%row: i32, %perGroup: i32, %warpFirst: i32, %tokenLast: i32, %before: i32, %openLast: i32, %groupHeads: i32, %sequenceTokens: i32, %heads: i32, %size: i32) -> (i32, i32, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %inWarp = arith.divui %row, %perGroup : i32
    %head = arith.remui %row, %perGroup : i32
    %token = arith.addi %warpFirst, %inWarp : i32
    %clamped = arith.minsi %token, %tokenLast : i32
    %own = arith.addi %before, %clamped : i32
    %last = arith.minsi %own, %openLast : i32
    %tokenRow = arith.addi %sequenceTokens, %clamped : i32
    %tokenHeads = arith.muli %tokenRow, %heads : i32
    %groupHead = arith.addi %groupHeads, %head : i32
    %queryRow = arith.addi %tokenHeads, %groupHead : i32
    %queryRow64 = arith.extui %queryRow : i32 to i64
    %size64 = arith.extui %size : i32 to i64
    %query = arith.muli %queryRow64, %size64 : i64
    return %token, %last, %query : i32, i32, i64
  }
  // a sum over the 4 lanes of a row
  func.func private @strict_device_attention_quad(%part: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %fromOne = func.call @strict_device_shuffles(%part, %c1) : (f32, i32) -> f32
    %pair = arith.addf %part, %fromOne : f32
    %fromTwo = func.call @strict_device_shuffles(%pair, %c2) : (f32, i32) -> f32
    %sum = arith.addf %pair, %fromTwo : f32
    return %sum : f32
  }
  // the two outputs at slot and the next times inverse (1 / sum of the row) into the result at at and the next (at is
  // even)
  func.func private @strict_device_attention_written(%result: !llvm.ptr, %wide: i1, %outputs: vector<64xf32>, %slot: i32, %at: i64, %inverse: f32, %writing: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %one = arith.constant 1 : i64
    cf.cond_br %writing, ^write, ^done
  ^write:
    %slotNext = arith.addi %slot, %c1 : i32
    %output = llvm.extractelement %outputs[%slot : i32] : vector<64xf32>
    %outputNext = llvm.extractelement %outputs[%slotNext : i32] : vector<64xf32>
    %value = arith.mulf %output, %inverse : f32
    %valueNext = arith.mulf %outputNext, %inverse : f32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %empty = llvm.mlir.undef : vector<2xf32>
    %withValue = llvm.insertelement %value, %empty[%c0 : i32] : vector<2xf32>
    %pair = llvm.insertelement %valueNext, %withValue[%c1 : i32] : vector<2xf32>
    cf.cond_br %wide, ^double, ^single
  ^double:
    %widePair = arith.extf %pair : vector<2xf32> to vector<2xf64>
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %widePair, %doubleSlot {alignment = 16 : i64} : vector<2xf64>, !llvm.ptr
    cf.br ^done
  ^single:
    %singleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %pair, %singleSlot {alignment = 8 : i64} : vector<2xf32>, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  // one number at at of a list of doubles, 32 or 16 bit floats, bfloat16 or 8 bit floats (bits 7) as a 32 bit float
  func.func private @strict_device_attention_number(%data: !llvm.ptr, %bits: i64, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.switch %bits : i64, [
      default: ^double,
      32: ^single,
      16: ^half,
      17: ^brain,
      7: ^byte
    ]
  ^double:
    %doubleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %double = llvm.load %doubleSlot invariant : !llvm.ptr -> f64
    %narrowed = arith.truncf %double : f64 to f32
    return %narrowed : f32
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %single = llvm.load %singleSlot invariant : !llvm.ptr -> f32
    return %single : f32
  ^half:
    %halfSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %half = llvm.load %halfSlot invariant : !llvm.ptr -> f16
    %widened = arith.extf %half : f16 to f32
    return %widened : f32
  ^brain:
    %brainSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brain = llvm.load %brainSlot invariant : !llvm.ptr -> bf16
    %fromBrain = arith.extf %brain : bf16 to f32
    return %fromBrain : f32
  ^byte:
    %byteSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %byteSlot invariant : !llvm.ptr -> i8
    %number = func.call @strict_device_float8_number(%byte) : (i8) -> f32
    return %number : f32
  }
  // the numbers at at and the next of a list (at even) as the 16 bit pair of an mma operand, read at once
  func.func private @strict_device_attention_adjacent(%data: !llvm.ptr, %bits: i64, %at: i32) -> vector<2xf16> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %at64 = arith.extui %at : i32 to i64
    cf.switch %bits : i64, [
      default: ^double,
      32: ^single,
      16: ^half,
      17: ^brain,
      7: ^byte
    ]
  ^double:
    %doubleSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubles = llvm.load %doubleSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<2xf64>
    %fromDoubles = arith.truncf %doubles : vector<2xf64> to vector<2xf16>
    return %fromDoubles : vector<2xf16>
  ^single:
    %singleSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singles = llvm.load %singleSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xf32>
    %fromSingles = arith.truncf %singles : vector<2xf32> to vector<2xf16>
    return %fromSingles : vector<2xf16>
  ^half:
    %halfSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halves = llvm.load %halfSlot invariant {alignment = 4 : i64} : !llvm.ptr -> vector<2xf16>
    return %halves : vector<2xf16>
  ^brain:
    %brainSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainSlot invariant {alignment = 4 : i64} : !llvm.ptr -> vector<2xbf16>
    %brainSingles = arith.extf %brains : vector<2xbf16> to vector<2xf32>
    %fromBrains = arith.truncf %brainSingles : vector<2xf32> to vector<2xf16>
    return %fromBrains : vector<2xf16>
  ^byte:
    %byteSlot = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %bytes = llvm.load %byteSlot invariant {alignment = 2 : i64} : !llvm.ptr -> vector<2xi8>
    %first = llvm.extractelement %bytes[%c0 : i32] : vector<2xi8>
    %second = llvm.extractelement %bytes[%c1 : i32] : vector<2xi8>
    %firstNumber = func.call @strict_device_float8_number(%first) : (i8) -> f32
    %secondNumber = func.call @strict_device_float8_number(%second) : (i8) -> f32
    %pairBits = func.call @strict_device_attention_pair(%firstNumber, %secondNumber) : (f32, f32) -> i32
    %pair = llvm.bitcast %pairBits : i32 to vector<2xf16>
    return %pair : vector<2xf16>
  }
  // the numbers at first and second of a list as the 16 bit pair of an mma operand
  func.func private @strict_device_attention_numbers(%data: !llvm.ptr, %bits: i64, %first: i32, %second: i32) -> vector<2xf16> attributes {llvm.passthrough = ["alwaysinline"]} {
    %first64 = arith.extui %first : i32 to i64
    %second64 = arith.extui %second : i32 to i64
    %firstNumber = func.call @strict_device_attention_number(%data, %bits, %first64) : (!llvm.ptr, i64, i64) -> f32
    %secondNumber = func.call @strict_device_attention_number(%data, %bits, %second64) : (!llvm.ptr, i64, i64) -> f32
    %pairBits = func.call @strict_device_attention_pair(%firstNumber, %secondNumber) : (f32, f32) -> i32
    %pair = llvm.bitcast %pairBits : i32 to vector<2xf16>
    return %pair : vector<2xf16>
  }
  // the key and value operands like strict_device_attention_loaded, with the bits of both lists known for 32 and 8
  // bit caches the loads are not separated by branches and wait for memory together
  func.func private @strict_device_attention_operands(%keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %first: i32, %positionLast: i32, %stride: i32, %base: i32, %size: i32, %row: i32, %quad: i32, %viewOffset: i32, %viewStride: i32) -> (vector<16xi32>, vector<16xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %singles = arith.constant 32 : i64
    %bytes = arith.constant 7 : i64
    %same = arith.cmpi eq, %keyBits, %valueBits : i64
    cf.cond_br %same, ^known, ^mixed
  ^known:
    cf.switch %keyBits : i64, [
      default: ^mixed,
      32: ^single,
      7: ^byte
    ]
  ^single:
    %singleKeys, %singleValues = func.call @strict_device_attention_loaded(%keys, %singles, %values, %singles, %first, %positionLast, %stride, %base, %size, %row, %quad, %viewOffset, %viewStride) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %singleKeys, %singleValues : vector<16xi32>, vector<16xi32>
  ^byte:
    %byteKeys, %byteValues = func.call @strict_device_attention_loaded(%keys, %bytes, %values, %bytes, %first, %positionLast, %stride, %base, %size, %row, %quad, %viewOffset, %viewStride) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %byteKeys, %byteValues : vector<16xi32>, vector<16xi32>
  ^mixed:
    %mixedKeys, %mixedValues = func.call @strict_device_attention_loaded(%keys, %keyBits, %values, %valueBits, %first, %positionLast, %stride, %base, %size, %row, %quad, %viewOffset, %viewStride) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %mixedKeys, %mixedValues : vector<16xi32>, vector<16xi32>
  }
  // the key and value operands of 8 positions from first for a lane (keys of position first + lane / 4, numbers
  // 2 * (lane % 4) and the next of each step of 8, values of positions first + 2 * (lane % 4) and the next, number
  // lane / 4 of each 8), all loads first so they wait for memory together
  func.func private @strict_device_attention_loaded(%keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %first: i32, %positionLast: i32, %stride: i32, %base: i32, %size: i32, %row: i32, %quad: i32, %viewOffset: i32, %viewStride: i32) -> (vector<16xi32>, vector<16xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %zeros = arith.constant dense<0> : vector<16xi32>
    %steps = arith.shrui %size, %c3 : i32
    %quadPositions = arith.shli %quad, %c1 : i32
    %keyPosition = arith.addi %first, %row : i32
    %keyClamped = arith.minsi %keyPosition, %positionLast : i32
    %keyRow = arith.muli %keyClamped, %stride : i32
    %keyGroup = arith.addi %keyRow, %base : i32
    %keyStart = arith.addi %keyGroup, %quadPositions : i32
    %position = arith.addi %first, %quadPositions : i32
    %positionNext = arith.addi %position, %c1 : i32
    %valuePosition = arith.minsi %position, %positionLast : i32
    %valuePositionNext = arith.minsi %positionNext, %positionLast : i32
    %viewBase = arith.addi %base, %viewOffset : i32
    %valueRow = arith.muli %valuePosition, %viewStride : i32
    %valueRowNext = arith.muli %valuePositionNext, %viewStride : i32
    %valueGroup = arith.addi %valueRow, %viewBase : i32
    %valueGroupNext = arith.addi %valueRowNext, %viewBase : i32
    %valueStart = arith.addi %valueGroup, %row : i32
    %valueStartNext = arith.addi %valueGroupNext, %row : i32
    cf.br ^step(%c0, %zeros, %zeros : i32, vector<16xi32>, vector<16xi32>)
  ^step(%step: i32, %keyPairs: vector<16xi32>, %valuePairs: vector<16xi32>):
    %more = arith.cmpi slt, %step, %c16 : i32
    cf.cond_br %more, ^check, ^done
  ^check:
    %used = arith.cmpi slt, %step, %steps : i32
    cf.cond_br %used, ^body, ^next(%keyPairs, %valuePairs : vector<16xi32>, vector<16xi32>)
  ^body:
    %stepNumbers = arith.shli %step, %c3 : i32
    %keyAt = arith.addi %keyStart, %stepNumbers : i32
    %key = func.call @strict_device_attention_adjacent(%keys, %keyBits, %keyAt) : (!llvm.ptr, i64, i32) -> vector<2xf16>
    %valueAt = arith.addi %valueStart, %stepNumbers : i32
    %valueAtNext = arith.addi %valueStartNext, %stepNumbers : i32
    %value = func.call @strict_device_attention_numbers(%values, %valueBits, %valueAt, %valueAtNext) : (!llvm.ptr, i64, i32, i32) -> vector<2xf16>
    %keyBitsPair = llvm.bitcast %key : vector<2xf16> to i32
    %valueBitsPair = llvm.bitcast %value : vector<2xf16> to i32
    %withKey = llvm.insertelement %keyBitsPair, %keyPairs[%step : i32] : vector<16xi32>
    %withValue = llvm.insertelement %valueBitsPair, %valuePairs[%step : i32] : vector<16xi32>
    cf.br ^next(%withKey, %withValue : vector<16xi32>, vector<16xi32>)
  ^next(%nextKeys: vector<16xi32>, %nextValues: vector<16xi32>):
    %following = arith.addi %step, %c1 : i32
    llvm.br ^step(%following, %nextKeys, %nextValues : i32, vector<16xi32>, vector<16xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %keyPairs, %valuePairs : vector<16xi32>, vector<16xi32>
  }
  // 8 positions from first for a warp while decoding (key and value operands loaded before): the scores of the two
  // rows of a lane, the new largest scores and sums, the outputs rescaled and the shares times the values added
  // (positions after last hidden)
  func.func private @strict_device_attention_chunk(%queries: vector<32xi32>, %outputs: vector<64xf32>, %state: vector<4xf32>, %keyPairs: vector<16xi32>, %valuePairs: vector<16xi32>, %first: i32, %lastLower: i32, %lastUpper: i32, %size: i32, %quad: i32) -> (vector<64xf32>, vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %steps = arith.shrui %size, %c3 : i32
    %quadPositions = arith.shli %quad, %c1 : i32
    cf.br ^step(%c0, %zero, %zero, %zero, %zero : i32, f32, f32, f32, f32)
  ^step(%step: i32, %sum0: f32, %sum1: f32, %sum2: f32, %sum3: f32):
    %moreSteps = arith.cmpi slt, %step, %c16 : i32
    cf.cond_br %moreSteps, ^stepCheck, ^scored
  ^stepCheck:
    %used = arith.cmpi slt, %step, %steps : i32
    cf.cond_br %used, ^stepBody, ^stepNext(%sum0, %sum1, %sum2, %sum3 : f32, f32, f32, f32)
  ^stepBody:
    %keyPairBits = llvm.extractelement %keyPairs[%step : i32] : vector<16xi32>
    %key = llvm.bitcast %keyPairBits : i32 to vector<2xf16>
    %slot = arith.shli %step, %c1 : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %lowerBits = llvm.extractelement %queries[%slot : i32] : vector<32xi32>
    %upperBits = llvm.extractelement %queries[%slotNext : i32] : vector<32xi32>
    %lowerQuery = llvm.bitcast %lowerBits : i32 to vector<2xf16>
    %upperQuery = llvm.bitcast %upperBits : i32 to vector<2xf16>
    %dot = func.call @strict_device_halves_mma(%lowerQuery, %upperQuery, %key, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(f32, f32, f32, f32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(f32, f32, f32, f32)>
    %dot2 = llvm.extractvalue %dot[2] : !llvm.struct<(f32, f32, f32, f32)>
    %dot3 = llvm.extractvalue %dot[3] : !llvm.struct<(f32, f32, f32, f32)>
    cf.br ^stepNext(%dot0, %dot1, %dot2, %dot3 : f32, f32, f32, f32)
  ^stepNext(%next0: f32, %next1: f32, %next2: f32, %next3: f32):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %next0, %next1, %next2, %next3 : i32, f32, f32, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scored:
    %position = arith.addi %first, %quadPositions : i32
    %positionNext = arith.addi %position, %c1 : i32
    %lowerVisible = arith.cmpi sle, %position, %lastLower : i32
    %lowerVisibleNext = arith.cmpi sle, %positionNext, %lastLower : i32
    %upperVisible = arith.cmpi sle, %position, %lastUpper : i32
    %upperVisibleNext = arith.cmpi sle, %positionNext, %lastUpper : i32
    %lowerShown = arith.select %lowerVisible, %sum0, %hidden : f32
    %lowerShownNext = arith.select %lowerVisibleNext, %sum1, %hidden : f32
    %upperShown = arith.select %upperVisible, %sum2, %hidden : f32
    %upperShownNext = arith.select %upperVisibleNext, %sum3, %hidden : f32
    %lowerPair = arith.maximumf %lowerShown, %lowerShownNext : f32
    %upperPair = arith.maximumf %upperShown, %upperShownNext : f32
    %lowerOne = func.call @strict_device_shuffles(%lowerPair, %c1) : (f32, i32) -> f32
    %upperOne = func.call @strict_device_shuffles(%upperPair, %c1) : (f32, i32) -> f32
    %lowerTwo = arith.maximumf %lowerPair, %lowerOne : f32
    %upperTwo = arith.maximumf %upperPair, %upperOne : f32
    %lowerFour = func.call @strict_device_shuffles(%lowerTwo, %c2) : (f32, i32) -> f32
    %upperFour = func.call @strict_device_shuffles(%upperTwo, %c2) : (f32, i32) -> f32
    %lowerLargest = arith.maximumf %lowerTwo, %lowerFour : f32
    %upperLargest = arith.maximumf %upperTwo, %upperFour : f32
    %oldLower = llvm.extractelement %state[%c0 : i32] : vector<4xf32>
    %lowerSum = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %oldUpper = llvm.extractelement %state[%c2 : i32] : vector<4xf32>
    %upperSum = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %newLower = arith.maximumf %oldLower, %lowerLargest : f32
    %newUpper = arith.maximumf %oldUpper, %upperLargest : f32
    %lowerShift = arith.subf %oldLower, %newLower : f32
    %upperShift = arith.subf %oldUpper, %newUpper : f32
    %lowerScale = llvm.call_intrinsic "llvm.exp2.f32"(%lowerShift) : (f32) -> f32
    %upperScale = llvm.call_intrinsic "llvm.exp2.f32"(%upperShift) : (f32) -> f32
    %lowerShare, %lowerShareNext = func.call @strict_device_attention_pairShares(%sum0, %sum1, %position, %lastLower, %newLower) : (f32, f32, i32, i32, f32) -> (f32, f32)
    %upperShare, %upperShareNext = func.call @strict_device_attention_pairShares(%sum2, %sum3, %position, %lastUpper, %newUpper) : (f32, f32, i32, i32, f32) -> (f32, f32)
    %lowerTotal = arith.addf %lowerShare, %lowerShareNext : f32
    %upperTotal = arith.addf %upperShare, %upperShareNext : f32
    %lowerScaledSum = arith.mulf %lowerSum, %lowerScale : f32
    %upperScaledSum = arith.mulf %upperSum, %upperScale : f32
    %lowerNewSum = arith.addf %lowerScaledSum, %lowerTotal : f32
    %upperNewSum = arith.addf %upperScaledSum, %upperTotal : f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %newLower, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %lowerNewSum, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %newUpper, %state1[%c2 : i32] : vector<4xf32>
    %newState = llvm.insertelement %upperNewSum, %state2[%c3 : i32] : vector<4xf32>
    %lowerSharesBits = func.call @strict_device_attention_pair(%lowerShare, %lowerShareNext) : (f32, f32) -> i32
    %upperSharesBits = func.call @strict_device_attention_pair(%upperShare, %upperShareNext) : (f32, f32) -> i32
    %lowerShares = llvm.bitcast %lowerSharesBits : i32 to vector<2xf16>
    %upperShares = llvm.bitcast %upperSharesBits : i32 to vector<2xf16>
    cf.br ^column(%c0, %outputs : i32, vector<64xf32>)
  ^column(%column: i32, %columnOutputs: vector<64xf32>):
    %moreColumns = arith.cmpi slt, %column, %c16 : i32
    cf.cond_br %moreColumns, ^columnCheck, ^done
  ^columnCheck:
    %usedColumn = arith.cmpi slt, %column, %steps : i32
    cf.cond_br %usedColumn, ^columnBody, ^columnNext(%columnOutputs : vector<64xf32>)
  ^columnBody:
    %valuePairBits = llvm.extractelement %valuePairs[%column : i32] : vector<16xi32>
    %value = llvm.bitcast %valuePairBits : i32 to vector<2xf16>
    %outputSlot = arith.shli %column, %c2 : i32
    %outputSlot1 = arith.addi %outputSlot, %c1 : i32
    %outputSlot2 = arith.addi %outputSlot, %c2 : i32
    %outputSlot3 = arith.addi %outputSlot, %c3 : i32
    %output0 = llvm.extractelement %columnOutputs[%outputSlot : i32] : vector<64xf32>
    %output1 = llvm.extractelement %columnOutputs[%outputSlot1 : i32] : vector<64xf32>
    %output2 = llvm.extractelement %columnOutputs[%outputSlot2 : i32] : vector<64xf32>
    %output3 = llvm.extractelement %columnOutputs[%outputSlot3 : i32] : vector<64xf32>
    %scaled0 = arith.mulf %output0, %lowerScale : f32
    %scaled1 = arith.mulf %output1, %lowerScale : f32
    %scaled2 = arith.mulf %output2, %upperScale : f32
    %scaled3 = arith.mulf %output3, %upperScale : f32
    %weighted = func.call @strict_device_halves_mma(%lowerShares, %upperShares, %value, %scaled0, %scaled1, %scaled2, %scaled3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %weighted0 = llvm.extractvalue %weighted[0] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted1 = llvm.extractvalue %weighted[1] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted2 = llvm.extractvalue %weighted[2] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted3 = llvm.extractvalue %weighted[3] : !llvm.struct<(f32, f32, f32, f32)>
    %stored0 = llvm.insertelement %weighted0, %columnOutputs[%outputSlot : i32] : vector<64xf32>
    %stored1 = llvm.insertelement %weighted1, %stored0[%outputSlot1 : i32] : vector<64xf32>
    %stored2 = llvm.insertelement %weighted2, %stored1[%outputSlot2 : i32] : vector<64xf32>
    %stored3 = llvm.insertelement %weighted3, %stored2[%outputSlot3 : i32] : vector<64xf32>
    cf.br ^columnNext(%stored3 : vector<64xf32>)
  ^columnNext(%nextOutputs: vector<64xf32>):
    %columnFollowing = arith.addi %column, %c1 : i32
    llvm.br ^column(%columnFollowing, %nextOutputs : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %columnOutputs, %newState : vector<64xf32>, vector<4xf32>
  }
  // the shares of the scores of positions position and the next (2 to the power of score minus largest, 0 after last)
  func.func private @strict_device_attention_pairShares(%score: f32, %scoreNext: f32, %position: i32, %last: i32, %largest: f32) -> (f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %positionNext = arith.addi %position, %c1 : i32
    %shifted = arith.subf %score, %largest : f32
    %shiftedNext = arith.subf %scoreNext, %largest : f32
    %share = llvm.call_intrinsic "llvm.exp2.f32"(%shifted) : (f32) -> f32
    %shareNext = llvm.call_intrinsic "llvm.exp2.f32"(%shiftedNext) : (f32) -> f32
    %visible = arith.cmpi sle, %position, %last : i32
    %visibleNext = arith.cmpi sle, %positionNext, %last : i32
    %shown = arith.select %visible, %share, %zero : f32
    %shownNext = arith.select %visibleNext, %shareNext, %zero : f32
    return %shown, %shownNext : f32, f32
  }
  // the outputs and state of another warp or block (the same rows and numbers in the same lanes) merged into these
  func.func private @strict_device_attention_merge(%outputs: vector<64xf32>, %state: vector<4xf32>, %otherOutputs: vector<64xf32>, %otherState: vector<4xf32>) -> (vector<64xf32>, vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %lowerLargest = llvm.extractelement %state[%c0 : i32] : vector<4xf32>
    %lowerSum = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperLargest = llvm.extractelement %state[%c2 : i32] : vector<4xf32>
    %upperSum = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %otherLowerLargest = llvm.extractelement %otherState[%c0 : i32] : vector<4xf32>
    %otherLowerSum = llvm.extractelement %otherState[%c1 : i32] : vector<4xf32>
    %otherUpperLargest = llvm.extractelement %otherState[%c2 : i32] : vector<4xf32>
    %otherUpperSum = llvm.extractelement %otherState[%c3 : i32] : vector<4xf32>
    %lower = arith.maximumf %lowerLargest, %otherLowerLargest : f32
    %upper = arith.maximumf %upperLargest, %otherUpperLargest : f32
    %lowerShift = arith.subf %lowerLargest, %lower : f32
    %otherLowerShift = arith.subf %otherLowerLargest, %lower : f32
    %upperShift = arith.subf %upperLargest, %upper : f32
    %otherUpperShift = arith.subf %otherUpperLargest, %upper : f32
    %lowerScale = llvm.call_intrinsic "llvm.exp2.f32"(%lowerShift) : (f32) -> f32
    %otherLowerScale = llvm.call_intrinsic "llvm.exp2.f32"(%otherLowerShift) : (f32) -> f32
    %upperScale = llvm.call_intrinsic "llvm.exp2.f32"(%upperShift) : (f32) -> f32
    %otherUpperScale = llvm.call_intrinsic "llvm.exp2.f32"(%otherUpperShift) : (f32) -> f32
    %lowerScaled = arith.mulf %lowerSum, %lowerScale : f32
    %otherLowerScaled = arith.mulf %otherLowerSum, %otherLowerScale : f32
    %upperScaled = arith.mulf %upperSum, %upperScale : f32
    %otherUpperScaled = arith.mulf %otherUpperSum, %otherUpperScale : f32
    %lowerTotal = arith.addf %lowerScaled, %otherLowerScaled : f32
    %upperTotal = arith.addf %upperScaled, %otherUpperScaled : f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %lower, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %lowerTotal, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %upper, %state1[%c2 : i32] : vector<4xf32>
    %merged = llvm.insertelement %upperTotal, %state2[%c3 : i32] : vector<4xf32>
    cf.br ^number(%c0, %outputs : i32, vector<64xf32>)
  ^number(%number: i32, %current: vector<64xf32>):
    %more = arith.cmpi slt, %number, %c64 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %inRow = arith.andi %number, %c3 : i32
    %isLower = arith.cmpi slt, %inRow, %c2 : i32
    %scale = arith.select %isLower, %lowerScale, %upperScale : f32
    %otherScale = arith.select %isLower, %otherLowerScale, %otherUpperScale : f32
    %output = llvm.extractelement %current[%number : i32] : vector<64xf32>
    %otherOutput = llvm.extractelement %otherOutputs[%number : i32] : vector<64xf32>
    %scaledOutput = arith.mulf %output, %scale : f32
    %scaledOther = arith.mulf %otherOutput, %otherScale : f32
    %sum = arith.addf %scaledOutput, %scaledOther : f32
    %updated = llvm.insertelement %sum, %current[%number : i32] : vector<64xf32>
    %following = arith.addi %number, %c1 : i32
    llvm.br ^number(%following, %updated : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %current, %merged : vector<64xf32>, vector<4xf32>
  }
  // the 64 outputs and the state of a lane to to (17 parts of 4 numbers, each 32 parts apart so the lanes write
  // neighbors)
  func.func private @strict_device_attention_put(%to: !llvm.ptr, %lane: i32, %outputs: vector<64xf32>, %state: vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    cf.br ^part(%c0 : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c16 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %first = arith.shli %part, %c2 : i32
    %second = arith.addi %first, %c1 : i32
    %third = arith.addi %first, %c2 : i32
    %fourth = arith.addi %first, %c3 : i32
    %output0 = llvm.extractelement %outputs[%first : i32] : vector<64xf32>
    %output1 = llvm.extractelement %outputs[%second : i32] : vector<64xf32>
    %output2 = llvm.extractelement %outputs[%third : i32] : vector<64xf32>
    %output3 = llvm.extractelement %outputs[%fourth : i32] : vector<64xf32>
    %with0 = llvm.insertelement %output0, %empty[%c0 : i32] : vector<4xf32>
    %with1 = llvm.insertelement %output1, %with0[%c1 : i32] : vector<4xf32>
    %with2 = llvm.insertelement %output2, %with1[%c2 : i32] : vector<4xf32>
    %numbers = llvm.insertelement %output3, %with2[%c3 : i32] : vector<4xf32>
    func.call @strict_device_attention_part(%to, %part, %lane, %numbers) : (!llvm.ptr, i32, i32, vector<4xf32>) -> ()
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    func.call @strict_device_attention_part(%to, %c16, %lane, %state) : (!llvm.ptr, i32, i32, vector<4xf32>) -> ()
    return
  }
  // where part number part of a lane starts
  func.func private @strict_device_attention_slot(%start: !llvm.ptr, %part: i32, %lane: i32) -> !llvm.ptr attributes {llvm.passthrough = ["alwaysinline"]} {
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %row = arith.shli %part, %c5 : i32
    %at = arith.addi %row, %lane : i32
    %numbers = arith.shli %at, %c2 : i32
    %numbers64 = arith.extui %numbers : i32 to i64
    %slot = llvm.getelementptr %start[%numbers64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    return %slot : !llvm.ptr
  }
  func.func private @strict_device_attention_part(%to: !llvm.ptr, %part: i32, %lane: i32, %numbers: vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = func.call @strict_device_attention_slot(%to, %part, %lane) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    llvm.store %numbers, %slot {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    return
  }
  // the 64 outputs and the state of a lane written by strict_device_attention_put, fresh past the cache of the SM for
  // the outputs of other blocks
  func.func private @strict_device_attention_get(%from: !llvm.ptr, %lane: i32, %fresh: i1) -> (vector<64xf32>, vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %outputs = arith.constant dense<0.0> : vector<64xf32>
    cf.br ^part(%c0, %outputs : i32, vector<64xf32>)
  ^part(%part: i32, %readOutputs: vector<64xf32>):
    %more = arith.cmpi slt, %part, %c16 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %numbers = func.call @strict_device_attention_read_part(%from, %part, %lane, %fresh) : (!llvm.ptr, i32, i32, i1) -> vector<4xf32>
    %first = arith.shli %part, %c2 : i32
    %second = arith.addi %first, %c1 : i32
    %third = arith.addi %first, %c2 : i32
    %fourth = arith.addi %first, %c3 : i32
    %number0 = llvm.extractelement %numbers[%c0 : i32] : vector<4xf32>
    %number1 = llvm.extractelement %numbers[%c1 : i32] : vector<4xf32>
    %number2 = llvm.extractelement %numbers[%c2 : i32] : vector<4xf32>
    %number3 = llvm.extractelement %numbers[%c3 : i32] : vector<4xf32>
    %with0 = llvm.insertelement %number0, %readOutputs[%first : i32] : vector<64xf32>
    %with1 = llvm.insertelement %number1, %with0[%second : i32] : vector<64xf32>
    %with2 = llvm.insertelement %number2, %with1[%third : i32] : vector<64xf32>
    %with3 = llvm.insertelement %number3, %with2[%fourth : i32] : vector<64xf32>
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following, %with3 : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    %state = func.call @strict_device_attention_read_part(%from, %c16, %lane, %fresh) : (!llvm.ptr, i32, i32, i1) -> vector<4xf32>
    return %readOutputs, %state : vector<64xf32>, vector<4xf32>
  }
  func.func private @strict_device_attention_read_part(%from: !llvm.ptr, %part: i32, %lane: i32, %fresh: i1) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = func.call @strict_device_attention_slot(%from, %part, %lane) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    cf.cond_br %fresh, ^uncached, ^cached
  ^uncached:
    %freshNumbers = llvm.load volatile %slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %freshNumbers : vector<4xf32>
  ^cached:
    %cachedNumbers = llvm.load %slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %cachedNumbers : vector<4xf32>
  }
  // where warps 4 to 7 (then 2 and 3, then 1) put their outputs for the warp 4 (2, 1) below: 2 regions in each tile
  func.func private @strict_device_attention_region(%region: i32) -> !llvm.ptr attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c2176 = llvm.mlir.constant(2176 : i32) : i32
    %keyTile = llvm.mlir.addressof @strict_device_attention_keys : !llvm.ptr<3>
    %valueTile = llvm.mlir.addressof @strict_device_attention_values : !llvm.ptr<3>
    %keyStart = llvm.addrspacecast %keyTile : !llvm.ptr<3> to !llvm.ptr
    %valueStart = llvm.addrspacecast %valueTile : !llvm.ptr<3> to !llvm.ptr
    %inKeys = arith.cmpi ult, %region, %c2 : i32
    %tile = arith.select %inKeys, %keyStart, %valueStart : !llvm.ptr
    %local = arith.andi %region, %c1 : i32
    %offset = arith.muli %local, %c2176 : i32
    %offset64 = arith.extui %offset : i32 to i64
    %start = llvm.getelementptr %tile[%offset64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    return %start : !llvm.ptr
  }
  // the outputs and states of the 8 warps of a block merged into warp 0 through the shared tiles: warps 4 to 7 into
  // 0 to 3, then 1 to 3 into 0
  func.func private @strict_device_attention_warps(%outputs: vector<64xf32>, %state: vector<4xf32>, %warp: i32, %lane: i32) -> (vector<64xf32>, vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    cf.br ^round(%c8, %c4, %outputs, %state : i32, i32, vector<64xf32>, vector<4xf32>)
  ^round(%active: i32, %receivers: i32, %current: vector<64xf32>, %currentState: vector<4xf32>):
    %more = arith.cmpi sgt, %receivers, %c0 : i32
    cf.cond_br %more, ^roundBody, ^done
  ^roundBody:
    func.call @strict_device_barrier() : () -> ()
    %above = arith.cmpi sge, %warp, %receivers : i32
    %below = arith.cmpi slt, %warp, %active : i32
    %isSender = arith.andi %above, %below : i1
    cf.cond_br %isSender, ^send, ^sent
  ^send:
    %sendRegion = arith.subi %warp, %receivers : i32
    %sendStart = func.call @strict_device_attention_region(%sendRegion) : (i32) -> !llvm.ptr
    func.call @strict_device_attention_put(%sendStart, %lane, %current, %currentState) : (!llvm.ptr, i32, vector<64xf32>, vector<4xf32>) -> ()
    cf.br ^sent
  ^sent:
    func.call @strict_device_barrier() : () -> ()
    %regions = arith.subi %active, %receivers : i32
    cf.br ^receive(%warp, %current, %currentState : i32, vector<64xf32>, vector<4xf32>)
  ^receive(%region: i32, %received: vector<64xf32>, %receivedState: vector<4xf32>):
    %isReceiver = arith.cmpi slt, %warp, %receivers : i32
    %moreRegions = arith.cmpi slt, %region, %regions : i32
    %receiving = arith.andi %isReceiver, %moreRegions : i1
    cf.cond_br %receiving, ^receiveBody, ^roundNext(%received, %receivedState : vector<64xf32>, vector<4xf32>)
  ^receiveBody:
    %receiveStart = func.call @strict_device_attention_region(%region) : (i32) -> !llvm.ptr
    %cached = arith.constant false
    %other, %otherState = func.call @strict_device_attention_get(%receiveStart, %lane, %cached) : (!llvm.ptr, i32, i1) -> (vector<64xf32>, vector<4xf32>)
    %merged, %mergedState = func.call @strict_device_attention_merge(%received, %receivedState, %other, %otherState) : (vector<64xf32>, vector<4xf32>, vector<64xf32>, vector<4xf32>) -> (vector<64xf32>, vector<4xf32>)
    %regionNext = arith.addi %region, %receivers : i32
    cf.br ^receive(%regionNext, %merged, %mergedState : i32, vector<64xf32>, vector<4xf32>)
  ^roundNext(%next: vector<64xf32>, %nextState: vector<4xf32>):
    %receiversNext = arith.shrui %receivers, %c2 : i32
    cf.br ^round(%receivers, %receiversNext, %next, %nextState : i32, i32, vector<64xf32>, vector<4xf32>)
  ^done:
    return %current, %currentState : vector<64xf32>, vector<4xf32>
  }
  // the merged outputs of a block while decoding with more blocks per group (68 numbers per lane, at most 512 blocks)
  // and how many blocks of each group are done (the last one merges them all and sets the counter back to 0)
  llvm.mlir.global internal @strict_device_attention_partials() {addr_space = 1 : i32, alignment = 16 : i64} : !llvm.array<1114112 x f32>
  llvm.mlir.global internal @strict_device_attention_counters(dense<0> : tensor<512xi32>) {addr_space = 1 : i32} : !llvm.array<512 x i32>
  llvm.mlir.global internal @strict_device_attention_last() {addr_space = 3 : i32} : i32
  // the start of the partial outputs of block number at
  func.func private @strict_device_attention_partial(%at: i32) -> !llvm.ptr attributes {llvm.passthrough = ["alwaysinline"]} {
    %c2176 = llvm.mlir.constant(2176 : i32) : i32
    %partials = llvm.mlir.addressof @strict_device_attention_partials : !llvm.ptr<1>
    %start = llvm.addrspacecast %partials : !llvm.ptr<1> to !llvm.ptr
    %offset = arith.muli %at, %c2176 : i32
    %offset64 = arith.extui %offset : i32 to i64
    %slot = llvm.getelementptr %start[%offset64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    return %slot : !llvm.ptr
  }
  // `Attention.Merged` of a few tokens (the rows of a group in tiles of 16, one per warp, up to 4 tiles verify drafts)
  // like flash decoding: the 8 warps of a block and the blocks of a tile split the positions 8 at a time, read keys and
  // values straight into the mma operands and merge their largest scores, sums and outputs, the last block writes them
  func.func private @strict_device_attention_decode(%result: !llvm.ptr, %queries: !llvm.ptr, %queryBits: i64, %keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %heads: i32, %groups: i32, %size: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1, %viewOffset: i32, %viewStride: i32) {
    %attentionHalves = arith.constant false
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %logarithm = arith.constant 1.44269504 : f32
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %blocks = func.call @strict_device_block_count() : () -> i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %perGroup = arith.divui %heads, %groups : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %warpTokens = arith.divui %c16, %perGroup : i32
    %spareTokens = arith.subi %warpTokens, %c1 : i32
    %tokensUp = arith.addi %tokens, %spareTokens : i32
    %tiles = arith.divui %tokensUp, %warpTokens : i32
    %groupSlots = arith.muli %sequences, %groups : i32
    %slots = arith.muli %groupSlots, %tiles : i32
    %splits = arith.divui %blocks, %slots : i32
    %split = arith.remui %block, %splits : i32
    %slot = arith.divui %block, %splits : i32
    %tile = arith.remui %slot, %tiles : i32
    %groupSlot = arith.divui %slot, %tiles : i32
    %group = arith.remui %groupSlot, %groups : i32
    %sequence = arith.divui %groupSlot, %groups : i32
    %warpFirst = arith.muli %tile, %warpTokens : i32
    %tokenLast = arith.subi %tokens, %c1 : i32
    %before, %openLast = func.call @strict_device_attention_before(%positions, %tokens, %causal) : (i32, i32, i1) -> (i32, i32)
    %groupHeads = arith.muli %group, %perGroup : i32
    %sequenceTokens = arith.muli %sequence, %tokens : i32
    %upperRow = arith.addi %row, %c8 : i32
    %lowerToken, %lowerLast, %lowerQuery = func.call @strict_device_attention_row(%row, %perGroup, %warpFirst, %tokenLast, %before, %openLast, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %upperToken, %upperLast, %upperQuery = func.call @strict_device_attention_row(%upperRow, %perGroup, %warpFirst, %tokenLast, %before, %openLast, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %groupsSize = arith.muli %groups, %size : i32
    %stride = arith.muli %groupsSize, %sequences : i32
    %sequenceGroups = arith.muli %sequence, %groups : i32
    %groupStart = arith.addi %sequenceGroups, %group : i32
    %base = arith.muli %groupStart, %size : i32
    %positionLast = arith.subi %positions, %c1 : i32
    %positionsUp = arith.addi %positions, %c7 : i32
    %chunks = arith.shrui %positionsUp, %c3 : i32
    %workers = arith.shli %splits, %c3 : i32
    %splitWarps = arith.shli %split, %c3 : i32
    %worker = arith.addi %splitWarps, %warp : i32
    %workerChunks = arith.muli %worker, %chunks : i32
    %chunkFirst = arith.divui %workerChunks, %workers : i32
    %nextWorkerChunks = arith.addi %workerChunks, %chunks : i32
    %chunkEnd = arith.divui %nextWorkerChunks, %workers : i32
    %firstPosition = arith.shli %chunkFirst, %c3 : i32
    %firstKeys, %firstValues = func.call @strict_device_attention_operands(%keys, %keyBits, %values, %valueBits, %firstPosition, %positionLast, %stride, %base, %size, %row, %quad, %viewOffset, %viewStride) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    %sizeFloat = arith.sitofp %size : i32 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeFloat) : (f32) -> f32
    %scale = arith.divf %logarithm, %root : f32
    %queryPairs = func.call @strict_device_attention_queries(%queries, %queryBits, %lowerQuery, %upperQuery, %size, %quad, %scale, %attentionHalves) : (!llvm.ptr, i64, i64, i64, i32, i32, f32, i1) -> vector<32xi32>
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %hidden, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %zero, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %hidden, %state1[%c2 : i32] : vector<4xf32>
    %started = llvm.insertelement %zero, %state2[%c3 : i32] : vector<4xf32>
    cf.br ^chunk(%chunkFirst, %zeros, %started, %firstKeys, %firstValues : i32, vector<64xf32>, vector<4xf32>, vector<16xi32>, vector<16xi32>)
  ^chunk(%chunk: i32, %outputs: vector<64xf32>, %state: vector<4xf32>, %keyPairs: vector<16xi32>, %valuePairs: vector<16xi32>):
    %moreChunks = arith.cmpi slt, %chunk, %chunkEnd : i32
    cf.cond_br %moreChunks, ^chunkBody, ^merge
  ^chunkBody:
    %first = arith.shli %chunk, %c3 : i32
    %chunkFollowing = arith.addi %chunk, %c1 : i32
    %nextFirst = arith.shli %chunkFollowing, %c3 : i32
    %nextKeys, %nextValues = func.call @strict_device_attention_operands(%keys, %keyBits, %values, %valueBits, %nextFirst, %positionLast, %stride, %base, %size, %row, %quad, %viewOffset, %viewStride) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    %newOutputs, %newState = func.call @strict_device_attention_chunk(%queryPairs, %outputs, %state, %keyPairs, %valuePairs, %first, %lowerLast, %upperLast, %size, %quad) : (vector<32xi32>, vector<64xf32>, vector<4xf32>, vector<16xi32>, vector<16xi32>, i32, i32, i32, i32, i32) -> (vector<64xf32>, vector<4xf32>)
    cf.br ^chunk(%chunkFollowing, %newOutputs, %newState, %nextKeys, %nextValues : i32, vector<64xf32>, vector<4xf32>, vector<16xi32>, vector<16xi32>)
  ^merge:
    %merged, %mergedState = func.call @strict_device_attention_warps(%outputs, %state, %warp, %lane) : (vector<64xf32>, vector<4xf32>, i32, i32) -> (vector<64xf32>, vector<4xf32>)
    %isFirstWarp = arith.cmpi eq, %warp, %c0 : i32
    %alone = arith.cmpi eq, %splits, %c1 : i32
    cf.cond_br %alone, ^single, ^shared
  ^single:
    cf.cond_br %isFirstWarp, ^write(%merged, %mergedState : vector<64xf32>, vector<4xf32>), ^done
  ^shared:
    %slotParts = arith.muli %slot, %splits : i32
    %partialAt = arith.addi %slotParts, %split : i32
    cf.cond_br %isFirstWarp, ^store, ^stored
  ^store:
    %partialStart = func.call @strict_device_attention_partial(%partialAt) : (i32) -> !llvm.ptr
    func.call @strict_device_attention_put(%partialStart, %lane, %merged, %mergedState) : (!llvm.ptr, i32, vector<64xf32>, vector<4xf32>) -> ()
    func.call @strict_device_fence() : () -> ()
    cf.br ^stored
  ^stored:
    func.call @strict_device_barrier() : () -> ()
    %lastSlot = llvm.mlir.addressof @strict_device_attention_last : !llvm.ptr<3>
    %isFirstThread = arith.cmpi eq, %thread, %c0 : i32
    cf.cond_br %isFirstThread, ^count, ^counted
  ^count:
    %counters = llvm.mlir.addressof @strict_device_attention_counters : !llvm.ptr<1>
    %slot64 = arith.extui %slot : i32 to i64
    %counter = llvm.getelementptr %counters[0, %slot64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<512 x i32>
    %done = llvm.atomicrmw add %counter, %c1 acq_rel : !llvm.ptr<1>, i32
    %lastSplit = arith.subi %splits, %c1 : i32
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
    %collecting = arith.andi %isLastBlock, %isFirstWarp : i1
    cf.cond_br %collecting, ^collect, ^done
  ^collect:
    func.call @strict_device_fence() : () -> ()
    cf.br ^gather(%c0, %zeros, %started : i32, vector<64xf32>, vector<4xf32>)
  ^gather(%part: i32, %gathered: vector<64xf32>, %gatheredState: vector<4xf32>):
    %moreParts = arith.cmpi slt, %part, %splits : i32
    cf.cond_br %moreParts, ^gatherBody, ^write(%gathered, %gatheredState : vector<64xf32>, vector<4xf32>)
  ^gatherBody:
    %partAt = arith.addi %slotParts, %part : i32
    %partStart = func.call @strict_device_attention_partial(%partAt) : (i32) -> !llvm.ptr
    %fresh = arith.constant true
    %partOutputs, %partState = func.call @strict_device_attention_get(%partStart, %lane, %fresh) : (!llvm.ptr, i32, i1) -> (vector<64xf32>, vector<4xf32>)
    %combined, %combinedState = func.call @strict_device_attention_merge(%gathered, %gatheredState, %partOutputs, %partState) : (vector<64xf32>, vector<4xf32>, vector<64xf32>, vector<4xf32>) -> (vector<64xf32>, vector<4xf32>)
    %partFollowing = arith.addi %part, %c1 : i32
    cf.br ^gather(%partFollowing, %combined, %combinedState : i32, vector<64xf32>, vector<4xf32>)
  ^write(%final: vector<64xf32>, %finalState: vector<4xf32>):
    func.call @strict_device_attention_finish(%result, %wide, %final, %finalState, %lowerToken, %upperToken, %tokenLast, %lowerQuery, %upperQuery, %size, %quad) : (!llvm.ptr, i1, vector<64xf32>, vector<4xf32>, i32, i32, i32, i64, i64, i32, i32) -> ()
    cf.br ^done
  ^done:
    return
  }
  // the outputs of the two rows of a lane divided by their sums (over the 4 lanes of a row) into the result, rows
  // after the last token are left out
  func.func private @strict_device_attention_finish(%result: !llvm.ptr, %wide: i1, %outputs: vector<64xf32>, %state: vector<4xf32>, %lowerToken: i32, %upperToken: i32, %tokenLast: i32, %lowerQuery: i64, %upperQuery: i64, %size: i32, %quad: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %lowerPart = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperPart = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %lowerSum = func.call @strict_device_attention_quad(%lowerPart) : (f32) -> f32
    %upperSum = func.call @strict_device_attention_quad(%upperPart) : (f32) -> f32
    %unit = arith.constant 1.0 : f32
    %lowerInverse = arith.divf %unit, %lowerSum : f32
    %upperInverse = arith.divf %unit, %upperSum : f32
    %lowerWriting = arith.cmpi sle, %lowerToken, %tokenLast : i32
    %upperWriting = arith.cmpi sle, %upperToken, %tokenLast : i32
    %steps = arith.shrui %size, %c3 : i32
    %quadNumbers = arith.shli %quad, %c1 : i32
    cf.br ^column(%c0 : i32)
  ^column(%column: i32):
    %moreColumns = arith.cmpi slt, %column, %c16 : i32
    cf.cond_br %moreColumns, ^columnCheck, ^done
  ^columnCheck:
    %usedColumn = arith.cmpi slt, %column, %steps : i32
    cf.cond_br %usedColumn, ^columnBody, ^columnNext
  ^columnBody:
    %columnStart = arith.shli %column, %c3 : i32
    %number = arith.addi %columnStart, %quadNumbers : i32
    %number64 = arith.extui %number : i32 to i64
    %slot = arith.shli %column, %c2 : i32
    %lowerAt = arith.addi %lowerQuery, %number64 : i64
    %upperAt = arith.addi %upperQuery, %number64 : i64
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot, %lowerAt, %lowerInverse, %lowerWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    %upperSlot = arith.addi %slot, %c2 : i32
    func.call @strict_device_attention_written(%result, %wide, %outputs, %upperSlot, %upperAt, %upperInverse, %upperWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    cf.br ^columnNext
  ^columnNext:
    %columnFollowing = arith.addi %column, %c1 : i32
    llvm.br ^column(%columnFollowing : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // strict_device_bytes_attention: 8 bit keys and values (size 128, a head per group, no causal mask) on the 8 bit
  // tensor cores like Sol-Attn, a block of 128 threads per 64 tokens of a head; two tiles of 32 positions at a time go
  // through four buffers by cp.async (the keys, the values 4736 bytes later, both laid out like
  // strict_device_attention_stored_eights), a routed block first plans its tiles (strict_device_bytes_planned)
  llvm.mlir.global internal @strict_device_bytes_tiles() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<37888 x i8>
  llvm.mlir.global internal @strict_device_bytes_plan() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<544 x i16>
  llvm.mlir.global internal @strict_device_bytes_pools() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x i16>
  llvm.mlir.global internal @strict_device_bytes_pool_scores() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1024 x f16>
  llvm.mlir.global internal @strict_device_bytes_counts() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<4 x i32>
  // the keys of tile buffer (0 to 3)
  func.func private @strict_device_bytes_buffer(%buffer: i32) -> !llvm.ptr<3> attributes {llvm.passthrough = ["alwaysinline"]} {
    %tiles = llvm.mlir.addressof @strict_device_bytes_tiles : !llvm.ptr<3>
    %bufferBytes = llvm.mlir.constant(9472 : i32) : i32
    %start = arith.muli %buffer, %bufferBytes : i32
    %start64 = arith.extui %start : i32 to i64
    %tile = llvm.getelementptr %tiles[%start64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    return %tile : !llvm.ptr<3>
  }
  // the 32 positions from first of 8 bit numbers (a scale per 32 numbers after the count of positions times stride)
  // into a tile without waiting: two 16 byte copies per thread, zeros after the last position, the first 32 threads
  // copy the scales
  func.func private @strict_device_bytes_copied(%tile: !llvm.ptr<3>, %data: !llvm.ptr, %first: i32, %positions: i32, %stride: i32, %base: i32, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c4608 = llvm.mlir.constant(4608 : i32) : i32
    %positionLast = arith.subi %positions, %c1 : i32
    cf.br ^part(%c0 : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c2 : i32
    cf.cond_br %more, ^body, ^scales
  ^body:
    %offset = arith.muli %part, %c128 : i32
    %piece = arith.addi %thread, %offset : i32
    %row = arith.shrui %piece, %c3 : i32
    %chunk = arith.andi %piece, %c7 : i32
    %chunkBytes = arith.shli %chunk, %c4 : i32
    %position = arith.addi %first, %row : i32
    %inside = arith.cmpi sle, %position, %positionLast : i32
    %read = arith.minsi %position, %positionLast : i32
    %rowStart = arith.muli %read, %stride : i32
    %rowBase = arith.addi %rowStart, %base : i32
    %at = arith.addi %rowBase, %chunkBytes : i32
    %at64 = arith.extui %at : i32 to i64
    %source = llvm.getelementptr %data[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowTarget = arith.muli %row, %c144 : i32
    %targetAt = arith.addi %rowTarget, %chunkBytes : i32
    %targetAt64 = arith.extui %targetAt : i32 to i64
    %target = llvm.getelementptr %tile[%targetAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    func.call @strict_device_copy_filled(%target, %source, %inside) : (!llvm.ptr<3>, !llvm.ptr, i1) -> ()
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^scales:
    %scaleThread = arith.cmpi slt, %thread, %c32 : i32
    cf.cond_br %scaleThread, ^scale, ^done
  ^scale:
    %count = arith.muli %positions, %stride : i32
    %count64 = arith.extui %count : i32 to i64
    %scales = llvm.getelementptr %data[%count64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scalePosition = arith.addi %first, %thread : i32
    %scaleRead = arith.minsi %scalePosition, %positionLast : i32
    %scaleStart = arith.muli %scaleRead, %stride : i32
    %scaleRow = arith.addi %scaleStart, %base : i32
    %scaleIndex = arith.shrui %scaleRow, %c5 : i32
    %scaleIndex64 = arith.extui %scaleIndex : i32 to i64
    %scaleSource = llvm.getelementptr %scales[%scaleIndex64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %scaleBytes = arith.shli %thread, %c2 : i32
    %scaleAt = arith.addi %c4608, %scaleBytes : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleTarget = llvm.getelementptr %tile[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    func.call @strict_device_copy_word(%scaleTarget, %scaleSource) : (!llvm.ptr<3>, !llvm.ptr) -> ()
    cf.br ^done
  ^done:
    return
  }
  // the first position of entry index (a tile, without a plan the index itself), whether it is a pooled chunk and which
  func.func private @strict_device_bytes_entry(%routed: i1, %index: i32) -> (i32, i1, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c32768 = llvm.mlir.constant(32768 : i32) : i32
    %false = arith.constant false
    %tileFirst = arith.shli %index, %c5 : i32
    cf.cond_br %routed, ^plan, ^done(%tileFirst, %false, %c0 : i32, i1, i32)
  ^plan:
    %plan = llvm.mlir.addressof @strict_device_bytes_plan : !llvm.ptr<3>
    %index64 = arith.extui %index : i32 to i64
    %slot = llvm.getelementptr %plan[%index64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    %entry16 = llvm.load %slot : !llvm.ptr<3> -> i16
    %entry = arith.extui %entry16 : i16 to i32
    %isPooled = arith.cmpi sge, %entry, %c32768 : i32
    %chunk = arith.subi %entry, %c32768 : i32
    %entryFirst = arith.shli %entry, %c5 : i32
    %first = arith.select %isPooled, %c0, %entryFirst : i32
    %pooledChunk = arith.select %isPooled, %chunk, %c0 : i32
    cf.br ^done(%first, %isPooled, %pooledChunk : i32, i1, i32)
  ^done(%chosenFirst: i32, %chosenPooled: i1, %chosenChunk: i32):
    return %chosenFirst, %chosenPooled, %chosenChunk : i32, i1, i32
  }
  // the count of pooled key blocks of the plan
  func.func private @strict_device_bytes_pooled_count() -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %counts = llvm.mlir.addressof @strict_device_bytes_counts : !llvm.ptr<3>
    %pooledCountSlot = llvm.getelementptr %counts[1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
    %pooledCount = llvm.load %pooledCountSlot : !llvm.ptr<3> -> i32
    return %pooledCount : i32
  }
  // the mean values of pooled chunk chunk (32 bit floats of strict_device_attention_pools from valueRows) as 8 bit
  // numbers into a value tile, a scale per position: 4 threads per position with 32 numbers each, zeros after the
  // pooled count
  func.func private @strict_device_bytes_pooled(%tile: !llvm.ptr<3>, %pooled: !llvm.ptr, %chunk: i32, %valueRows: i32, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %c4608 = llvm.mlir.constant(4608 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %noNumbers = arith.constant dense<0.0> : vector<32xf32>
    %noWords = arith.constant dense<0> : vector<8xi32>
    %pooledCount = func.call @strict_device_bytes_pooled_count() : () -> i32
    %row = arith.shrui %thread, %c2 : i32
    %part = arith.andi %thread, %c3 : i32
    %partNumbers = arith.shli %part, %c5 : i32
    %chunkStart = arith.shli %chunk, %c5 : i32
    %entry = arith.addi %chunkStart, %row : i32
    %inChunk = arith.cmpi slt, %entry, %pooledCount : i32
    cf.cond_br %inChunk, ^start, ^quantize(%noNumbers, %zero : vector<32xf32>, f32)
  ^start:
    %pools = llvm.mlir.addressof @strict_device_bytes_pools : !llvm.ptr<3>
    %entry64 = arith.extui %entry : i32 to i64
    %poolSlot = llvm.getelementptr %pools[%entry64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    %keyBlock16 = llvm.load %poolSlot : !llvm.ptr<3> -> i16
    %keyBlock = arith.extui %keyBlock16 : i16 to i32
    %blockRow = arith.shli %keyBlock, %c7 : i32
    %rowStart = arith.addi %valueRows, %blockRow : i32
    %partStart = arith.addi %rowStart, %partNumbers : i32
    cf.br ^load(%c0, %noNumbers, %zero : i32, vector<32xf32>, f32)
  ^load(%index: i32, %numbers: vector<32xf32>, %largest: f32):
    %more = arith.cmpi slt, %index, %c8 : i32
    cf.cond_br %more, ^read, ^quantize(%numbers, %largest : vector<32xf32>, f32)
  ^read:
    %offset = arith.shli %index, %c2 : i32
    %at = arith.addi %partStart, %offset : i32
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %pooled[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %four = llvm.load %slot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %offset1 = arith.addi %offset, %c1 : i32
    %offset2 = arith.addi %offset, %c2 : i32
    %offset3 = arith.addi %offset, %c3 : i32
    %read0 = llvm.extractelement %four[%c0 : i32] : vector<4xf32>
    %read1 = llvm.extractelement %four[%c1 : i32] : vector<4xf32>
    %read2 = llvm.extractelement %four[%c2 : i32] : vector<4xf32>
    %read3 = llvm.extractelement %four[%c3 : i32] : vector<4xf32>
    %with0 = llvm.insertelement %read0, %numbers[%offset : i32] : vector<32xf32>
    %with1 = llvm.insertelement %read1, %with0[%offset1 : i32] : vector<32xf32>
    %with2 = llvm.insertelement %read2, %with1[%offset2 : i32] : vector<32xf32>
    %with3 = llvm.insertelement %read3, %with2[%offset3 : i32] : vector<32xf32>
    %magnitudes = llvm.call_intrinsic "llvm.fabs.v4f32"(%four) : (vector<4xf32>) -> vector<4xf32>
    %fourLargest = llvm.call_intrinsic "llvm.vector.reduce.fmax.v4f32"(%magnitudes) : (vector<4xf32>) -> f32
    %larger = arith.maxnumf %largest, %fourLargest : f32
    %following = arith.addi %index, %c1 : i32
    llvm.br ^load(%following, %with3, %larger : i32, vector<32xf32>, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^quantize(%loaded: vector<32xf32>, %partLargest: f32):
    %rowLargest = func.call @strict_device_bytes_quad_largest(%partLargest) : (f32) -> f32
    %factor = func.call @strict_device_attention_inverse(%rowLargest) : (f32) -> f32
    cf.br ^pack(%c0, %noWords : i32, vector<8xi32>)
  ^pack(%word: i32, %words: vector<8xi32>):
    %morePacks = arith.cmpi slt, %word, %c8 : i32
    cf.cond_br %morePacks, ^packBody, ^store
  ^packBody:
    %packSlot0 = arith.shli %word, %c2 : i32
    %packSlot1 = arith.addi %packSlot0, %c1 : i32
    %packSlot2 = arith.addi %packSlot0, %c2 : i32
    %packSlot3 = arith.addi %packSlot0, %c3 : i32
    %number0 = llvm.extractelement %loaded[%packSlot0 : i32] : vector<32xf32>
    %number1 = llvm.extractelement %loaded[%packSlot1 : i32] : vector<32xf32>
    %number2 = llvm.extractelement %loaded[%packSlot2 : i32] : vector<32xf32>
    %number3 = llvm.extractelement %loaded[%packSlot3 : i32] : vector<32xf32>
    %packed = func.call @strict_device_attention_packed(%number0, %number1, %number2, %number3, %factor) : (f32, f32, f32, f32, f32) -> i32
    %withWord = llvm.insertelement %packed, %words[%word : i32] : vector<8xi32>
    %packFollowing = arith.addi %word, %c1 : i32
    llvm.br ^pack(%packFollowing, %withWord : i32, vector<8xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^store:
    %rowTarget = arith.muli %row, %c144 : i32
    %partBytes = arith.shli %part, %c5 : i32
    %targetAt = arith.addi %rowTarget, %partBytes : i32
    %targetAt64 = arith.extui %targetAt : i32 to i64
    %target = llvm.getelementptr %tile[%targetAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %lowWords = llvm.shufflevector %words, %words [0, 1, 2, 3] : vector<8xi32>
    %highWords = llvm.shufflevector %words, %words [4, 5, 6, 7] : vector<8xi32>
    llvm.store %lowWords, %target {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %highTarget = llvm.getelementptr %target[16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    llvm.store %highWords, %highTarget {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %isFirst = arith.cmpi eq, %part, %c0 : i32
    cf.cond_br %isFirst, ^scale, ^done
  ^scale:
    %rowScale = arith.divf %rowLargest, %most : f32
    %rowBytes = arith.shli %row, %c2 : i32
    %scaleAt = arith.addi %c4608, %rowBytes : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleTarget = llvm.getelementptr %tile[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    llvm.store %rowScale, %scaleTarget {alignment = 4 : i64} : f32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // starts loading entry index of count (a tile, or the values of a pooled chunk) into buffer, nothing past the last;
  // dense blocks of 256 threads copy the keys with the first 128 and the transposed values with the others
  func.func private @strict_device_bytes_issue(%buffer: i32, %index: i32, %count: i32, %routed: i1, %keys: !llvm.ptr, %values: !llvm.ptr, %pooled: !llvm.ptr, %positions: i32, %stride: i32, %base: i32, %valueRows: i32, %thread: i32, %dense: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %inPlan = arith.cmpi slt, %index, %count : i32
    cf.cond_br %inPlan, ^entry, ^commit
  ^entry:
    %first, %isPooled, %chunk = func.call @strict_device_bytes_entry(%routed, %index) : (i1, i32) -> (i32, i1, i32)
    %keyTile = func.call @strict_device_bytes_buffer(%buffer) : (i32) -> !llvm.ptr<3>
    %valueTile = llvm.getelementptr %keyTile[4736] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    cf.cond_br %isPooled, ^pooledValues, ^tile
  ^pooledValues:
    func.call @strict_device_bytes_pooled(%valueTile, %pooled, %chunk, %valueRows, %thread) : (!llvm.ptr<3>, !llvm.ptr, i32, i32, i32) -> ()
    cf.br ^commit
  ^tile:
    cf.cond_br %dense, ^half, ^both
  ^half:
    %c127 = llvm.mlir.constant(127 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %upper = arith.cmpi sge, %thread, %c128 : i32
    %halfThread = arith.andi %thread, %c127 : i32
    cf.cond_br %upper, ^denseValues, ^denseKeys
  ^denseKeys:
    func.call @strict_device_bytes_copied(%keyTile, %keys, %first, %positions, %stride, %base, %halfThread) : (!llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    cf.br ^commit
  ^denseValues:
    func.call @strict_device_dense_copied(%valueTile, %values, %first, %positions, %base, %halfThread) : (!llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32) -> ()
    cf.br ^commit
  ^both:
    func.call @strict_device_bytes_copied(%keyTile, %keys, %first, %positions, %stride, %base, %thread) : (!llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    func.call @strict_device_bytes_copied(%valueTile, %values, %first, %positions, %stride, %base, %thread) : (!llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    cf.br ^commit
  ^commit:
    return
  }
  // starts loading the two entries of step (into buffers 0 and 1 or 2 and 3) and commits their copies as a group
  func.func private @strict_device_bytes_issued(%step: i32, %count: i32, %routed: i1, %keys: !llvm.ptr, %values: !llvm.ptr, %pooled: !llvm.ptr, %positions: i32, %stride: i32, %base: i32, %valueRows: i32, %thread: i32, %dense: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %pair = arith.andi %step, %c1 : i32
    %firstBuffer = arith.shli %pair, %c1 : i32
    %secondBuffer = arith.addi %firstBuffer, %c1 : i32
    %entry = arith.shli %step, %c1 : i32
    %otherEntry = arith.addi %entry, %c1 : i32
    func.call @strict_device_bytes_issue(%firstBuffer, %entry, %count, %routed, %keys, %values, %pooled, %positions, %stride, %base, %valueRows, %thread, %dense) : (i32, i32, i32, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i1) -> ()
    func.call @strict_device_bytes_issue(%secondBuffer, %otherEntry, %count, %routed, %keys, %values, %pooled, %positions, %stride, %base, %valueRows, %thread, %dense) : (i32, i32, i32, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i1) -> ()
    func.call @strict_device_copies_commit() : () -> ()
    return
  }
  // the last visible position of pooled chunk chunk (below 0 for a chunk after the pooled count)
  func.func private @strict_device_bytes_chunk_last(%chunk: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %pooledCount = func.call @strict_device_bytes_pooled_count() : () -> i32
    %chunkStart = arith.shli %chunk, %c5 : i32
    %left = arith.subi %pooledCount, %chunkStart : i32
    %leftLast = arith.subi %left, %c1 : i32
    %last = arith.minsi %leftLast, %c31 : i32
    return %last : i32
  }
  // strict_device_attention_softmax of two tiles at once (from first and from second, the last visible positions of
  // the lower and upper rows in each), last the largest share of each row
  func.func private @strict_device_bytes_softmax(%scores: vector<16xf32>, %others: vector<16xf32>, %state: vector<4xf32>, %first: i32, %second: i32, %lowerLast: i32, %upperLast: i32, %lowerOtherLast: i32, %upperOtherLast: i32, %quad: i32) -> (vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %lowerFirst = func.call @strict_device_attention_largest(%scores, %c0, %first, %quad, %lowerLast) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %upperFirst = func.call @strict_device_attention_largest(%scores, %c2, %first, %quad, %upperLast) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %lowerSecond = func.call @strict_device_attention_largest(%others, %c0, %second, %quad, %lowerOtherLast) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %upperSecond = func.call @strict_device_attention_largest(%others, %c2, %second, %quad, %upperOtherLast) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %lowerLargest = arith.maximumf %lowerFirst, %lowerSecond : f32
    %upperLargest = arith.maximumf %upperFirst, %upperSecond : f32
    %newLower, %newUpper, %lowerScale, %upperScale = func.call @strict_device_attention_raised(%state, %lowerLargest, %upperLargest) : (vector<4xf32>, f32, f32) -> (f32, f32, f32, f32)
    %shared, %lowerTotal, %upperTotal = func.call @strict_device_attention_tile_shares(%scores, %first, %lowerLast, %upperLast, %newLower, %newUpper, %quad) : (vector<16xf32>, i32, i32, i32, f32, f32, i32) -> (vector<16xf32>, f32, f32)
    %otherShared, %lowerOther, %upperOther = func.call @strict_device_attention_tile_shares(%others, %second, %lowerOtherLast, %upperOtherLast, %newLower, %newUpper, %quad) : (vector<16xf32>, i32, i32, i32, f32, f32, i32) -> (vector<16xf32>, f32, f32)
    %lowerBoth = arith.addf %lowerTotal, %lowerOther : f32
    %upperBoth = arith.addf %upperTotal, %upperOther : f32
    %newState = func.call @strict_device_attention_summed(%state, %newLower, %newUpper, %lowerScale, %upperScale, %lowerBoth, %upperBoth) : (vector<4xf32>, f32, f32, f32, f32, f32, f32) -> vector<4xf32>
    %lowerBelow = arith.subf %lowerLargest, %newLower : f32
    %upperBelow = arith.subf %upperLargest, %newUpper : f32
    %lowerShare = func.call @strict_device_binary_power(%lowerBelow) : (f32) -> f32
    %upperShare = func.call @strict_device_binary_power(%upperBelow) : (f32) -> f32
    return %shared, %otherShared, %newState, %lowerScale, %upperScale, %lowerShare, %upperShare : vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32
  }
  // the largest of part over the 4 lanes of a row
  func.func private @strict_device_bytes_quad_largest(%part: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %fromOne = func.call @strict_device_shuffles(%part, %c1) : (f32, i32) -> f32
    %pair = arith.maxnumf %part, %fromOne : f32
    %fromTwo = func.call @strict_device_shuffles(%pair, %c2) : (f32, i32) -> f32
    %largest = arith.maxnumf %pair, %fromTwo : f32
    return %largest : f32
  }
  // the scores of pooled chunk chunk like strict_device_attention_scores: the pooled scores of the block plus the shift
  // of each row
  func.func private @strict_device_bytes_scores_pooled(%chunk: i32, %lane: i32, %lowerShift: f32, %upperShift: f32) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %noScores = arith.constant dense<0.0> : vector<16xf32>
    %poolScores = llvm.mlir.addressof @strict_device_bytes_pool_scores : !llvm.ptr<3>
    %quad = arith.andi %lane, %c3 : i32
    %quadPositions = arith.shli %quad, %c1 : i32
    %chunkStart = arith.shli %chunk, %c5 : i32
    cf.br ^part(%c0, %noScores : i32, vector<16xf32>)
  ^part(%part: i32, %scores: vector<16xf32>):
    %more = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partPositions = arith.shli %part, %c3 : i32
    %position = arith.addi %partPositions, %quadPositions : i32
    %at = arith.addi %chunkStart, %position : i32
    %at64 = arith.extui %at : i32 to i64
    %scoreSlot = llvm.getelementptr %poolScores[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %halves = llvm.load %scoreSlot {alignment = 4 : i64} : !llvm.ptr<3> -> vector<2xf16>
    %pair = arith.extf %halves : vector<2xf16> to vector<2xf32>
    %score = llvm.extractelement %pair[%c0 : i32] : vector<2xf32>
    %scoreNext = llvm.extractelement %pair[%c1 : i32] : vector<2xf32>
    %lower = arith.addf %score, %lowerShift : f32
    %lowerNext = arith.addf %scoreNext, %lowerShift : f32
    %upper = arith.addf %score, %upperShift : f32
    %upperNext = arith.addf %scoreNext, %upperShift : f32
    %slot = arith.shli %part, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %with0 = llvm.insertelement %lower, %scores[%slot : i32] : vector<16xf32>
    %with1 = llvm.insertelement %lowerNext, %with0[%slot1 : i32] : vector<16xf32>
    %with2 = llvm.insertelement %upper, %with1[%slot2 : i32] : vector<16xf32>
    %with3 = llvm.insertelement %upperNext, %with2[%slot3 : i32] : vector<16xf32>
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following, %with3 : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %scores : vector<16xf32>
  }
  // the two weights at slot and the next times the scales of their positions, and the larger of largest and them
  func.func private @strict_device_bytes_scaled(%weights: vector<16xf32>, %slot: i32, %scales: vector<2xf32>, %largest: f32) -> (vector<16xf32>, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %weight = llvm.extractelement %weights[%slot : i32] : vector<16xf32>
    %weightNext = llvm.extractelement %weights[%slotNext : i32] : vector<16xf32>
    %scale = llvm.extractelement %scales[%c0 : i32] : vector<2xf32>
    %scaleNext = llvm.extractelement %scales[%c1 : i32] : vector<2xf32>
    %scaled = arith.mulf %weight, %scale : f32
    %scaledNext = arith.mulf %weightNext, %scaleNext : f32
    %larger = arith.maxnumf %largest, %scaled : f32
    %largerNext = arith.maxnumf %larger, %scaledNext : f32
    %withScaled = llvm.insertelement %scaled, %weights[%slot : i32] : vector<16xf32>
    %withBoth = llvm.insertelement %scaledNext, %withScaled[%slotNext : i32] : vector<16xf32>
    return %withBoth, %largerNext : vector<16xf32>, f32
  }
  // the 8 bit operand of the weights at slot, the next, slot + 4 and the next (positions 2 * (lane % 4) and the next of
  // two parts of 8) times factor
  func.func private @strict_device_bytes_operand(%weights: vector<16xf32>, %slot: i32, %factor: f32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot4 = arith.addi %slot, %c4 : i32
    %slot5 = arith.addi %slot, %c5 : i32
    %weight0 = llvm.extractelement %weights[%slot : i32] : vector<16xf32>
    %weight1 = llvm.extractelement %weights[%slot1 : i32] : vector<16xf32>
    %weight4 = llvm.extractelement %weights[%slot4 : i32] : vector<16xf32>
    %weight5 = llvm.extractelement %weights[%slot5 : i32] : vector<16xf32>
    %word = func.call @strict_device_attention_packed(%weight0, %weight1, %weight4, %weight5, %factor) : (f32, f32, f32, f32, f32) -> i32
    return %word : i32
  }
  // bytes 0 and 2 of first and of second (odd: bytes 1 and 3): of the two numbers of two positions each matrix of an
  // ldmatrix.trans gives a lane, the 4 positions of one number
  func.func private @strict_device_bytes_permuted(%first: i32, %second: i32, %odd: i1) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %firstBytes = llvm.bitcast %first : i32 to vector<4xi8>
    %secondBytes = llvm.bitcast %second : i32 to vector<4xi8>
    %evenBytes = llvm.shufflevector %firstBytes, %secondBytes [0, 2, 4, 6] : vector<4xi8>
    %oddBytes = llvm.shufflevector %firstBytes, %secondBytes [1, 3, 5, 7] : vector<4xi8>
    %chosen = arith.select %odd, %oddBytes, %evenBytes : vector<4xi8>
    %word = llvm.bitcast %chosen : vector<4xi8> to i32
    return %word : i32
  }
  // the sum at slot shrunk by scale plus an integer product times the unit of its row, the product biased like
  // strict_device_attention_unbiased and offset -1.5 * 2^23 times unit: two fmas
  func.func private @strict_device_bytes_added(%sums: vector<64xf32>, %slot: i32, %product: i32, %unit: f32, %scale: f32, %offset: f32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %sum = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    %shrunk = llvm.intr.fma(%sum, %scale, %offset) : (f32, f32, f32) -> f32
    %biased = arith.bitcast %product : i32 to f32
    %added = llvm.intr.fma(%biased, %unit, %shrunk) : (f32, f32, f32) -> f32
    %withAdded = llvm.insertelement %added, %sums[%slot : i32] : vector<64xf32>
    return %withAdded : vector<64xf32>
  }
  // the 8 sums at slot (4 numbers of the lower row, then of the upper row) with the products of their even and odd
  // numbers
  func.func private @strict_device_bytes_folded(%sums: vector<64xf32>, %slot: i32, %even: !llvm.struct<(i32, i32, i32, i32)>, %odd: !llvm.struct<(i32, i32, i32, i32)>, %lowerUnit: f32, %upperUnit: f32, %lowerScale: f32, %upperScale: f32, %lowerOffset: f32, %upperOffset: f32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %even0 = llvm.extractvalue %even[0] : !llvm.struct<(i32, i32, i32, i32)>
    %even1 = llvm.extractvalue %even[1] : !llvm.struct<(i32, i32, i32, i32)>
    %even2 = llvm.extractvalue %even[2] : !llvm.struct<(i32, i32, i32, i32)>
    %even3 = llvm.extractvalue %even[3] : !llvm.struct<(i32, i32, i32, i32)>
    %odd0 = llvm.extractvalue %odd[0] : !llvm.struct<(i32, i32, i32, i32)>
    %odd1 = llvm.extractvalue %odd[1] : !llvm.struct<(i32, i32, i32, i32)>
    %odd2 = llvm.extractvalue %odd[2] : !llvm.struct<(i32, i32, i32, i32)>
    %odd3 = llvm.extractvalue %odd[3] : !llvm.struct<(i32, i32, i32, i32)>
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %slot4 = arith.addi %slot, %c4 : i32
    %slot5 = arith.addi %slot, %c5 : i32
    %slot6 = arith.addi %slot, %c6 : i32
    %slot7 = arith.addi %slot, %c7 : i32
    %with0 = func.call @strict_device_bytes_added(%sums, %slot, %even0, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with1 = func.call @strict_device_bytes_added(%with0, %slot1, %odd0, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with2 = func.call @strict_device_bytes_added(%with1, %slot2, %even1, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with3 = func.call @strict_device_bytes_added(%with2, %slot3, %odd1, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with4 = func.call @strict_device_bytes_added(%with3, %slot4, %even2, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with5 = func.call @strict_device_bytes_added(%with4, %slot5, %odd2, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with6 = func.call @strict_device_bytes_added(%with5, %slot6, %even3, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with7 = func.call @strict_device_bytes_added(%with6, %slot7, %odd3, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    return %with7 : vector<64xf32>
  }
  // the weights of a tile: its shares times the scales of their positions in the value tile, and the largest of each
  // row of the lane
  func.func private @strict_device_bytes_weights(%shares: vector<16xf32>, %valueTile: !llvm.ptr<3>, %lane: i32) -> (vector<16xf32>, f32, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c4608 = llvm.mlir.constant(4608 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %quad = arith.andi %lane, %c3 : i32
    %quadPositions = arith.shli %quad, %c1 : i32
    cf.br ^scaled(%c0, %shares, %zero, %zero : i32, vector<16xf32>, f32, f32)
  ^scaled(%part: i32, %weights: vector<16xf32>, %lowerLargest: f32, %upperLargest: f32):
    %moreParts = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %moreParts, ^scaleBody, ^done
  ^scaleBody:
    %partPositions = arith.shli %part, %c3 : i32
    %position = arith.addi %partPositions, %quadPositions : i32
    %positionBytes = arith.shli %position, %c2 : i32
    %scaleAt = arith.addi %c4608, %positionBytes : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleSlot = llvm.getelementptr %valueTile[%scaleAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %scales = llvm.load %scaleSlot {alignment = 8 : i64} : !llvm.ptr<3> -> vector<2xf32>
    %partSlot = arith.shli %part, %c2 : i32
    %upperSlot = arith.addi %partSlot, %c2 : i32
    %lowerScaled, %lowerLarger = func.call @strict_device_bytes_scaled(%weights, %partSlot, %scales, %lowerLargest) : (vector<16xf32>, i32, vector<2xf32>, f32) -> (vector<16xf32>, f32)
    %bothScaled, %upperLarger = func.call @strict_device_bytes_scaled(%lowerScaled, %upperSlot, %scales, %upperLargest) : (vector<16xf32>, i32, vector<2xf32>, f32) -> (vector<16xf32>, f32)
    %partFollowing = arith.addi %part, %c1 : i32
    llvm.br ^scaled(%partFollowing, %bothScaled, %lowerLarger, %upperLarger : i32, vector<16xf32>, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %weights, %lowerLargest, %upperLargest : vector<16xf32>, f32, f32
  }
  // the 4 integer sums of an m16n8k32 of the operands of a tile and the value operands of a number (low and high,
  // strict_device_bytes_permuted) added to sums
  func.func private @strict_device_bytes_product(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %low: i32, %high: i32, %sums: !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.passthrough = ["alwaysinline"]} {
    %sum0 = llvm.extractvalue %sums[0] : !llvm.struct<(i32, i32, i32, i32)>
    %sum1 = llvm.extractvalue %sums[1] : !llvm.struct<(i32, i32, i32, i32)>
    %sum2 = llvm.extractvalue %sums[2] : !llvm.struct<(i32, i32, i32, i32)>
    %sum3 = llvm.extractvalue %sums[3] : !llvm.struct<(i32, i32, i32, i32)>
    %product = func.call @strict_device_mma16(%a0, %a1, %a2, %a3, %low, %high, %sum0, %sum1, %sum2, %sum3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    return %product : !llvm.struct<(i32, i32, i32, i32)>
  }
  // the outputs of the two rows of a lane (8 per 16 numbers: 4 of row lane / 4 from 4 * (lane % 4), then those of the
  // row 8 below) shrunk by their scales plus the shares of two tiles times their 8 bit values: the weights as 8 bits
  // per row, for 8 even or 8 odd numbers of 16 an m16n8k32 per tile adding up in 32 bits from the bits of 1.5 * 2^23
  func.func private @strict_device_bytes_weighted(%shares: vector<16xf32>, %others: vector<16xf32>, %outputs: vector<64xf32>, %lowerScale: f32, %upperScale: f32, %valueTile: !llvm.ptr<3>, %otherTile: !llvm.ptr<3>, %lane: i32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c10 = llvm.mlir.constant(10 : i32) : i32
    %c144 = llvm.mlir.constant(144 : i32) : i32
    %bias = llvm.mlir.constant(1262485504 : i32) : i32
    %most = arith.constant 127.0 : f32
    %unbias = arith.constant -12582912.0 : f32
    %false = arith.constant false
    %true = arith.constant true
    %weights, %lowerFirst, %upperFirst = func.call @strict_device_bytes_weights(%shares, %valueTile, %lane) : (vector<16xf32>, !llvm.ptr<3>, i32) -> (vector<16xf32>, f32, f32)
    %otherWeights, %lowerSecond, %upperSecond = func.call @strict_device_bytes_weights(%others, %otherTile, %lane) : (vector<16xf32>, !llvm.ptr<3>, i32) -> (vector<16xf32>, f32, f32)
    %lowerLargest = arith.maxnumf %lowerFirst, %lowerSecond : f32
    %upperLargest = arith.maxnumf %upperFirst, %upperSecond : f32
    %lowerRow = func.call @strict_device_bytes_quad_largest(%lowerLargest) : (f32) -> f32
    %upperRow = func.call @strict_device_bytes_quad_largest(%upperLargest) : (f32) -> f32
    %lowerFactor = func.call @strict_device_attention_inverse(%lowerRow) : (f32) -> f32
    %upperFactor = func.call @strict_device_attention_inverse(%upperRow) : (f32) -> f32
    %lowerUnit = arith.divf %lowerRow, %most : f32
    %upperUnit = arith.divf %upperRow, %most : f32
    %lowerOffset = arith.mulf %lowerUnit, %unbias : f32
    %upperOffset = arith.mulf %upperUnit, %unbias : f32
    %a0 = func.call @strict_device_bytes_operand(%weights, %c0, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %a1 = func.call @strict_device_bytes_operand(%weights, %c2, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %a2 = func.call @strict_device_bytes_operand(%weights, %c8, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %a3 = func.call @strict_device_bytes_operand(%weights, %c10, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %b0 = func.call @strict_device_bytes_operand(%otherWeights, %c0, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %b1 = func.call @strict_device_bytes_operand(%otherWeights, %c2, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %b2 = func.call @strict_device_bytes_operand(%otherWeights, %c8, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %b3 = func.call @strict_device_bytes_operand(%otherWeights, %c10, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %bias0 = llvm.insertvalue %bias, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %bias1 = llvm.insertvalue %bias, %bias0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %bias2 = llvm.insertvalue %bias, %bias1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %biased = llvm.insertvalue %bias, %bias2[3] : !llvm.struct<(i32, i32, i32, i32)>
    %rowStart = arith.muli %lane, %c144 : i32
    cf.br ^block(%c0, %outputs : i32, vector<64xf32>)
  ^block(%block: i32, %sums: vector<64xf32>):
    %moreBlocks = arith.cmpi slt, %block, %c8 : i32
    cf.cond_br %moreBlocks, ^blockBody, ^done
  ^blockBody:
    %blockBytes = arith.shli %block, %c4 : i32
    %at = arith.addi %rowStart, %blockBytes : i32
    %at64 = arith.extui %at : i32 to i64
    %valueSlot = llvm.getelementptr %valueTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %otherSlot = llvm.getelementptr %otherTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %matrices = func.call @strict_device_matrices_transposed(%valueSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %otherMatrices = func.call @strict_device_matrices_transposed(%otherSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %matrix0 = llvm.extractvalue %matrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix1 = llvm.extractvalue %matrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix2 = llvm.extractvalue %matrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix3 = llvm.extractvalue %matrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %other0 = llvm.extractvalue %otherMatrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %other1 = llvm.extractvalue %otherMatrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %other2 = llvm.extractvalue %otherMatrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %other3 = llvm.extractvalue %otherMatrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %evenLow = func.call @strict_device_bytes_permuted(%matrix0, %matrix1, %false) : (i32, i32, i1) -> i32
    %evenHigh = func.call @strict_device_bytes_permuted(%matrix2, %matrix3, %false) : (i32, i32, i1) -> i32
    %oddLow = func.call @strict_device_bytes_permuted(%matrix0, %matrix1, %true) : (i32, i32, i1) -> i32
    %oddHigh = func.call @strict_device_bytes_permuted(%matrix2, %matrix3, %true) : (i32, i32, i1) -> i32
    %otherEvenLow = func.call @strict_device_bytes_permuted(%other0, %other1, %false) : (i32, i32, i1) -> i32
    %otherEvenHigh = func.call @strict_device_bytes_permuted(%other2, %other3, %false) : (i32, i32, i1) -> i32
    %otherOddLow = func.call @strict_device_bytes_permuted(%other0, %other1, %true) : (i32, i32, i1) -> i32
    %otherOddHigh = func.call @strict_device_bytes_permuted(%other2, %other3, %true) : (i32, i32, i1) -> i32
    %evenFirst = func.call @strict_device_bytes_product(%a0, %a1, %a2, %a3, %evenLow, %evenHigh, %biased) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %oddFirst = func.call @strict_device_bytes_product(%a0, %a1, %a2, %a3, %oddLow, %oddHigh, %biased) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %even = func.call @strict_device_bytes_product(%b0, %b1, %b2, %b3, %otherEvenLow, %otherEvenHigh, %evenFirst) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %odd = func.call @strict_device_bytes_product(%b0, %b1, %b2, %b3, %otherOddLow, %otherOddHigh, %oddFirst) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %blockSlot = arith.shli %block, %c3 : i32
    %folded = func.call @strict_device_bytes_folded(%sums, %blockSlot, %even, %odd, %lowerUnit, %upperUnit, %lowerScale, %upperScale, %lowerOffset, %upperOffset) : (vector<64xf32>, i32, !llvm.struct<(i32, i32, i32, i32)>, !llvm.struct<(i32, i32, i32, i32)>, f32, f32, f32, f32, f32, f32) -> vector<64xf32>
    %blockFollowing = arith.addi %block, %c1 : i32
    llvm.br ^block(%blockFollowing, %folded : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %sums : vector<64xf32>
  }
  // the outputs of strict_device_bytes_weighted divided by the sums of their rows into the result
  func.func private @strict_device_bytes_finish(%result: !llvm.ptr, %wide: i1, %outputs: vector<64xf32>, %state: vector<4xf32>, %lowerToken: i32, %upperToken: i32, %tokenLast: i32, %lowerQuery: i64, %upperQuery: i64, %quad: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %two = arith.constant 2 : i64
    %unit = arith.constant 1.0 : f32
    %lowerPart = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperPart = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %lowerSum = func.call @strict_device_attention_quad(%lowerPart) : (f32) -> f32
    %upperSum = func.call @strict_device_attention_quad(%upperPart) : (f32) -> f32
    %lowerInverse = arith.divf %unit, %lowerSum : f32
    %upperInverse = arith.divf %unit, %upperSum : f32
    %lowerWriting = arith.cmpi sle, %lowerToken, %tokenLast : i32
    %upperWriting = arith.cmpi sle, %upperToken, %tokenLast : i32
    %quadNumbers = arith.shli %quad, %c2 : i32
    cf.br ^block(%c0 : i32)
  ^block(%block: i32):
    %moreBlocks = arith.cmpi slt, %block, %c8 : i32
    cf.cond_br %moreBlocks, ^blockBody, ^done
  ^blockBody:
    %blockNumbers = arith.shli %block, %c4 : i32
    %number = arith.addi %blockNumbers, %quadNumbers : i32
    %number64 = arith.extui %number : i32 to i64
    %slot = arith.shli %block, %c3 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot4 = arith.addi %slot, %c4 : i32
    %slot6 = arith.addi %slot, %c6 : i32
    %lowerAt = arith.addi %lowerQuery, %number64 : i64
    %lowerNext = arith.addi %lowerAt, %two : i64
    %upperAt = arith.addi %upperQuery, %number64 : i64
    %upperNext = arith.addi %upperAt, %two : i64
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot, %lowerAt, %lowerInverse, %lowerWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot2, %lowerNext, %lowerInverse, %lowerWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot4, %upperAt, %upperInverse, %upperWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot6, %upperNext, %upperInverse, %upperWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    %blockFollowing = arith.addi %block, %c1 : i32
    llvm.br ^block(%blockFollowing : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // the sum of number over the lanes before lane and over the whole warp
  func.func private @strict_device_bytes_scan(%number: i32, %lane: i32) -> (i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    cf.br ^step(%c1, %c0, %number : i32, i32, i32)
  ^step(%offset: i32, %before: i32, %total: i32):
    %more = arith.cmpi slt, %offset, %c32 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %other = func.call @strict_device_swap(%total, %offset) : (i32, i32) -> i32
    %laneBit = arith.andi %lane, %offset : i32
    %isUpper = arith.cmpi ne, %laneBit, %c0 : i32
    %withOther = arith.addi %before, %other : i32
    %nextBefore = arith.select %isUpper, %withOther, %before : i32
    %nextTotal = arith.addi %total, %other : i32
    %nextOffset = arith.shli %offset, %c1 : i32
    llvm.br ^step(%nextOffset, %nextBefore, %nextTotal : i32, i32, i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %before, %total : i32, i32
  }
  // the plan of strict_device_bytes_planned by its first warp, 32 key blocks at a time in order: the exact tiles, then
  // a marker 32768 + chunk per 32 pooled key blocks, an even count of them (length -1 when the tiles do not fit, the
  // block attends densely)
  func.func private @strict_device_bytes_planning(%scratch: !llvm.ptr<3>, %queryBlock: i32, %keyBlocks: i32, %positions: i32, %scale: f32, %exact: i32, %lane: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %c1536 = llvm.mlir.constant(1536 : i32) : i32
    %c1538 = llvm.mlir.constant(1538 : i32) : i32
    %c32768 = llvm.mlir.constant(32768 : i32) : i32
    %c65535 = llvm.mlir.constant(65535 : i32) : i32
    %true = arith.constant true
    %plan = llvm.mlir.addressof @strict_device_bytes_plan : !llvm.ptr<3>
    %pools = llvm.mlir.addressof @strict_device_bytes_pools : !llvm.ptr<3>
    %poolScores = llvm.mlir.addressof @strict_device_bytes_pool_scores : !llvm.ptr<3>
    %counts = llvm.mlir.addressof @strict_device_bytes_counts : !llvm.ptr<3>
    %threshold = func.call @strict_device_attention_scratch(%scratch, %c1536) : (!llvm.ptr<3>, i32) -> f32
    %meaned = func.call @strict_device_attention_scratch(%scratch, %c1538) : (!llvm.ptr<3>, i32) -> f32
    %nearFirst = arith.subi %queryBlock, %c1 : i32
    %nearLast = arith.addi %queryBlock, %c1 : i32
    %queryStart = arith.shli %queryBlock, %c6 : i32
    %leadingQueries = arith.cmpi slt, %queryStart, %exact : i32
    %blockLast = arith.subi %keyBlocks, %c1 : i32
    cf.br ^blocks(%c0, %c0, %c0 : i32, i32, i32)
  ^blocks(%start: i32, %length: i32, %pooledCount: i32):
    %moreBlocks = arith.cmpi slt, %start, %keyBlocks : i32
    cf.cond_br %moreBlocks, ^route, ^chunks
  ^route:
    %keyBlock = arith.addi %start, %lane : i32
    %valid = arith.cmpi slt, %keyBlock, %keyBlocks : i32
    %read = arith.minsi %keyBlock, %blockLast : i32
    %scoreAt = arith.addi %c512, %read : i32
    %raw = func.call @strict_device_attention_scratch(%scratch, %scoreAt) : (!llvm.ptr<3>, i32) -> f32
    %centered = arith.subf %raw, %meaned : f32
    %score = arith.mulf %centered, %scale : f32
    %above = arith.cmpf ogt, %score, %threshold : f32
    %afterFirst = arith.cmpi sge, %keyBlock, %nearFirst : i32
    %beforeLast = arith.cmpi sle, %keyBlock, %nearLast : i32
    %near = arith.andi %afterFirst, %beforeLast : i1
    %keyStart = arith.shli %keyBlock, %c6 : i32
    %leadingKey = arith.cmpi slt, %keyStart, %exact : i32
    %routedExact = arith.ori %above, %near : i1
    %routedOrLeading = arith.ori %routedExact, %leadingKey : i1
    %attended = arith.ori %routedOrLeading, %leadingQueries : i1
    %isExact = arith.andi %valid, %attended : i1
    %unattended = arith.xori %attended, %true : i1
    %isPooled = arith.andi %valid, %unattended : i1
    %exactOne = arith.extui %isExact : i1 to i32
    %pooledOne = arith.extui %isPooled : i1 to i32
    %pooledShifted = arith.shli %pooledOne, %c16 : i32
    %counted = arith.ori %exactOne, %pooledShifted : i32
    %before, %total = func.call @strict_device_bytes_scan(%counted, %lane) : (i32, i32) -> (i32, i32)
    %exactBefore = arith.andi %before, %c65535 : i32
    %pooledBefore = arith.shrui %before, %c16 : i32
    %exactTotal = arith.andi %total, %c65535 : i32
    %pooledTotal = arith.shrui %total, %c16 : i32
    cf.cond_br %isExact, ^exactCheck, ^pooledCheck
  ^exactCheck:
    %exactTwice = arith.shli %exactBefore, %c1 : i32
    %exactAt = arith.addi %length, %exactTwice : i32
    %exactEnd = arith.addi %exactAt, %c2 : i32
    %fitsPlan = arith.cmpi sle, %exactEnd, %c512 : i32
    cf.cond_br %fitsPlan, ^exactStore, ^stepped
  ^exactStore:
    %firstTile = arith.shli %keyBlock, %c1 : i32
    %secondTile = arith.addi %firstTile, %c1 : i32
    %exactAt64 = arith.extui %exactAt : i32 to i64
    %firstSlot = llvm.getelementptr %plan[%exactAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    %firstEntry = arith.trunci %firstTile : i32 to i16
    %secondEntry = arith.trunci %secondTile : i32 to i16
    llvm.store %firstEntry, %firstSlot : i16, !llvm.ptr<3>
    %secondSlot = llvm.getelementptr %firstSlot[1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i16
    llvm.store %secondEntry, %secondSlot : i16, !llvm.ptr<3>
    cf.br ^stepped
  ^pooledCheck:
    cf.cond_br %isPooled, ^pooledStore, ^stepped
  ^pooledStore:
    %poolAt = arith.addi %pooledCount, %pooledBefore : i32
    %poolAt64 = arith.extui %poolAt : i32 to i64
    %poolSlot = llvm.getelementptr %pools[%poolAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    %poolEntry = arith.trunci %keyBlock : i32 to i16
    llvm.store %poolEntry, %poolSlot : i16, !llvm.ptr<3>
    %blockLeft = arith.subi %positions, %keyStart : i32
    %blockLength = arith.minsi %blockLeft, %c64 : i32
    %lengthFloat = arith.sitofp %blockLength : i32 to f32
    %logarithm = func.call @strict_device_binary_logarithm(%lengthFloat) : (f32) -> f32
    %pooledScore = arith.addf %score, %logarithm : f32
    %pooledHalf = arith.truncf %pooledScore : f32 to f16
    %scoreSlot = llvm.getelementptr %poolScores[%poolAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    llvm.store %pooledHalf, %scoreSlot : f16, !llvm.ptr<3>
    cf.br ^stepped
  ^stepped:
    %exactTiles = arith.shli %exactTotal, %c1 : i32
    %nextLength = arith.addi %length, %exactTiles : i32
    %nextPooled = arith.addi %pooledCount, %pooledTotal : i32
    %nextStart = arith.addi %start, %c32 : i32
    cf.br ^blocks(%nextStart, %nextLength, %nextPooled : i32, i32, i32)
  ^chunks:
    %overflow = arith.cmpi sgt, %length, %c512 : i32
    %isFirst = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %overflow, ^dense, ^markers
  ^dense:
    cf.cond_br %isFirst, ^denseStore, ^done
  ^denseStore:
    %unplanned = llvm.mlir.constant(-1 : i32) : i32
    llvm.store %unplanned, %counts : i32, !llvm.ptr<3>
    cf.br ^done
  ^markers:
    %pooledUp = arith.addi %pooledCount, %c31 : i32
    %chunks = arith.shrui %pooledUp, %c5 : i32
    %chunksUp = arith.addi %chunks, %c1 : i32
    %evenMask = llvm.mlir.constant(-2 : i32) : i32
    %chunkCount = arith.andi %chunksUp, %evenMask : i32
    %isMarker = arith.cmpi slt, %lane, %chunkCount : i32
    cf.cond_br %isMarker, ^marker, ^counted
  ^marker:
    %marker = arith.addi %lane, %c32768 : i32
    %markerEntry = arith.trunci %marker : i32 to i16
    %markerAt = arith.addi %length, %lane : i32
    %markerAt64 = arith.extui %markerAt : i32 to i64
    %markerSlot = llvm.getelementptr %plan[%markerAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    llvm.store %markerEntry, %markerSlot : i16, !llvm.ptr<3>
    cf.br ^counted
  ^counted:
    cf.cond_br %isFirst, ^countStore, ^done
  ^countStore:
    %planLength = arith.addi %length, %chunkCount : i32
    llvm.store %planLength, %counts : i32, !llvm.ptr<3>
    %pooledCountSlot = llvm.getelementptr %counts[1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
    llvm.store %pooledCount, %pooledCountSlot : i32, !llvm.ptr<3>
    cf.br ^done
  ^done:
    return
  }
  // the 4 numbers from start (lane * 4) of 8 rows of 128 pooled numbers from first, 4 rows apart (0 past rows)
  func.func private @strict_device_bytes_rows(%pooled: !llvm.ptr, %start: i32, %first: i32, %rows: i32) -> vector<32xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %noFour = arith.constant dense<0.0> : vector<4xf32>
    %noNumbers = arith.constant dense<0.0> : vector<32xf32>
    %rowLast = arith.subi %rows, %c1 : i32
    cf.br ^row(%c0, %noNumbers : i32, vector<32xf32>)
  ^row(%index: i32, %numbers: vector<32xf32>):
    %more = arith.cmpi slt, %index, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %apart = arith.shli %index, %c2 : i32
    %current = arith.addi %first, %apart : i32
    %inside = arith.cmpi slt, %current, %rows : i32
    %read = arith.minsi %current, %rowLast : i32
    %readRow = arith.shli %read, %c7 : i32
    %at = arith.addi %start, %readRow : i32
    %at64 = arith.extui %at : i32 to i64
    %slot = llvm.getelementptr %pooled[%at64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %four = llvm.load %slot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %shown = arith.select %inside, %four, %noFour : vector<4xf32>
    %slot1 = arith.addi %apart, %c1 : i32
    %slot2 = arith.addi %apart, %c2 : i32
    %slot3 = arith.addi %apart, %c3 : i32
    %number0 = llvm.extractelement %shown[%c0 : i32] : vector<4xf32>
    %number1 = llvm.extractelement %shown[%c1 : i32] : vector<4xf32>
    %number2 = llvm.extractelement %shown[%c2 : i32] : vector<4xf32>
    %number3 = llvm.extractelement %shown[%c3 : i32] : vector<4xf32>
    %with0 = llvm.insertelement %number0, %numbers[%apart : i32] : vector<32xf32>
    %with1 = llvm.insertelement %number1, %with0[%slot1 : i32] : vector<32xf32>
    %with2 = llvm.insertelement %number2, %with1[%slot2 : i32] : vector<32xf32>
    %with3 = llvm.insertelement %number3, %with2[%slot3 : i32] : vector<32xf32>
    %following = arith.addi %index, %c1 : i32
    llvm.br ^row(%following, %with3 : i32, vector<32xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %numbers : vector<32xf32>
  }
  // the 4 numbers at slot of the rows of strict_device_bytes_rows
  func.func private @strict_device_bytes_four(%rows: vector<32xf32>, %slot: i32) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %number0 = llvm.extractelement %rows[%slot : i32] : vector<32xf32>
    %number1 = llvm.extractelement %rows[%slot1 : i32] : vector<32xf32>
    %number2 = llvm.extractelement %rows[%slot2 : i32] : vector<32xf32>
    %number3 = llvm.extractelement %rows[%slot3 : i32] : vector<32xf32>
    %with0 = llvm.insertelement %number0, %empty[%c0 : i32] : vector<4xf32>
    %with1 = llvm.insertelement %number1, %with0[%c1 : i32] : vector<4xf32>
    %with2 = llvm.insertelement %number2, %with1[%c2 : i32] : vector<4xf32>
    %with3 = llvm.insertelement %number3, %with2[%c3 : i32] : vector<4xf32>
    return %with3 : vector<4xf32>
  }
  // the sums and sums of squares of the numbers of a lane with the rows of strict_device_bytes_rows
  func.func private @strict_device_bytes_moments(%rows: vector<32xf32>, %sums: vector<4xf32>, %squares: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    cf.br ^row(%c0, %sums, %squares : i32, vector<4xf32>, vector<4xf32>)
  ^row(%index: i32, %summed: vector<4xf32>, %squared: vector<4xf32>):
    %more = arith.cmpi slt, %index, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %slot = arith.shli %index, %c2 : i32
    %four = func.call @strict_device_bytes_four(%rows, %slot) : (vector<32xf32>, i32) -> vector<4xf32>
    %nextSummed = arith.addf %summed, %four : vector<4xf32>
    %nextSquared = llvm.intr.fma(%four, %four, %squared) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %following = arith.addi %index, %c1 : i32
    llvm.br ^row(%following, %nextSummed, %nextSquared : i32, vector<4xf32>, vector<4xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %summed, %squared : vector<4xf32>, vector<4xf32>
  }
  // the scores of the 8 key blocks of the rows of strict_device_bytes_rows (from first, 4 apart) with the centroid
  // numbers of the lane into the scratch at 512 by the first lane
  func.func private @strict_device_bytes_dots(%rows: vector<32xf32>, %centroid: vector<4xf32>, %scratch: !llvm.ptr<3>, %first: i32, %keyBlocks: i32, %lane: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %isFirstLane = arith.cmpi eq, %lane, %c0 : i32
    cf.br ^row(%c0 : i32)
  ^row(%index: i32):
    %more = arith.cmpi slt, %index, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %slot = arith.shli %index, %c2 : i32
    %four = func.call @strict_device_bytes_four(%rows, %slot) : (vector<32xf32>, i32) -> vector<4xf32>
    %products = arith.mulf %four, %centroid : vector<4xf32>
    %part = llvm.call_intrinsic "llvm.vector.reduce.fadd.v4f32"(%zero, %products) : (f32, vector<4xf32>) -> f32
    %dot = func.call @strict_device_attention_warp_sum(%part) : (f32) -> f32
    %keyBlock = arith.addi %first, %slot : i32
    %isKeyBlock = arith.cmpi slt, %keyBlock, %keyBlocks : i32
    %storing = arith.andi %isFirstLane, %isKeyBlock : i1
    %following = arith.addi %index, %c1 : i32
    cf.cond_br %storing, ^store, ^next
  ^store:
    %dotAt = arith.addi %c512, %keyBlock : i32
    func.call @strict_device_attention_scratch_put(%scratch, %dotAt, %dot) : (!llvm.ptr<3>, i32, f32) -> ()
    cf.br ^next
  ^next:
    llvm.br ^row(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // the plan of a routed block like Sol-Attn's routing: a key block is attended exactly when the centered score of the
  // query block's centroid with it clears the threshold or it is next to the query block, else pooled (its score plus
  // the logarithm of its length and its mean value stand for its positions); the scratch is the third buffer of tiles:
  // the mean and variance per number of the pooled keys of the group at 0 and 128, the centroid at 256, threshold and
  // mean score at 1536 and 1538 (strict_device_attention_threshold), the scores of the key blocks at 512 (both by warps
  // reading 8 rows of 128 pooled numbers at a time, the sums per warp in the fourth buffer); returns the plan's length
  func.func private @strict_device_bytes_planned(%pooled: !llvm.ptr, %routing: f32, %scale: f32, %queryBlock: i32, %group: i32, %sequence: i32, %heads: i32, %sequences: i32, %tokens: i32, %positions: i32, %thread: i32, %exact: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c63 = llvm.mlir.constant(63 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %scratch = func.call @strict_device_bytes_buffer(%c2) : (i32) -> !llvm.ptr<3>
    %positionsUp = arith.addi %positions, %c63 : i32
    %keyBlocks = arith.shrui %positionsUp, %c6 : i32
    %tokensUp = arith.addi %tokens, %c63 : i32
    %queryBlocks = arith.shrui %tokensUp, %c6 : i32
    %sequenceHeads = arith.muli %sequence, %heads : i32
    %groupStart = arith.addi %sequenceHeads, %group : i32
    %groupBlocks = arith.muli %groupStart, %keyBlocks : i32
    %keyRows = arith.muli %groupBlocks, %c128 : i32
    %allGroups = arith.muli %sequences, %heads : i32
    %allBlocks = arith.muli %allGroups, %keyBlocks : i32
    %section = arith.muli %allBlocks, %c128 : i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %laneNumbers = arith.shli %lane, %c2 : i32
    %laneStart = arith.addi %keyRows, %laneNumbers : i32
    %noFour = arith.constant dense<0.0> : vector<4xf32>
    cf.br ^moment(%c0, %noFour, %noFour : i32, vector<4xf32>, vector<4xf32>)
  ^moment(%momentBase: i32, %sums: vector<4xf32>, %squares: vector<4xf32>):
    %moreMoments = arith.cmpi slt, %momentBase, %keyBlocks : i32
    cf.cond_br %moreMoments, ^momentBody, ^momentsDone
  ^momentBody:
    %momentFirst = arith.addi %momentBase, %warp : i32
    %momentRows = func.call @strict_device_bytes_rows(%pooled, %laneStart, %momentFirst, %keyBlocks) : (!llvm.ptr, i32, i32, i32) -> vector<32xf32>
    %nextSums, %nextSquares = func.call @strict_device_bytes_moments(%momentRows, %sums, %squares) : (vector<32xf32>, vector<4xf32>, vector<4xf32>) -> (vector<4xf32>, vector<4xf32>)
    %nextMomentBase = arith.addi %momentBase, %c32 : i32
    cf.br ^moment(%nextMomentBase, %nextSums, %nextSquares : i32, vector<4xf32>, vector<4xf32>)
  ^momentsDone:
    %partials = func.call @strict_device_bytes_buffer(%c3) : (i32) -> !llvm.ptr<3>
    %warpPartials = arith.shli %warp, %c8 : i32
    %sumsAt = arith.addi %warpPartials, %laneNumbers : i32
    %squaresAt = arith.addi %sumsAt, %c128 : i32
    %sumsAt64 = arith.extui %sumsAt : i32 to i64
    %squaresAt64 = arith.extui %squaresAt : i32 to i64
    %sumsSlot = llvm.getelementptr %partials[%sumsAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %squaresSlot = llvm.getelementptr %partials[%squaresAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %sums, %sumsSlot {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr<3>
    llvm.store %squares, %squaresSlot {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr<3>
    func.call @strict_device_barrier() : () -> ()
    cf.br ^partial(%c0, %zero, %zero : i32, f32, f32)
  ^partial(%partialWarp: i32, %sum: f32, %summedSquares: f32):
    %morePartials = arith.cmpi slt, %partialWarp, %c4 : i32
    cf.cond_br %morePartials, ^partialBody, ^partialsDone
  ^partialBody:
    %partialStart = arith.shli %partialWarp, %c8 : i32
    %partialAt = arith.addi %partialStart, %thread : i32
    %partialSquaresAt = arith.addi %partialAt, %c128 : i32
    %partialSum = func.call @strict_device_attention_scratch(%partials, %partialAt) : (!llvm.ptr<3>, i32) -> f32
    %partialSquares = func.call @strict_device_attention_scratch(%partials, %partialSquaresAt) : (!llvm.ptr<3>, i32) -> f32
    %nextSum = arith.addf %sum, %partialSum : f32
    %nextSummedSquares = arith.addf %summedSquares, %partialSquares : f32
    %partialFollowing = arith.addi %partialWarp, %c1 : i32
    llvm.br ^partial(%partialFollowing, %nextSum, %nextSummedSquares : i32, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^partialsDone:
    %blockCount = arith.sitofp %keyBlocks : i32 to f32
    %meanKey = arith.divf %sum, %blockCount : f32
    %meanSquare = arith.divf %summedSquares, %blockCount : f32
    %meanSquared = arith.mulf %meanKey, %meanKey : f32
    %variance = arith.subf %meanSquare, %meanSquared : f32
    %varianceSlot = arith.addi %thread, %c128 : i32
    func.call @strict_device_attention_scratch_put(%scratch, %thread, %meanKey) : (!llvm.ptr<3>, i32, f32) -> ()
    func.call @strict_device_attention_scratch_put(%scratch, %varianceSlot, %variance) : (!llvm.ptr<3>, i32, f32) -> ()
    %headBlocks = arith.muli %groupStart, %queryBlocks : i32
    %queryRow = arith.addi %headBlocks, %queryBlock : i32
    %queryRowStart = arith.muli %queryRow, %c128 : i32
    %twoSections = arith.shli %section, %c1 : i32
    %centroidRow = arith.addi %queryRowStart, %twoSections : i32
    %centroidAt = arith.addi %centroidRow, %thread : i32
    %centroid = func.call @strict_device_attention_pooled_at(%pooled, %centroidAt) : (!llvm.ptr, i32) -> f32
    %centroidSlot = arith.addi %thread, %c256 : i32
    func.call @strict_device_attention_scratch_put(%scratch, %centroidSlot, %centroid) : (!llvm.ptr<3>, i32, f32) -> ()
    func.call @strict_device_barrier() : () -> ()
    %isFirstWarp = arith.cmpi eq, %warp, %c0 : i32
    cf.cond_br %isFirstWarp, ^threshold, ^scores
  ^threshold:
    func.call @strict_device_attention_threshold(%scratch, %c256, %lane, %routing, %scale) : (!llvm.ptr<3>, i32, i32, f32, f32) -> ()
    cf.br ^scores
  ^scores:
    %centroidStart = arith.addi %c256, %laneNumbers : i32
    %centroidStart64 = arith.extui %centroidStart : i32 to i64
    %laneCentroid = llvm.getelementptr %scratch[%centroidStart64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %centroidNumbers = llvm.load %laneCentroid {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
    cf.br ^scoreBlock(%c0 : i32)
  ^scoreBlock(%scoredBase: i32):
    %moreScores = arith.cmpi slt, %scoredBase, %keyBlocks : i32
    cf.cond_br %moreScores, ^dot, ^dotsDone
  ^dot:
    %scoredFirst = arith.addi %scoredBase, %warp : i32
    %scoredRows = func.call @strict_device_bytes_rows(%pooled, %laneStart, %scoredFirst, %keyBlocks) : (!llvm.ptr, i32, i32, i32) -> vector<32xf32>
    func.call @strict_device_bytes_dots(%scoredRows, %centroidNumbers, %scratch, %scoredFirst, %keyBlocks, %lane) : (vector<32xf32>, vector<4xf32>, !llvm.ptr<3>, i32, i32, i32) -> ()
    %scoredNext = arith.addi %scoredBase, %c32 : i32
    cf.br ^scoreBlock(%scoredNext : i32)
  ^dotsDone:
    func.call @strict_device_barrier() : () -> ()
    cf.cond_br %isFirstWarp, ^plan, ^planDone
  ^plan:
    func.call @strict_device_bytes_planning(%scratch, %queryBlock, %keyBlocks, %positions, %scale, %exact, %lane) : (!llvm.ptr<3>, i32, i32, i32, f32, i32, i32) -> ()
    cf.br ^planDone
  ^planDone:
    func.call @strict_device_barrier() : () -> ()
    %counts = llvm.mlir.addressof @strict_device_bytes_counts : !llvm.ptr<3>
    %length = llvm.load %counts : !llvm.ptr<3> -> i32
    return %length : i32
  }
  // strict_device_bytes_attention of the 8 bit keys and values strict_gpu_attention quantized (failed when they are
  // not), routed for routing above 0 by the 32 bit pooled means of strict_device_attention_pools
  func.func private @strict_device_bytes_run(%result: !llvm.ptr, %wide: i1, %dense: i1, %failed: !llvm.ptr, %queries: !llvm.ptr, %queriesCount: i64, %queryBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %headsNumber: f64, %sequencesNumber: f64, %pooled: !llvm.ptr, %pooledBits: i64, %routingNumber: f64, %exactNumber: f64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %eightBits = arith.constant 8 : i64
    %eightKeys = arith.cmpi eq, %keyBits, %eightBits : i64
    %eightValues = arith.cmpi eq, %valueBits, %eightBits : i64
    %eights = arith.andi %eightKeys, %eightValues : i1
    cf.cond_br %eights, ^run, ^fail
  ^fail:
    %one = arith.constant 1 : i64
    llvm.store %one, %failed : i64, !llvm.ptr
    return
  ^run:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c63 = llvm.mlir.constant(63 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %false = arith.constant false
    %true = arith.constant true
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %logarithm = arith.constant 1.44269504 : f32
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %row = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %heads = arith.fptosi %headsNumber : f64 to i32
    %sequences = arith.fptosi %sequencesNumber : f64 to i32
    %headsSize = arith.muli %heads, %c128 : i32
    %stride = arith.muli %headsSize, %sequences : i32
    %queries32 = arith.trunci %queriesCount : i64 to i32
    %tokens = arith.divui %queries32, %stride : i32
    %keys32 = arith.trunci %keysCount : i64 to i32
    %positions = arith.divui %keys32, %stride : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c127 = llvm.mlir.constant(127 : i32) : i32
    %blockShift = arith.select %dense, %c7, %c6 : i32
    %blockSpare = arith.select %dense, %c127, %c63 : i32
    %tokensUp = arith.addi %tokens, %blockSpare : i32
    %queryBlocks = arith.shrui %tokensUp, %blockShift : i32
    %queryBlock = arith.remui %block, %queryBlocks : i32
    %rest = arith.divui %block, %queryBlocks : i32
    %group = arith.remui %rest, %heads : i32
    %sequence = arith.divui %rest, %heads : i32
    %blockFirst = arith.shli %queryBlock, %blockShift : i32
    %warpStart = arith.shli %warp, %c4 : i32
    %warpFirst = arith.addi %blockFirst, %warpStart : i32
    %tokenLast = arith.subi %tokens, %c1 : i32
    %positionLast = arith.subi %positions, %c1 : i32
    %sequenceTokens = arith.muli %sequence, %tokens : i32
    %upperRow = arith.addi %row, %c8 : i32
    %lowerToken, %lowerLast, %lowerQuery = func.call @strict_device_attention_row(%row, %c1, %warpFirst, %tokenLast, %positionLast, %positionLast, %group, %sequenceTokens, %heads, %c128) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %upperToken, %upperLast, %upperQuery = func.call @strict_device_attention_row(%upperRow, %c1, %warpFirst, %tokenLast, %positionLast, %positionLast, %group, %sequenceTokens, %heads, %c128) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %sizeFloat = arith.sitofp %c128 : i32 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeFloat) : (f32) -> f32
    %scale = arith.divf %logarithm, %root : f32
    %queryPairs = func.call @strict_device_attention_quantized(%queries, %queryBits, %lowerQuery, %upperQuery, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, f32) -> vector<32xi32>
    %sequenceHeads = arith.muli %sequence, %heads : i32
    %groupStart = arith.addi %sequenceHeads, %group : i32
    %base = arith.muli %groupStart, %c128 : i32
    %positionsUp = arith.addi %positions, %c63 : i32
    %keyBlockCount = arith.shrui %positionsUp, %c6 : i32
    %tileCount = arith.shli %keyBlockCount, %c1 : i32
    %noRouting = arith.constant 0.0 : f64
    %routingWanted = arith.cmpf ogt, %routingNumber, %noRouting : f64
    %singleBits = arith.constant 32 : i64
    %singlePooled = arith.cmpi eq, %pooledBits, %singleBits : i64
    %routable = arith.andi %routingWanted, %singlePooled : i1
    %sparse = arith.xori %dense, %true : i1
    %routed = arith.andi %routable, %sparse : i1
    cf.cond_br %routed, ^planning, ^planned(%tileCount, %false, %zero, %zero, %c0 : i32, i1, f32, f32, i32)
  ^planning:
    %routing = arith.truncf %routingNumber : f64 to f32
    %exact = arith.fptosi %exactNumber : f64 to i32
    %planLength = func.call @strict_device_bytes_planned(%pooled, %routing, %scale, %queryBlock, %group, %sequence, %heads, %sequences, %tokens, %positions, %thread, %exact) : (!llvm.ptr, f32, f32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %fitted = arith.cmpi sge, %planLength, %c0 : i32
    cf.cond_br %fitted, ^routedPlan, ^planned(%tileCount, %false, %zero, %zero, %c0 : i32, i1, f32, f32, i32)
  ^routedPlan:
    %means = func.call @strict_device_bytes_buffer(%c2) : (i32) -> !llvm.ptr<3>
    %shiftLower = func.call @strict_device_attention_shift(%queries, %queryBits, %lowerQuery, %quad, %scale, %means) : (!llvm.ptr, i64, i64, i32, f32, !llvm.ptr<3>) -> f32
    %shiftUpper = func.call @strict_device_attention_shift(%queries, %queryBits, %upperQuery, %quad, %scale, %means) : (!llvm.ptr, i64, i64, i32, f32, !llvm.ptr<3>) -> f32
    %blocksUp = arith.addi %positions, %c63 : i32
    %keyBlocks = arith.shrui %blocksUp, %c6 : i32
    %allGroups = arith.muli %sequences, %heads : i32
    %allBlocks = arith.muli %allGroups, %keyBlocks : i32
    %keySection = arith.muli %allBlocks, %c128 : i32
    %groupBlocks = arith.muli %groupStart, %keyBlocks : i32
    %groupRows = arith.muli %groupBlocks, %c128 : i32
    %pooledValueRows = arith.addi %keySection, %groupRows : i32
    cf.br ^planned(%planLength, %true, %shiftLower, %shiftUpper, %pooledValueRows : i32, i1, f32, f32, i32)
  ^planned(%count: i32, %routedHere: i1, %lowerShift: f32, %upperShift: f32, %valueRows: i32):
    func.call @strict_device_bytes_issued(%c0, %count, %routedHere, %keys, %values, %pooled, %positions, %stride, %base, %valueRows, %thread, %dense) : (i32, i32, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i1) -> ()
    %warpActive = arith.cmpi sle, %warpFirst, %tokenLast : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %hidden, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %zero, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %hidden, %state1[%c2 : i32] : vector<4xf32>
    %started = llvm.insertelement %zero, %state2[%c3 : i32] : vector<4xf32>
    cf.br ^step(%c0, %zeros, %started : i32, vector<64xf32>, vector<4xf32>)
  ^step(%step: i32, %outputs: vector<64xf32>, %state: vector<4xf32>):
    %entry = arith.shli %step, %c1 : i32
    %moreSteps = arith.cmpi slt, %entry, %count : i32
    cf.cond_br %moreSteps, ^wait, ^finish
  ^wait:
    func.call @strict_device_copies_waited() : () -> ()
    func.call @strict_device_barrier() : () -> ()
    %stepFollowing = arith.addi %step, %c1 : i32
    func.call @strict_device_bytes_issued(%stepFollowing, %count, %routedHere, %keys, %values, %pooled, %positions, %stride, %base, %valueRows, %thread, %dense) : (i32, i32, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i1) -> ()
    cf.cond_br %warpActive, ^compute, ^step(%stepFollowing, %outputs, %state : i32, vector<64xf32>, vector<4xf32>)
  ^compute:
    %pair = arith.andi %step, %c1 : i32
    %firstBuffer = arith.shli %pair, %c1 : i32
    %secondBuffer = arith.addi %firstBuffer, %c1 : i32
    %keyTile = func.call @strict_device_bytes_buffer(%firstBuffer) : (i32) -> !llvm.ptr<3>
    %valueTile = llvm.getelementptr %keyTile[4736] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %otherKeyTile = func.call @strict_device_bytes_buffer(%secondBuffer) : (i32) -> !llvm.ptr<3>
    %otherValueTile = llvm.getelementptr %otherKeyTile[4736] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %otherEntry = arith.addi %entry, %c1 : i32
    %first, %isPooled, %chunk = func.call @strict_device_bytes_entry(%routedHere, %entry) : (i1, i32) -> (i32, i1, i32)
    %second, %otherPooled, %otherChunk = func.call @strict_device_bytes_entry(%routedHere, %otherEntry) : (i1, i32) -> (i32, i1, i32)
    cf.cond_br %isPooled, ^poolScores, ^tileScores
  ^poolScores:
    %pooledScores = func.call @strict_device_bytes_scores_pooled(%chunk, %lane, %lowerShift, %upperShift) : (i32, i32, f32, f32) -> vector<16xf32>
    %otherPooledScores = func.call @strict_device_bytes_scores_pooled(%otherChunk, %lane, %lowerShift, %upperShift) : (i32, i32, f32, f32) -> vector<16xf32>
    cf.br ^scored(%pooledScores, %otherPooledScores : vector<16xf32>, vector<16xf32>)
  ^tileScores:
    %tileScored = func.call @strict_device_attention_scores_eights(%queryPairs, %keyTile, %lane) : (vector<32xi32>, !llvm.ptr<3>, i32) -> vector<16xf32>
    %otherScored = func.call @strict_device_attention_scores_eights(%queryPairs, %otherKeyTile, %lane) : (vector<32xi32>, !llvm.ptr<3>, i32) -> vector<16xf32>
    cf.br ^scored(%tileScored, %otherScored : vector<16xf32>, vector<16xf32>)
  ^scored(%scores: vector<16xf32>, %others: vector<16xf32>):
    %chunkLast = func.call @strict_device_bytes_chunk_last(%chunk) : (i32) -> i32
    %otherChunkLast = func.call @strict_device_bytes_chunk_last(%otherChunk) : (i32) -> i32
    %lowerFirstLast = arith.select %isPooled, %chunkLast, %lowerLast : i32
    %upperFirstLast = arith.select %isPooled, %chunkLast, %upperLast : i32
    %lowerSecondLast = arith.select %isPooled, %otherChunkLast, %lowerLast : i32
    %upperSecondLast = arith.select %isPooled, %otherChunkLast, %upperLast : i32
    %stepEnd = arith.addi %second, %c31 : i32
    %lowerWhole = arith.cmpi sle, %stepEnd, %lowerSecondLast : i32
    %upperWhole = arith.cmpi sle, %stepEnd, %upperSecondLast : i32
    %wholeStep = arith.andi %lowerWhole, %upperWhole : i1
    cf.cond_br %wholeStep, ^whole, ^masked
  ^whole:
    %unmasked = llvm.mlir.constant(2147483647 : i32) : i32
    %wholeShares, %wholeOthers, %wholeState, %wholeLower, %wholeUpper, %wholeLowerShare, %wholeUpperShare = func.call @strict_device_bytes_softmax(%scores, %others, %state, %first, %second, %unmasked, %unmasked, %unmasked, %unmasked, %quad) : (vector<16xf32>, vector<16xf32>, vector<4xf32>, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32)
    cf.br ^weigh(%wholeShares, %wholeOthers, %wholeState, %wholeLower, %wholeUpper, %wholeLowerShare, %wholeUpperShare : vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32)
  ^masked:
    %maskedShares, %maskedOthers, %maskedState, %maskedLower, %maskedUpper, %maskedLowerShare, %maskedUpperShare = func.call @strict_device_bytes_softmax(%scores, %others, %state, %first, %second, %lowerFirstLast, %upperFirstLast, %lowerSecondLast, %upperSecondLast, %quad) : (vector<16xf32>, vector<16xf32>, vector<4xf32>, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32)
    cf.br ^weigh(%maskedShares, %maskedOthers, %maskedState, %maskedLower, %maskedUpper, %maskedLowerShare, %maskedUpperShare : vector<16xf32>, vector<16xf32>, vector<4xf32>, f32, f32, f32, f32)
  ^weigh(%shares: vector<16xf32>, %otherShares: vector<16xf32>, %newState: vector<4xf32>, %lowerScale: f32, %upperScale: f32, %lowerShare: f32, %upperShare: f32):
    cf.cond_br %dense, ^denseWeigh, ^bytesWeigh
  ^denseWeigh:
    %denseWeighted = func.call @strict_device_dense_weighted(%shares, %otherShares, %outputs, %lowerScale, %upperScale, %lowerShare, %upperShare, %valueTile, %otherValueTile, %lane) : (vector<16xf32>, vector<16xf32>, vector<64xf32>, f32, f32, f32, f32, !llvm.ptr<3>, !llvm.ptr<3>, i32) -> vector<64xf32>
    cf.br ^step(%stepFollowing, %denseWeighted, %newState : i32, vector<64xf32>, vector<4xf32>)
  ^bytesWeigh:
    %weighted = func.call @strict_device_bytes_weighted(%shares, %otherShares, %outputs, %lowerScale, %upperScale, %valueTile, %otherValueTile, %lane) : (vector<16xf32>, vector<16xf32>, vector<64xf32>, f32, f32, !llvm.ptr<3>, !llvm.ptr<3>, i32) -> vector<64xf32>
    cf.br ^step(%stepFollowing, %weighted, %newState : i32, vector<64xf32>, vector<4xf32>)
  ^finish:
    cf.cond_br %dense, ^denseFinish, ^bytesFinish
  ^denseFinish:
    %spareMask = llvm.mlir.constant(-32 : i32) : i32
    %paddedUp = arith.addi %positions, %c31 : i32
    %padded = arith.andi %paddedUp, %spareMask : i32
    %padded64 = arith.extui %padded : i32 to i64
    %stride64 = arith.extui %stride : i32 to i64
    %scalesStart = arith.muli %stride64, %padded64 : i64
    %scales = llvm.getelementptr %values[%scalesStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    func.call @strict_device_dense_finish(%result, %wide, %outputs, %state, %lowerToken, %upperToken, %tokenLast, %lowerQuery, %upperQuery, %quad, %scales, %base) : (!llvm.ptr, i1, vector<64xf32>, vector<4xf32>, i32, i32, i32, i64, i64, i32, !llvm.ptr, i32) -> ()
    return
  ^bytesFinish:
    func.call @strict_device_bytes_finish(%result, %wide, %outputs, %state, %lowerToken, %upperToken, %tokenLast, %lowerQuery, %upperQuery, %quad) : (!llvm.ptr, i1, vector<64xf32>, vector<4xf32>, i32, i32, i32, i64, i64, i32) -> ()
    return
  }
  llvm.func @strict_device_bytes_attention(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    %dense = llvm.mlir.constant(false) : i1
    func.call @strict_device_bytes_run(%result, %wide, %dense, %failed, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %sequences, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, i1, i1, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_bytes_attentions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>, nvvm.minctasm = 2 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    %dense = llvm.mlir.constant(false) : i1
    func.call @strict_device_bytes_run(%result, %wide, %dense, %failed, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %sequences, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, i1, i1, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  // the value of number at position (0 past positions) of rows of width numbers
  func.func private @strict_device_values_number(%source: !llvm.ptr, %bits: i64, %position: i32, %positions: i32, %width: i32, %number: i32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %inside = arith.cmpi slt, %position, %positions : i32
    %last = arith.subi %positions, %c1 : i32
    %read = arith.minsi %position, %last : i32
    %read64 = arith.extui %read : i32 to i64
    %width64 = arith.extui %width : i32 to i64
    %number64 = arith.extui %number : i32 to i64
    %row = arith.muli %read64, %width64 : i64
    %at = arith.addi %row, %number64 : i64
    %value = func.call @strict_device_attention_number(%source, %bits, %at) : (!llvm.ptr, i64, i64) -> f32
    %chosen = arith.select %inside, %value, %zero : f32
    return %chosen : f32
  }
  // the 16 values of number from position first on (0 past positions) times inverse as 4 words of 8 bit numbers, word q
  // holds positions 2q, 2q + 1, 8 + 2q and 9 + 2q like the shares of an m16n8k32 operand (strict_device_bytes_operand)
  func.func private @strict_device_values_words(%source: !llvm.ptr, %bits: i64, %first: i32, %positions: i32, %width: i32, %number: i32, %inverse: f32) -> vector<4xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c9 = llvm.mlir.constant(9 : i32) : i32
    %noWords = arith.constant dense<0> : vector<4xi32>
    cf.br ^word(%c0, %noWords : i32, vector<4xi32>)
  ^word(%quad: i32, %words: vector<4xi32>):
    %more = arith.cmpi slt, %quad, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %pair = arith.shli %quad, %c1 : i32
    %at0 = arith.addi %first, %pair : i32
    %at1 = arith.addi %at0, %c1 : i32
    %at2 = arith.addi %at0, %c8 : i32
    %at3 = arith.addi %at0, %c9 : i32
    %value0 = func.call @strict_device_values_number(%source, %bits, %at0, %positions, %width, %number) : (!llvm.ptr, i64, i32, i32, i32, i32) -> f32
    %value1 = func.call @strict_device_values_number(%source, %bits, %at1, %positions, %width, %number) : (!llvm.ptr, i64, i32, i32, i32, i32) -> f32
    %value2 = func.call @strict_device_values_number(%source, %bits, %at2, %positions, %width, %number) : (!llvm.ptr, i64, i32, i32, i32, i32) -> f32
    %value3 = func.call @strict_device_values_number(%source, %bits, %at3, %positions, %width, %number) : (!llvm.ptr, i64, i32, i32, i32, i32) -> f32
    %word = func.call @strict_device_attention_packed(%value0, %value1, %value2, %value3, %inverse) : (f32, f32, f32, f32, f32) -> i32
    %withWord = llvm.insertelement %word, %words[%quad : i32] : vector<4xi32>
    %following = arith.addi %quad, %c1 : i32
    llvm.br ^word(%following, %withWord : i32, vector<4xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %words : vector<4xi32>
  }
  llvm.mlir.global internal @strict_device_values_largest() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<288 x f32>
  // a block of strict_device_values_transposed: the largest magnitude of its 32 numbers over all positions (each warp
  // every 8th), then the 8 bit numbers of every 32 positions (each warp every 8th tile of them)
  func.func private @strict_device_values_transposed_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %spareMask = llvm.mlir.constant(-32 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %lane = arith.andi %thread, %c31 : i32
    %warp = arith.shrui %thread, %c5 : i32
    %width32 = arith.trunci %width : i64 to i32
    %positions64 = arith.divui %count, %width : i64
    %positions = arith.trunci %positions64 : i64 to i32
    %positionsUp = arith.addi %positions, %c31 : i32
    %padded = arith.andi %positionsUp, %spareMask : i32
    %padded64 = arith.extui %padded : i32 to i64
    %groupStart = arith.shli %block, %c5 : i32
    %number = arith.addi %groupStart, %lane : i32
    %number64 = arith.extui %number : i32 to i64
    cf.br ^largest(%warp, %zero : i32, f32)
  ^largest(%position: i32, %largest: f32):
    %morePositions = arith.cmpi slt, %position, %positions : i32
    cf.cond_br %morePositions, ^largestBody, ^largestDone
  ^largestBody:
    %value = func.call @strict_device_values_number(%source, %bits, %position, %positions, %width32, %number) : (!llvm.ptr, i64, i32, i32, i32, i32) -> f32
    %magnitude = llvm.intr.fabs(%value) : (f32) -> f32
    %larger = arith.maxnumf %largest, %magnitude : f32
    %nextPosition = arith.addi %position, %c8 : i32
    cf.br ^largest(%nextPosition, %larger : i32, f32)
  ^largestDone:
    %shared = llvm.mlir.addressof @strict_device_values_largest : !llvm.ptr<3>
    %thread64 = arith.extui %thread : i32 to i64
    %partSlot = llvm.getelementptr %shared[%thread64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %largest, %partSlot : f32, !llvm.ptr<3>
    func.call @strict_device_barrier() : () -> ()
    %isFirstWarp = arith.cmpi eq, %warp, %c0 : i32
    cf.cond_br %isFirstWarp, ^combine(%c0, %zero : i32, f32), ^combined
  ^combine(%part: i32, %total: f32):
    %moreParts = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %moreParts, ^combineBody, ^combineDone
  ^combineBody:
    %partStart = arith.shli %part, %c5 : i32
    %partAt = arith.addi %partStart, %lane : i32
    %partAt64 = arith.extui %partAt : i32 to i64
    %partRead = llvm.getelementptr %shared[%partAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %partLargest = llvm.load %partRead : !llvm.ptr<3> -> f32
    %combined = arith.maxnumf %total, %partLargest : f32
    %nextPart = arith.addi %part, %c1 : i32
    llvm.br ^combine(%nextPart, %combined : i32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^combineDone:
    %totalAt = arith.addi %c256, %lane : i32
    %totalAt64 = arith.extui %totalAt : i32 to i64
    %totalSlot = llvm.getelementptr %shared[%totalAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %total, %totalSlot : f32, !llvm.ptr<3>
    %scale = arith.divf %total, %most : f32
    %width64 = arith.extui %width32 : i32 to i64
    %scalesStart = arith.muli %width64, %padded64 : i64
    %scales = llvm.getelementptr %target[%scalesStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scaleSlot = llvm.getelementptr %scales[%number64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %scaleSlot : f32, !llvm.ptr
    cf.br ^combined
  ^combined:
    func.call @strict_device_barrier() : () -> ()
    %blockAt = arith.addi %c256, %lane : i32
    %blockAt64 = arith.extui %blockAt : i32 to i64
    %blockSlot = llvm.getelementptr %shared[%blockAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %blockLargest = llvm.load %blockSlot : !llvm.ptr<3> -> f32
    %inverse = func.call @strict_device_attention_inverse(%blockLargest) : (f32) -> f32
    %tiles = arith.shrui %padded, %c5 : i32
    %rowStart = arith.muli %number64, %padded64 : i64
    cf.br ^tile(%warp : i32)
  ^tile(%tile: i32):
    %moreTiles = arith.cmpi slt, %tile, %tiles : i32
    cf.cond_br %moreTiles, ^tileBody, ^done
  ^tileBody:
    %first = arith.shli %tile, %c5 : i32
    %second = arith.addi %first, %c16 : i32
    %low = func.call @strict_device_values_words(%source, %bits, %first, %positions, %width32, %number, %inverse) : (!llvm.ptr, i64, i32, i32, i32, i32, f32) -> vector<4xi32>
    %high = func.call @strict_device_values_words(%source, %bits, %second, %positions, %width32, %number, %inverse) : (!llvm.ptr, i64, i32, i32, i32, i32, f32) -> vector<4xi32>
    %first64 = arith.extui %first : i32 to i64
    %at = arith.addi %rowStart, %first64 : i64
    %lowSlot = llvm.getelementptr %target[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %low, %lowSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr
    %highSlot = llvm.getelementptr %lowSlot[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %high, %highSlot {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr
    %nextTile = arith.addi %tile, %c8 : i32
    cf.br ^tile(%nextTile : i32)
  ^done:
    return
  }
  // values of 32 bit floats or bfloat16 (count / width positions of width numbers) as 8 bit numbers transposed for
  // strict_device_dense_attention: per number its positions padded to 32, each 16 in the order of
  // strict_device_values_words, then a 32 bit scale per number (its largest magnitude / 127); a block of 256 threads per
  // 32 numbers
  llvm.func @strict_device_values_transposed(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %brainBits = arith.constant 17 : i64
    %singleBits = arith.constant 32 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^single
  ^brain:
    func.call @strict_device_values_transposed_run(%source, %count, %width, %target, %brainBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^single:
    func.call @strict_device_values_transposed_run(%source, %count, %width, %target, %singleBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  }
  // the 32 positions from first of the 8 bit values of strict_device_values_transposed, a row of 32 bytes per number of
  // the group (base): two 16 byte copies per thread, the halves of rows 4 to 7 of every 8 swapped so the rows of an
  // ldmatrix lie in different banks, zeros past the padded positions
  func.func private @strict_device_dense_copied(%tile: !llvm.ptr<3>, %data: !llvm.ptr, %first: i32, %positions: i32, %base: i32, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %spareMask = llvm.mlir.constant(-32 : i32) : i32
    %positionsUp = arith.addi %positions, %c31 : i32
    %padded = arith.andi %positionsUp, %spareMask : i32
    %padded64 = arith.extui %padded : i32 to i64
    %first64 = arith.extui %first : i32 to i64
    %inside = arith.cmpi slt, %first, %padded : i32
    cf.br ^part(%c0 : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c2 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %offset = arith.shli %part, %c7 : i32
    %piece = arith.addi %thread, %offset : i32
    %row = arith.shrui %piece, %c1 : i32
    %chunk = arith.andi %piece, %c1 : i32
    %number = arith.addi %base, %row : i32
    %number64 = arith.extui %number : i32 to i64
    %rowStart = arith.muli %number64, %padded64 : i64
    %chunkBytes = arith.shli %chunk, %c4 : i32
    %chunkBytes64 = arith.extui %chunkBytes : i32 to i64
    %rowFirst = arith.addi %rowStart, %first64 : i64
    %at = arith.addi %rowFirst, %chunkBytes64 : i64
    %source = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %quarter = arith.shrui %row, %c2 : i32
    %flip = arith.andi %quarter, %c1 : i32
    %swizzled = arith.xori %chunk, %flip : i32
    %rowBytes = arith.shli %row, %c5 : i32
    %swizzledBytes = arith.shli %swizzled, %c4 : i32
    %targetAt = arith.addi %rowBytes, %swizzledBytes : i32
    %targetAt64 = arith.extui %targetAt : i32 to i64
    %target = llvm.getelementptr %tile[%targetAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    func.call @strict_device_copy_filled(%target, %source, %inside) : (!llvm.ptr<3>, !llvm.ptr, i1) -> ()
    %following = arith.addi %part, %c1 : i32
    llvm.br ^part(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // the 4 sums at slot (2 numbers of the lower row, then of the upper row) with the products of an m16n8k32
  func.func private @strict_device_dense_folded(%sums: vector<64xf32>, %slot: i32, %products: !llvm.struct<(i32, i32, i32, i32)>, %lowerUnit: f32, %upperUnit: f32, %lowerScale: f32, %upperScale: f32, %lowerOffset: f32, %upperOffset: f32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %product0 = llvm.extractvalue %products[0] : !llvm.struct<(i32, i32, i32, i32)>
    %product1 = llvm.extractvalue %products[1] : !llvm.struct<(i32, i32, i32, i32)>
    %product2 = llvm.extractvalue %products[2] : !llvm.struct<(i32, i32, i32, i32)>
    %product3 = llvm.extractvalue %products[3] : !llvm.struct<(i32, i32, i32, i32)>
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %with0 = func.call @strict_device_bytes_added(%sums, %slot, %product0, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with1 = func.call @strict_device_bytes_added(%with0, %slot1, %product1, %lowerUnit, %lowerScale, %lowerOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with2 = func.call @strict_device_bytes_added(%with1, %slot2, %product2, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    %with3 = func.call @strict_device_bytes_added(%with2, %slot3, %product3, %upperUnit, %upperScale, %upperOffset) : (vector<64xf32>, i32, i32, f32, f32, f32) -> vector<64xf32>
    return %with3 : vector<64xf32>
  }
  // strict_device_bytes_weighted for the values of strict_device_values_transposed: the outputs of the two rows of a lane
  // (numbers 8j + 2 * (lane % 4) and the next of every 8, lower then upper row) shrunk by their scales plus the shares
  // of two tiles as 8 bits per row (up to the largest share of the row) times the values, an ldmatrix of 16 numbers and
  // 4 m16n8k32 per 16 numbers
  func.func private @strict_device_dense_weighted(%shares: vector<16xf32>, %others: vector<16xf32>, %outputs: vector<64xf32>, %lowerScale: f32, %upperScale: f32, %lowerRow: f32, %upperRow: f32, %valueTile: !llvm.ptr<3>, %otherTile: !llvm.ptr<3>, %lane: i32) -> vector<64xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c10 = llvm.mlir.constant(10 : i32) : i32
    %bias = llvm.mlir.constant(1262485504 : i32) : i32
    %step = arith.constant 0.007874016 : f32
    %unbias = arith.constant -12582912.0 : f32
    %lowerFactor = func.call @strict_device_attention_inverse(%lowerRow) : (f32) -> f32
    %upperFactor = func.call @strict_device_attention_inverse(%upperRow) : (f32) -> f32
    %lowerUnit = arith.mulf %lowerRow, %step : f32
    %upperUnit = arith.mulf %upperRow, %step : f32
    %lowerOffset = arith.mulf %lowerUnit, %unbias : f32
    %upperOffset = arith.mulf %upperUnit, %unbias : f32
    %a0 = func.call @strict_device_bytes_operand(%shares, %c0, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %a1 = func.call @strict_device_bytes_operand(%shares, %c2, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %a2 = func.call @strict_device_bytes_operand(%shares, %c8, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %a3 = func.call @strict_device_bytes_operand(%shares, %c10, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %b0 = func.call @strict_device_bytes_operand(%others, %c0, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %b1 = func.call @strict_device_bytes_operand(%others, %c2, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %b2 = func.call @strict_device_bytes_operand(%others, %c8, %lowerFactor) : (vector<16xf32>, i32, f32) -> i32
    %b3 = func.call @strict_device_bytes_operand(%others, %c10, %upperFactor) : (vector<16xf32>, i32, f32) -> i32
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %bias0 = llvm.insertvalue %bias, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %bias1 = llvm.insertvalue %bias, %bias0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %bias2 = llvm.insertvalue %bias, %bias1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %biased = llvm.insertvalue %bias, %bias2[3] : !llvm.struct<(i32, i32, i32, i32)>
    %laneRow = arith.andi %lane, %c7 : i32
    %laneHalf = arith.shrui %lane, %c4 : i32
    %halfRows = arith.shli %laneHalf, %c3 : i32
    %rowInBlock = arith.addi %laneRow, %halfRows : i32
    %laneChunk = arith.shrui %lane, %c3 : i32
    %chunk = arith.andi %laneChunk, %c1 : i32
    %quarter = arith.shrui %rowInBlock, %c2 : i32
    %flip = arith.andi %quarter, %c1 : i32
    %swizzled = arith.xori %chunk, %flip : i32
    %swizzledBytes = arith.shli %swizzled, %c4 : i32
    cf.br ^block(%c0, %outputs : i32, vector<64xf32>)
  ^block(%block: i32, %sums: vector<64xf32>):
    %moreBlocks = arith.cmpi slt, %block, %c8 : i32
    cf.cond_br %moreBlocks, ^blockBody, ^done
  ^blockBody:
    %blockRows = arith.shli %block, %c4 : i32
    %row = arith.addi %blockRows, %rowInBlock : i32
    %rowBytes = arith.shli %row, %c5 : i32
    %at = arith.addi %rowBytes, %swizzledBytes : i32
    %at64 = arith.extui %at : i32 to i64
    %valueSlot = llvm.getelementptr %valueTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %otherSlot = llvm.getelementptr %otherTile[%at64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %matrices = func.call @strict_device_matrices(%valueSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %otherMatrices = func.call @strict_device_matrices(%otherSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %matrix0 = llvm.extractvalue %matrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix1 = llvm.extractvalue %matrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix2 = llvm.extractvalue %matrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %matrix3 = llvm.extractvalue %matrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %other0 = llvm.extractvalue %otherMatrices[0] : !llvm.struct<(i32, i32, i32, i32)>
    %other1 = llvm.extractvalue %otherMatrices[1] : !llvm.struct<(i32, i32, i32, i32)>
    %other2 = llvm.extractvalue %otherMatrices[2] : !llvm.struct<(i32, i32, i32, i32)>
    %other3 = llvm.extractvalue %otherMatrices[3] : !llvm.struct<(i32, i32, i32, i32)>
    %lowFirst = func.call @strict_device_bytes_product(%a0, %a1, %a2, %a3, %matrix0, %matrix1, %biased) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %low = func.call @strict_device_bytes_product(%b0, %b1, %b2, %b3, %other0, %other1, %lowFirst) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %highFirst = func.call @strict_device_bytes_product(%a0, %a1, %a2, %a3, %matrix2, %matrix3, %biased) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %high = func.call @strict_device_bytes_product(%b0, %b1, %b2, %b3, %other2, %other3, %highFirst) : (i32, i32, i32, i32, i32, i32, !llvm.struct<(i32, i32, i32, i32)>) -> !llvm.struct<(i32, i32, i32, i32)>
    %lowSlot = arith.shli %block, %c3 : i32
    %highSlot = arith.addi %lowSlot, %c4 : i32
    %foldedLow = func.call @strict_device_dense_folded(%sums, %lowSlot, %low, %lowerUnit, %upperUnit, %lowerScale, %upperScale, %lowerOffset, %upperOffset) : (vector<64xf32>, i32, !llvm.struct<(i32, i32, i32, i32)>, f32, f32, f32, f32, f32, f32) -> vector<64xf32>
    %folded = func.call @strict_device_dense_folded(%foldedLow, %highSlot, %high, %lowerUnit, %upperUnit, %lowerScale, %upperScale, %lowerOffset, %upperOffset) : (vector<64xf32>, i32, !llvm.struct<(i32, i32, i32, i32)>, f32, f32, f32, f32, f32, f32) -> vector<64xf32>
    %blockFollowing = arith.addi %block, %c1 : i32
    llvm.br ^block(%blockFollowing, %folded : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %sums : vector<64xf32>
  }
  // the outputs of strict_device_dense_weighted times the scales of their numbers (strict_device_values_transposed),
  // divided by the sums of their rows into the result
  func.func private @strict_device_dense_finish(%result: !llvm.ptr, %wide: i1, %outputs: vector<64xf32>, %state: vector<4xf32>, %lowerToken: i32, %upperToken: i32, %tokenLast: i32, %lowerQuery: i64, %upperQuery: i64, %quad: i32, %scales: !llvm.ptr, %base: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %unit = arith.constant 1.0 : f32
    %lowerPart = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperPart = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %lowerSum = func.call @strict_device_attention_quad(%lowerPart) : (f32) -> f32
    %upperSum = func.call @strict_device_attention_quad(%upperPart) : (f32) -> f32
    %lowerInverse = arith.divf %unit, %lowerSum : f32
    %upperInverse = arith.divf %unit, %upperSum : f32
    %lowerWriting = arith.cmpi sle, %lowerToken, %tokenLast : i32
    %upperWriting = arith.cmpi sle, %upperToken, %tokenLast : i32
    %quadNumbers = arith.shli %quad, %c1 : i32
    cf.br ^block(%c0 : i32)
  ^block(%block: i32):
    %moreBlocks = arith.cmpi slt, %block, %c16 : i32
    cf.cond_br %moreBlocks, ^blockBody, ^done
  ^blockBody:
    %blockNumbers = arith.shli %block, %c3 : i32
    %number = arith.addi %blockNumbers, %quadNumbers : i32
    %scaleAt = arith.addi %base, %number : i32
    %scaleAt64 = arith.extui %scaleAt : i32 to i64
    %scaleSlot = llvm.getelementptr %scales[%scaleAt64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pair = llvm.load %scaleSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xf32>
    %scale = llvm.extractelement %pair[%c0 : i32] : vector<2xf32>
    %scaleNext = llvm.extractelement %pair[%c1 : i32] : vector<2xf32>
    %slot = arith.shli %block, %c2 : i32
    %slot1 = arith.addi %slot, %c1 : i32
    %slot2 = arith.addi %slot, %c2 : i32
    %slot3 = arith.addi %slot, %c3 : i32
    %output0 = llvm.extractelement %outputs[%slot : i32] : vector<64xf32>
    %output1 = llvm.extractelement %outputs[%slot1 : i32] : vector<64xf32>
    %output2 = llvm.extractelement %outputs[%slot2 : i32] : vector<64xf32>
    %output3 = llvm.extractelement %outputs[%slot3 : i32] : vector<64xf32>
    %scaled0 = arith.mulf %output0, %scale : f32
    %scaled1 = arith.mulf %output1, %scaleNext : f32
    %scaled2 = arith.mulf %output2, %scale : f32
    %scaled3 = arith.mulf %output3, %scaleNext : f32
    %with0 = llvm.insertelement %scaled0, %outputs[%slot : i32] : vector<64xf32>
    %with1 = llvm.insertelement %scaled1, %with0[%slot1 : i32] : vector<64xf32>
    %with2 = llvm.insertelement %scaled2, %with1[%slot2 : i32] : vector<64xf32>
    %scaledOutputs = llvm.insertelement %scaled3, %with2[%slot3 : i32] : vector<64xf32>
    %number64 = arith.extui %number : i32 to i64
    %lowerAt = arith.addi %lowerQuery, %number64 : i64
    %upperAt = arith.addi %upperQuery, %number64 : i64
    func.call @strict_device_attention_written(%result, %wide, %scaledOutputs, %slot, %lowerAt, %lowerInverse, %lowerWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    func.call @strict_device_attention_written(%result, %wide, %scaledOutputs, %slot2, %upperAt, %upperInverse, %upperWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    %blockFollowing = arith.addi %block, %c1 : i32
    llvm.br ^block(%blockFollowing : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // strict_device_bytes_attention without routing in blocks of 256 threads per 128 tokens: every tile of keys and
  // values read from L2 serves twice the queries
  llvm.func @strict_device_dense_attention(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(true) : i1
    %dense = llvm.mlir.constant(true) : i1
    func.call @strict_device_bytes_run(%result, %wide, %dense, %failed, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %sequences, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, i1, i1, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_dense_attentions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %wide = llvm.mlir.constant(false) : i1
    %dense = llvm.mlir.constant(true) : i1
    func.call @strict_device_bytes_run(%result, %wide, %dense, %failed, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %sequences, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, i1, i1, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_attention(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %causal, %wide, %valueOffset, %valueStride, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, i1, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_attentions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %causal, %wide, %valueOffset, %valueStride, %pooled, %pooledBits, %routing, %exact) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, i1, f64, f64, !llvm.ptr, i64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_attention_pooled(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_attention_pooling(%result, %wide, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %valueOffset, %valueStride) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, f64) -> ()
    llvm.return
  }
  llvm.func @strict_device_attention_pools(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64, %valueOffset: f64, %valueStride: f64, %pooled: !llvm.ptr, %pooledCount: i64, %pooledBits: i64, %routing: f64, %exact: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_attention_pooling(%result, %wide, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %valueOffset, %valueStride) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, f64) -> ()
    llvm.return
  }
