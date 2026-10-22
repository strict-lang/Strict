  // two tiles of 32 positions of keys and three of values in 16 bit floats, 136 apart per position (padded so the rows
  // of an ldmatrix lie in different banks)
  llvm.mlir.global internal @strict_device_attention_keys() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8704 x f16>
  llvm.mlir.global internal @strict_device_attention_values() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<13056 x f16>
  // 4 numbers from at of a list of doubles, 32 or 16 bit floats or 8 bit floats (bits 7) as 32 bit floats
  func.func private @strict_device_attention_read(%data: !llvm.ptr, %bits: i64, %at: i64) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    cf.switch %bits : i64, [
      default: ^doubles,
      32: ^singles,
      16: ^halves,
      7: ^bytes
    ]
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
      7: ^bytes
    ]
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
  func.func private @strict_device_attention_stored(%tile: !llvm.ptr<3>, %numbers: vector<16xi32>, %bits: i64, %size: i32, %thread: i32) attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_attention_scores(%queries: vector<32xi32>, %keyTile: !llvm.ptr<3>, %size: i32, %lane: i32) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %withFirst = func.call @strict_device_attention_product(%query0, %query1, %query2, %query3, %matrix0, %matrix1, %pairScores, %firstScores) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<16xf32>, i32) -> vector<16xf32>
    %withSecond = func.call @strict_device_attention_product(%query0, %query1, %query2, %query3, %matrix2, %matrix3, %withFirst, %secondScores) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32, i32, vector<16xf32>, i32) -> vector<16xf32>
    %pairFollowing = arith.addi %pair, %c1 : i32
    llvm.br ^pair(%pairFollowing, %withSecond : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^stepNext(%nextScores: vector<16xf32>):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %nextScores : i32, vector<16xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %scores : vector<16xf32>
  }
  // an m16n8k16 adding the a operands times the b operands (bits of 16 bit pairs) to the 4 sums at slot of sums
  func.func private @strict_device_attention_product(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %bits0: i32, %bits1: i32, %sums: vector<16xf32>, %slot: i32) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %product = func.call @strict_device_halves_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
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
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %lowerLargest = func.call @strict_device_attention_largest(%scores, %c0, %first, %quad, %lastLower) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %upperLargest = func.call @strict_device_attention_largest(%scores, %c2, %first, %quad, %lastUpper) : (vector<16xf32>, i32, i32, i32, i32) -> f32
    %oldLower = llvm.extractelement %state[%c0 : i32] : vector<4xf32>
    %lowerSum = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %oldUpper = llvm.extractelement %state[%c2 : i32] : vector<4xf32>
    %upperSum = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %newLower = arith.maximumf %oldLower, %lowerLargest : f32
    %newUpper = arith.maximumf %oldUpper, %upperLargest : f32
    %lowerShift = arith.subf %oldLower, %newLower : f32
    %upperShift = arith.subf %oldUpper, %newUpper : f32
    %lowerScale = func.call @strict_device_binary_power(%lowerShift) : (f32) -> f32
    %upperScale = func.call @strict_device_binary_power(%upperShift) : (f32) -> f32
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
    %lowerScaledSum = arith.mulf %lowerSum, %lowerScale : f32
    %upperScaledSum = arith.mulf %upperSum, %upperScale : f32
    %lowerNewSum = arith.addf %lowerScaledSum, %lowerTotal : f32
    %upperNewSum = arith.addf %upperScaledSum, %upperTotal : f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %newLower, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %lowerNewSum, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %newUpper, %state1[%c2 : i32] : vector<4xf32>
    %newState = llvm.insertelement %upperNewSum, %state2[%c3 : i32] : vector<4xf32>
    return %shared, %newState, %lowerScale, %upperScale : vector<16xf32>, vector<4xf32>, f32, f32
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
  func.func private @strict_device_attention_queries(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %size: i32, %quad: i32, %scale: f32) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %singles = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    cf.switch %bits : i64, [
      default: ^double,
      32: ^single,
      17: ^brain
    ]
  ^single:
    %singlePairs = func.call @strict_device_attention_query(%queries, %singles, %first, %second, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
    return %singlePairs : vector<32xi32>
  ^brain:
    %brainPairs = func.call @strict_device_attention_query(%queries, %brainBits, %first, %second, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
    return %brainPairs : vector<32xi32>
  ^double:
    %doubles = arith.constant 64 : i64
    %doublePairs = func.call @strict_device_attention_query(%queries, %doubles, %first, %second, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
    return %doublePairs : vector<32xi32>
  }
  // the query numbers of the two rows of a lane (rows at first and second, row lane / 4 and the one 8 below) as the 16
  // bit operands of all 16 steps of 8 numbers, times scale; steps after size / 8 stay zero
  func.func private @strict_device_attention_query(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %size: i32, %quad: i32, %scale: f32) -> vector<32xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %firstPair = func.call @strict_device_attention_two(%queries, %bits, %firstAt, %scale) : (!llvm.ptr, i64, i64, f32) -> i32
    %secondPair = func.call @strict_device_attention_two(%queries, %bits, %secondAt, %scale) : (!llvm.ptr, i64, i64, f32) -> i32
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
  func.func private @strict_device_attention_two(%data: !llvm.ptr, %bits: i64, %at: i64, %scale: f32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_attention_run(%result: !llvm.ptr, %queries: !llvm.ptr, %queriesCount: i64, %queryBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %headsNumber: f64, %groupsNumber: f64, %sizeNumber: f64, %sequencesNumber: f64, %causalNumber: f64, %wide: i1) {
    %zeroNumber = arith.constant 0.0 : f64
    %causal = arith.cmpf une, %causalNumber, %zeroNumber : f64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c63 = llvm.mlir.constant(63 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
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
    %keys32 = arith.trunci %keysCount : i64 to i32
    %positions = arith.divui %keys32, %stride : i32
    %warpTokens = arith.divui %c16, %perGroup : i32
    %decodeTokens = arith.shli %warpTokens, %c2 : i32
    %decoding = arith.cmpi sle, %tokens, %decodeTokens : i32
    cf.cond_br %decoding, ^decode, ^prompt
  ^decode:
    func.call @strict_device_attention_decode(%result, %queries, %queryBits, %keys, %keyBits, %values, %valueBits, %heads, %groups, %size, %sequences, %tokens, %positions, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1) -> ()
    return
  ^prompt:
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %singleBits = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    %isWhole = arith.cmpi eq, %size, %c128 : i32
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
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %singleBits, %values, %singleBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1) -> ()
    return
  ^brainPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %brainBits, %values, %brainBits, %heads, %groups, %c128, %sequences, %tokens, %positions, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1) -> ()
    return
  ^anyPrompt:
    func.call @strict_device_attention_prompt(%result, %queries, %queryBits, %keys, %keyBits, %values, %valueBits, %heads, %groups, %size, %sequences, %tokens, %positions, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i1, i1) -> ()
    return
  }
  // `Attention.Merged` of a prompt like FlashAttention 2 on the 16 bit tensor cores: a block takes a sequence, a group
  // of keys and the rows (token and head) of up to 8 x 16 of its queries, walks the visible positions in tiles of 32
  // (two in shared memory, the next one loads while the block computes the current one) and keeps the largest score
  // and sum of every row, so the scores are never stored
  func.func private @strict_device_attention_prompt(%result: !llvm.ptr, %queries: !llvm.ptr, %queryBits: i64, %keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %heads: i32, %groups: i32, %size: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1) attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %queryPairs = func.call @strict_device_attention_queries(%queries, %queryBits, %lowerQuery, %upperQuery, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
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
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %hidden, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %zero, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %hidden, %state1[%c2 : i32] : vector<4xf32>
    %started = llvm.insertelement %zero, %state2[%c3 : i32] : vector<4xf32>
    %keyTile = llvm.mlir.addressof @strict_device_attention_keys : !llvm.ptr<3>
    %valueTile = llvm.mlir.addressof @strict_device_attention_values : !llvm.ptr<3>
    %firstKeys = func.call @strict_device_attention_fetch(%keys, %keyBits, %c0, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %firstValues = func.call @strict_device_attention_fetch(%values, %valueBits, %c0, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    func.call @strict_device_attention_stored(%keyTile, %firstKeys, %keyBits, %size, %thread) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32) -> ()
    func.call @strict_device_attention_stored(%valueTile, %firstValues, %valueBits, %size, %thread) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32) -> ()
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
    %nextFirst = arith.addi %first, %c32 : i32
    %nextKeys = func.call @strict_device_attention_fetch(%keys, %keyBits, %nextFirst, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %nextValues = func.call @strict_device_attention_fetch(%values, %valueBits, %nextFirst, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> vector<16xi32>
    %tileFollowing = arith.addi %tile, %c1 : i32
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
    %scores = func.call @strict_device_attention_scores(%queryPairs, %keysNow, %size, %lane) : (vector<32xi32>, !llvm.ptr<3>, i32, i32) -> vector<16xf32>
    %shares, %newState, %lowerScale, %upperScale = func.call @strict_device_attention_softmax(%scores, %state, %first, %lowerLast, %upperLast, %quad) : (vector<16xf32>, vector<4xf32>, i32, i32, i32, i32) -> (vector<16xf32>, vector<4xf32>, f32, f32)
    %always = arith.constant true
    cf.cond_br %late, ^computed(%doneOutputs, %newState, %shares, %lowerScale, %upperScale, %always : vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1), ^early
  ^early:
    %weighted = func.call @strict_device_attention_weighted(%shares, %doneOutputs, %lowerScale, %upperScale, %valuesNow, %size, %lane) : (vector<16xf32>, vector<64xf32>, f32, f32, !llvm.ptr<3>, i32, i32) -> vector<64xf32>
    cf.br ^computed(%weighted, %newState, %noShares, %unit, %unit, %nothing : vector<64xf32>, vector<4xf32>, vector<16xf32>, f32, f32, i1)
  ^computed(%newOutputs: vector<64xf32>, %computedState: vector<4xf32>, %nextPending: vector<16xf32>, %nextLower: f32, %nextUpper: f32, %nextIsPending: i1):
    func.call @strict_device_attention_stored(%keysNext, %nextKeys, %keyBits, %size, %thread) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32) -> ()
    func.call @strict_device_attention_stored(%valuesNext, %nextValues, %valueBits, %size, %thread) : (!llvm.ptr<3>, vector<16xi32>, i64, i32, i32) -> ()
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
  func.func private @strict_device_attention_operands(%keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %first: i32, %positionLast: i32, %stride: i32, %base: i32, %size: i32, %row: i32, %quad: i32) -> (vector<16xi32>, vector<16xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %singleKeys, %singleValues = func.call @strict_device_attention_loaded(%keys, %singles, %values, %singles, %first, %positionLast, %stride, %base, %size, %row, %quad) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %singleKeys, %singleValues : vector<16xi32>, vector<16xi32>
  ^byte:
    %byteKeys, %byteValues = func.call @strict_device_attention_loaded(%keys, %bytes, %values, %bytes, %first, %positionLast, %stride, %base, %size, %row, %quad) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %byteKeys, %byteValues : vector<16xi32>, vector<16xi32>
  ^mixed:
    %mixedKeys, %mixedValues = func.call @strict_device_attention_loaded(%keys, %keyBits, %values, %valueBits, %first, %positionLast, %stride, %base, %size, %row, %quad) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    return %mixedKeys, %mixedValues : vector<16xi32>, vector<16xi32>
  }
  // the key and value operands of 8 positions from first for a lane (keys of position first + lane / 4, numbers
  // 2 * (lane % 4) and the next of each step of 8, values of positions first + 2 * (lane % 4) and the next, number
  // lane / 4 of each 8), all loads first so they wait for memory together
  func.func private @strict_device_attention_loaded(%keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %first: i32, %positionLast: i32, %stride: i32, %base: i32, %size: i32, %row: i32, %quad: i32) -> (vector<16xi32>, vector<16xi32>) attributes {llvm.passthrough = ["alwaysinline"]} {
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
    %valueRow = arith.muli %valuePosition, %stride : i32
    %valueRowNext = arith.muli %valuePositionNext, %stride : i32
    %valueGroup = arith.addi %valueRow, %base : i32
    %valueGroupNext = arith.addi %valueRowNext, %base : i32
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
  func.func private @strict_device_attention_decode(%result: !llvm.ptr, %queries: !llvm.ptr, %queryBits: i64, %keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %heads: i32, %groups: i32, %size: i32, %sequences: i32, %tokens: i32, %positions: i32, %causal: i1, %wide: i1) {
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
    %firstKeys, %firstValues = func.call @strict_device_attention_operands(%keys, %keyBits, %values, %valueBits, %firstPosition, %positionLast, %stride, %base, %size, %row, %quad) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
    %sizeFloat = arith.sitofp %size : i32 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeFloat) : (f32) -> f32
    %scale = arith.divf %logarithm, %root : f32
    %queryPairs = func.call @strict_device_attention_queries(%queries, %queryBits, %lowerQuery, %upperQuery, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
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
    %nextKeys, %nextValues = func.call @strict_device_attention_operands(%keys, %keyBits, %values, %valueBits, %nextFirst, %positionLast, %stride, %base, %size, %row, %quad) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32, i32) -> (vector<16xi32>, vector<16xi32>)
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
  llvm.func @strict_device_attention(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_attentions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64, %causal: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %causal, %wide) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, f64, i1) -> ()
    llvm.return
  }