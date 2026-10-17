  // a tile of 64 positions of keys and values in 16 bit floats: keys 136 apart per position, values transposed and 72
  // apart per number of a head (both padded so the 32 lanes of a warp read different banks)
  llvm.mlir.global internal @strict_device_attention_keys() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8704 x f16>
  llvm.mlir.global internal @strict_device_attention_values() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<9216 x f16>
  // 4 numbers from at of a list of doubles, 32 or 16 bit floats or 8 bit floats (bits 7) as 32 bit floats
  func.func private @strict_device_attention_read(%data: !llvm.ptr, %bits: i64, %at: i64) -> vector<4xf32> attributes {passthrough = ["alwaysinline"]} {
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
  // two 32 bit floats as the 16 bit pair of an mma operand
  func.func private @strict_device_attention_pair(%first: f32, %second: f32) -> i32 attributes {passthrough = ["alwaysinline"]} {
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
  // the keys and values of positions first to first + 63 (zeros after the last position) into shared memory, every
  // thread of the block takes parts of 4 numbers
  func.func private @strict_device_attention_tile(%keys: !llvm.ptr, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %first: i32, %positions: i32, %stride: i32, %base: i32, %size: i32, %thread: i32) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %c72 = llvm.mlir.constant(72 : i32) : i32
    %c136 = llvm.mlir.constant(136 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %zeros = arith.constant dense<0.0> : vector<4xf32>
    %keyTile = llvm.mlir.addressof @strict_device_attention_keys : !llvm.ptr<3>
    %valueTile = llvm.mlir.addressof @strict_device_attention_values : !llvm.ptr<3>
    %parts = arith.shrui %size, %c2 : i32
    %total = arith.muli %parts, %c64 : i32
    cf.br ^part(%thread : i32)
  ^part(%part: i32):
    %morePart = arith.cmpi slt, %part, %total : i32
    cf.cond_br %morePart, ^body, ^done
  ^body:
    %row = arith.divui %part, %parts : i32
    %partInRow = arith.remui %part, %parts : i32
    %number = arith.shli %partInRow, %c2 : i32
    %position = arith.addi %first, %row : i32
    %inside = arith.cmpi slt, %position, %positions : i32
    cf.cond_br %inside, ^load, ^store(%zeros, %zeros : vector<4xf32>, vector<4xf32>)
  ^load:
    %positionStart = arith.muli %position, %stride : i32
    %rowStart = arith.addi %positionStart, %base : i32
    %at = arith.addi %rowStart, %number : i32
    %at64 = arith.extui %at : i32 to i64
    %keyNumbers = func.call @strict_device_attention_read(%keys, %keyBits, %at64) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %valueNumbers = func.call @strict_device_attention_read(%values, %valueBits, %at64) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    cf.br ^store(%keyNumbers, %valueNumbers : vector<4xf32>, vector<4xf32>)
  ^store(%keyPart: vector<4xf32>, %valuePart: vector<4xf32>):
    %key0 = llvm.extractelement %keyPart[%c0 : i32] : vector<4xf32>
    %key1 = llvm.extractelement %keyPart[%c1 : i32] : vector<4xf32>
    %key2 = llvm.extractelement %keyPart[%c2 : i32] : vector<4xf32>
    %key3 = llvm.extractelement %keyPart[%c3 : i32] : vector<4xf32>
    %keyLow = func.call @strict_device_attention_pair(%key0, %key1) : (f32, f32) -> i32
    %keyHigh = func.call @strict_device_attention_pair(%key2, %key3) : (f32, f32) -> i32
    %keyRow = arith.muli %row, %c136 : i32
    %keyAt = arith.addi %keyRow, %number : i32
    %keyAt64 = arith.extui %keyAt : i32 to i64
    %keySlot = llvm.getelementptr %keyTile[0, %keyAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8704 x f16>
    llvm.store %keyLow, %keySlot {alignment = 4 : i64} : i32, !llvm.ptr<3>
    %keyNext = llvm.getelementptr %keySlot[2] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f16
    llvm.store %keyHigh, %keyNext {alignment = 4 : i64} : i32, !llvm.ptr<3>
    cf.br ^value(%c0 : i32)
  ^value(%index: i32):
    %moreValues = arith.cmpi slt, %index, %c4 : i32
    cf.cond_br %moreValues, ^valueBody, ^next
  ^valueBody:
    %valueNumber = llvm.extractelement %valuePart[%index : i32] : vector<4xf32>
    %valueHalf = arith.truncf %valueNumber : f32 to f16
    %column = arith.addi %number, %index : i32
    %valueRow = arith.muli %column, %c72 : i32
    %valueAt = arith.addi %valueRow, %row : i32
    %valueAt64 = arith.extui %valueAt : i32 to i64
    %valueSlot = llvm.getelementptr %valueTile[0, %valueAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<9216 x f16>
    llvm.store %valueHalf, %valueSlot : f16, !llvm.ptr<3>
    %following = arith.addi %index, %c1 : i32
    llvm.br ^value(%following : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^next:
    %nextPart = arith.addi %part, %c256 : i32
    cf.br ^part(%nextPart : i32)
  ^done:
    return
  }
  // the query numbers of the two rows of a lane (rows at first and second, row lane / 4 and the one 8 below) as the 16
  // bit operands of all 16 steps of 8 numbers, times scale; steps after size / 8 stay zero
  func.func private @strict_device_attention_queries(%queries: !llvm.ptr, %bits: i64, %first: i64, %second: i64, %size: i32, %quad: i32, %scale: f32) -> vector<32xi32> attributes {passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_attention_two(%data: !llvm.ptr, %bits: i64, %at: i64, %scale: f32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^double
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleFirst = llvm.load %singleSlot invariant : !llvm.ptr -> f32
    %singleNext = llvm.getelementptr %singleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %singleSecond = llvm.load %singleNext invariant : !llvm.ptr -> f32
    cf.br ^scaled(%singleFirst, %singleSecond : f32, f32)
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
  // the scores of the 16 rows of a warp with the 64 positions of the tile in shared memory (8 parts of 8 positions, a
  // lane has 4 of each: row lane / 4 and the one 8 below, positions 2 * (lane % 4) and the next), queries scaled
  func.func private @strict_device_attention_scores(%queries: vector<32xi32>, %size: i32, %row: i32, %quad: i32) -> vector<32xf32> attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c136 = llvm.mlir.constant(136 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : vector<32xf32>
    %keyTile = llvm.mlir.addressof @strict_device_attention_keys : !llvm.ptr<3>
    %steps = arith.shrui %size, %c3 : i32
    %quadNumbers = arith.shli %quad, %c1 : i32
    cf.br ^part(%c0, %zeros : i32, vector<32xf32>)
  ^part(%part: i32, %scores: vector<32xf32>):
    %moreParts = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %moreParts, ^partBody, ^done
  ^partBody:
    %partStart = arith.shli %part, %c3 : i32
    %position = arith.addi %partStart, %row : i32
    %keyRow = arith.muli %position, %c136 : i32
    %keyStart = arith.addi %keyRow, %quadNumbers : i32
    cf.br ^step(%c0, %zero, %zero, %zero, %zero : i32, f32, f32, f32, f32)
  ^step(%step: i32, %sum0: f32, %sum1: f32, %sum2: f32, %sum3: f32):
    %moreSteps = arith.cmpi slt, %step, %c16 : i32
    cf.cond_br %moreSteps, ^stepCheck, ^partDone
  ^stepCheck:
    %used = arith.cmpi slt, %step, %steps : i32
    cf.cond_br %used, ^stepBody, ^stepNext(%sum0, %sum1, %sum2, %sum3 : f32, f32, f32, f32)
  ^stepBody:
    %stepNumbers = arith.shli %step, %c3 : i32
    %keyAt = arith.addi %keyStart, %stepNumbers : i32
    %keyAt64 = arith.extui %keyAt : i32 to i64
    %keySlot = llvm.getelementptr %keyTile[0, %keyAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8704 x f16>
    %keyBits = llvm.load %keySlot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    %key = llvm.bitcast %keyBits : i32 to vector<2xf16>
    %slot = arith.shli %step, %c1 : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %firstBits = llvm.extractelement %queries[%slot : i32] : vector<32xi32>
    %secondBits = llvm.extractelement %queries[%slotNext : i32] : vector<32xi32>
    %first = llvm.bitcast %firstBits : i32 to vector<2xf16>
    %second = llvm.bitcast %secondBits : i32 to vector<2xf16>
    %dot = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f32.f32"(%first, %second, %key, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %dot0 = llvm.extractvalue %dot[0] : !llvm.struct<(f32, f32, f32, f32)>
    %dot1 = llvm.extractvalue %dot[1] : !llvm.struct<(f32, f32, f32, f32)>
    %dot2 = llvm.extractvalue %dot[2] : !llvm.struct<(f32, f32, f32, f32)>
    %dot3 = llvm.extractvalue %dot[3] : !llvm.struct<(f32, f32, f32, f32)>
    cf.br ^stepNext(%dot0, %dot1, %dot2, %dot3 : f32, f32, f32, f32)
  ^stepNext(%next0: f32, %next1: f32, %next2: f32, %next3: f32):
    %stepFollowing = arith.addi %step, %c1 : i32
    llvm.br ^step(%stepFollowing, %next0, %next1, %next2, %next3 : i32, f32, f32, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^partDone:
    %slots = arith.shli %part, %c2 : i32
    %slot1 = arith.addi %slots, %c1 : i32
    %slot2 = arith.addi %slots, %c2 : i32
    %slot3 = arith.addi %slots, %c3 : i32
    %with0 = llvm.insertelement %sum0, %scores[%slots : i32] : vector<32xf32>
    %with1 = llvm.insertelement %sum1, %with0[%slot1 : i32] : vector<32xf32>
    %with2 = llvm.insertelement %sum2, %with1[%slot2 : i32] : vector<32xf32>
    %with3 = llvm.insertelement %sum3, %with2[%slot3 : i32] : vector<32xf32>
    %partFollowing = arith.addi %part, %c1 : i32
    llvm.br ^part(%partFollowing, %with3 : i32, vector<32xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %scores : vector<32xf32>
  }
  // the largest of the 32 scores of a lane for its row lane / 4 (lower) or the one 8 below, over the 4 lanes of a row,
  // positions after last hidden
  func.func private @strict_device_attention_largest(%scores: vector<32xf32>, %lower: i32, %first: i32, %quad: i32, %last: i32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %quadPositions = arith.shli %quad, %c1 : i32
    %rowStart = arith.addi %first, %quadPositions : i32
    cf.br ^part(%c0, %hidden : i32, f32)
  ^part(%part: i32, %largest: f32):
    %more = arith.cmpi slt, %part, %c8 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %c3 : i32
    %position = arith.addi %rowStart, %partStart : i32
    %positionNext = arith.addi %position, %c1 : i32
    %slots = arith.shli %part, %c2 : i32
    %slot = arith.addi %slots, %lower : i32
    %slotNext = arith.addi %slot, %c1 : i32
    %score = llvm.extractelement %scores[%slot : i32] : vector<32xf32>
    %scoreNext = llvm.extractelement %scores[%slotNext : i32] : vector<32xf32>
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
  // one tile for a warp: the scores, the new largest scores and sums of the two rows of a lane (state: largest and sum
  // of the lower row, then of the upper one), the outputs rescaled and the shares times the values added
  func.func private @strict_device_attention_step(%queries: vector<32xi32>, %outputs: vector<64xf32>, %state: vector<4xf32>, %first: i32, %lastLower: i32, %lastUpper: i32, %size: i32, %row: i32, %quad: i32) -> (vector<64xf32>, vector<4xf32>) attributes {passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c72 = llvm.mlir.constant(72 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %valueTile = llvm.mlir.addressof @strict_device_attention_values : !llvm.ptr<3>
    %scores = func.call @strict_device_attention_scores(%queries, %size, %row, %quad) : (vector<32xi32>, i32, i32, i32) -> vector<32xf32>
    %lowerLargest = func.call @strict_device_attention_largest(%scores, %c0, %first, %quad, %lastLower) : (vector<32xf32>, i32, i32, i32, i32) -> f32
    %upperLargest = func.call @strict_device_attention_largest(%scores, %c2, %first, %quad, %lastUpper) : (vector<32xf32>, i32, i32, i32, i32) -> f32
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
    %quadPositions = arith.shli %quad, %c1 : i32
    %rowStart = arith.addi %first, %quadPositions : i32
    cf.br ^shares(%c0, %zero, %zero, %scores : i32, f32, f32, vector<32xf32>)
  ^shares(%sharePart: i32, %lowerTotal: f32, %upperTotal: f32, %shared: vector<32xf32>):
    %moreShares = arith.cmpi slt, %sharePart, %c8 : i32
    cf.cond_br %moreShares, ^shareBody, ^sharesDone
  ^shareBody:
    %partStart = arith.shli %sharePart, %c3 : i32
    %position = arith.addi %rowStart, %partStart : i32
    %positionNext = arith.addi %position, %c1 : i32
    %slots = arith.shli %sharePart, %c2 : i32
    %lowerShare, %lowerShareNext, %afterLower = func.call @strict_device_attention_shares(%shared, %slots, %position, %positionNext, %lastLower, %newLower) : (vector<32xf32>, i32, i32, i32, i32, f32) -> (f32, f32, vector<32xf32>)
    %upperSlots = arith.addi %slots, %c2 : i32
    %upperShare, %upperShareNext, %afterUpper = func.call @strict_device_attention_shares(%afterLower, %upperSlots, %position, %positionNext, %lastUpper, %newUpper) : (vector<32xf32>, i32, i32, i32, i32, f32) -> (f32, f32, vector<32xf32>)
    %lowerPair = arith.addf %lowerShare, %lowerShareNext : f32
    %upperPair = arith.addf %upperShare, %upperShareNext : f32
    %lowerMore = arith.addf %lowerTotal, %lowerPair : f32
    %upperMore = arith.addf %upperTotal, %upperPair : f32
    %shareFollowing = arith.addi %sharePart, %c1 : i32
    llvm.br ^shares(%shareFollowing, %lowerMore, %upperMore, %afterUpper : i32, f32, f32, vector<32xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
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
    %steps = arith.shrui %size, %c3 : i32
    cf.br ^column(%c0, %outputs : i32, vector<64xf32>)
  ^column(%column: i32, %columnOutputs: vector<64xf32>):
    %moreColumns = arith.cmpi slt, %column, %c16 : i32
    cf.cond_br %moreColumns, ^columnCheck, ^done
  ^columnCheck:
    %usedColumn = arith.cmpi slt, %column, %steps : i32
    cf.cond_br %usedColumn, ^columnBody, ^columnNext(%columnOutputs : vector<64xf32>)
  ^columnBody:
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
    %columnStart = arith.shli %column, %c3 : i32
    %valueColumn = arith.addi %columnStart, %row : i32
    %valueRow = arith.muli %valueColumn, %c72 : i32
    %valueStart = arith.addi %valueRow, %quadPositions : i32
    cf.br ^valuePart(%c0, %scaled0, %scaled1, %scaled2, %scaled3 : i32, f32, f32, f32, f32)
  ^valuePart(%valuePart: i32, %sum0: f32, %sum1: f32, %sum2: f32, %sum3: f32):
    %moreValueParts = arith.cmpi slt, %valuePart, %c8 : i32
    cf.cond_br %moreValueParts, ^valueBody, ^columnStore
  ^valueBody:
    %valuePartStart = arith.shli %valuePart, %c3 : i32
    %valueAt = arith.addi %valueStart, %valuePartStart : i32
    %valueAt64 = arith.extui %valueAt : i32 to i64
    %valueSlot = llvm.getelementptr %valueTile[0, %valueAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<9216 x f16>
    %valueBits = llvm.load %valueSlot {alignment = 4 : i64} : !llvm.ptr<3> -> i32
    %value = llvm.bitcast %valueBits : i32 to vector<2xf16>
    %shareSlot = arith.shli %valuePart, %c2 : i32
    %shareSlot1 = arith.addi %shareSlot, %c1 : i32
    %shareSlot2 = arith.addi %shareSlot, %c2 : i32
    %shareSlot3 = arith.addi %shareSlot, %c3 : i32
    %share0 = llvm.extractelement %shared[%shareSlot : i32] : vector<32xf32>
    %share1 = llvm.extractelement %shared[%shareSlot1 : i32] : vector<32xf32>
    %share2 = llvm.extractelement %shared[%shareSlot2 : i32] : vector<32xf32>
    %share3 = llvm.extractelement %shared[%shareSlot3 : i32] : vector<32xf32>
    %lowerShares = func.call @strict_device_attention_pair(%share0, %share1) : (f32, f32) -> i32
    %upperShares = func.call @strict_device_attention_pair(%share2, %share3) : (f32, f32) -> i32
    %lowerOperand = llvm.bitcast %lowerShares : i32 to vector<2xf16>
    %upperOperand = llvm.bitcast %upperShares : i32 to vector<2xf16>
    %weighted = llvm.call_intrinsic "llvm.nvvm.mma.m16n8k8.row.col.f32.f32"(%lowerOperand, %upperOperand, %value, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %weighted0 = llvm.extractvalue %weighted[0] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted1 = llvm.extractvalue %weighted[1] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted2 = llvm.extractvalue %weighted[2] : !llvm.struct<(f32, f32, f32, f32)>
    %weighted3 = llvm.extractvalue %weighted[3] : !llvm.struct<(f32, f32, f32, f32)>
    %valueFollowing = arith.addi %valuePart, %c1 : i32
    llvm.br ^valuePart(%valueFollowing, %weighted0, %weighted1, %weighted2, %weighted3 : i32, f32, f32, f32, f32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^columnStore:
    %stored0 = llvm.insertelement %sum0, %columnOutputs[%outputSlot : i32] : vector<64xf32>
    %stored1 = llvm.insertelement %sum1, %stored0[%outputSlot1 : i32] : vector<64xf32>
    %stored2 = llvm.insertelement %sum2, %stored1[%outputSlot2 : i32] : vector<64xf32>
    %stored3 = llvm.insertelement %sum3, %stored2[%outputSlot3 : i32] : vector<64xf32>
    cf.br ^columnNext(%stored3 : vector<64xf32>)
  ^columnNext(%nextOutputs: vector<64xf32>):
    %columnFollowing = arith.addi %column, %c1 : i32
    llvm.br ^column(%columnFollowing, %nextOutputs : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %columnOutputs, %newState : vector<64xf32>, vector<4xf32>
  }
  // the shares (2 to the power of score minus largest, 0 after last) of the two scores at slot and the next, written
  // back for the values
  func.func private @strict_device_attention_shares(%scores: vector<32xf32>, %slot: i32, %position: i32, %positionNext: i32, %last: i32, %largest: f32) -> (f32, f32, vector<32xf32>) attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %slotNext = arith.addi %slot, %c1 : i32
    %score = llvm.extractelement %scores[%slot : i32] : vector<32xf32>
    %scoreNext = llvm.extractelement %scores[%slotNext : i32] : vector<32xf32>
    %shifted = arith.subf %score, %largest : f32
    %shiftedNext = arith.subf %scoreNext, %largest : f32
    %share = llvm.call_intrinsic "llvm.exp2.f32"(%shifted) : (f32) -> f32
    %shareNext = llvm.call_intrinsic "llvm.exp2.f32"(%shiftedNext) : (f32) -> f32
    %visible = arith.cmpi sle, %position, %last : i32
    %visibleNext = arith.cmpi sle, %positionNext, %last : i32
    %shown = arith.select %visible, %share, %zero : f32
    %shownNext = arith.select %visibleNext, %shareNext, %zero : f32
    %withShare = llvm.insertelement %shown, %scores[%slot : i32] : vector<32xf32>
    %withBoth = llvm.insertelement %shownNext, %withShare[%slotNext : i32] : vector<32xf32>
    return %shown, %shownNext, %withBoth : f32, f32, vector<32xf32>
  }
  func.func private @strict_device_attention_store(%result: !llvm.ptr, %wide: i1, %at: i64, %value: f32) attributes {passthrough = ["alwaysinline"]} {
    cf.cond_br %wide, ^double, ^single
  ^double:
    %wideValue = arith.extf %value : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %wideValue, %doubleSlot : f64, !llvm.ptr
    return
  ^single:
    %singleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %singleSlot : f32, !llvm.ptr
    return
  }
  // `Attention.Merged` like FlashAttention 2 on the 16 bit tensor cores: a block takes a sequence, a group of keys and
  // the rows (token and head) of up to 8 x 16 of its queries, walks the visible positions in tiles of 64 in shared
  // memory and keeps the largest score and sum of every row, so the scores are never stored
  func.func private @strict_device_attention_run(%result: !llvm.ptr, %queries: !llvm.ptr, %queriesCount: i64, %queryBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keyBits: i64, %values: !llvm.ptr, %valueBits: i64, %headsNumber: f64, %groupsNumber: f64, %sizeNumber: f64, %sequencesNumber: f64, %wide: i1) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c63 = llvm.mlir.constant(63 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %hidden = arith.constant -1.0e+30 : f32
    %zero = arith.constant 0.0 : f32
    %logarithm = arith.constant 1.44269504 : f32
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
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
    %before = arith.subi %positions, %tokens : i32
    %groupHeads = arith.muli %group, %perGroup : i32
    %sequenceTokens = arith.muli %sequence, %tokens : i32
    %upperRow = arith.addi %row, %c8 : i32
    %lowerToken, %lowerLast, %lowerQuery = func.call @strict_device_attention_row(%row, %perGroup, %warpFirst, %tokenLast, %before, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %upperToken, %upperLast, %upperQuery = func.call @strict_device_attention_row(%upperRow, %perGroup, %warpFirst, %tokenLast, %before, %groupHeads, %sequenceTokens, %heads, %size) : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i64)
    %sizeFloat = arith.sitofp %size : i32 to f32
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%sizeFloat) : (f32) -> f32
    %scale = arith.divf %logarithm, %root : f32
    %queryPairs = func.call @strict_device_attention_queries(%queries, %queryBits, %lowerQuery, %upperQuery, %size, %quad, %scale) : (!llvm.ptr, i64, i64, i64, i32, i32, f32) -> vector<32xi32>
    %blockEnd = arith.addi %blockFirst, %spareTokens : i32
    %blockLastToken = arith.minsi %blockEnd, %tokenLast : i32
    %blockLast = arith.addi %before, %blockLastToken : i32
    %blockPositions = arith.addi %blockLast, %c64 : i32
    %tiles = arith.shrui %blockPositions, %c6 : i32
    %warpSpare = arith.subi %warpTokens, %c1 : i32
    %warpEnd = arith.addi %warpFirst, %warpSpare : i32
    %warpLastToken = arith.minsi %warpEnd, %tokenLast : i32
    %warpLast = arith.addi %before, %warpLastToken : i32
    %warpActive = arith.cmpi sle, %warpFirst, %tokenLast : i32
    %sequenceGroups = arith.muli %sequence, %groups : i32
    %groupStart = arith.addi %sequenceGroups, %group : i32
    %base = arith.muli %groupStart, %size : i32
    %empty = llvm.mlir.undef : vector<4xf32>
    %state0 = llvm.insertelement %hidden, %empty[%c0 : i32] : vector<4xf32>
    %state1 = llvm.insertelement %zero, %state0[%c1 : i32] : vector<4xf32>
    %state2 = llvm.insertelement %hidden, %state1[%c2 : i32] : vector<4xf32>
    %started = llvm.insertelement %zero, %state2[%c3 : i32] : vector<4xf32>
    cf.br ^tile(%c0, %zeros, %started : i32, vector<64xf32>, vector<4xf32>)
  ^tile(%tile: i32, %outputs: vector<64xf32>, %state: vector<4xf32>):
    %moreTiles = arith.cmpi slt, %tile, %tiles : i32
    cf.cond_br %moreTiles, ^load, ^finish
  ^load:
    %first = arith.shli %tile, %c6 : i32
    nvvm.barrier0
    func.call @strict_device_attention_tile(%keys, %keyBits, %values, %valueBits, %first, %positions, %stride, %base, %size, %thread) : (!llvm.ptr, i64, !llvm.ptr, i64, i32, i32, i32, i32, i32, i32) -> ()
    nvvm.barrier0
    %visible = arith.cmpi sle, %first, %warpLast : i32
    %computing = arith.andi %warpActive, %visible : i1
    cf.cond_br %computing, ^compute, ^tileNext(%outputs, %state : vector<64xf32>, vector<4xf32>)
  ^compute:
    %newOutputs, %newState = func.call @strict_device_attention_step(%queryPairs, %outputs, %state, %first, %lowerLast, %upperLast, %size, %row, %quad) : (vector<32xi32>, vector<64xf32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> (vector<64xf32>, vector<4xf32>)
    cf.br ^tileNext(%newOutputs, %newState : vector<64xf32>, vector<4xf32>)
  ^tileNext(%nextOutputs: vector<64xf32>, %nextState: vector<4xf32>):
    %tileFollowing = arith.addi %tile, %c1 : i32
    cf.br ^tile(%tileFollowing, %nextOutputs, %nextState : i32, vector<64xf32>, vector<4xf32>)
  ^finish:
    %lowerPart = llvm.extractelement %state[%c1 : i32] : vector<4xf32>
    %upperPart = llvm.extractelement %state[%c3 : i32] : vector<4xf32>
    %lowerSum = func.call @strict_device_attention_quad(%lowerPart) : (f32) -> f32
    %upperSum = func.call @strict_device_attention_quad(%upperPart) : (f32) -> f32
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
    func.call @strict_device_attention_written(%result, %wide, %outputs, %slot, %lowerAt, %lowerSum, %lowerWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    %upperSlot = arith.addi %slot, %c2 : i32
    func.call @strict_device_attention_written(%result, %wide, %outputs, %upperSlot, %upperAt, %upperSum, %upperWriting) : (!llvm.ptr, i1, vector<64xf32>, i32, i64, f32, i1) -> ()
    cf.br ^columnNext
  ^columnNext:
    %columnFollowing = arith.addi %column, %c1 : i32
    llvm.br ^column(%columnFollowing : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // the token of a row of a warp (16 rows of tokens times the heads of a group), its last visible position and where
  // its query starts
  func.func private @strict_device_attention_row(%row: i32, %perGroup: i32, %warpFirst: i32, %tokenLast: i32, %before: i32, %groupHeads: i32, %sequenceTokens: i32, %heads: i32, %size: i32) -> (i32, i32, i64) attributes {passthrough = ["alwaysinline"]} {
    %inWarp = arith.divui %row, %perGroup : i32
    %head = arith.remui %row, %perGroup : i32
    %token = arith.addi %warpFirst, %inWarp : i32
    %clamped = arith.minsi %token, %tokenLast : i32
    %last = arith.addi %before, %clamped : i32
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
  func.func private @strict_device_attention_quad(%part: f32) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %fromOne = func.call @strict_device_shuffles(%part, %c1) : (f32, i32) -> f32
    %pair = arith.addf %part, %fromOne : f32
    %fromTwo = func.call @strict_device_shuffles(%pair, %c2) : (f32, i32) -> f32
    %sum = arith.addf %pair, %fromTwo : f32
    return %sum : f32
  }
  // the two outputs at slot and the next divided by sum into the result at at and the next
  func.func private @strict_device_attention_written(%result: !llvm.ptr, %wide: i1, %outputs: vector<64xf32>, %slot: i32, %at: i64, %sum: f32, %writing: i1) attributes {passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %one = arith.constant 1 : i64
    cf.cond_br %writing, ^write, ^done
  ^write:
    %slotNext = arith.addi %slot, %c1 : i32
    %output = llvm.extractelement %outputs[%slot : i32] : vector<64xf32>
    %outputNext = llvm.extractelement %outputs[%slotNext : i32] : vector<64xf32>
    %value = arith.divf %output, %sum : f32
    %valueNext = arith.divf %outputNext, %sum : f32
    %atNext = arith.addi %at, %one : i64
    func.call @strict_device_attention_store(%result, %wide, %at, %value) : (!llvm.ptr, i1, i64, f32) -> ()
    func.call @strict_device_attention_store(%result, %wide, %atNext, %valueNext) : (!llvm.ptr, i1, i64, f32) -> ()
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_attention(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %wide) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, i1) -> ()
    llvm.return
  }
  llvm.func @strict_device_attentions(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %queries: !llvm.ptr, %queriesCount: i64, %queriesBits: i64, %keys: !llvm.ptr, %keysCount: i64, %keysBits: i64, %values: !llvm.ptr, %valuesCount: i64, %valuesBits: i64, %heads: f64, %groups: f64, %size: f64, %sequences: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
    func.call @strict_device_attention_run(%result, %queries, %queriesCount, %queriesBits, %keys, %keysCount, %keysBits, %values, %valuesBits, %heads, %groups, %size, %sequences, %wide) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, f64, f64, f64, f64, i1) -> ()
    llvm.return
  }