  // a number of a list at at as a 32 bit float, 0 when outside is set (at is not read then)
  func.func private @strict_device_products_number(%data: !llvm.ptr, %count: i64, %bits: i64, %at: i64, %outside: i1) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f32
    %safe = arith.select %outside, %none, %at : i64
    %number = func.call @strict_device_elements(%data, %bits, %safe, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    %chosen = arith.select %outside, %zero, %number : f32
    return %chosen : f32
  }
  // two 32 bit floats as the 16 bit pair of an mma operand
  func.func private @strict_device_products_pair(%first: f32, %second: f32) -> vector<2xf16> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %firstHalf = arith.truncf %first : f32 to f16
    %secondHalf = arith.truncf %second : f32 to f16
    %empty = llvm.mlir.undef : vector<2xf16>
    %withFirst = llvm.insertelement %firstHalf, %empty[%c0 : i32] : vector<2xf16>
    %pair = llvm.insertelement %secondHalf, %withFirst[%c1 : i32] : vector<2xf16>
    return %pair : vector<2xf16>
  }
  // one output at row and column of a batch into the result unless it is past the edges
  func.func private @strict_device_products_written(%result: !llvm.ptr, %results: i64, %value: f32, %batch: i64, %row: i64, %column: i64, %rows: i64, %columns: i64) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %doubleBits = arith.constant 64 : i64
    %wide = arith.cmpi eq, %results, %doubleBits : i64
    %rowInside = arith.cmpi slt, %row, %rows : i64
    %columnInside = arith.cmpi slt, %column, %columns : i64
    %inside = arith.andi %rowInside, %columnInside : i1
    cf.cond_br %inside, ^write, ^done
  ^write:
    %batchRow = arith.muli %batch, %rows : i64
    %fullRow = arith.addi %batchRow, %row : i64
    %rowStart = arith.muli %fullRow, %columns : i64
    %at = arith.addi %rowStart, %column : i64
    cf.cond_br %wide, ^double, ^narrow
  ^double:
    %double = arith.extf %value : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    cf.br ^done
  ^narrow:
    func.call @strict_device_stored(%result, %at, %value, %results) : (!llvm.ptr, i64, f32, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  // `Products.Multiplied` on the 16 bit tensor cores: a warp computes 16 rows by 8 columns of one batch, the numbers of
  // left and right read at their strides 8 along inner at a time (zeros past the edges), sums in 32 bit floats
  func.func private @strict_device_products_run(%result: !llvm.ptr, %results: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %shape: vector<4xf64>, %strides: vector<8xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %sixteen = arith.constant 16 : i64
    %thirtyTwo = arith.constant 32 : i64
    %zero = arith.constant 0.0 : f32
    %shapeInts = arith.fptosi %shape : vector<4xf64> to vector<4xi64>
    %strideInts = arith.fptosi %strides : vector<8xf64> to vector<8xi64>
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %batches = llvm.extractelement %shapeInts[%c0 : i32] : vector<4xi64>
    %rows = llvm.extractelement %shapeInts[%c1 : i32] : vector<4xi64>
    %inner = llvm.extractelement %shapeInts[%c2 : i32] : vector<4xi64>
    %columns = llvm.extractelement %shapeInts[%c3 : i32] : vector<4xi64>
    %leftStart = llvm.extractelement %strideInts[%c0 : i32] : vector<8xi64>
    %leftBatch = llvm.extractelement %strideInts[%c1 : i32] : vector<8xi64>
    %leftRow = llvm.extractelement %strideInts[%c2 : i32] : vector<8xi64>
    %leftInner = llvm.extractelement %strideInts[%c3 : i32] : vector<8xi64>
    %rightStart = llvm.extractelement %strideInts[%c4 : i32] : vector<8xi64>
    %rightBatch = llvm.extractelement %strideInts[%c5 : i32] : vector<8xi64>
    %rightInner = llvm.extractelement %strideInts[%c6 : i32] : vector<8xi64>
    %rightColumn = llvm.extractelement %strideInts[%c7 : i32] : vector<8xi64>
    %thread32 = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %width32 = func.call @strict_device_block_size() : () -> i32
    %thread = arith.extui %thread32 : i32 to i64
    %block = arith.extui %block32 : i32 to i64
    %width = arith.extui %width32 : i32 to i64
    %blockStart = arith.muli %block, %width : i64
    %global = arith.addi %blockStart, %thread : i64
    %warp = arith.divui %global, %thirtyTwo : i64
    %lane = arith.remui %global, %thirtyTwo : i64
    %rowsUp = arith.addi %rows, %fifteen : i64
    %rowTiles = arith.divui %rowsUp, %sixteen : i64
    %columnsUp = arith.addi %columns, %seven : i64
    %columnTiles = arith.divui %columnsUp, %eight : i64
    %perBatch = arith.muli %rowTiles, %columnTiles : i64
    %tiles = arith.muli %batches, %perBatch : i64
    %working = arith.cmpi ult, %warp, %tiles : i64
    cf.cond_br %working, ^tile, ^done
  ^tile:
    %batch = arith.divui %warp, %perBatch : i64
    %within = arith.remui %warp, %perBatch : i64
    %rowTile = arith.divui %within, %columnTiles : i64
    %columnTile = arith.remui %within, %columnTiles : i64
    %group = arith.divui %lane, %four : i64
    %quad = arith.remui %lane, %four : i64
    %rowFirst = arith.muli %rowTile, %sixteen : i64
    %lowerRow = arith.addi %rowFirst, %group : i64
    %upperRow = arith.addi %lowerRow, %eight : i64
    %columnFirst = arith.muli %columnTile, %eight : i64
    %column = arith.addi %columnFirst, %group : i64
    %leftBatchStart = arith.muli %batch, %leftBatch : i64
    %leftBase = arith.addi %leftStart, %leftBatchStart : i64
    %rightBatchStart = arith.muli %batch, %rightBatch : i64
    %rightBase = arith.addi %rightStart, %rightBatchStart : i64
    %lowerOffset = arith.muli %lowerRow, %leftRow : i64
    %upperOffset = arith.muli %upperRow, %leftRow : i64
    %lowerStart = arith.addi %leftBase, %lowerOffset : i64
    %upperStart = arith.addi %leftBase, %upperOffset : i64
    %columnOffset = arith.muli %column, %rightColumn : i64
    %columnStart = arith.addi %rightBase, %columnOffset : i64
    %lowerOutside = arith.cmpi sge, %lowerRow, %rows : i64
    %upperOutside = arith.cmpi sge, %upperRow, %rows : i64
    %columnOutside = arith.cmpi sge, %column, %columns : i64
    %quadDouble = arith.muli %quad, %two : i64
    cf.br ^step(%none, %zero, %zero, %zero, %zero : i64, f32, f32, f32, f32)
  ^step(%step: i64, %sum0: f32, %sum1: f32, %sum2: f32, %sum3: f32):
    %more = arith.cmpi slt, %step, %inner : i64
    cf.cond_br %more, ^multiply, ^write
  ^multiply:
    %first = arith.addi %step, %quadDouble : i64
    %second = arith.addi %first, %one : i64
    %firstOutside = arith.cmpi sge, %first, %inner : i64
    %secondOutside = arith.cmpi sge, %second, %inner : i64
    %firstLeft = arith.muli %first, %leftInner : i64
    %secondLeft = arith.muli %second, %leftInner : i64
    %firstRight = arith.muli %first, %rightInner : i64
    %secondRight = arith.muli %second, %rightInner : i64
    %lowerFirstAt = arith.addi %lowerStart, %firstLeft : i64
    %lowerSecondAt = arith.addi %lowerStart, %secondLeft : i64
    %upperFirstAt = arith.addi %upperStart, %firstLeft : i64
    %upperSecondAt = arith.addi %upperStart, %secondLeft : i64
    %rightFirstAt = arith.addi %columnStart, %firstRight : i64
    %rightSecondAt = arith.addi %columnStart, %secondRight : i64
    %lowerFirstOut = arith.ori %lowerOutside, %firstOutside : i1
    %lowerSecondOut = arith.ori %lowerOutside, %secondOutside : i1
    %upperFirstOut = arith.ori %upperOutside, %firstOutside : i1
    %upperSecondOut = arith.ori %upperOutside, %secondOutside : i1
    %rightFirstOut = arith.ori %columnOutside, %firstOutside : i1
    %rightSecondOut = arith.ori %columnOutside, %secondOutside : i1
    %lowerFirst = func.call @strict_device_products_number(%left, %leftCount, %leftBits, %lowerFirstAt, %lowerFirstOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %lowerSecond = func.call @strict_device_products_number(%left, %leftCount, %leftBits, %lowerSecondAt, %lowerSecondOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %upperFirst = func.call @strict_device_products_number(%left, %leftCount, %leftBits, %upperFirstAt, %upperFirstOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %upperSecond = func.call @strict_device_products_number(%left, %leftCount, %leftBits, %upperSecondAt, %upperSecondOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %rightFirst = func.call @strict_device_products_number(%right, %rightCount, %rightBits, %rightFirstAt, %rightFirstOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %rightSecond = func.call @strict_device_products_number(%right, %rightCount, %rightBits, %rightSecondAt, %rightSecondOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %lowerPair = func.call @strict_device_products_pair(%lowerFirst, %lowerSecond) : (f32, f32) -> vector<2xf16>
    %upperPair = func.call @strict_device_products_pair(%upperFirst, %upperSecond) : (f32, f32) -> vector<2xf16>
    %rightPair = func.call @strict_device_products_pair(%rightFirst, %rightSecond) : (f32, f32) -> vector<2xf16>
    %sums = func.call @strict_device_halves_mma(%lowerPair, %upperPair, %rightPair, %sum0, %sum1, %sum2, %sum3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %next0 = llvm.extractvalue %sums[0] : !llvm.struct<(f32, f32, f32, f32)>
    %next1 = llvm.extractvalue %sums[1] : !llvm.struct<(f32, f32, f32, f32)>
    %next2 = llvm.extractvalue %sums[2] : !llvm.struct<(f32, f32, f32, f32)>
    %next3 = llvm.extractvalue %sums[3] : !llvm.struct<(f32, f32, f32, f32)>
    %nextStep = arith.addi %step, %eight : i64
    cf.br ^step(%nextStep, %next0, %next1, %next2, %next3 : i64, f32, f32, f32, f32)
  ^write:
    %outFirst = arith.addi %columnFirst, %quadDouble : i64
    %outSecond = arith.addi %outFirst, %one : i64
    func.call @strict_device_products_written(%result, %results, %sum0, %batch, %lowerRow, %outFirst, %rows, %columns) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %results, %sum1, %batch, %lowerRow, %outSecond, %rows, %columns) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %results, %sum2, %batch, %upperRow, %outFirst, %rows, %columns) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %results, %sum3, %batch, %upperRow, %outSecond, %rows, %columns) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_products(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64) attributes {nvvm.kernel} {
    %results = llvm.mlir.constant(64 : i64) : i64
    %shape0 = llvm.mlir.undef : vector<4xf64>
    %shapeAt0 = llvm.mlir.constant(0 : i32) : i32
    %shape1 = llvm.insertelement %batches, %shape0[%shapeAt0 : i32] : vector<4xf64>
    %shapeAt1 = llvm.mlir.constant(1 : i32) : i32
    %shape2 = llvm.insertelement %rows, %shape1[%shapeAt1 : i32] : vector<4xf64>
    %shapeAt2 = llvm.mlir.constant(2 : i32) : i32
    %shape3 = llvm.insertelement %inner, %shape2[%shapeAt2 : i32] : vector<4xf64>
    %shapeAt3 = llvm.mlir.constant(3 : i32) : i32
    %shape4 = llvm.insertelement %columnCount, %shape3[%shapeAt3 : i32] : vector<4xf64>
    %strides0 = llvm.mlir.undef : vector<8xf64>
    %strideAt0 = llvm.mlir.constant(0 : i32) : i32
    %strides1 = llvm.insertelement %leftStart, %strides0[%strideAt0 : i32] : vector<8xf64>
    %strideAt1 = llvm.mlir.constant(1 : i32) : i32
    %strides2 = llvm.insertelement %leftBatch, %strides1[%strideAt1 : i32] : vector<8xf64>
    %strideAt2 = llvm.mlir.constant(2 : i32) : i32
    %strides3 = llvm.insertelement %leftRow, %strides2[%strideAt2 : i32] : vector<8xf64>
    %strideAt3 = llvm.mlir.constant(3 : i32) : i32
    %strides4 = llvm.insertelement %leftInner, %strides3[%strideAt3 : i32] : vector<8xf64>
    %strideAt4 = llvm.mlir.constant(4 : i32) : i32
    %strides5 = llvm.insertelement %rightStart, %strides4[%strideAt4 : i32] : vector<8xf64>
    %strideAt5 = llvm.mlir.constant(5 : i32) : i32
    %strides6 = llvm.insertelement %rightBatch, %strides5[%strideAt5 : i32] : vector<8xf64>
    %strideAt6 = llvm.mlir.constant(6 : i32) : i32
    %strides7 = llvm.insertelement %rightInner, %strides6[%strideAt6 : i32] : vector<8xf64>
    %strideAt7 = llvm.mlir.constant(7 : i32) : i32
    %strides8 = llvm.insertelement %rightColumn, %strides7[%strideAt7 : i32] : vector<8xf64>
    func.call @strict_device_products_run(%result, %results, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
  llvm.func @strict_device_productss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64, %results: i64) attributes {nvvm.kernel} {
    %shape0 = llvm.mlir.undef : vector<4xf64>
    %shapeAt0 = llvm.mlir.constant(0 : i32) : i32
    %shape1 = llvm.insertelement %batches, %shape0[%shapeAt0 : i32] : vector<4xf64>
    %shapeAt1 = llvm.mlir.constant(1 : i32) : i32
    %shape2 = llvm.insertelement %rows, %shape1[%shapeAt1 : i32] : vector<4xf64>
    %shapeAt2 = llvm.mlir.constant(2 : i32) : i32
    %shape3 = llvm.insertelement %inner, %shape2[%shapeAt2 : i32] : vector<4xf64>
    %shapeAt3 = llvm.mlir.constant(3 : i32) : i32
    %shape4 = llvm.insertelement %columnCount, %shape3[%shapeAt3 : i32] : vector<4xf64>
    %strides0 = llvm.mlir.undef : vector<8xf64>
    %strideAt0 = llvm.mlir.constant(0 : i32) : i32
    %strides1 = llvm.insertelement %leftStart, %strides0[%strideAt0 : i32] : vector<8xf64>
    %strideAt1 = llvm.mlir.constant(1 : i32) : i32
    %strides2 = llvm.insertelement %leftBatch, %strides1[%strideAt1 : i32] : vector<8xf64>
    %strideAt2 = llvm.mlir.constant(2 : i32) : i32
    %strides3 = llvm.insertelement %leftRow, %strides2[%strideAt2 : i32] : vector<8xf64>
    %strideAt3 = llvm.mlir.constant(3 : i32) : i32
    %strides4 = llvm.insertelement %leftInner, %strides3[%strideAt3 : i32] : vector<8xf64>
    %strideAt4 = llvm.mlir.constant(4 : i32) : i32
    %strides5 = llvm.insertelement %rightStart, %strides4[%strideAt4 : i32] : vector<8xf64>
    %strideAt5 = llvm.mlir.constant(5 : i32) : i32
    %strides6 = llvm.insertelement %rightBatch, %strides5[%strideAt5 : i32] : vector<8xf64>
    %strideAt6 = llvm.mlir.constant(6 : i32) : i32
    %strides7 = llvm.insertelement %rightInner, %strides6[%strideAt6 : i32] : vector<8xf64>
    %strideAt7 = llvm.mlir.constant(7 : i32) : i32
    %strides8 = llvm.insertelement %rightColumn, %strides7[%strideAt7 : i32] : vector<8xf64>
    func.call @strict_device_products_run(%result, %results, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
  // `Products.Multiplied` of 64 rows and 32 columns or more on the 16 bit tensor cores: a block of 8 warps takes 128 rows
  // by 128 columns of one batch, 32 numbers along inner of both at a time in shared memory (80 bytes per row or column so
  // the rows of an ldmatrix lie in distinct banks, two stages in turn), each warp 64 rows by 32 columns
  llvm.mlir.global internal @strict_device_product_lefts() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<20480 x i8>
  llvm.mlir.global internal @strict_device_product_rights() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<20480 x i8>
  // 4 numbers of a list from at, step apart, as 16 bit floats: the first valid of them and zeros after, one vector load
  // of 32 bit floats when fast and all 4 are valid
  func.func private @strict_device_product_quad(%data: !llvm.ptr, %count: i64, %bits: i64, %at: i64, %step: i64, %valid: i64, %fast: i1) -> vector<4xf16> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %whole = arith.cmpi eq, %valid, %four : i64
    %vectored = arith.andi %whole, %fast : i1
    cf.cond_br %vectored, ^vector, ^single
  ^vector:
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brainVector, ^singleVector
  ^brainVector:
    %brainSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainSlot invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fromBrains = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    %brainHalves = arith.truncf %fromBrains : vector<4xf32> to vector<4xf16>
    return %brainHalves : vector<4xf16>
  ^singleVector:
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %floats = llvm.load %slot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %halves = arith.truncf %floats : vector<4xf32> to vector<4xf16>
    return %halves : vector<4xf16>
  ^single:
    %secondAt = arith.addi %at, %step : i64
    %thirdAt = arith.addi %secondAt, %step : i64
    %fourthAt = arith.addi %thirdAt, %step : i64
    %firstOut = arith.cmpi sle, %valid, %none : i64
    %secondOut = arith.cmpi sle, %valid, %one : i64
    %thirdOut = arith.cmpi sle, %valid, %two : i64
    %fourthOut = arith.cmpi sle, %valid, %three : i64
    %first = func.call @strict_device_products_number(%data, %count, %bits, %at, %firstOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %second = func.call @strict_device_products_number(%data, %count, %bits, %secondAt, %secondOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %third = func.call @strict_device_products_number(%data, %count, %bits, %thirdAt, %thirdOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %fourth = func.call @strict_device_products_number(%data, %count, %bits, %fourthAt, %fourthOut) : (!llvm.ptr, i64, i64, i64, i1) -> f32
    %empty = llvm.mlir.undef : vector<4xf32>
    %with0 = llvm.insertelement %first, %empty[%c0 : i32] : vector<4xf32>
    %with1 = llvm.insertelement %second, %with0[%c1 : i32] : vector<4xf32>
    %with2 = llvm.insertelement %third, %with1[%c2 : i32] : vector<4xf32>
    %with3 = llvm.insertelement %fourth, %with2[%c3 : i32] : vector<4xf32>
    %singles = arith.truncf %with3 : vector<4xf32> to vector<4xf16>
    return %singles : vector<4xf16>
  }
  // the 16 numbers a thread copies of 128 rows (or columns) from first by the 32 numbers of step along inner of one
  // operand (zeros past rows and inner): 4 times 4 numbers next to each other along inner, along the rows when byRows
  func.func private @strict_device_product_fetch(%data: !llvm.ptr, %count: i64, %bits: i64, %base: i64, %rowStride: i64, %innerStride: i64, %rows: i64, %first: i64, %inner: i64, %step: i64, %thread: i32, %byRows: i1, %fast: i1) -> vector<16xf16> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %thread64 = arith.extui %thread : i32 to i64
    %innerFirst = arith.shli %step, %five : i64
    %empty = llvm.mlir.undef : vector<16xf16>
    cf.br ^part(%none, %empty : i64, vector<16xf16>)
  ^part(%part: i64, %halves: vector<16xf16>):
    %more = arith.cmpi slt, %part, %four : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %eight : i64
    %linear = arith.addi %thread64, %partStart : i64
    %innerRow = arith.shrui %linear, %three : i64
    %innerQuad = arith.andi %linear, %seven : i64
    %innerOffset = arith.shli %innerQuad, %two : i64
    %rowsInner = arith.shrui %linear, %five : i64
    %rowsQuad = arith.andi %linear, %thirtyOne : i64
    %rowsOffset = arith.shli %rowsQuad, %two : i64
    %localRow = arith.select %byRows, %rowsOffset, %innerRow : i64
    %localInner = arith.select %byRows, %rowsInner, %innerOffset : i64
    %row = arith.addi %first, %localRow : i64
    %index = arith.addi %innerFirst, %localInner : i64
    %rowAt = arith.muli %row, %rowStride : i64
    %innerAt = arith.muli %index, %innerStride : i64
    %rowBase = arith.addi %base, %rowAt : i64
    %at = arith.addi %rowBase, %innerAt : i64
    %rowsLeft = arith.subi %rows, %row : i64
    %innerLeft = arith.subi %inner, %index : i64
    %rowInside = arith.cmpi slt, %row, %rows : i64
    %innerInside = arith.cmpi slt, %index, %inner : i64
    %along = arith.select %byRows, %rowsLeft, %innerLeft : i64
    %across = arith.select %byRows, %innerInside, %rowInside : i1
    %capped = arith.minsi %along, %four : i64
    %positive = arith.maxsi %capped, %none : i64
    %valid = arith.select %across, %positive, %none : i64
    %stepSize = arith.select %byRows, %rowStride, %innerStride : i64
    %quad = func.call @strict_device_product_quad(%data, %count, %bits, %at, %stepSize, %valid, %fast) : (!llvm.ptr, i64, i64, i64, i64, i64, i1) -> vector<4xf16>
    %part32 = arith.trunci %part : i64 to i32
    %slot0 = arith.shli %part32, %c2 : i32
    %slot1 = arith.addi %slot0, %c1 : i32
    %slot2 = arith.addi %slot0, %c2 : i32
    %slot3 = arith.addi %slot0, %c3 : i32
    %half0 = llvm.extractelement %quad[%c0 : i32] : vector<4xf16>
    %half1 = llvm.extractelement %quad[%c1 : i32] : vector<4xf16>
    %half2 = llvm.extractelement %quad[%c2 : i32] : vector<4xf16>
    %half3 = llvm.extractelement %quad[%c3 : i32] : vector<4xf16>
    %with0 = llvm.insertelement %half0, %halves[%slot0 : i32] : vector<16xf16>
    %with1 = llvm.insertelement %half1, %with0[%slot1 : i32] : vector<16xf16>
    %with2 = llvm.insertelement %half2, %with1[%slot2 : i32] : vector<16xf16>
    %with3 = llvm.insertelement %half3, %with2[%slot3 : i32] : vector<16xf16>
    %nextPart = arith.addi %part, %one : i64
    cf.br ^part(%nextPart, %with3 : i64, vector<16xf16>)
  ^done:
    return %halves : vector<16xf16>
  }
  // the numbers strict_device_product_fetch loaded into stage buffer (0 or 1) of array: row by row with 80 bytes each,
  // 4 numbers along inner at once, one by one when they lie along the rows
  func.func private @strict_device_product_put(%halves: vector<16xf16>, %array: !llvm.ptr<3>, %buffer: i32, %thread: i32, %byRows: i1) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %bufferStart = arith.muli %buffer, %c10240 : i32
    cf.br ^part(%c0 : i32)
  ^part(%part: i32):
    %more = arith.cmpi slt, %part, %c4 : i32
    cf.cond_br %more, ^body, ^done
  ^body:
    %partStart = arith.shli %part, %c8 : i32
    %linear = arith.addi %thread, %partStart : i32
    %slot0 = arith.shli %part, %c2 : i32
    %slot1 = arith.addi %slot0, %c1 : i32
    %slot2 = arith.addi %slot0, %c2 : i32
    %slot3 = arith.addi %slot0, %c3 : i32
    %half0 = llvm.extractelement %halves[%slot0 : i32] : vector<16xf16>
    %half1 = llvm.extractelement %halves[%slot1 : i32] : vector<16xf16>
    %half2 = llvm.extractelement %halves[%slot2 : i32] : vector<16xf16>
    %half3 = llvm.extractelement %halves[%slot3 : i32] : vector<16xf16>
    %nextPart = arith.addi %part, %c1 : i32
    cf.cond_br %byRows, ^alongRows, ^alongInner
  ^alongInner:
    %row = arith.shrui %linear, %c3 : i32
    %quad = arith.andi %linear, %c7 : i32
    %rowStart = arith.muli %row, %c80 : i32
    %quadBytes = arith.shli %quad, %c3 : i32
    %rowOffset = arith.addi %rowStart, %quadBytes : i32
    %offset = arith.addi %bufferStart, %rowOffset : i32
    %offset64 = arith.extui %offset : i32 to i64
    %slot = llvm.getelementptr %array[%offset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %empty = llvm.mlir.undef : vector<4xf16>
    %with0 = llvm.insertelement %half0, %empty[%c0 : i32] : vector<4xf16>
    %with1 = llvm.insertelement %half1, %with0[%c1 : i32] : vector<4xf16>
    %with2 = llvm.insertelement %half2, %with1[%c2 : i32] : vector<4xf16>
    %with3 = llvm.insertelement %half3, %with2[%c3 : i32] : vector<4xf16>
    llvm.store %with3, %slot {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
    llvm.br ^part(%nextPart : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^alongRows:
    %index = arith.shrui %linear, %c5 : i32
    %rowQuad = arith.andi %linear, %c31 : i32
    %firstRow = arith.shli %rowQuad, %c2 : i32
    %firstStart = arith.muli %firstRow, %c80 : i32
    %indexBytes = arith.shli %index, %c1 : i32
    %firstOffset = arith.addi %firstStart, %indexBytes : i32
    %firstAt = arith.addi %bufferStart, %firstOffset : i32
    %secondAt = arith.addi %firstAt, %c80 : i32
    %thirdAt = arith.addi %secondAt, %c80 : i32
    %fourthAt = arith.addi %thirdAt, %c80 : i32
    %firstAt64 = arith.extui %firstAt : i32 to i64
    %secondAt64 = arith.extui %secondAt : i32 to i64
    %thirdAt64 = arith.extui %thirdAt : i32 to i64
    %fourthAt64 = arith.extui %fourthAt : i32 to i64
    %firstSlot = llvm.getelementptr %array[%firstAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %secondSlot = llvm.getelementptr %array[%secondAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %thirdSlot = llvm.getelementptr %array[%thirdAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %fourthSlot = llvm.getelementptr %array[%fourthAt64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    llvm.store %half0, %firstSlot : f16, !llvm.ptr<3>
    llvm.store %half1, %secondSlot : f16, !llvm.ptr<3>
    llvm.store %half2, %thirdSlot : f16, !llvm.ptr<3>
    llvm.store %half3, %fourthSlot : f16, !llvm.ptr<3>
    llvm.br ^part(%nextPart : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  // m16n8k16 of 16 bit floats, or of bfloat16 when brain (natively from sm_80 on, before and on AMD converted to 16 bit
  // floats first)
  func.func private @strict_device_product_dot(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32, %brain: i1) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %arch = func.call @strict_device_arch() : () -> i32
    %c800 = arith.constant 800 : i32
    %isWide = arith.cmpi sge, %arch, %c800 : i32
    %native = arith.andi %brain, %isWide : i1
    cf.cond_br %native, ^brains, ^converted
  ^brains:
    %product = func.call @strict_device_attention_mma16(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3, %brain) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, i1) -> !llvm.struct<(f32, f32, f32, f32)>
    return %product : !llvm.struct<(f32, f32, f32, f32)>
  ^converted:
    %h0 = func.call @strict_device_product_halves(%a0, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %h1 = func.call @strict_device_product_halves(%a1, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %h2 = func.call @strict_device_product_halves(%a2, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %h3 = func.call @strict_device_product_halves(%a3, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %k0 = func.call @strict_device_product_halves(%b0, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %k1 = func.call @strict_device_product_halves(%b1, %brain) : (vector<2xf16>, i1) -> vector<2xf16>
    %halves = func.call @strict_device_halves_mma16(%h0, %h1, %h2, %h3, %k0, %k1, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %halves : !llvm.struct<(f32, f32, f32, f32)>
  }
  // the two numbers of a pair as 16 bit floats, from bfloat16 bits when brain
  func.func private @strict_device_product_halves(%pair: vector<2xf16>, %brain: i1) -> vector<2xf16> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %brains = llvm.bitcast %pair : vector<2xf16> to vector<2xbf16>
    %wide = arith.extf %brains : vector<2xbf16> to vector<2xf32>
    %narrow = arith.truncf %wide : vector<2xf32> to vector<2xf16>
    %chosen = arith.select %brain, %narrow, %pair : vector<2xf16>
    return %chosen : vector<2xf16>
  }
  // the 4 sums from slot plus the 16 x 8 by 8 products of 16 numbers along inner (a: rows g and g + 8 of numbers 0 to 7,
  // then of 8 to 15, b: 0 to 7 and 8 to 15 of column g)
  func.func private @strict_device_product_mma(%sums: vector<64xf32>, %slot: i32, %a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %brain: i1) -> vector<64xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
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
    %h0 = llvm.bitcast %a0 : i32 to vector<2xf16>
    %h1 = llvm.bitcast %a1 : i32 to vector<2xf16>
    %h2 = llvm.bitcast %a2 : i32 to vector<2xf16>
    %h3 = llvm.bitcast %a3 : i32 to vector<2xf16>
    %k0 = llvm.bitcast %b0 : i32 to vector<2xf16>
    %k1 = llvm.bitcast %b1 : i32 to vector<2xf16>
    %dot = func.call @strict_device_product_dot(%h0, %h1, %h2, %h3, %k0, %k1, %old0, %old1, %old2, %old3, %brain) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, i1) -> !llvm.struct<(f32, f32, f32, f32)>
    %new0 = llvm.extractvalue %dot[0] : !llvm.struct<(f32, f32, f32, f32)>
    %new1 = llvm.extractvalue %dot[1] : !llvm.struct<(f32, f32, f32, f32)>
    %new2 = llvm.extractvalue %dot[2] : !llvm.struct<(f32, f32, f32, f32)>
    %new3 = llvm.extractvalue %dot[3] : !llvm.struct<(f32, f32, f32, f32)>
    %with0 = llvm.insertelement %new0, %sums[%slot : i32] : vector<64xf32>
    %with1 = llvm.insertelement %new1, %with0[%slot1 : i32] : vector<64xf32>
    %with2 = llvm.insertelement %new2, %with1[%slot2 : i32] : vector<64xf32>
    %with3 = llvm.insertelement %new3, %with2[%slot3 : i32] : vector<64xf32>
    return %with3 : vector<64xf32>
  }
  // 16 numbers along inner (half 0 or 1 of a stage) of stage buffer for the 64 rows and 32 columns of a warp: an
  // ldmatrix.x4 per 16 rows and per 16 columns, 16 mma; sums holds the 4 results (rows g and g + 8 of the lane by columns
  // 2 * (lane % 4) and the next) of the 4 x 4 tiles of 16 rows by 8 columns, row tile first
  func.func private @strict_device_product_half(%sums: vector<64xf32>, %buffer: i32, %half: i32, %warpRow: i32, %warpColumn: i32, %lane: i32, %brain: i1) -> vector<64xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %lefts = llvm.mlir.addressof @strict_device_product_lefts : !llvm.ptr<3>
    %rights = llvm.mlir.addressof @strict_device_product_rights : !llvm.ptr<3>
    %bufferStart = arith.muli %buffer, %c10240 : i32
    %matrix = arith.shrui %lane, %c3 : i32
    %matrixRow = arith.andi %lane, %c7 : i32
    %matrixLow = arith.andi %matrix, %c1 : i32
    %matrixHigh = arith.shrui %matrix, %c1 : i32
    %halfBytes = arith.shli %half, %c5 : i32
    %lowRows = arith.shli %matrixLow, %c3 : i32
    %highRows = arith.shli %matrixHigh, %c3 : i32
    %lowBytes = arith.shli %matrixLow, %c4 : i32
    %highBytes = arith.shli %matrixHigh, %c4 : i32
    %leftInTile = arith.addi %matrixRow, %lowRows : i32
    %leftColumn = arith.addi %halfBytes, %highBytes : i32
    %rightInTile = arith.addi %matrixRow, %highRows : i32
    %rightColumn = arith.addi %halfBytes, %lowBytes : i32
    %firstRight = arith.addi %warpColumn, %rightInTile : i32
    %secondRight = arith.addi %firstRight, %c16 : i32
    %firstRightStart = arith.muli %firstRight, %c80 : i32
    %secondRightStart = arith.muli %secondRight, %c80 : i32
    %firstRightAt = arith.addi %firstRightStart, %rightColumn : i32
    %secondRightAt = arith.addi %secondRightStart, %rightColumn : i32
    %firstRightOffset = arith.addi %bufferStart, %firstRightAt : i32
    %secondRightOffset = arith.addi %bufferStart, %secondRightAt : i32
    %firstRightOffset64 = arith.extui %firstRightOffset : i32 to i64
    %secondRightOffset64 = arith.extui %secondRightOffset : i32 to i64
    %firstRightSlot = llvm.getelementptr %rights[%firstRightOffset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %secondRightSlot = llvm.getelementptr %rights[%secondRightOffset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %firstColumns = func.call @strict_device_matrices(%firstRightSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %secondColumns = func.call @strict_device_matrices(%secondRightSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %r0 = llvm.extractvalue %firstColumns[0] : !llvm.struct<(i32, i32, i32, i32)>
    %r1 = llvm.extractvalue %firstColumns[1] : !llvm.struct<(i32, i32, i32, i32)>
    %r2 = llvm.extractvalue %firstColumns[2] : !llvm.struct<(i32, i32, i32, i32)>
    %r3 = llvm.extractvalue %firstColumns[3] : !llvm.struct<(i32, i32, i32, i32)>
    %r4 = llvm.extractvalue %secondColumns[0] : !llvm.struct<(i32, i32, i32, i32)>
    %r5 = llvm.extractvalue %secondColumns[1] : !llvm.struct<(i32, i32, i32, i32)>
    %r6 = llvm.extractvalue %secondColumns[2] : !llvm.struct<(i32, i32, i32, i32)>
    %r7 = llvm.extractvalue %secondColumns[3] : !llvm.struct<(i32, i32, i32, i32)>
    %firstQuad = func.call @strict_device_words(%r0, %r1, %r2, %r3) : (i32, i32, i32, i32) -> vector<4xi32>
    %secondQuad = func.call @strict_device_words(%r4, %r5, %r6, %r7) : (i32, i32, i32, i32) -> vector<4xi32>
    %columnOperands = llvm.shufflevector %firstQuad, %secondQuad [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xi32>
    cf.br ^rowTile(%c0, %sums : i32, vector<64xf32>)
  ^rowTile(%rowTileIndex: i32, %rowSums: vector<64xf32>):
    %moreRowTiles = arith.cmpi slt, %rowTileIndex, %c4 : i32
    cf.cond_br %moreRowTiles, ^rowTileBody, ^done
  ^rowTileBody:
    %tileRows = arith.shli %rowTileIndex, %c4 : i32
    %tileRow = arith.addi %warpRow, %tileRows : i32
    %laneRow = arith.addi %tileRow, %leftInTile : i32
    %rowStart = arith.muli %laneRow, %c80 : i32
    %rowAt = arith.addi %rowStart, %leftColumn : i32
    %rowOffset = arith.addi %bufferStart, %rowAt : i32
    %rowOffset64 = arith.extui %rowOffset : i32 to i64
    %rowSlot = llvm.getelementptr %lefts[%rowOffset64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %rowOperands = func.call @strict_device_matrices(%rowSlot) : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    %a0 = llvm.extractvalue %rowOperands[0] : !llvm.struct<(i32, i32, i32, i32)>
    %a1 = llvm.extractvalue %rowOperands[1] : !llvm.struct<(i32, i32, i32, i32)>
    %a2 = llvm.extractvalue %rowOperands[2] : !llvm.struct<(i32, i32, i32, i32)>
    %a3 = llvm.extractvalue %rowOperands[3] : !llvm.struct<(i32, i32, i32, i32)>
    %tileSlots = arith.shli %rowTileIndex, %c4 : i32
    cf.br ^columnTile(%c0, %rowSums : i32, vector<64xf32>)
  ^columnTile(%columnTileIndex: i32, %columnSums: vector<64xf32>):
    %moreColumnTiles = arith.cmpi slt, %columnTileIndex, %c4 : i32
    cf.cond_br %moreColumnTiles, ^columnTileBody, ^rowTileNext
  ^columnTileBody:
    %lowOperand = arith.shli %columnTileIndex, %c1 : i32
    %highOperand = arith.addi %lowOperand, %c1 : i32
    %b0 = llvm.extractelement %columnOperands[%lowOperand : i32] : vector<8xi32>
    %b1 = llvm.extractelement %columnOperands[%highOperand : i32] : vector<8xi32>
    %columnSlots = arith.shli %columnTileIndex, %c2 : i32
    %slot = arith.addi %tileSlots, %columnSlots : i32
    %nextSums = func.call @strict_device_product_mma(%columnSums, %slot, %a0, %a1, %a2, %a3, %b0, %b1, %brain) : (vector<64xf32>, i32, i32, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %columnTileNext = arith.addi %columnTileIndex, %c1 : i32
    llvm.br ^columnTile(%columnTileNext, %nextSums : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^rowTileNext:
    %rowTileFollowing = arith.addi %rowTileIndex, %c1 : i32
    llvm.br ^rowTile(%rowTileFollowing, %columnSums : i32, vector<64xf32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %rowSums : vector<64xf32>
  }
  // true when the numbers of an operand can be read as 32 bit floats 4 at a time: their stride along inner is 1 (along
  // the rows when byRows), the other stride, the start and the data are aligned to 4 numbers
  func.func private @strict_device_product_fast(%data: !llvm.ptr, %bits: i64, %base: i64, %rowStride: i64, %innerStride: i64, %byRows: i1) -> i1 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %brainBits = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %along = arith.select %byRows, %rowStride, %innerStride : i64
    %across = arith.select %byRows, %innerStride, %rowStride : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    %isVector = arith.ori %isSingle, %isBrain : i1
    %isNext = arith.cmpi eq, %along, %one : i64
    %acrossRest = arith.andi %across, %three : i64
    %baseRest = arith.andi %base, %three : i64
    %address = llvm.ptrtoint %data : !llvm.ptr to i64
    %addressMask = arith.select %isBrain, %seven, %fifteen : i64
    %addressRest = arith.andi %address, %addressMask : i64
    %rests = arith.ori %acrossRest, %baseRest : i64
    %allRests = arith.ori %rests, %addressRest : i64
    %aligned = arith.cmpi eq, %allRests, %none : i64
    %singleNext = arith.andi %isVector, %isNext : i1
    %fast = arith.andi %singleNext, %aligned : i1
    return %fast : i1
  }
  // 4 aligned vector<2xi32> loads (4 bfloat16 numbers each) along inner without conversion so global loads overlap with mma
  func.func private @strict_device_product_load_brains(%data: !llvm.ptr, %base: i64, %rowStride: i64, %rows: i64, %first: i64, %step: i64, %thread: i32) -> vector<8xi32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %thirtyTwo = arith.constant 32 : i64
    %sixtyFour = arith.constant 64 : i64
    %ninetySix = arith.constant 96 : i64
    %thread64 = arith.extui %thread : i32 to i64
    %localRow0 = arith.shrui %thread64, %three : i64
    %quad = arith.andi %thread64, %seven : i64
    %quadCol = arith.shli %quad, %two : i64
    %innerFirst = arith.shli %step, %five : i64
    %col = arith.addi %innerFirst, %quadCol : i64
    %baseCol = arith.addi %base, %col : i64
    %row0 = arith.addi %first, %localRow0 : i64
    %row1 = arith.addi %row0, %thirtyTwo : i64
    %row2 = arith.addi %row0, %sixtyFour : i64
    %row3 = arith.addi %row0, %ninetySix : i64
    %in0 = arith.cmpi slt, %row0, %rows : i64
    %in1 = arith.cmpi slt, %row1, %rows : i64
    %in2 = arith.cmpi slt, %row2, %rows : i64
    %in3 = arith.cmpi slt, %row3, %rows : i64
    %safe0 = arith.select %in0, %row0, %none : i64
    %safe1 = arith.select %in1, %row1, %none : i64
    %safe2 = arith.select %in2, %row2, %none : i64
    %safe3 = arith.select %in3, %row3, %none : i64
    %off0 = arith.muli %safe0, %rowStride : i64
    %off1 = arith.muli %safe1, %rowStride : i64
    %off2 = arith.muli %safe2, %rowStride : i64
    %off3 = arith.muli %safe3, %rowStride : i64
    %at0 = arith.addi %baseCol, %off0 : i64
    %at1 = arith.addi %baseCol, %off1 : i64
    %at2 = arith.addi %baseCol, %off2 : i64
    %at3 = arith.addi %baseCol, %off3 : i64
    %slot0 = llvm.getelementptr %data[%at0] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %slot1 = llvm.getelementptr %data[%at1] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %slot2 = llvm.getelementptr %data[%at2] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %slot3 = llvm.getelementptr %data[%at3] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %v0 = llvm.load %slot0 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %v1 = llvm.load %slot1 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %v2 = llvm.load %slot2 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %v3 = llvm.load %slot3 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<2xi32>
    %v01 = llvm.shufflevector %v0, %v1 [0, 1, 2, 3] : vector<2xi32>
    %v23 = llvm.shufflevector %v2, %v3 [0, 1, 2, 3] : vector<2xi32>
    %all = llvm.shufflevector %v01, %v23 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xi32>
    return %all : vector<8xi32>
  }
  // converts raw bfloat16 vector<8xi32> into 4 masked vector<4xf16> quads and stores them into stage buffer of array
  func.func private @strict_device_product_put_brains(%raw: vector<8xi32>, %rows: i64, %first: i64, %array: !llvm.ptr<3>, %buffer: i32, %thread: i32) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c80 = llvm.mlir.constant(80 : i32) : i32
    %c2560 = llvm.mlir.constant(2560 : i32) : i32
    %c5120 = llvm.mlir.constant(5120 : i32) : i32
    %c7680 = llvm.mlir.constant(7680 : i32) : i32
    %c10240 = llvm.mlir.constant(10240 : i32) : i32
    %thirtyTwo = arith.constant 32 : i64
    %sixtyFour = arith.constant 64 : i64
    %ninetySix = arith.constant 96 : i64
    %zero4 = arith.constant dense<0.0> : vector<4xf16>
    %localRow = arith.shrui %thread, %c3 : i32
    %quad = arith.andi %thread, %c7 : i32
    %localRow64 = arith.extui %localRow : i32 to i64
    %row0 = arith.addi %first, %localRow64 : i64
    %row1 = arith.addi %row0, %thirtyTwo : i64
    %row2 = arith.addi %row0, %sixtyFour : i64
    %row3 = arith.addi %row0, %ninetySix : i64
    %in0 = arith.cmpi slt, %row0, %rows : i64
    %in1 = arith.cmpi slt, %row1, %rows : i64
    %in2 = arith.cmpi slt, %row2, %rows : i64
    %in3 = arith.cmpi slt, %row3, %rows : i64
    %v0 = llvm.shufflevector %raw, %raw [0, 1] : vector<8xi32>
    %v1 = llvm.shufflevector %raw, %raw [2, 3] : vector<8xi32>
    %v2 = llvm.shufflevector %raw, %raw [4, 5] : vector<8xi32>
    %v3 = llvm.shufflevector %raw, %raw [6, 7] : vector<8xi32>
    %b0 = llvm.bitcast %v0 : vector<2xi32> to vector<4xbf16>
    %b1 = llvm.bitcast %v1 : vector<2xi32> to vector<4xbf16>
    %b2 = llvm.bitcast %v2 : vector<2xi32> to vector<4xbf16>
    %b3 = llvm.bitcast %v3 : vector<2xi32> to vector<4xbf16>
    %f0 = arith.extf %b0 : vector<4xbf16> to vector<4xf32>
    %f1 = arith.extf %b1 : vector<4xbf16> to vector<4xf32>
    %f2 = arith.extf %b2 : vector<4xbf16> to vector<4xf32>
    %f3 = arith.extf %b3 : vector<4xbf16> to vector<4xf32>
    %h0 = arith.truncf %f0 : vector<4xf32> to vector<4xf16>
    %h1 = arith.truncf %f1 : vector<4xf32> to vector<4xf16>
    %h2 = arith.truncf %f2 : vector<4xf32> to vector<4xf16>
    %h3 = arith.truncf %f3 : vector<4xf32> to vector<4xf16>
    %m0 = arith.select %in0, %h0, %zero4 : vector<4xf16>
    %m1 = arith.select %in1, %h1, %zero4 : vector<4xf16>
    %m2 = arith.select %in2, %h2, %zero4 : vector<4xf16>
    %m3 = arith.select %in3, %h3, %zero4 : vector<4xf16>
    %bufferStart = arith.muli %buffer, %c10240 : i32
    %rowBytes = arith.muli %localRow, %c80 : i32
    %quadBytes = arith.shli %quad, %c3 : i32
    %baseOffset = arith.addi %bufferStart, %rowBytes : i32
    %off0 = arith.addi %baseOffset, %quadBytes : i32
    %off1 = arith.addi %off0, %c2560 : i32
    %off2 = arith.addi %off0, %c5120 : i32
    %off3 = arith.addi %off0, %c7680 : i32
    %off0_64 = arith.extui %off0 : i32 to i64
    %off1_64 = arith.extui %off1 : i32 to i64
    %off2_64 = arith.extui %off2 : i32 to i64
    %off3_64 = arith.extui %off3 : i32 to i64
    %s0 = llvm.getelementptr %array[%off0_64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %s1 = llvm.getelementptr %array[%off1_64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %s2 = llvm.getelementptr %array[%off2_64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %s3 = llvm.getelementptr %array[%off3_64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    llvm.store %m0, %s0 {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
    llvm.store %m1, %s1 {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
    llvm.store %m2, %s2 {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
    llvm.store %m3, %s3 {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
    return
  }
  // 4 back-to-back aligned vector loads (bfloat16 or f32) along inner in a single basic block
  func.func private @strict_device_product_fetch_fast(%data: !llvm.ptr, %bits: i64, %base: i64, %rowStride: i64, %rows: i64, %first: i64, %step: i64, %thread: i32) -> vector<16xf16> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64
    %brainBits = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %sixtyFour = arith.constant 64 : i64
    %ninetySix = arith.constant 96 : i64
    %zero4 = arith.constant dense<0.0> : vector<4xf16>
    %thread64 = arith.extui %thread : i32 to i64
    %localRow0 = arith.shrui %thread64, %three : i64
    %quad = arith.andi %thread64, %seven : i64
    %quadCol = arith.shli %quad, %two : i64
    %innerFirst = arith.shli %step, %five : i64
    %col = arith.addi %innerFirst, %quadCol : i64
    %baseCol = arith.addi %base, %col : i64
    %row0 = arith.addi %first, %localRow0 : i64
    %row1 = arith.addi %row0, %thirtyTwo : i64
    %row2 = arith.addi %row0, %sixtyFour : i64
    %row3 = arith.addi %row0, %ninetySix : i64
    %in0 = arith.cmpi slt, %row0, %rows : i64
    %in1 = arith.cmpi slt, %row1, %rows : i64
    %in2 = arith.cmpi slt, %row2, %rows : i64
    %in3 = arith.cmpi slt, %row3, %rows : i64
    %safe0 = arith.select %in0, %row0, %none : i64
    %safe1 = arith.select %in1, %row1, %none : i64
    %safe2 = arith.select %in2, %row2, %none : i64
    %safe3 = arith.select %in3, %row3, %none : i64
    %off0 = arith.muli %safe0, %rowStride : i64
    %off1 = arith.muli %safe1, %rowStride : i64
    %off2 = arith.muli %safe2, %rowStride : i64
    %off3 = arith.muli %safe3, %rowStride : i64
    %at0 = arith.addi %baseCol, %off0 : i64
    %at1 = arith.addi %baseCol, %off1 : i64
    %at2 = arith.addi %baseCol, %off2 : i64
    %at3 = arith.addi %baseCol, %off3 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^single
  ^brain:
    %bs0 = llvm.getelementptr %data[%at0] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bs1 = llvm.getelementptr %data[%at1] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bs2 = llvm.getelementptr %data[%at2] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bs3 = llvm.getelementptr %data[%at3] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bv0 = llvm.load %bs0 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %bv1 = llvm.load %bs1 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %bv2 = llvm.load %bs2 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %bv3 = llvm.load %bs3 invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %bf0 = arith.extf %bv0 : vector<4xbf16> to vector<4xf32>
    %bf1 = arith.extf %bv1 : vector<4xbf16> to vector<4xf32>
    %bf2 = arith.extf %bv2 : vector<4xbf16> to vector<4xf32>
    %bf3 = arith.extf %bv3 : vector<4xbf16> to vector<4xf32>
    %bh0 = arith.truncf %bf0 : vector<4xf32> to vector<4xf16>
    %bh1 = arith.truncf %bf1 : vector<4xf32> to vector<4xf16>
    %bh2 = arith.truncf %bf2 : vector<4xf32> to vector<4xf16>
    %bh3 = arith.truncf %bf3 : vector<4xf32> to vector<4xf16>
    cf.br ^mask(%bh0, %bh1, %bh2, %bh3 : vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>)
  ^single:
    %ss0 = llvm.getelementptr %data[%at0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %ss1 = llvm.getelementptr %data[%at1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %ss2 = llvm.getelementptr %data[%at2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %ss3 = llvm.getelementptr %data[%at3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sv0 = llvm.load %ss0 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sv1 = llvm.load %ss1 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sv2 = llvm.load %ss2 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sv3 = llvm.load %ss3 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sh0 = arith.truncf %sv0 : vector<4xf32> to vector<4xf16>
    %sh1 = arith.truncf %sv1 : vector<4xf32> to vector<4xf16>
    %sh2 = arith.truncf %sv2 : vector<4xf32> to vector<4xf16>
    %sh3 = arith.truncf %sv3 : vector<4xf32> to vector<4xf16>
    cf.br ^mask(%sh0, %sh1, %sh2, %sh3 : vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>)
  ^mask(%q0: vector<4xf16>, %q1: vector<4xf16>, %q2: vector<4xf16>, %q3: vector<4xf16>):
    %m0 = arith.select %in0, %q0, %zero4 : vector<4xf16>
    %m1 = arith.select %in1, %q1, %zero4 : vector<4xf16>
    %m2 = arith.select %in2, %q2, %zero4 : vector<4xf16>
    %m3 = arith.select %in3, %q3, %zero4 : vector<4xf16>
    %m01 = llvm.shufflevector %m0, %m1 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf16>
    %m23 = llvm.shufflevector %m2, %m3 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf16>
    %all = llvm.shufflevector %m01, %m23 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf16>
    return %all : vector<16xf16>
  }
  // the block computes the tile batch, rows from rowFirst and columns from columnFirst over all stages of 32 numbers
  // along inner (the next stage loaded into registers while one computes), each warp its 64 rows by 32 columns
  func.func private @strict_device_product_stages(%left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %leftBase: i64, %leftRow: i64, %leftInner: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %rightBase: i64, %rightColumn: i64, %rightInner: i64, %rows: i64, %columns: i64, %inner: i64, %rowFirst: i64, %columnFirst: i64, %thread: i32) -> vector<64xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %halvesOnly = arith.constant false
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %false = arith.constant false
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %brainBits = arith.constant 17 : i64
    %thirtyOne = arith.constant 31 : i64
    %zeros = arith.constant dense<0.0> : vector<64xf32>
    %lefts = llvm.mlir.addressof @strict_device_product_lefts : !llvm.ptr<3>
    %rights = llvm.mlir.addressof @strict_device_product_rights : !llvm.ptr<3>
    %leftNext = arith.cmpi eq, %leftInner, %one : i64
    %leftRowNext = arith.cmpi eq, %leftRow, %one : i64
    %leftNotNext = arith.xori %leftNext, %leftRowNext : i1
    %leftByRows = arith.andi %leftNotNext, %leftRowNext : i1
    %rightNext = arith.cmpi eq, %rightInner, %one : i64
    %rightColumnNext = arith.cmpi eq, %rightColumn, %one : i64
    %rightNotNext = arith.xori %rightNext, %rightColumnNext : i1
    %rightByRows = arith.andi %rightNotNext, %rightColumnNext : i1
    %leftFast = func.call @strict_device_product_fast(%left, %leftBits, %leftBase, %leftRow, %leftInner, %leftByRows) : (!llvm.ptr, i64, i64, i64, i64, i1) -> i1
    %rightFast = func.call @strict_device_product_fast(%right, %rightBits, %rightBase, %rightColumn, %rightInner, %rightByRows) : (!llvm.ptr, i64, i64, i64, i64, i1) -> i1
    %innerUp = arith.addi %inner, %thirtyOne : i64
    %stages = arith.shrui %innerUp, %five : i64
    %lastStage = arith.subi %stages, %one : i64
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpColumnIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.shli %warpRowIndex, %c6 : i32
    %warpColumn = arith.shli %warpColumnIndex, %c5 : i32
    %bothFast = arith.andi %leftFast, %rightFast : i1
    %anyByRows = arith.ori %leftByRows, %rightByRows : i1
    %neitherByRows = arith.cmpi eq, %anyByRows, %false : i1
    %innerRest = arith.andi %inner, %thirtyOne : i64
    %wholeStages = arith.cmpi eq, %innerRest, %none : i64
    %someStages = arith.cmpi sgt, %stages, %none : i64
    %hasRows = arith.cmpi sgt, %rows, %none : i64
    %hasColumns = arith.cmpi sgt, %columns, %none : i64
    %hasShape = arith.andi %hasRows, %hasColumns : i1
    %validStages = arith.andi %wholeStages, %someStages : i1
    %fastFlags = arith.andi %bothFast, %neitherByRows : i1
    %fastShape = arith.andi %hasShape, %validStages : i1
    %fastStream = arith.andi %fastFlags, %fastShape : i1
    cf.cond_br %fastStream, ^checkBrains, ^generalInit
  ^checkBrains:
    %leftBrain = arith.cmpi eq, %leftBits, %brainBits : i64
    %rightBrain = arith.cmpi eq, %rightBits, %brainBits : i64
    %bothBrains = arith.andi %leftBrain, %rightBrain : i1
    cf.cond_br %bothBrains, ^brainInit, ^fastInit
  ^brainInit:
    %rawFirstL = func.call @strict_device_product_load_brains(%left, %leftBase, %leftRow, %rows, %rowFirst, %none, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i32) -> vector<8xi32>
    %rawFirstR = func.call @strict_device_product_load_brains(%right, %rightBase, %rightColumn, %columns, %columnFirst, %none, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i32) -> vector<8xi32>
    func.call @strict_device_product_put_brains(%rawFirstL, %rows, %rowFirst, %lefts, %c0, %thread) : (vector<8xi32>, i64, i64, !llvm.ptr<3>, i32, i32) -> ()
    func.call @strict_device_product_put_brains(%rawFirstR, %columns, %columnFirst, %rights, %c0, %thread) : (vector<8xi32>, i64, i64, !llvm.ptr<3>, i32, i32) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^brainStage(%none, %zeros : i64, vector<64xf32>)
  ^brainStage(%bStage: i64, %bSums: vector<64xf32>):
    %bMore = arith.cmpi slt, %bStage, %stages : i64
    cf.cond_br %bMore, ^brainCompute, ^done(%bSums : vector<64xf32>)
  ^brainCompute:
    %bNext = arith.addi %bStage, %one : i64
    %bFetched = arith.minsi %bNext, %lastStage : i64
    %bRawL = func.call @strict_device_product_load_brains(%left, %leftBase, %leftRow, %rows, %rowFirst, %bFetched, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i32) -> vector<8xi32>
    %bRawR = func.call @strict_device_product_load_brains(%right, %rightBase, %rightColumn, %columns, %columnFirst, %bFetched, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i32) -> vector<8xi32>
    %bParity = arith.andi %bStage, %one : i64
    %bBuffer = arith.trunci %bParity : i64 to i32
    %bFirst = func.call @strict_device_product_half(%bSums, %bBuffer, %c0, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %bBoth = func.call @strict_device_product_half(%bFirst, %bBuffer, %c1, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %bHasNext = arith.cmpi slt, %bNext, %stages : i64
    cf.cond_br %bHasNext, ^brainPut, ^brainStage(%bNext, %bBoth : i64, vector<64xf32>)
  ^brainPut:
    %bOther = arith.xori %bBuffer, %c1 : i32
    func.call @strict_device_product_put_brains(%bRawL, %rows, %rowFirst, %lefts, %bOther, %thread) : (vector<8xi32>, i64, i64, !llvm.ptr<3>, i32, i32) -> ()
    func.call @strict_device_product_put_brains(%bRawR, %columns, %columnFirst, %rights, %bOther, %thread) : (vector<8xi32>, i64, i64, !llvm.ptr<3>, i32, i32) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^brainStage(%bNext, %bBoth : i64, vector<64xf32>)
  ^fastInit:
    %fastFirstL = func.call @strict_device_product_fetch_fast(%left, %leftBits, %leftBase, %leftRow, %rows, %rowFirst, %none, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i32) -> vector<16xf16>
    %fastFirstR = func.call @strict_device_product_fetch_fast(%right, %rightBits, %rightBase, %rightColumn, %columns, %columnFirst, %none, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i32) -> vector<16xf16>
    func.call @strict_device_product_put(%fastFirstL, %lefts, %c0, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%fastFirstR, %rights, %c0, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^fastStage(%none, %zeros : i64, vector<64xf32>)
  ^fastStage(%fStage: i64, %fSums: vector<64xf32>):
    %fMore = arith.cmpi slt, %fStage, %stages : i64
    cf.cond_br %fMore, ^fastCompute, ^done(%fSums : vector<64xf32>)
  ^fastCompute:
    %fNext = arith.addi %fStage, %one : i64
    %fFetched = arith.minsi %fNext, %lastStage : i64
    %fNextL = func.call @strict_device_product_fetch_fast(%left, %leftBits, %leftBase, %leftRow, %rows, %rowFirst, %fFetched, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i32) -> vector<16xf16>
    %fNextR = func.call @strict_device_product_fetch_fast(%right, %rightBits, %rightBase, %rightColumn, %columns, %columnFirst, %fFetched, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i32) -> vector<16xf16>
    %fParity = arith.andi %fStage, %one : i64
    %fBuffer = arith.trunci %fParity : i64 to i32
    %fFirst = func.call @strict_device_product_half(%fSums, %fBuffer, %c0, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %fBoth = func.call @strict_device_product_half(%fFirst, %fBuffer, %c1, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %fHasNext = arith.cmpi slt, %fNext, %stages : i64
    cf.cond_br %fHasNext, ^fastPut, ^fastStage(%fNext, %fBoth : i64, vector<64xf32>)
  ^fastPut:
    %fOther = arith.xori %fBuffer, %c1 : i32
    func.call @strict_device_product_put(%fNextL, %lefts, %fOther, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%fNextR, %rights, %fOther, %thread, %false) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^fastStage(%fNext, %fBoth : i64, vector<64xf32>)
  ^generalInit:
    %firstLefts = func.call @strict_device_product_fetch(%left, %leftCount, %leftBits, %leftBase, %leftRow, %leftInner, %rows, %rowFirst, %inner, %none, %thread, %leftByRows, %leftFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    %firstRights = func.call @strict_device_product_fetch(%right, %rightCount, %rightBits, %rightBase, %rightColumn, %rightInner, %columns, %columnFirst, %inner, %none, %thread, %rightByRows, %rightFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    func.call @strict_device_product_put(%firstLefts, %lefts, %c0, %thread, %leftByRows) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%firstRights, %rights, %c0, %thread, %rightByRows) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%none, %zeros : i64, vector<64xf32>)
  ^stage(%stage: i64, %sums: vector<64xf32>):
    %moreStages = arith.cmpi slt, %stage, %stages : i64
    cf.cond_br %moreStages, ^compute, ^done(%sums : vector<64xf32>)
  ^compute:
    %nextStage = arith.addi %stage, %one : i64
    %fetchedStage = arith.minsi %nextStage, %lastStage : i64
    %nextLefts = func.call @strict_device_product_fetch(%left, %leftCount, %leftBits, %leftBase, %leftRow, %leftInner, %rows, %rowFirst, %inner, %fetchedStage, %thread, %leftByRows, %leftFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    %nextRights = func.call @strict_device_product_fetch(%right, %rightCount, %rightBits, %rightBase, %rightColumn, %rightInner, %columns, %columnFirst, %inner, %fetchedStage, %thread, %rightByRows, %rightFast) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i1, i1) -> vector<16xf16>
    %stageParity = arith.andi %stage, %one : i64
    %buffer = arith.trunci %stageParity : i64 to i32
    %firstHalf = func.call @strict_device_product_half(%sums, %buffer, %c0, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %bothHalves = func.call @strict_device_product_half(%firstHalf, %buffer, %c1, %warpRow, %warpColumn, %lane, %halvesOnly) : (vector<64xf32>, i32, i32, i32, i32, i32, i1) -> vector<64xf32>
    %hasNext = arith.cmpi slt, %nextStage, %stages : i64
    cf.cond_br %hasNext, ^put, ^stage(%nextStage, %bothHalves : i64, vector<64xf32>)
  ^put:
    %otherBuffer = arith.xori %buffer, %c1 : i32
    func.call @strict_device_product_put(%nextLefts, %lefts, %otherBuffer, %thread, %leftByRows) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_product_put(%nextRights, %rights, %otherBuffer, %thread, %rightByRows) : (vector<16xf16>, !llvm.ptr<3>, i32, i32, i1) -> ()
    func.call @strict_device_barrier() : () -> ()
    cf.br ^stage(%nextStage, %bothHalves : i64, vector<64xf32>)
  ^done(%finalSums: vector<64xf32>):
    return %finalSums : vector<64xf32>
  }
  // a block of 256 threads per tile of 128 rows by 128 columns of a batch, the column tiles of the same rows next to each
  // other (they read the same rows of left)
  func.func private @strict_device_product_tiles_run(%result: !llvm.ptr, %results: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %shape: vector<4xf64>, %strides: vector<8xf64>) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c31 = llvm.mlir.constant(31 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %one = arith.constant 1 : i64
    %seven = arith.constant 7 : i64
    %hundredTwentySeven = arith.constant 127 : i64
    %shapeInts = arith.fptosi %shape : vector<4xf64> to vector<4xi64>
    %strideInts = arith.fptosi %strides : vector<8xf64> to vector<8xi64>
    %batches = llvm.extractelement %shapeInts[%c0 : i32] : vector<4xi64>
    %rows = llvm.extractelement %shapeInts[%c1 : i32] : vector<4xi64>
    %inner = llvm.extractelement %shapeInts[%c2 : i32] : vector<4xi64>
    %columns = llvm.extractelement %shapeInts[%c3 : i32] : vector<4xi64>
    %leftStart = llvm.extractelement %strideInts[%c0 : i32] : vector<8xi64>
    %leftBatch = llvm.extractelement %strideInts[%c1 : i32] : vector<8xi64>
    %leftRow = llvm.extractelement %strideInts[%c2 : i32] : vector<8xi64>
    %leftInner = llvm.extractelement %strideInts[%c3 : i32] : vector<8xi64>
    %rightStart = llvm.extractelement %strideInts[%c4 : i32] : vector<8xi64>
    %rightBatch = llvm.extractelement %strideInts[%c5 : i32] : vector<8xi64>
    %rightInner = llvm.extractelement %strideInts[%c6 : i32] : vector<8xi64>
    %rightColumn = llvm.extractelement %strideInts[%c7 : i32] : vector<8xi64>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block32 = func.call @strict_device_block_index() : () -> i32
    %block = arith.extui %block32 : i32 to i64
    %rowsUp = arith.addi %rows, %hundredTwentySeven : i64
    %rowTiles = arith.shrui %rowsUp, %seven : i64
    %columnsUp = arith.addi %columns, %hundredTwentySeven : i64
    %columnTiles = arith.shrui %columnsUp, %seven : i64
    %perBatch = arith.muli %rowTiles, %columnTiles : i64
    %tiles = arith.muli %batches, %perBatch : i64
    %working = arith.cmpi ult, %block, %tiles : i64
    cf.cond_br %working, ^tile, ^done
  ^tile:
    %batch = arith.divui %block, %perBatch : i64
    %within = arith.remui %block, %perBatch : i64
    %rowTile = arith.divui %within, %columnTiles : i64
    %columnTile = arith.remui %within, %columnTiles : i64
    %rowFirst = arith.shli %rowTile, %seven : i64
    %columnFirst = arith.shli %columnTile, %seven : i64
    %leftBatchStart = arith.muli %batch, %leftBatch : i64
    %leftBase = arith.addi %leftStart, %leftBatchStart : i64
    %rightBatchStart = arith.muli %batch, %rightBatch : i64
    %rightBase = arith.addi %rightStart, %rightBatchStart : i64
    %sums = func.call @strict_device_product_stages(%left, %leftCount, %leftBits, %leftBase, %leftRow, %leftInner, %right, %rightCount, %rightBits, %rightBase, %rightColumn, %rightInner, %rows, %columns, %inner, %rowFirst, %columnFirst, %thread) : (!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> vector<64xf32>
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %warpRowIndex = arith.andi %warp, %c1 : i32
    %warpColumnIndex = arith.shrui %warp, %c1 : i32
    %warpRow = arith.shli %warpRowIndex, %c6 : i32
    %warpColumn = arith.shli %warpColumnIndex, %c5 : i32
    %group = arith.shrui %lane, %c2 : i32
    %quad = arith.andi %lane, %c3 : i32
    %pairColumn = arith.shli %quad, %c1 : i32
    cf.br ^write(%c0 : i32)
  ^write(%slot: i32):
    %moreSlots = arith.cmpi slt, %slot, %c64 : i32
    cf.cond_br %moreSlots, ^writeBody, ^done
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
    %row = arith.addi %rowFirst, %rowLocal64 : i64
    %column = arith.addi %columnFirst, %columnLocal64 : i64
    %value = llvm.extractelement %sums[%slot : i32] : vector<64xf32>
    func.call @strict_device_products_written(%result, %results, %value, %batch, %row, %column, %rows, %columns) : (!llvm.ptr, i64, f32, i64, i64, i64, i64, i64) -> ()
    %slotNext = arith.addi %slot, %c1 : i32
    llvm.br ^write(%slotNext : i32) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  llvm.func @strict_device_product_tiles(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %results = llvm.mlir.constant(64 : i64) : i64
    %shape0 = llvm.mlir.undef : vector<4xf64>
    %shapeAt0 = llvm.mlir.constant(0 : i32) : i32
    %shape1 = llvm.insertelement %batches, %shape0[%shapeAt0 : i32] : vector<4xf64>
    %shapeAt1 = llvm.mlir.constant(1 : i32) : i32
    %shape2 = llvm.insertelement %rows, %shape1[%shapeAt1 : i32] : vector<4xf64>
    %shapeAt2 = llvm.mlir.constant(2 : i32) : i32
    %shape3 = llvm.insertelement %inner, %shape2[%shapeAt2 : i32] : vector<4xf64>
    %shapeAt3 = llvm.mlir.constant(3 : i32) : i32
    %shape4 = llvm.insertelement %columnCount, %shape3[%shapeAt3 : i32] : vector<4xf64>
    %strides0 = llvm.mlir.undef : vector<8xf64>
    %strideAt0 = llvm.mlir.constant(0 : i32) : i32
    %strides1 = llvm.insertelement %leftStart, %strides0[%strideAt0 : i32] : vector<8xf64>
    %strideAt1 = llvm.mlir.constant(1 : i32) : i32
    %strides2 = llvm.insertelement %leftBatch, %strides1[%strideAt1 : i32] : vector<8xf64>
    %strideAt2 = llvm.mlir.constant(2 : i32) : i32
    %strides3 = llvm.insertelement %leftRow, %strides2[%strideAt2 : i32] : vector<8xf64>
    %strideAt3 = llvm.mlir.constant(3 : i32) : i32
    %strides4 = llvm.insertelement %leftInner, %strides3[%strideAt3 : i32] : vector<8xf64>
    %strideAt4 = llvm.mlir.constant(4 : i32) : i32
    %strides5 = llvm.insertelement %rightStart, %strides4[%strideAt4 : i32] : vector<8xf64>
    %strideAt5 = llvm.mlir.constant(5 : i32) : i32
    %strides6 = llvm.insertelement %rightBatch, %strides5[%strideAt5 : i32] : vector<8xf64>
    %strideAt6 = llvm.mlir.constant(6 : i32) : i32
    %strides7 = llvm.insertelement %rightInner, %strides6[%strideAt6 : i32] : vector<8xf64>
    %strideAt7 = llvm.mlir.constant(7 : i32) : i32
    %strides8 = llvm.insertelement %rightColumn, %strides7[%strideAt7 : i32] : vector<8xf64>
    func.call @strict_device_product_tiles_run(%result, %results, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
  llvm.func @strict_device_product_tiless(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64, %results: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %shape0 = llvm.mlir.undef : vector<4xf64>
    %shapeAt0 = llvm.mlir.constant(0 : i32) : i32
    %shape1 = llvm.insertelement %batches, %shape0[%shapeAt0 : i32] : vector<4xf64>
    %shapeAt1 = llvm.mlir.constant(1 : i32) : i32
    %shape2 = llvm.insertelement %rows, %shape1[%shapeAt1 : i32] : vector<4xf64>
    %shapeAt2 = llvm.mlir.constant(2 : i32) : i32
    %shape3 = llvm.insertelement %inner, %shape2[%shapeAt2 : i32] : vector<4xf64>
    %shapeAt3 = llvm.mlir.constant(3 : i32) : i32
    %shape4 = llvm.insertelement %columnCount, %shape3[%shapeAt3 : i32] : vector<4xf64>
    %strides0 = llvm.mlir.undef : vector<8xf64>
    %strideAt0 = llvm.mlir.constant(0 : i32) : i32
    %strides1 = llvm.insertelement %leftStart, %strides0[%strideAt0 : i32] : vector<8xf64>
    %strideAt1 = llvm.mlir.constant(1 : i32) : i32
    %strides2 = llvm.insertelement %leftBatch, %strides1[%strideAt1 : i32] : vector<8xf64>
    %strideAt2 = llvm.mlir.constant(2 : i32) : i32
    %strides3 = llvm.insertelement %leftRow, %strides2[%strideAt2 : i32] : vector<8xf64>
    %strideAt3 = llvm.mlir.constant(3 : i32) : i32
    %strides4 = llvm.insertelement %leftInner, %strides3[%strideAt3 : i32] : vector<8xf64>
    %strideAt4 = llvm.mlir.constant(4 : i32) : i32
    %strides5 = llvm.insertelement %rightStart, %strides4[%strideAt4 : i32] : vector<8xf64>
    %strideAt5 = llvm.mlir.constant(5 : i32) : i32
    %strides6 = llvm.insertelement %rightBatch, %strides5[%strideAt5 : i32] : vector<8xf64>
    %strideAt6 = llvm.mlir.constant(6 : i32) : i32
    %strides7 = llvm.insertelement %rightInner, %strides6[%strideAt6 : i32] : vector<8xf64>
    %strideAt7 = llvm.mlir.constant(7 : i32) : i32
    %strides8 = llvm.insertelement %rightColumn, %strides7[%strideAt7 : i32] : vector<8xf64>
    func.call @strict_device_product_tiles_run(%result, %results, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
