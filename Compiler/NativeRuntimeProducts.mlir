  // a number of a list at at as a 32 bit float, 0 when outside is set (at is not read then)
  func.func private @strict_device_products_number(%data: !llvm.ptr, %count: i64, %bits: i64, %at: i64, %outside: i1) -> f32 attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f32
    %safe = arith.select %outside, %none, %at : i64
    %number = func.call @strict_device_elements(%data, %bits, %safe, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    %chosen = arith.select %outside, %zero, %number : f32
    return %chosen : f32
  }
  // two 32 bit floats as the 16 bit pair of an mma operand
  func.func private @strict_device_products_pair(%first: f32, %second: f32) -> vector<2xf16> attributes {passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_products_written(%result: !llvm.ptr, %wide: i1, %value: f32, %batch: i64, %row: i64, %column: i64, %rows: i64, %columns: i64) attributes {passthrough = ["alwaysinline"]} {
    %rowInside = arith.cmpi slt, %row, %rows : i64
    %columnInside = arith.cmpi slt, %column, %columns : i64
    %inside = arith.andi %rowInside, %columnInside : i1
    cf.cond_br %inside, ^write, ^done
  ^write:
    %batchRow = arith.muli %batch, %rows : i64
    %fullRow = arith.addi %batchRow, %row : i64
    %rowStart = arith.muli %fullRow, %columns : i64
    %at = arith.addi %rowStart, %column : i64
    cf.cond_br %wide, ^double, ^single
  ^double:
    %double = arith.extf %value : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    cf.br ^done
  ^single:
    %singleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %singleSlot : f32, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  // `Products.Multiplied` on the 16 bit tensor cores: a warp computes 16 rows by 8 columns of one batch, the numbers of
  // left and right read at their strides 8 along inner at a time (zeros past the edges), sums in 32 bit floats
  func.func private @strict_device_products_run(%result: !llvm.ptr, %wide: i1, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %shape: vector<4xf64>, %strides: vector<8xf64>) {
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
    func.call @strict_device_products_written(%result, %wide, %sum0, %batch, %lowerRow, %outFirst, %rows, %columns) : (!llvm.ptr, i1, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %wide, %sum1, %batch, %lowerRow, %outSecond, %rows, %columns) : (!llvm.ptr, i1, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %wide, %sum2, %batch, %upperRow, %outFirst, %rows, %columns) : (!llvm.ptr, i1, f32, i64, i64, i64, i64, i64) -> ()
    func.call @strict_device_products_written(%result, %wide, %sum3, %batch, %upperRow, %outSecond, %rows, %columns) : (!llvm.ptr, i1, f32, i64, i64, i64, i64, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  llvm.func @strict_device_products(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(true) : i1
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
    func.call @strict_device_products_run(%result, %wide, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
  llvm.func @strict_device_productss(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %left: !llvm.ptr, %leftCount: i64, %leftBits: i64, %right: !llvm.ptr, %rightCount: i64, %rightBits: i64, %batches: f64, %rows: f64, %inner: f64, %columnCount: f64, %leftStart: f64, %leftBatch: f64, %leftRow: f64, %leftInner: f64, %rightStart: f64, %rightBatch: f64, %rightInner: f64, %rightColumn: f64) attributes {nvvm.kernel} {
    %wide = llvm.mlir.constant(false) : i1
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
    func.call @strict_device_products_run(%result, %wide, %left, %leftCount, %leftBits, %right, %rightCount, %rightBits, %shape4, %strides8) : (!llvm.ptr, i1, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, vector<4xf64>, vector<8xf64>) -> ()
    llvm.return
  }
