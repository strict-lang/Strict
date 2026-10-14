  func.func private @strict_device_element(%data: !llvm.ptr, %bits: i64, %at: i64, %count: i64) -> f64 {
    %sixteen = arith.constant 16 : i64
    %isHalf = arith.cmpi eq, %bits, %sixteen : i64
    cf.cond_br %isHalf, ^half, ^wider
  ^half:
    %halfSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfValue = llvm.load %halfSlot : !llvm.ptr -> f16
    %halfNumber = arith.extf %halfValue : f16 to f64
    return %halfNumber : f64
  ^wider:
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^other
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleValue = llvm.load %singleSlot : !llvm.ptr -> f32
    %singleNumber = arith.extf %singleValue : f32 to f64
    return %singleNumber : f64
  ^other:
    %eight = arith.constant 8 : i64
    %isQuantized = arith.cmpi eq, %bits, %eight : i64
    cf.cond_br %isQuantized, ^quantized, ^double
  ^quantized:
    %byteSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %byteSlot : !llvm.ptr -> i8
    %byteNumber = arith.sitofp %byte : i8 to f64
    %three = arith.constant 3 : i64
    %alignMask = arith.constant -4 : i64
    %padded = arith.addi %count, %three : i64
    %aligned = arith.andi %padded, %alignMask : i64
    %scales = llvm.getelementptr %data[%aligned] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %five = arith.constant 5 : i64
    %block = arith.shrui %at, %five : i64
    %scaleSlot = llvm.getelementptr %scales[%block] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %scale = llvm.load %scaleSlot : !llvm.ptr -> f32
    %wideScale = arith.extf %scale : f32 to f64
    %quantizedNumber = arith.mulf %byteNumber, %wideScale : f64
    return %quantizedNumber : f64
  ^double:
    %doubleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubleValue = llvm.load %doubleSlot : !llvm.ptr -> f64
    return %doubleValue : f64
  }
  func.func private @strict_device_bytes(%count: i64, %bits: i64) -> i64 {
    %eight = arith.constant 8 : i64
    %three = arith.constant 3 : i64
    %thirtyOne = arith.constant 31 : i64
    %five = arith.constant 5 : i64
    %two = arith.constant 2 : i64
    %alignMask = arith.constant -4 : i64
    %padded = arith.addi %count, %three : i64
    %aligned = arith.andi %padded, %alignMask : i64
    %rounded = arith.addi %count, %thirtyOne : i64
    %blocks = arith.shrui %rounded, %five : i64
    %scaleBytes = arith.shli %blocks, %two : i64
    %quantizedBytes = arith.addi %aligned, %scaleBytes : i64
    %cells = arith.muli %count, %bits : i64
    %plainBytes = arith.divui %cells, %eight : i64
    %isQuantized = arith.cmpi eq, %bits, %eight : i64
    %bytes = arith.select %isQuantized, %quantizedBytes, %plainBytes : i64
    return %bytes : i64
  }
  func.func private @strict_device_sum(%value: f64) -> f64 {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %thirtyTwo = arith.constant 32 : i64
    %sixteen = arith.constant 16 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    cf.br ^round(%value, %sixteen : f64, i32)
  ^round(%sum: f64, %offset: i32):
    %more = arith.cmpi sge, %offset, %one : i32
    cf.cond_br %more, ^shuffle, ^done
  ^shuffle:
    %bits = arith.bitcast %sum : f64 to i64
    %low = arith.trunci %bits : i64 to i32
    %shifted = arith.shrui %bits, %thirtyTwo : i64
    %high = arith.trunci %shifted : i64 to i32
    %otherLow = nvvm.shfl.sync bfly %all, %low, %offset, %clamp : i32 -> i32
    %otherHigh = nvvm.shfl.sync bfly %all, %high, %offset, %clamp : i32 -> i32
    %lowWide = arith.extui %otherLow : i32 to i64
    %highWide = arith.extui %otherHigh : i32 to i64
    %highPlaced = arith.shli %highWide, %thirtyTwo : i64
    %combined = arith.ori %highPlaced, %lowWide : i64
    %other = arith.bitcast %combined : i64 to f64
    %added = arith.addf %sum, %other : f64
    %half = arith.divui %offset, %two : i32
    cf.br ^round(%added, %half : f64, i32)
  ^done:
    return %sum : f64
  }