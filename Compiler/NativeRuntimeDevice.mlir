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
  // mirrors strict_packed_data of the host runtime: the bytes of the numbers, after them come the scales
  func.func private @strict_device_data(%count: i64, %bits: i64) -> i64 {
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %spare = arith.constant 255 : i64
    %fourBlock = arith.constant 144 : i64
    %sixBlock = arith.constant 208 : i64
    %alignMask = arith.constant -4 : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %rounded = arith.addi %count, %spare : i64
    %blocks = arith.shrui %rounded, %eight : i64
    %fourBytes = arith.muli %blocks, %fourBlock : i64
    %sixBytes = arith.muli %blocks, %sixBlock : i64
    %padded = arith.addi %count, %three : i64
    %aligned = arith.andi %padded, %alignMask : i64
    %orSix = arith.select %isSix, %sixBytes, %aligned : i64
    %data = arith.select %isFour, %fourBytes, %orSix : i64
    return %data : i64
  }
  // mirrors strict_k4_scale of the host runtime, the 6 bit numbers become floats without a conversion instruction
  func.func private @strict_k4_scale(%scales: !llvm.ptr, %group: i64) -> (f32, f32) {
    %four = arith.constant 4 : i64
    %sixBits = arith.constant 63 : i32
    %fourBits = arith.constant 15 : i32
    %six = arith.constant 6 : i32
    %shift = arith.constant 4 : i32
    %magic = arith.constant 1258291200 : i32
    %magicFloat = arith.constant 8388608.0 : f32
    %isLow = arith.cmpi ult, %group, %four : i64
    %later = arith.addi %group, %four : i64
    %below = arith.subi %group, %four : i64
    %earlier = arith.select %isLow, %group, %below : i64
    %ownSlot = llvm.getelementptr %scales[%group] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %ownByte = llvm.load %ownSlot : !llvm.ptr -> i8
    %own = arith.extui %ownByte : i8 to i32
    %laterSlot = llvm.getelementptr %scales[%later] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %laterByte = llvm.load %laterSlot : !llvm.ptr -> i8
    %next = arith.extui %laterByte : i8 to i32
    %earlierSlot = llvm.getelementptr %scales[%earlier] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %earlierByte = llvm.load %earlierSlot : !llvm.ptr -> i8
    %previous = arith.extui %earlierByte : i8 to i32
    %lowScale = arith.andi %own, %sixBits : i32
    %lowMinimum = arith.andi %next, %sixBits : i32
    %nextLow = arith.andi %next, %fourBits : i32
    %previousTop = arith.shrui %previous, %six : i32
    %previousPlaced = arith.shli %previousTop, %shift : i32
    %highScale = arith.ori %nextLow, %previousPlaced : i32
    %nextHigh = arith.shrui %next, %shift : i32
    %ownTop = arith.shrui %own, %six : i32
    %ownPlaced = arith.shli %ownTop, %shift : i32
    %highMinimum = arith.ori %nextHigh, %ownPlaced : i32
    %scale = arith.select %isLow, %lowScale, %highScale : i32
    %minimum = arith.select %isLow, %lowMinimum, %highMinimum : i32
    %scaleMagic = arith.ori %scale, %magic : i32
    %scaleFloat = arith.bitcast %scaleMagic : i32 to f32
    %scaleNumber = arith.subf %scaleFloat, %magicFloat : f32
    %minimumMagic = arith.ori %minimum, %magic : i32
    %minimumFloat = arith.bitcast %minimumMagic : i32 to f32
    %minimumNumber = arith.subf %minimumFloat, %magicFloat : f32
    return %scaleNumber, %minimumNumber : f32, f32
  }
  // a row of a matrix and its width, behind the header of strict_gpu_rows (row stride, one width byte per row)
  func.func private @strict_device_row(%source: !llvm.ptr, %at: i64, %count: i64) -> (!llvm.ptr, i64) {
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %alignMask = arith.constant -16 : i64
    %stride = llvm.load %source : !llvm.ptr -> i64
    %widthAt = arith.addi %at, %eight : i64
    %widthSlot = llvm.getelementptr %source[%widthAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %width = llvm.load %widthSlot : !llvm.ptr -> i8
    %bits = arith.extui %width : i8 to i64
    %withTable = arith.addi %count, %eight : i64
    %paddedHeader = arith.addi %withTable, %fifteen : i64
    %header = arith.andi %paddedHeader, %alignMask : i64
    %rowOffset = arith.muli %at, %stride : i64
    %offset = arith.addi %header, %rowOffset : i64
    %row = llvm.getelementptr %source[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    return %row, %bits : !llvm.ptr, i64
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
  func.func private @strict_device_elements(%data: !llvm.ptr, %bits: i64, %at: i64, %count: i64) -> f32 {
    %sixteen = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %isHalf = arith.cmpi eq, %bits, %sixteen : i64
    cf.cond_br %isHalf, ^half, ^wider
  ^half:
    %halfSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfValue = llvm.load %halfSlot : !llvm.ptr -> f16
    %halfNumber = arith.extf %halfValue : f16 to f32
    return %halfNumber : f32
  ^wider:
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^other
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleValue = llvm.load %singleSlot : !llvm.ptr -> f32
    return %singleValue : f32
  ^other:
    %eight = arith.constant 8 : i64
    %six = arith.constant 6 : i64
    %four = arith.constant 4 : i64
    %isQuantized = arith.cmpi eq, %bits, %eight : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %aligned = func.call @strict_device_data(%count, %bits) : (i64, i64) -> i64
    %scales = llvm.getelementptr %data[%aligned] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %block = arith.shrui %at, %eight : i64
    %blockMask = arith.constant 255 : i64
    %inBlock = arith.andi %at, %blockMask : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %partSize = arith.constant 32 : i64
    %thirtyOne = arith.constant 31 : i64
    %fifteen = arith.constant 15 : i32
    %number = arith.andi %inBlock, %thirtyOne : i64
    cf.cond_br %isQuantized, ^quantized, ^blocked
  ^quantized:
    %byteSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %byteSlot : !llvm.ptr -> i8
    %byteNumber = arith.sitofp %byte : i8 to f32
    %scaleIndex = arith.shrui %at, %five : i64
    %scaleSlot = llvm.getelementptr %scales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %scale = llvm.load %scaleSlot : !llvm.ptr -> f32
    %quantizedNumber = arith.mulf %byteNumber, %scale : f32
    return %quantizedNumber : f32
  ^blocked:
    cf.cond_br %isFour, ^four, ^sixOrDouble
  ^sixOrDouble:
    cf.cond_br %isSix, ^six, ^double
  ^four:
    %fourBytes = arith.constant 144 : i64
    %fourOffset = arith.muli %block, %fourBytes : i64
    %fourStart = llvm.getelementptr %data[%fourOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halves = llvm.load %fourStart : !llvm.ptr -> vector<2xf16>
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %halfScale = llvm.extractelement %halves[%c0 : i32] : vector<2xf16>
    %halfMinimum = llvm.extractelement %halves[%c1 : i32] : vector<2xf16>
    %blockScale = arith.extf %halfScale : f16 to f32
    %blockMinimum = arith.extf %halfMinimum : f16 to f32
    %group = arith.shrui %inBlock, %five : i64
    %groupScales = llvm.getelementptr %fourStart[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %levels:2 = func.call @strict_k4_scale(%groupScales, %group) : (!llvm.ptr, i64) -> (f32, f32)
    %part = arith.shrui %group, %one : i64
    %partBytes = arith.muli %part, %partSize : i64
    %nibbleStart = arith.addi %partBytes, %number : i64
    %nibbleAt = arith.addi %nibbleStart, %sixteen : i64
    %nibbleSlot = llvm.getelementptr %fourStart[%nibbleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %nibbleByte = llvm.load %nibbleSlot : !llvm.ptr -> i8
    %nibbleWide = arith.extui %nibbleByte : i8 to i32
    %isHigh = arith.andi %group, %one : i64
    %highShift = arith.shli %isHigh, %two : i64
    %nibbleShift = arith.trunci %highShift : i64 to i32
    %nibbleShifted = arith.shrui %nibbleWide, %nibbleShift : i32
    %nibble = arith.andi %nibbleShifted, %fifteen : i32
    %level = arith.uitofp %nibble : i32 to f32
    %factor = arith.mulf %blockScale, %levels#0 : f32
    %offset = arith.mulf %blockMinimum, %levels#1 : f32
    %scaledLevel = arith.mulf %factor, %level : f32
    %fourNumber = arith.subf %scaledLevel, %offset : f32
    return %fourNumber : f32
  ^six:
    %sixBytes = arith.constant 208 : i64
    %sixOffset = arith.muli %block, %sixBytes : i64
    %sixStart = llvm.getelementptr %data[%sixOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %seven = arith.constant 7 : i64
    %three = arith.constant 3 : i64
    %half = arith.shrui %inBlock, %seven : i64
    %quarterBits = arith.shrui %inBlock, %five : i64
    %quarter = arith.andi %quarterBits, %three : i64
    %isSecond = arith.andi %quarter, %one : i64
    %secondBytes = arith.muli %isSecond, %partSize : i64
    %halfLow = arith.shli %half, %six : i64
    %lowBase = arith.addi %halfLow, %secondBytes : i64
    %lowAt = arith.addi %lowBase, %number : i64
    %lowSlot = llvm.getelementptr %sixStart[%lowAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowByte = llvm.load %lowSlot : !llvm.ptr -> i8
    %lowWide = arith.extui %lowByte : i8 to i32
    %isUpper = arith.shrui %quarter, %one : i64
    %upperShift = arith.shli %isUpper, %two : i64
    %lowShift = arith.trunci %upperShift : i64 to i32
    %lowShifted = arith.shrui %lowWide, %lowShift : i32
    %low = arith.andi %lowShifted, %fifteen : i32
    %highStart = arith.constant 128 : i64
    %halfHigh = arith.shli %half, %five : i64
    %highBase = arith.addi %highStart, %halfHigh : i64
    %highAt = arith.addi %highBase, %number : i64
    %highSlot = llvm.getelementptr %sixStart[%highAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %highByte = llvm.load %highSlot : !llvm.ptr -> i8
    %highWide = arith.extui %highByte : i8 to i32
    %pairShift = arith.shli %quarter, %one : i64
    %highShiftBits = arith.trunci %pairShift : i64 to i32
    %highShifted = arith.shrui %highWide, %highShiftBits : i32
    %pairMask = arith.constant 3 : i32
    %high = arith.andi %highShifted, %pairMask : i32
    %fourBits = arith.constant 4 : i32
    %highPlaced = arith.shli %high, %fourBits : i32
    %combined = arith.ori %low, %highPlaced : i32
    %middle = arith.constant 32 : i32
    %centered = arith.subi %combined, %middle : i32
    %sixLevel = arith.sitofp %centered : i32 to f32
    %scalesStart = arith.constant 192 : i64
    %halfScales = arith.shli %half, %three : i64
    %quarterScales = arith.shli %quarter, %one : i64
    %sixteenth = arith.shrui %number, %four : i64
    %groupBase = arith.addi %scalesStart, %halfScales : i64
    %groupQuarter = arith.addi %groupBase, %quarterScales : i64
    %groupAt = arith.addi %groupQuarter, %sixteenth : i64
    %groupSlot = llvm.getelementptr %sixStart[%groupAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %groupByte = llvm.load %groupSlot : !llvm.ptr -> i8
    %groupScale = arith.sitofp %groupByte : i8 to f32
    %dSlot = llvm.getelementptr %scales[%block] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %dHalf = llvm.load %dSlot : !llvm.ptr -> f16
    %d = arith.extf %dHalf : f16 to f32
    %sixFactor = arith.mulf %d, %groupScale : f32
    %sixNumber = arith.mulf %sixFactor, %sixLevel : f32
    return %sixNumber : f32
  ^double:
    %doubleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubleValue = llvm.load %doubleSlot : !llvm.ptr -> f64
    %narrowed = arith.truncf %doubleValue : f64 to f32
    return %narrowed : f32
  }
  func.func private @strict_device_sums(%value: f32) -> f32 {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %sixteen = arith.constant 16 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    cf.br ^round(%value, %sixteen : f32, i32)
  ^round(%sum: f32, %offset: i32):
    %more = arith.cmpi sge, %offset, %one : i32
    cf.cond_br %more, ^shuffle, ^done
  ^shuffle:
    %bits = arith.bitcast %sum : f32 to i32
    %otherBits = nvvm.shfl.sync bfly %all, %bits, %offset, %clamp : i32 -> i32
    %other = arith.bitcast %otherBits : i32 to f32
    %added = arith.addf %sum, %other : f32
    %half = arith.divui %offset, %two : i32
    cf.br ^round(%added, %half : f32, i32)
  ^done:
    return %sum : f32
  }
  // exact double powers need libdevice, so the launch fails and the CPU runs the loop
  func.func private @strict_device_power(%failed: !llvm.ptr, %left: f64, %right: f64) -> f64 {
    %one = arith.constant 1 : i64
    llvm.store %one, %failed : i64, !llvm.ptr
    return %left : f64
  }
  // 32 bit powers: square roots and small whole exponents exactly, powers of e and other bases with the approximate
  // exponential and logarithm of base 2 (ex2.approx and lg2.approx, about 2 units in the last place)
  func.func private @strict_device_powers(%failed: !llvm.ptr, %left: f32, %right: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %isRoot = arith.cmpf oeq, %right, %half : f32
    cf.cond_br %isRoot, ^root, ^whole
  ^root:
    %root = llvm.call_intrinsic "llvm.sqrt.f32"(%left) : (f32) -> f32
    return %root : f32
  ^whole:
    %isZero = arith.cmpf oeq, %right, %zero : f32
    %isOne = arith.cmpf oeq, %right, %one : f32
    %isTwo = arith.cmpf oeq, %right, %two : f32
    %squared = arith.mulf %left, %left : f32
    %small = arith.select %isTwo, %squared, %left : f32
    %smallOrOne = arith.select %isZero, %one, %small : f32
    %isSmall = arith.ori %isZero, %isOne : i1
    %isExact = arith.ori %isSmall, %isTwo : i1
    cf.cond_br %isExact, ^done(%smallOrOne : f32), ^base
  ^base:
    %e = arith.constant 2.71828175 : f32
    %isE = arith.cmpf oeq, %left, %e : f32
    %log2e = arith.constant 1.44269504 : f32
    cf.cond_br %isE, ^exponential(%log2e, %one : f32, f32), ^signed
  ^signed:
    %size = llvm.intr.fabs(%left) : (f32) -> f32
    %logarithm = llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"(%size) : (f32) -> f32
    %isNegative = arith.cmpf olt, %left, %zero : f32
    %floored = llvm.call_intrinsic "llvm.floor.f32"(%right) : (f32) -> f32
    %isInteger = arith.cmpf oeq, %floored, %right : f32
    %parity = arith.remf %right, %two : f32
    %isOdd = arith.cmpf one, %parity, %zero : f32
    %minusOne = arith.constant -1.0 : f32
    %oddSign = arith.select %isOdd, %minusOne, %one : f32
    %nan = arith.constant 0x7FC00000 : f32
    %negativeSign = arith.select %isInteger, %oddSign, %nan : f32
    %sign = arith.select %isNegative, %negativeSign, %one : f32
    cf.br ^exponential(%logarithm, %sign : f32, f32)
  ^exponential(%factor: f32, %signValue: f32):
    %exponent = arith.mulf %right, %factor : f32
    %power = llvm.call_intrinsic "llvm.exp2.f32"(%exponent) : (f32) -> f32
    %signed = arith.mulf %power, %signValue : f32
    cf.br ^done(%signed : f32)
  ^done(%result: f32):
    return %result : f32
  }
  // eight 32 bit floats of a list from an index that is a multiple of eight, two loads of 16 bytes
  func.func private @strict_device_eight(%data: !llvm.ptr, %at: i64) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %lowSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lowFour = llvm.load %lowSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %highSlot = llvm.getelementptr %lowSlot[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %highFour = llvm.load %highSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %s0 = llvm.extractelement %lowFour[%c0 : i32] : vector<4xf32>
    %s1 = llvm.extractelement %lowFour[%c1 : i32] : vector<4xf32>
    %s2 = llvm.extractelement %lowFour[%c2 : i32] : vector<4xf32>
    %s3 = llvm.extractelement %lowFour[%c3 : i32] : vector<4xf32>
    %t0 = llvm.extractelement %highFour[%c0 : i32] : vector<4xf32>
    %t1 = llvm.extractelement %highFour[%c1 : i32] : vector<4xf32>
    %t2 = llvm.extractelement %highFour[%c2 : i32] : vector<4xf32>
    %t3 = llvm.extractelement %highFour[%c3 : i32] : vector<4xf32>
    return %s0, %s1, %s2, %s3, %t0, %t1, %t2, %t3 : f32, f32, f32, f32, f32, f32, f32, f32
  }
  // the dot product of a row and a list of 32 bit floats by all lanes of a warp, each lane decodes eight numbers of the
  // row per step (one 32 bit word of 4 bit numbers, two of bytes, eight halves or floats), the warp reads 128 bytes of
  // 4 bit numbers at once and sums the lanes up; lists of doubles take the generic path (a branch on the width of the
  // list inside the loop was turned into predicated conversions on the slow FP64 pipe)
  func.func private @strict_device_dots(%failed: !llvm.ptr, %row: !llvm.ptr, %rowBits: i64, %count: i64, %other: !llvm.ptr, %otherBits: i64, %otherCount: i64, %lane: i64, %lanes: i64) -> f32 {
    // a number of 4 or 8 bits becomes a float without a conversion instruction (they run on the slow FP64 pipe of
    // consumer GPUs): ored into the mantissa of 2^23 it is 2^23 + number, signed bytes are moved up by 128 first
    %magic = arith.constant 1258291200 : i32
    %magicFloat = arith.constant 8388608.0 : f32
    %signedFloat = arith.constant 8388736.0 : f32
    %byteMask = arith.constant 255 : i32
    %signBit = arith.constant 128 : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %zero = arith.constant 0.0 : f32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %two = arith.constant 2 : i64
    %sixteen = arith.constant 16 : i64
    %eightBits = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i32
    %fourBits = arith.constant 4 : i32
    %true = arith.constant true
    %short = arith.cmpi slt, %otherCount, %count : i64
    cf.cond_br %short, ^fail, ^check
  ^fail:
    %failure = arith.constant 1 : i64
    llvm.store %failure, %failed : i64, !llvm.ptr
    return %zero : f32
  ^check:
    %rest = arith.andi %count, %seven : i64
    %whole = arith.cmpi eq, %rest, %none : i64
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %otherBits, %thirtyTwo : i64
    %blockMask = arith.constant 255 : i64
    %blockRest = arith.andi %count, %blockMask : i64
    %isBlocks = arith.cmpi eq, %blockRest, %none : i64
    %isFourRow = arith.cmpi eq, %rowBits, %four : i64
    %isSixRow = arith.cmpi eq, %rowBits, %six : i64
    %isBlockedRow = arith.ori %isFourRow, %isSixRow : i1
    %notBlocked = arith.xori %isBlockedRow, %true : i1
    %blocksFit = arith.ori %notBlocked, %isBlocks : i1
    %wholeSingle = arith.andi %whole, %isSingle : i1
    %fast = arith.andi %wholeSingle, %blocksFit : i1
    %chunks = arith.shrui %count, %three : i64
    %sixteens = arith.shrui %count, %four : i64
    %aligned = func.call @strict_device_data(%count, %rowBits) : (i64, i64) -> i64
    %scales = llvm.getelementptr %row[%aligned] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.cond_br %fast, ^dispatch, ^generic(%lane, %zero : i64, f32)
  ^dispatch:
    cf.switch %rowBits : i64, [
      default: ^generic(%lane, %zero : i64, f32),
      4: ^four(%lane, %zero : i64, f32),
      6: ^sixes(%lane, %zero : i64, f32),
      8: ^bytes(%lane, %zero : i64, f32),
      16: ^halves(%lane, %zero : i64, f32),
      32: ^singles(%lane, %zero : i64, f32)
    ]
  ^four(%fourChunk: i64, %fourSum: f32):
    %fourMore = arith.cmpi slt, %fourChunk, %chunks : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
    %fourWordMask = arith.constant 31 : i64
    %fourBlockBytes = arith.constant 144 : i64
    %fourBlock = arith.shrui %fourChunk, %five : i64
    %fourWord = arith.andi %fourChunk, %fourWordMask : i64
    %fourPart = arith.shrui %fourWord, %three : i64
    %fourInPart = arith.andi %fourWord, %seven : i64
    %fourOffset = arith.muli %fourBlock, %fourBlockBytes : i64
    %fourStart = llvm.getelementptr %row[%fourOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %fourWordBytes = arith.shli %fourWord, %two : i64
    %fourNibbleAt = arith.addi %fourWordBytes, %sixteen : i64
    %fourNibbleSlot = llvm.getelementptr %fourStart[%fourNibbleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %fourNibbles = llvm.load %fourNibbleSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %fourHalves = llvm.load %fourStart {alignment = 4 : i64} : !llvm.ptr -> vector<2xf16>
    %fourHalfScale = llvm.extractelement %fourHalves[%c0 : i32] : vector<2xf16>
    %fourHalfMinimum = llvm.extractelement %fourHalves[%c1 : i32] : vector<2xf16>
    %fourScale = arith.extf %fourHalfScale : f16 to f32
    %fourMinimum = arith.extf %fourHalfMinimum : f16 to f32
    %fourScales = llvm.getelementptr %fourStart[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %fourLowGroup = arith.shli %fourPart, %one : i64
    %fourHighGroup = arith.addi %fourLowGroup, %one : i64
    %fourLow:2 = func.call @strict_k4_scale(%fourScales, %fourLowGroup) : (!llvm.ptr, i64) -> (f32, f32)
    %fourHigh:2 = func.call @strict_k4_scale(%fourScales, %fourHighGroup) : (!llvm.ptr, i64) -> (f32, f32)
    %fourBlockFirst = arith.shli %fourBlock, %eightBits : i64
    %fourPartFirst = arith.shli %fourPart, %six : i64
    %fourInFirst = arith.shli %fourInPart, %two : i64
    %fourPartStart = arith.addi %fourBlockFirst, %fourPartFirst : i64
    %fourLowAt = arith.addi %fourPartStart, %fourInFirst : i64
    %fourLowSlot = llvm.getelementptr %other[%fourLowAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %fourLowInputs = llvm.load %fourLowSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %fourHighSlot = llvm.getelementptr %fourLowSlot[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %fourHighInputs = llvm.load %fourHighSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %nx0 = llvm.extractelement %fourLowInputs[%c0 : i32] : vector<4xf32>
    %nx1 = llvm.extractelement %fourLowInputs[%c1 : i32] : vector<4xf32>
    %nx2 = llvm.extractelement %fourLowInputs[%c2 : i32] : vector<4xf32>
    %nx3 = llvm.extractelement %fourLowInputs[%c3 : i32] : vector<4xf32>
    %nx4 = llvm.extractelement %fourHighInputs[%c0 : i32] : vector<4xf32>
    %nx5 = llvm.extractelement %fourHighInputs[%c1 : i32] : vector<4xf32>
    %nx6 = llvm.extractelement %fourHighInputs[%c2 : i32] : vector<4xf32>
    %nx7 = llvm.extractelement %fourHighInputs[%c3 : i32] : vector<4xf32>
    %nq0s = arith.constant 0 : i32
    %nq0w = arith.shrui %fourNibbles, %nq0s : i32
    %nq0n = arith.andi %nq0w, %fifteen : i32
    %nq0m = arith.ori %nq0n, %magic : i32
    %nq0f = arith.bitcast %nq0m : i32 to f32
    %nq0 = arith.subf %nq0f, %magicFloat : f32
    %nq4s = arith.constant 4 : i32
    %nq4w = arith.shrui %fourNibbles, %nq4s : i32
    %nq4n = arith.andi %nq4w, %fifteen : i32
    %nq4m = arith.ori %nq4n, %magic : i32
    %nq4f = arith.bitcast %nq4m : i32 to f32
    %nq4 = arith.subf %nq4f, %magicFloat : f32
    %nq1s = arith.constant 8 : i32
    %nq1w = arith.shrui %fourNibbles, %nq1s : i32
    %nq1n = arith.andi %nq1w, %fifteen : i32
    %nq1m = arith.ori %nq1n, %magic : i32
    %nq1f = arith.bitcast %nq1m : i32 to f32
    %nq1 = arith.subf %nq1f, %magicFloat : f32
    %nq5s = arith.constant 12 : i32
    %nq5w = arith.shrui %fourNibbles, %nq5s : i32
    %nq5n = arith.andi %nq5w, %fifteen : i32
    %nq5m = arith.ori %nq5n, %magic : i32
    %nq5f = arith.bitcast %nq5m : i32 to f32
    %nq5 = arith.subf %nq5f, %magicFloat : f32
    %nq2s = arith.constant 16 : i32
    %nq2w = arith.shrui %fourNibbles, %nq2s : i32
    %nq2n = arith.andi %nq2w, %fifteen : i32
    %nq2m = arith.ori %nq2n, %magic : i32
    %nq2f = arith.bitcast %nq2m : i32 to f32
    %nq2 = arith.subf %nq2f, %magicFloat : f32
    %nq6s = arith.constant 20 : i32
    %nq6w = arith.shrui %fourNibbles, %nq6s : i32
    %nq6n = arith.andi %nq6w, %fifteen : i32
    %nq6m = arith.ori %nq6n, %magic : i32
    %nq6f = arith.bitcast %nq6m : i32 to f32
    %nq6 = arith.subf %nq6f, %magicFloat : f32
    %nq3s = arith.constant 24 : i32
    %nq3w = arith.shrui %fourNibbles, %nq3s : i32
    %nq3n = arith.andi %nq3w, %fifteen : i32
    %nq3m = arith.ori %nq3n, %magic : i32
    %nq3f = arith.bitcast %nq3m : i32 to f32
    %nq3 = arith.subf %nq3f, %magicFloat : f32
    %nq7s = arith.constant 28 : i32
    %nq7w = arith.shrui %fourNibbles, %nq7s : i32
    %nq7n = arith.andi %nq7w, %fifteen : i32
    %nq7m = arith.ori %nq7n, %magic : i32
    %nq7f = arith.bitcast %nq7m : i32 to f32
    %nq7 = arith.subf %nq7f, %magicFloat : f32
    %fourLowLevelsp0 = arith.mulf %nq0, %nx0 fastmath<contract> : f32
    %fourLowLevelsp1 = arith.mulf %nq1, %nx1 fastmath<contract> : f32
    %fourLowLevelsp2 = arith.mulf %nq2, %nx2 fastmath<contract> : f32
    %fourLowLevelsp3 = arith.mulf %nq3, %nx3 fastmath<contract> : f32
    %fourLowLevels0 = arith.addf %fourLowLevelsp0, %fourLowLevelsp1 fastmath<contract> : f32
    %fourLowLevels1 = arith.addf %fourLowLevels0, %fourLowLevelsp2 fastmath<contract> : f32
    %fourLowLevels = arith.addf %fourLowLevels1, %fourLowLevelsp3 fastmath<contract> : f32
    %fourHighLevelsp0 = arith.mulf %nq4, %nx4 fastmath<contract> : f32
    %fourHighLevelsp1 = arith.mulf %nq5, %nx5 fastmath<contract> : f32
    %fourHighLevelsp2 = arith.mulf %nq6, %nx6 fastmath<contract> : f32
    %fourHighLevelsp3 = arith.mulf %nq7, %nx7 fastmath<contract> : f32
    %fourHighLevels0 = arith.addf %fourHighLevelsp0, %fourHighLevelsp1 fastmath<contract> : f32
    %fourHighLevels1 = arith.addf %fourHighLevels0, %fourHighLevelsp2 fastmath<contract> : f32
    %fourHighLevels = arith.addf %fourHighLevels1, %fourHighLevelsp3 fastmath<contract> : f32
    %fourLowInputSum0 = arith.addf %nx0, %nx1 fastmath<contract> : f32
    %fourLowInputSum1 = arith.addf %fourLowInputSum0, %nx2 fastmath<contract> : f32
    %fourLowInputSum = arith.addf %fourLowInputSum1, %nx3 fastmath<contract> : f32
    %fourHighInputSum0 = arith.addf %nx4, %nx5 fastmath<contract> : f32
    %fourHighInputSum1 = arith.addf %fourHighInputSum0, %nx6 fastmath<contract> : f32
    %fourHighInputSum = arith.addf %fourHighInputSum1, %nx7 fastmath<contract> : f32
    %fourLowScaled = arith.mulf %fourLow#0, %fourLowLevels fastmath<contract> : f32
    %fourHighScaled = arith.mulf %fourHigh#0, %fourHighLevels fastmath<contract> : f32
    %fourLevels = arith.addf %fourLowScaled, %fourHighScaled fastmath<contract> : f32
    %fourLowShift = arith.mulf %fourLow#1, %fourLowInputSum fastmath<contract> : f32
    %fourHighShift = arith.mulf %fourHigh#1, %fourHighInputSum fastmath<contract> : f32
    %fourShifts = arith.addf %fourLowShift, %fourHighShift fastmath<contract> : f32
    %fourScaled = arith.mulf %fourScale, %fourLevels fastmath<contract> : f32
    %fourShifted = arith.mulf %fourMinimum, %fourShifts fastmath<contract> : f32
    %fourTerm = arith.subf %fourScaled, %fourShifted fastmath<contract> : f32
    %fourNext = arith.addf %fourSum, %fourTerm : f32
    %fourFollowing = arith.addi %fourChunk, %lanes : i64
    cf.br ^four(%fourFollowing, %fourNext : i64, f32)
  ^sixes(%sixChunk: i64, %sixSum: f32):
    %sixMore = arith.cmpi slt, %sixChunk, %sixteens : i64
    cf.cond_br %sixMore, ^sixesBody, ^reduce(%sixSum : f32)
  ^sixesBody:
    %sixBlockBytes = arith.constant 208 : i64
    %sixWordMask = arith.constant 15 : i64
    %sixBlock = arith.shrui %sixChunk, %four : i64
    %sixWord = arith.andi %sixChunk, %sixWordMask : i64
    %sixHalf = arith.shrui %sixWord, %three : i64
    %sixInHalf = arith.andi %sixWord, %seven : i64
    %sixFirst = arith.shli %sixInHalf, %two : i64
    %sixOffset = arith.muli %sixBlock, %sixBlockBytes : i64
    %sixStart = llvm.getelementptr %row[%sixOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixLowHalf = arith.shli %sixHalf, %six : i64
    %sixLowAt = arith.addi %sixLowHalf, %sixFirst : i64
    %sixLowSlot = llvm.getelementptr %sixStart[%sixLowAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixLow = llvm.load %sixLowSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixSecondSlot = llvm.getelementptr %sixLowSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixSecond = llvm.load %sixSecondSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixHighHalf = arith.shli %sixHalf, %five : i64
    %sixHighStart = arith.constant 128 : i64
    %sixHighBase = arith.addi %sixHighStart, %sixHighHalf : i64
    %sixHighAt = arith.addi %sixHighBase, %sixFirst : i64
    %sixHighSlot = llvm.getelementptr %sixStart[%sixHighAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixHigh = llvm.load %sixHighSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixScaleStart = arith.constant 192 : i64
    %sixScaleHalf = arith.shli %sixHalf, %three : i64
    %sixSixteenth = arith.shrui %sixInHalf, %two : i64
    %sixScaleBase = arith.addi %sixScaleStart, %sixScaleHalf : i64
    %sixScaleAt = arith.addi %sixScaleBase, %sixSixteenth : i64
    %sixScaleSlot = llvm.getelementptr %sixStart[%sixScaleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixHalves = llvm.getelementptr %row[%aligned] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixHalfSlot = llvm.getelementptr %sixHalves[%sixBlock] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %sixHalfScale = llvm.load %sixHalfSlot : !llvm.ptr -> f16
    %sixScale = arith.extf %sixHalfScale : f16 to f32
    %sixBlockFirst = arith.shli %sixBlock, %eightBits : i64
    %sixHalfFirst = arith.shli %sixHalf, %seven : i64
    %sixPartStart = arith.addi %sixBlockFirst, %sixHalfFirst : i64
    %sixInputAt = arith.addi %sixPartStart, %sixFirst : i64
    %sixInputSlot = llvm.getelementptr %other[%sixInputAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sixCentered = arith.constant 8388640.0 : f32
    %sixPairMask = arith.constant 3 : i32
    %sq0Slot = llvm.getelementptr %sixScaleSlot[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %sq0Byte = llvm.load %sq0Slot : !llvm.ptr -> i8
    %sq0Wide = arith.extui %sq0Byte : i8 to i32
    %sq0Moved = arith.xori %sq0Wide, %signBit : i32
    %sq0Magic = arith.ori %sq0Moved, %magic : i32
    %sq0Float = arith.bitcast %sq0Magic : i32 to f32
    %sq0 = arith.subf %sq0Float, %signedFloat : f32
    %sx0Slot = llvm.getelementptr %sixInputSlot[0] : (!llvm.ptr) -> !llvm.ptr, f32
    %sx0Four = llvm.load %sx0Slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sx00 = llvm.extractelement %sx0Four[%c0 : i32] : vector<4xf32>
    %sv00ls = arith.constant 0 : i32
    %sv00lw = arith.shrui %sixLow, %sv00ls : i32
    %sv00l = arith.andi %sv00lw, %fifteen : i32
    %sv00hs = arith.constant 0 : i32
    %sv00hw = arith.shrui %sixHigh, %sv00hs : i32
    %sv00h = arith.andi %sv00hw, %sixPairMask : i32
    %sv00hp = arith.shli %sv00h, %fourBits : i32
    %sv00c = arith.ori %sv00l, %sv00hp : i32
    %sv00m = arith.ori %sv00c, %magic : i32
    %sv00f = arith.bitcast %sv00m : i32 to f32
    %sv00 = arith.subf %sv00f, %sixCentered : f32
    %sx01 = llvm.extractelement %sx0Four[%c1 : i32] : vector<4xf32>
    %sv01ls = arith.constant 8 : i32
    %sv01lw = arith.shrui %sixLow, %sv01ls : i32
    %sv01l = arith.andi %sv01lw, %fifteen : i32
    %sv01hs = arith.constant 8 : i32
    %sv01hw = arith.shrui %sixHigh, %sv01hs : i32
    %sv01h = arith.andi %sv01hw, %sixPairMask : i32
    %sv01hp = arith.shli %sv01h, %fourBits : i32
    %sv01c = arith.ori %sv01l, %sv01hp : i32
    %sv01m = arith.ori %sv01c, %magic : i32
    %sv01f = arith.bitcast %sv01m : i32 to f32
    %sv01 = arith.subf %sv01f, %sixCentered : f32
    %sx02 = llvm.extractelement %sx0Four[%c2 : i32] : vector<4xf32>
    %sv02ls = arith.constant 16 : i32
    %sv02lw = arith.shrui %sixLow, %sv02ls : i32
    %sv02l = arith.andi %sv02lw, %fifteen : i32
    %sv02hs = arith.constant 16 : i32
    %sv02hw = arith.shrui %sixHigh, %sv02hs : i32
    %sv02h = arith.andi %sv02hw, %sixPairMask : i32
    %sv02hp = arith.shli %sv02h, %fourBits : i32
    %sv02c = arith.ori %sv02l, %sv02hp : i32
    %sv02m = arith.ori %sv02c, %magic : i32
    %sv02f = arith.bitcast %sv02m : i32 to f32
    %sv02 = arith.subf %sv02f, %sixCentered : f32
    %sx03 = llvm.extractelement %sx0Four[%c3 : i32] : vector<4xf32>
    %sv03ls = arith.constant 24 : i32
    %sv03lw = arith.shrui %sixLow, %sv03ls : i32
    %sv03l = arith.andi %sv03lw, %fifteen : i32
    %sv03hs = arith.constant 24 : i32
    %sv03hw = arith.shrui %sixHigh, %sv03hs : i32
    %sv03h = arith.andi %sv03hw, %sixPairMask : i32
    %sv03hp = arith.shli %sv03h, %fourBits : i32
    %sv03c = arith.ori %sv03l, %sv03hp : i32
    %sv03m = arith.ori %sv03c, %magic : i32
    %sv03f = arith.bitcast %sv03m : i32 to f32
    %sv03 = arith.subf %sv03f, %sixCentered : f32
    %sd0p0 = arith.mulf %sv00, %sx00 fastmath<contract> : f32
    %sd0p1 = arith.mulf %sv01, %sx01 fastmath<contract> : f32
    %sd0p2 = arith.mulf %sv02, %sx02 fastmath<contract> : f32
    %sd0p3 = arith.mulf %sv03, %sx03 fastmath<contract> : f32
    %sd00 = arith.addf %sd0p0, %sd0p1 fastmath<contract> : f32
    %sd01 = arith.addf %sd00, %sd0p2 fastmath<contract> : f32
    %sd0 = arith.addf %sd01, %sd0p3 fastmath<contract> : f32
    %ss0 = arith.mulf %sq0, %sd0 fastmath<contract> : f32
    %sq1Slot = llvm.getelementptr %sixScaleSlot[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %sq1Byte = llvm.load %sq1Slot : !llvm.ptr -> i8
    %sq1Wide = arith.extui %sq1Byte : i8 to i32
    %sq1Moved = arith.xori %sq1Wide, %signBit : i32
    %sq1Magic = arith.ori %sq1Moved, %magic : i32
    %sq1Float = arith.bitcast %sq1Magic : i32 to f32
    %sq1 = arith.subf %sq1Float, %signedFloat : f32
    %sx1Slot = llvm.getelementptr %sixInputSlot[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %sx1Four = llvm.load %sx1Slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sx10 = llvm.extractelement %sx1Four[%c0 : i32] : vector<4xf32>
    %sv10ls = arith.constant 0 : i32
    %sv10lw = arith.shrui %sixSecond, %sv10ls : i32
    %sv10l = arith.andi %sv10lw, %fifteen : i32
    %sv10hs = arith.constant 2 : i32
    %sv10hw = arith.shrui %sixHigh, %sv10hs : i32
    %sv10h = arith.andi %sv10hw, %sixPairMask : i32
    %sv10hp = arith.shli %sv10h, %fourBits : i32
    %sv10c = arith.ori %sv10l, %sv10hp : i32
    %sv10m = arith.ori %sv10c, %magic : i32
    %sv10f = arith.bitcast %sv10m : i32 to f32
    %sv10 = arith.subf %sv10f, %sixCentered : f32
    %sx11 = llvm.extractelement %sx1Four[%c1 : i32] : vector<4xf32>
    %sv11ls = arith.constant 8 : i32
    %sv11lw = arith.shrui %sixSecond, %sv11ls : i32
    %sv11l = arith.andi %sv11lw, %fifteen : i32
    %sv11hs = arith.constant 10 : i32
    %sv11hw = arith.shrui %sixHigh, %sv11hs : i32
    %sv11h = arith.andi %sv11hw, %sixPairMask : i32
    %sv11hp = arith.shli %sv11h, %fourBits : i32
    %sv11c = arith.ori %sv11l, %sv11hp : i32
    %sv11m = arith.ori %sv11c, %magic : i32
    %sv11f = arith.bitcast %sv11m : i32 to f32
    %sv11 = arith.subf %sv11f, %sixCentered : f32
    %sx12 = llvm.extractelement %sx1Four[%c2 : i32] : vector<4xf32>
    %sv12ls = arith.constant 16 : i32
    %sv12lw = arith.shrui %sixSecond, %sv12ls : i32
    %sv12l = arith.andi %sv12lw, %fifteen : i32
    %sv12hs = arith.constant 18 : i32
    %sv12hw = arith.shrui %sixHigh, %sv12hs : i32
    %sv12h = arith.andi %sv12hw, %sixPairMask : i32
    %sv12hp = arith.shli %sv12h, %fourBits : i32
    %sv12c = arith.ori %sv12l, %sv12hp : i32
    %sv12m = arith.ori %sv12c, %magic : i32
    %sv12f = arith.bitcast %sv12m : i32 to f32
    %sv12 = arith.subf %sv12f, %sixCentered : f32
    %sx13 = llvm.extractelement %sx1Four[%c3 : i32] : vector<4xf32>
    %sv13ls = arith.constant 24 : i32
    %sv13lw = arith.shrui %sixSecond, %sv13ls : i32
    %sv13l = arith.andi %sv13lw, %fifteen : i32
    %sv13hs = arith.constant 26 : i32
    %sv13hw = arith.shrui %sixHigh, %sv13hs : i32
    %sv13h = arith.andi %sv13hw, %sixPairMask : i32
    %sv13hp = arith.shli %sv13h, %fourBits : i32
    %sv13c = arith.ori %sv13l, %sv13hp : i32
    %sv13m = arith.ori %sv13c, %magic : i32
    %sv13f = arith.bitcast %sv13m : i32 to f32
    %sv13 = arith.subf %sv13f, %sixCentered : f32
    %sd1p0 = arith.mulf %sv10, %sx10 fastmath<contract> : f32
    %sd1p1 = arith.mulf %sv11, %sx11 fastmath<contract> : f32
    %sd1p2 = arith.mulf %sv12, %sx12 fastmath<contract> : f32
    %sd1p3 = arith.mulf %sv13, %sx13 fastmath<contract> : f32
    %sd10 = arith.addf %sd1p0, %sd1p1 fastmath<contract> : f32
    %sd11 = arith.addf %sd10, %sd1p2 fastmath<contract> : f32
    %sd1 = arith.addf %sd11, %sd1p3 fastmath<contract> : f32
    %ss1 = arith.mulf %sq1, %sd1 fastmath<contract> : f32
    %sq2Slot = llvm.getelementptr %sixScaleSlot[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %sq2Byte = llvm.load %sq2Slot : !llvm.ptr -> i8
    %sq2Wide = arith.extui %sq2Byte : i8 to i32
    %sq2Moved = arith.xori %sq2Wide, %signBit : i32
    %sq2Magic = arith.ori %sq2Moved, %magic : i32
    %sq2Float = arith.bitcast %sq2Magic : i32 to f32
    %sq2 = arith.subf %sq2Float, %signedFloat : f32
    %sx2Slot = llvm.getelementptr %sixInputSlot[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %sx2Four = llvm.load %sx2Slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sx20 = llvm.extractelement %sx2Four[%c0 : i32] : vector<4xf32>
    %sv20ls = arith.constant 4 : i32
    %sv20lw = arith.shrui %sixLow, %sv20ls : i32
    %sv20l = arith.andi %sv20lw, %fifteen : i32
    %sv20hs = arith.constant 4 : i32
    %sv20hw = arith.shrui %sixHigh, %sv20hs : i32
    %sv20h = arith.andi %sv20hw, %sixPairMask : i32
    %sv20hp = arith.shli %sv20h, %fourBits : i32
    %sv20c = arith.ori %sv20l, %sv20hp : i32
    %sv20m = arith.ori %sv20c, %magic : i32
    %sv20f = arith.bitcast %sv20m : i32 to f32
    %sv20 = arith.subf %sv20f, %sixCentered : f32
    %sx21 = llvm.extractelement %sx2Four[%c1 : i32] : vector<4xf32>
    %sv21ls = arith.constant 12 : i32
    %sv21lw = arith.shrui %sixLow, %sv21ls : i32
    %sv21l = arith.andi %sv21lw, %fifteen : i32
    %sv21hs = arith.constant 12 : i32
    %sv21hw = arith.shrui %sixHigh, %sv21hs : i32
    %sv21h = arith.andi %sv21hw, %sixPairMask : i32
    %sv21hp = arith.shli %sv21h, %fourBits : i32
    %sv21c = arith.ori %sv21l, %sv21hp : i32
    %sv21m = arith.ori %sv21c, %magic : i32
    %sv21f = arith.bitcast %sv21m : i32 to f32
    %sv21 = arith.subf %sv21f, %sixCentered : f32
    %sx22 = llvm.extractelement %sx2Four[%c2 : i32] : vector<4xf32>
    %sv22ls = arith.constant 20 : i32
    %sv22lw = arith.shrui %sixLow, %sv22ls : i32
    %sv22l = arith.andi %sv22lw, %fifteen : i32
    %sv22hs = arith.constant 20 : i32
    %sv22hw = arith.shrui %sixHigh, %sv22hs : i32
    %sv22h = arith.andi %sv22hw, %sixPairMask : i32
    %sv22hp = arith.shli %sv22h, %fourBits : i32
    %sv22c = arith.ori %sv22l, %sv22hp : i32
    %sv22m = arith.ori %sv22c, %magic : i32
    %sv22f = arith.bitcast %sv22m : i32 to f32
    %sv22 = arith.subf %sv22f, %sixCentered : f32
    %sx23 = llvm.extractelement %sx2Four[%c3 : i32] : vector<4xf32>
    %sv23ls = arith.constant 28 : i32
    %sv23lw = arith.shrui %sixLow, %sv23ls : i32
    %sv23l = arith.andi %sv23lw, %fifteen : i32
    %sv23hs = arith.constant 28 : i32
    %sv23hw = arith.shrui %sixHigh, %sv23hs : i32
    %sv23h = arith.andi %sv23hw, %sixPairMask : i32
    %sv23hp = arith.shli %sv23h, %fourBits : i32
    %sv23c = arith.ori %sv23l, %sv23hp : i32
    %sv23m = arith.ori %sv23c, %magic : i32
    %sv23f = arith.bitcast %sv23m : i32 to f32
    %sv23 = arith.subf %sv23f, %sixCentered : f32
    %sd2p0 = arith.mulf %sv20, %sx20 fastmath<contract> : f32
    %sd2p1 = arith.mulf %sv21, %sx21 fastmath<contract> : f32
    %sd2p2 = arith.mulf %sv22, %sx22 fastmath<contract> : f32
    %sd2p3 = arith.mulf %sv23, %sx23 fastmath<contract> : f32
    %sd20 = arith.addf %sd2p0, %sd2p1 fastmath<contract> : f32
    %sd21 = arith.addf %sd20, %sd2p2 fastmath<contract> : f32
    %sd2 = arith.addf %sd21, %sd2p3 fastmath<contract> : f32
    %ss2 = arith.mulf %sq2, %sd2 fastmath<contract> : f32
    %sq3Slot = llvm.getelementptr %sixScaleSlot[6] : (!llvm.ptr) -> !llvm.ptr, i8
    %sq3Byte = llvm.load %sq3Slot : !llvm.ptr -> i8
    %sq3Wide = arith.extui %sq3Byte : i8 to i32
    %sq3Moved = arith.xori %sq3Wide, %signBit : i32
    %sq3Magic = arith.ori %sq3Moved, %magic : i32
    %sq3Float = arith.bitcast %sq3Magic : i32 to f32
    %sq3 = arith.subf %sq3Float, %signedFloat : f32
    %sx3Slot = llvm.getelementptr %sixInputSlot[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %sx3Four = llvm.load %sx3Slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sx30 = llvm.extractelement %sx3Four[%c0 : i32] : vector<4xf32>
    %sv30ls = arith.constant 4 : i32
    %sv30lw = arith.shrui %sixSecond, %sv30ls : i32
    %sv30l = arith.andi %sv30lw, %fifteen : i32
    %sv30hs = arith.constant 6 : i32
    %sv30hw = arith.shrui %sixHigh, %sv30hs : i32
    %sv30h = arith.andi %sv30hw, %sixPairMask : i32
    %sv30hp = arith.shli %sv30h, %fourBits : i32
    %sv30c = arith.ori %sv30l, %sv30hp : i32
    %sv30m = arith.ori %sv30c, %magic : i32
    %sv30f = arith.bitcast %sv30m : i32 to f32
    %sv30 = arith.subf %sv30f, %sixCentered : f32
    %sx31 = llvm.extractelement %sx3Four[%c1 : i32] : vector<4xf32>
    %sv31ls = arith.constant 12 : i32
    %sv31lw = arith.shrui %sixSecond, %sv31ls : i32
    %sv31l = arith.andi %sv31lw, %fifteen : i32
    %sv31hs = arith.constant 14 : i32
    %sv31hw = arith.shrui %sixHigh, %sv31hs : i32
    %sv31h = arith.andi %sv31hw, %sixPairMask : i32
    %sv31hp = arith.shli %sv31h, %fourBits : i32
    %sv31c = arith.ori %sv31l, %sv31hp : i32
    %sv31m = arith.ori %sv31c, %magic : i32
    %sv31f = arith.bitcast %sv31m : i32 to f32
    %sv31 = arith.subf %sv31f, %sixCentered : f32
    %sx32 = llvm.extractelement %sx3Four[%c2 : i32] : vector<4xf32>
    %sv32ls = arith.constant 20 : i32
    %sv32lw = arith.shrui %sixSecond, %sv32ls : i32
    %sv32l = arith.andi %sv32lw, %fifteen : i32
    %sv32hs = arith.constant 22 : i32
    %sv32hw = arith.shrui %sixHigh, %sv32hs : i32
    %sv32h = arith.andi %sv32hw, %sixPairMask : i32
    %sv32hp = arith.shli %sv32h, %fourBits : i32
    %sv32c = arith.ori %sv32l, %sv32hp : i32
    %sv32m = arith.ori %sv32c, %magic : i32
    %sv32f = arith.bitcast %sv32m : i32 to f32
    %sv32 = arith.subf %sv32f, %sixCentered : f32
    %sx33 = llvm.extractelement %sx3Four[%c3 : i32] : vector<4xf32>
    %sv33ls = arith.constant 28 : i32
    %sv33lw = arith.shrui %sixSecond, %sv33ls : i32
    %sv33l = arith.andi %sv33lw, %fifteen : i32
    %sv33hs = arith.constant 30 : i32
    %sv33hw = arith.shrui %sixHigh, %sv33hs : i32
    %sv33h = arith.andi %sv33hw, %sixPairMask : i32
    %sv33hp = arith.shli %sv33h, %fourBits : i32
    %sv33c = arith.ori %sv33l, %sv33hp : i32
    %sv33m = arith.ori %sv33c, %magic : i32
    %sv33f = arith.bitcast %sv33m : i32 to f32
    %sv33 = arith.subf %sv33f, %sixCentered : f32
    %sd3p0 = arith.mulf %sv30, %sx30 fastmath<contract> : f32
    %sd3p1 = arith.mulf %sv31, %sx31 fastmath<contract> : f32
    %sd3p2 = arith.mulf %sv32, %sx32 fastmath<contract> : f32
    %sd3p3 = arith.mulf %sv33, %sx33 fastmath<contract> : f32
    %sd30 = arith.addf %sd3p0, %sd3p1 fastmath<contract> : f32
    %sd31 = arith.addf %sd30, %sd3p2 fastmath<contract> : f32
    %sd3 = arith.addf %sd31, %sd3p3 fastmath<contract> : f32
    %ss3 = arith.mulf %sq3, %sd3 fastmath<contract> : f32
    %sixLevels0 = arith.addf %ss0, %ss1 fastmath<contract> : f32
    %sixLevels1 = arith.addf %sixLevels0, %ss2 fastmath<contract> : f32
    %sixLevels = arith.addf %sixLevels1, %ss3 fastmath<contract> : f32
    %sixTerm = arith.mulf %sixScale, %sixLevels fastmath<contract> : f32
    %sixNext = arith.addf %sixSum, %sixTerm : f32
    %sixFollowing = arith.addi %sixChunk, %lanes : i64
    cf.br ^sixes(%sixFollowing, %sixNext : i64, f32)
  ^bytes(%byteChunk: i64, %byteSum: f32):
    %byteMore = arith.cmpi slt, %byteChunk, %chunks : i64
    cf.cond_br %byteMore, ^bytesBody, ^reduce(%byteSum : f32)
  ^bytesBody:
    %be = arith.shli %byteChunk, %three : i64
    %y0, %y1, %y2, %y3, %y4, %y5, %y6, %y7 = func.call @strict_device_eight(%other, %be) : (!llvm.ptr, i64) -> (f32, f32, f32, f32, f32, f32, f32, f32)
    %lowSlot = llvm.getelementptr %row[%be] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %low = llvm.load %lowSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %highSlot = llvm.getelementptr %lowSlot[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %high = llvm.load %highSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %bs0 = arith.constant 0 : i32
    %bw0 = arith.shrui %low, %bs0 : i32
    %bt0u = arith.andi %bw0, %byteMask : i32
    %bt0x = arith.xori %bt0u, %signBit : i32
    %bt0m = arith.ori %bt0x, %magic : i32
    %b0f = arith.bitcast %bt0m : i32 to f32
    %b0 = arith.subf %b0f, %signedFloat : f32
    %bs1 = arith.constant 8 : i32
    %bw1 = arith.shrui %low, %bs1 : i32
    %bt1u = arith.andi %bw1, %byteMask : i32
    %bt1x = arith.xori %bt1u, %signBit : i32
    %bt1m = arith.ori %bt1x, %magic : i32
    %b1f = arith.bitcast %bt1m : i32 to f32
    %b1 = arith.subf %b1f, %signedFloat : f32
    %bs2 = arith.constant 16 : i32
    %bw2 = arith.shrui %low, %bs2 : i32
    %bt2u = arith.andi %bw2, %byteMask : i32
    %bt2x = arith.xori %bt2u, %signBit : i32
    %bt2m = arith.ori %bt2x, %magic : i32
    %b2f = arith.bitcast %bt2m : i32 to f32
    %b2 = arith.subf %b2f, %signedFloat : f32
    %bs3 = arith.constant 24 : i32
    %bw3 = arith.shrui %low, %bs3 : i32
    %bt3u = arith.andi %bw3, %byteMask : i32
    %bt3x = arith.xori %bt3u, %signBit : i32
    %bt3m = arith.ori %bt3x, %magic : i32
    %b3f = arith.bitcast %bt3m : i32 to f32
    %b3 = arith.subf %b3f, %signedFloat : f32
    %bs4 = arith.constant 0 : i32
    %bw4 = arith.shrui %high, %bs4 : i32
    %bt4u = arith.andi %bw4, %byteMask : i32
    %bt4x = arith.xori %bt4u, %signBit : i32
    %bt4m = arith.ori %bt4x, %magic : i32
    %b4f = arith.bitcast %bt4m : i32 to f32
    %b4 = arith.subf %b4f, %signedFloat : f32
    %bs5 = arith.constant 8 : i32
    %bw5 = arith.shrui %high, %bs5 : i32
    %bt5u = arith.andi %bw5, %byteMask : i32
    %bt5x = arith.xori %bt5u, %signBit : i32
    %bt5m = arith.ori %bt5x, %magic : i32
    %b5f = arith.bitcast %bt5m : i32 to f32
    %b5 = arith.subf %b5f, %signedFloat : f32
    %bs6 = arith.constant 16 : i32
    %bw6 = arith.shrui %high, %bs6 : i32
    %bt6u = arith.andi %bw6, %byteMask : i32
    %bt6x = arith.xori %bt6u, %signBit : i32
    %bt6m = arith.ori %bt6x, %magic : i32
    %b6f = arith.bitcast %bt6m : i32 to f32
    %b6 = arith.subf %b6f, %signedFloat : f32
    %bs7 = arith.constant 24 : i32
    %bw7 = arith.shrui %high, %bs7 : i32
    %bt7u = arith.andi %bw7, %byteMask : i32
    %bt7x = arith.xori %bt7u, %signBit : i32
    %bt7m = arith.ori %bt7x, %magic : i32
    %b7f = arith.bitcast %bt7m : i32 to f32
    %b7 = arith.subf %b7f, %signedFloat : f32
    %quantizedp0 = arith.mulf %b0, %y0 fastmath<contract> : f32
    %quantizedp1 = arith.mulf %b1, %y1 fastmath<contract> : f32
    %quantizedp2 = arith.mulf %b2, %y2 fastmath<contract> : f32
    %quantizedp3 = arith.mulf %b3, %y3 fastmath<contract> : f32
    %quantizedp4 = arith.mulf %b4, %y4 fastmath<contract> : f32
    %quantizedp5 = arith.mulf %b5, %y5 fastmath<contract> : f32
    %quantizedp6 = arith.mulf %b6, %y6 fastmath<contract> : f32
    %quantizedp7 = arith.mulf %b7, %y7 fastmath<contract> : f32
    %quantized0 = arith.addf %quantizedp0, %quantizedp1 fastmath<contract> : f32
    %quantized1 = arith.addf %quantized0, %quantizedp2 fastmath<contract> : f32
    %quantized2 = arith.addf %quantized1, %quantizedp3 fastmath<contract> : f32
    %quantized3 = arith.addf %quantized2, %quantizedp4 fastmath<contract> : f32
    %quantized4 = arith.addf %quantized3, %quantizedp5 fastmath<contract> : f32
    %quantized5 = arith.addf %quantized4, %quantizedp6 fastmath<contract> : f32
    %quantized = arith.addf %quantized5, %quantizedp7 fastmath<contract> : f32
    %blockIndex = arith.shrui %be, %five : i64
    %blockSlot = llvm.getelementptr %scales[%blockIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %blockScale = llvm.load %blockSlot : !llvm.ptr -> f32
    %byteTerm = arith.mulf %blockScale, %quantized fastmath<contract> : f32
    %byteNext = arith.addf %byteSum, %byteTerm : f32
    %byteFollowing = arith.addi %byteChunk, %lanes : i64
    cf.br ^bytes(%byteFollowing, %byteNext : i64, f32)
  ^halves(%halfChunk: i64, %halfSum: f32):
    %halfMore = arith.cmpi slt, %halfChunk, %chunks : i64
    cf.cond_br %halfMore, ^halvesBody, ^reduce(%halfSum : f32)
  ^halvesBody:
    %he = arith.shli %halfChunk, %three : i64
    %z0, %z1, %z2, %z3, %z4, %z5, %z6, %z7 = func.call @strict_device_eight(%other, %he) : (!llvm.ptr, i64) -> (f32, f32, f32, f32, f32, f32, f32, f32)
    %halfSlot = llvm.getelementptr %row[%he] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfEight = llvm.load %halfSlot {alignment = 16 : i64} : !llvm.ptr -> vector<8xf16>
    %hc0 = llvm.mlir.constant(0 : i32) : i32
    %hh0 = llvm.extractelement %halfEight[%hc0 : i32] : vector<8xf16>
    %h0 = arith.extf %hh0 : f16 to f32
    %hc1 = llvm.mlir.constant(1 : i32) : i32
    %hh1 = llvm.extractelement %halfEight[%hc1 : i32] : vector<8xf16>
    %h1 = arith.extf %hh1 : f16 to f32
    %hc2 = llvm.mlir.constant(2 : i32) : i32
    %hh2 = llvm.extractelement %halfEight[%hc2 : i32] : vector<8xf16>
    %h2 = arith.extf %hh2 : f16 to f32
    %hc3 = llvm.mlir.constant(3 : i32) : i32
    %hh3 = llvm.extractelement %halfEight[%hc3 : i32] : vector<8xf16>
    %h3 = arith.extf %hh3 : f16 to f32
    %hc4 = llvm.mlir.constant(4 : i32) : i32
    %hh4 = llvm.extractelement %halfEight[%hc4 : i32] : vector<8xf16>
    %h4 = arith.extf %hh4 : f16 to f32
    %hc5 = llvm.mlir.constant(5 : i32) : i32
    %hh5 = llvm.extractelement %halfEight[%hc5 : i32] : vector<8xf16>
    %h5 = arith.extf %hh5 : f16 to f32
    %hc6 = llvm.mlir.constant(6 : i32) : i32
    %hh6 = llvm.extractelement %halfEight[%hc6 : i32] : vector<8xf16>
    %h6 = arith.extf %hh6 : f16 to f32
    %hc7 = llvm.mlir.constant(7 : i32) : i32
    %hh7 = llvm.extractelement %halfEight[%hc7 : i32] : vector<8xf16>
    %h7 = arith.extf %hh7 : f16 to f32
    %halfDotp0 = arith.mulf %h0, %z0 fastmath<contract> : f32
    %halfDotp1 = arith.mulf %h1, %z1 fastmath<contract> : f32
    %halfDotp2 = arith.mulf %h2, %z2 fastmath<contract> : f32
    %halfDotp3 = arith.mulf %h3, %z3 fastmath<contract> : f32
    %halfDotp4 = arith.mulf %h4, %z4 fastmath<contract> : f32
    %halfDotp5 = arith.mulf %h5, %z5 fastmath<contract> : f32
    %halfDotp6 = arith.mulf %h6, %z6 fastmath<contract> : f32
    %halfDotp7 = arith.mulf %h7, %z7 fastmath<contract> : f32
    %halfDot0 = arith.addf %halfDotp0, %halfDotp1 fastmath<contract> : f32
    %halfDot1 = arith.addf %halfDot0, %halfDotp2 fastmath<contract> : f32
    %halfDot2 = arith.addf %halfDot1, %halfDotp3 fastmath<contract> : f32
    %halfDot3 = arith.addf %halfDot2, %halfDotp4 fastmath<contract> : f32
    %halfDot4 = arith.addf %halfDot3, %halfDotp5 fastmath<contract> : f32
    %halfDot5 = arith.addf %halfDot4, %halfDotp6 fastmath<contract> : f32
    %halfDot = arith.addf %halfDot5, %halfDotp7 fastmath<contract> : f32
    %halfNext = arith.addf %halfSum, %halfDot : f32
    %halfFollowing = arith.addi %halfChunk, %lanes : i64
    cf.br ^halves(%halfFollowing, %halfNext : i64, f32)
  ^singles(%singleChunk: i64, %singleSum: f32):
    %singleMore = arith.cmpi slt, %singleChunk, %chunks : i64
    cf.cond_br %singleMore, ^singlesBody, ^reduce(%singleSum : f32)
  ^singlesBody:
    %se = arith.shli %singleChunk, %three : i64
    %w0, %w1, %w2, %w3, %w4, %w5, %w6, %w7 = func.call @strict_device_eight(%other, %se) : (!llvm.ptr, i64) -> (f32, f32, f32, f32, f32, f32, f32, f32)
    %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7 = func.call @strict_device_eight(%row, %se) : (!llvm.ptr, i64) -> (f32, f32, f32, f32, f32, f32, f32, f32)
    %singleDotp0 = arith.mulf %v0, %w0 fastmath<contract> : f32
    %singleDotp1 = arith.mulf %v1, %w1 fastmath<contract> : f32
    %singleDotp2 = arith.mulf %v2, %w2 fastmath<contract> : f32
    %singleDotp3 = arith.mulf %v3, %w3 fastmath<contract> : f32
    %singleDotp4 = arith.mulf %v4, %w4 fastmath<contract> : f32
    %singleDotp5 = arith.mulf %v5, %w5 fastmath<contract> : f32
    %singleDotp6 = arith.mulf %v6, %w6 fastmath<contract> : f32
    %singleDotp7 = arith.mulf %v7, %w7 fastmath<contract> : f32
    %singleDot0 = arith.addf %singleDotp0, %singleDotp1 fastmath<contract> : f32
    %singleDot1 = arith.addf %singleDot0, %singleDotp2 fastmath<contract> : f32
    %singleDot2 = arith.addf %singleDot1, %singleDotp3 fastmath<contract> : f32
    %singleDot3 = arith.addf %singleDot2, %singleDotp4 fastmath<contract> : f32
    %singleDot4 = arith.addf %singleDot3, %singleDotp5 fastmath<contract> : f32
    %singleDot5 = arith.addf %singleDot4, %singleDotp6 fastmath<contract> : f32
    %singleDot = arith.addf %singleDot5, %singleDotp7 fastmath<contract> : f32
    %singleNext = arith.addf %singleSum, %singleDot : f32
    %singleFollowing = arith.addi %singleChunk, %lanes : i64
    cf.br ^singles(%singleFollowing, %singleNext : i64, f32)
  ^generic(%index: i64, %genericSum: f32):
    %genericMore = arith.cmpi slt, %index, %count : i64
    cf.cond_br %genericMore, ^genericBody, ^reduce(%genericSum : f32)
  ^genericBody:
    %left = func.call @strict_device_elements(%row, %rowBits, %index, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    %right = func.call @strict_device_elements(%other, %otherBits, %index, %otherCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %genericProduct = arith.mulf %left, %right fastmath<contract> : f32
    %genericNext = arith.addf %genericSum, %genericProduct fastmath<contract> : f32
    %genericFollowing = arith.addi %index, %lanes : i64
    cf.br ^generic(%genericFollowing, %genericNext : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_sums(%total) : (f32) -> f32
    return %reduced : f32
  }
  // the dot product of count numbers of two lists from their starts, aligned 32 bit floats go through strict_device_dots
  func.func private @strict_device_offset_dots(%failed: !llvm.ptr, %x: !llvm.ptr, %xBits: i64, %xCount: i64, %xStart: i64, %y: !llvm.ptr, %yBits: i64, %yCount: i64, %yStart: i64, %count: i64, %lane: i64, %lanes: i64) -> f32 {
    %zero = arith.constant 0.0 : f32
    %none = arith.constant 0 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %thirtyTwo = arith.constant 32 : i64
    %xEnd = arith.addi %xStart, %count : i64
    %yEnd = arith.addi %yStart, %count : i64
    %xLow = arith.cmpi slt, %xStart, %none : i64
    %yLow = arith.cmpi slt, %yStart, %none : i64
    %xHigh = arith.cmpi sgt, %xEnd, %xCount : i64
    %yHigh = arith.cmpi sgt, %yEnd, %yCount : i64
    %xOutside = arith.ori %xLow, %xHigh : i1
    %yOutside = arith.ori %yLow, %yHigh : i1
    %outside = arith.ori %xOutside, %yOutside : i1
    cf.cond_br %outside, ^fail, ^check
  ^fail:
    %failure = arith.constant 1 : i64
    llvm.store %failure, %failed : i64, !llvm.ptr
    return %zero : f32
  ^check:
    %xFloats = arith.cmpi eq, %xBits, %thirtyTwo : i64
    %yFloats = arith.cmpi eq, %yBits, %thirtyTwo : i64
    %floats = arith.andi %xFloats, %yFloats : i1
    %xFirst = llvm.getelementptr %x[%xStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %yFirst = llvm.getelementptr %y[%yStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %xAddress = llvm.ptrtoint %xFirst : !llvm.ptr to i64
    %yAddress = llvm.ptrtoint %yFirst : !llvm.ptr to i64
    %addresses = arith.ori %xAddress, %yAddress : i64
    %misaligned = arith.andi %addresses, %fifteen : i64
    %isAligned = arith.cmpi eq, %misaligned, %none : i64
    %rest = arith.andi %count, %seven : i64
    %whole = arith.cmpi eq, %rest, %none : i64
    %alignedFloats = arith.andi %floats, %isAligned : i1
    %fast = arith.andi %alignedFloats, %whole : i1
    cf.cond_br %fast, ^dots, ^generic(%lane, %zero : i64, f32)
  ^dots:
    %dot = func.call @strict_device_dots(%failed, %xFirst, %thirtyTwo, %count, %yFirst, %thirtyTwo, %count, %lane, %lanes) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    return %dot : f32
  ^generic(%index: i64, %sum: f32):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^reduce(%sum : f32)
  ^body:
    %xAt = arith.addi %xStart, %index : i64
    %yAt = arith.addi %yStart, %index : i64
    %left = func.call @strict_device_elements(%x, %xBits, %xAt, %xCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %right = func.call @strict_device_elements(%y, %yBits, %yAt, %yCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %product = arith.mulf %left, %right fastmath<contract> : f32
    %next = arith.addf %sum, %product fastmath<contract> : f32
    %following = arith.addi %index, %lanes : i64
    cf.br ^generic(%following, %next : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_sums(%total) : (f32) -> f32
    return %reduced : f32
  }
