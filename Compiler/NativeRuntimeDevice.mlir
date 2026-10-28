  // strict_device_mma16 of unsigned 8 bit a (0 to 255) with the signed product: twice the product of a / 2 plus that of
  // the lowest bits of a
  func.func private @strict_device_mma16_halves(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %b0: i32, %b1: i32, %c0: i32, %c1: i32, %c2: i32, %c3: i32) -> !llvm.struct<(i32, i32, i32, i32)> attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %halfMask = arith.constant 2139062143 : i32
    %lowMask = arith.constant 16843009 : i32
    %shifted0 = arith.shrui %a0, %one : i32
    %shifted1 = arith.shrui %a1, %one : i32
    %shifted2 = arith.shrui %a2, %one : i32
    %shifted3 = arith.shrui %a3, %one : i32
    %half0 = arith.andi %shifted0, %halfMask : i32
    %half1 = arith.andi %shifted1, %halfMask : i32
    %half2 = arith.andi %shifted2, %halfMask : i32
    %half3 = arith.andi %shifted3, %halfMask : i32
    %low0 = arith.andi %a0, %lowMask : i32
    %low1 = arith.andi %a1, %lowMask : i32
    %low2 = arith.andi %a2, %lowMask : i32
    %low3 = arith.andi %a3, %lowMask : i32
    %halves = func.call @strict_device_mma16(%half0, %half1, %half2, %half3, %b0, %b1, %zero, %zero, %zero, %zero) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %lows = func.call @strict_device_mma16(%low0, %low1, %low2, %low3, %b0, %b1, %c0, %c1, %c2, %c3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %halves0 = llvm.extractvalue %halves[0] : !llvm.struct<(i32, i32, i32, i32)>
    %halves1 = llvm.extractvalue %halves[1] : !llvm.struct<(i32, i32, i32, i32)>
    %halves2 = llvm.extractvalue %halves[2] : !llvm.struct<(i32, i32, i32, i32)>
    %halves3 = llvm.extractvalue %halves[3] : !llvm.struct<(i32, i32, i32, i32)>
    %lows0 = llvm.extractvalue %lows[0] : !llvm.struct<(i32, i32, i32, i32)>
    %lows1 = llvm.extractvalue %lows[1] : !llvm.struct<(i32, i32, i32, i32)>
    %lows2 = llvm.extractvalue %lows[2] : !llvm.struct<(i32, i32, i32, i32)>
    %lows3 = llvm.extractvalue %lows[3] : !llvm.struct<(i32, i32, i32, i32)>
    %twice0 = arith.shli %halves0, %one : i32
    %twice1 = arith.shli %halves1, %one : i32
    %twice2 = arith.shli %halves2, %one : i32
    %twice3 = arith.shli %halves3, %one : i32
    %d0 = arith.addi %lows0, %twice0 : i32
    %d1 = arith.addi %lows1, %twice1 : i32
    %d2 = arith.addi %lows2, %twice2 : i32
    %d3 = arith.addi %lows3, %twice3 : i32
    %empty = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %with0 = llvm.insertvalue %d0, %empty[0] : !llvm.struct<(i32, i32, i32, i32)>
    %with1 = llvm.insertvalue %d1, %with0[1] : !llvm.struct<(i32, i32, i32, i32)>
    %with2 = llvm.insertvalue %d2, %with1[2] : !llvm.struct<(i32, i32, i32, i32)>
    %with3 = llvm.insertvalue %d3, %with2[3] : !llvm.struct<(i32, i32, i32, i32)>
    return %with3 : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func private @strict_device_element(%data: !llvm.ptr, %bits: i64, %at: i64, %count: i64) -> f64 {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^notBrain
  ^brain:
    %brainSingle = func.call @strict_device_bfloat16_read(%data, %at) : (!llvm.ptr, i64) -> f32
    %brainNumber = arith.extf %brainSingle : f32 to f64
    return %brainNumber : f64
  ^notBrain:
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
    cf.cond_br %isQuantized, ^quantized, ^unscaled
  ^unscaled:
    %seven = arith.constant 7 : i64
    %isFloat8 = arith.cmpi eq, %bits, %seven : i64
    cf.cond_br %isFloat8, ^float8, ^bytes
  ^bytes:
    %nine = arith.constant 9 : i64
    %isBytes = arith.cmpi eq, %bits, %nine : i64
    cf.cond_br %isBytes, ^byte, ^double
  ^byte:
    %byteSingle = func.call @strict_device_byte_read(%data, %at) : (!llvm.ptr, i64) -> f32
    %unsignedNumber = arith.extf %byteSingle : f32 to f64
    return %unsignedNumber : f64
  ^float8:
    %float8Single = func.call @strict_device_float8_read(%data, %at) : (!llvm.ptr, i64) -> f32
    %float8Number = arith.extf %float8Single : f32 to f64
    return %float8Number : f64
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
  // an 8 bit float of a list in format 7 (E4M3: sign, 4 exponent bits with bias 7, 3 mantissa bits, at most 448)
  func.func private @strict_device_float8_number(%byte: i8) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %bits = arith.extui %byte : i8 to i32
    %three = arith.constant 3 : i32
    %seven = arith.constant 7 : i32
    %fifteen = arith.constant 15 : i32
    %twenty = arith.constant 20 : i32
    %twentyThree = arith.constant 23 : i32
    %rebias = arith.constant 120 : i32
    %signBit = arith.constant 128 : i32
    %none = arith.constant 0 : i32
    %smallest = arith.constant 0.001953125 : f32
    %shifted = arith.shrui %bits, %three : i32
    %exponent = arith.andi %shifted, %fifteen : i32
    %mantissa = arith.andi %bits, %seven : i32
    %biased = arith.addi %exponent, %rebias : i32
    %exponentBits = arith.shli %biased, %twentyThree : i32
    %mantissaBits = arith.shli %mantissa, %twenty : i32
    %normalBits = arith.ori %exponentBits, %mantissaBits : i32
    %normal = arith.bitcast %normalBits : i32 to f32
    %mantissaNumber = arith.uitofp %mantissa : i32 to f32
    %subnormal = arith.mulf %mantissaNumber, %smallest : f32
    %isSubnormal = arith.cmpi eq, %exponent, %none : i32
    %magnitude = arith.select %isSubnormal, %subnormal, %normal : f32
    %sign = arith.andi %bits, %signBit : i32
    %isNegative = arith.cmpi ne, %sign, %none : i32
    %negated = arith.negf %magnitude : f32
    %number = arith.select %isNegative, %negated, %magnitude : f32
    return %number : f32
  }
  // number at of a list in format 7, kept out of the element readers so their callers keep their registers
  func.func private @strict_device_float8_read(%data: !llvm.ptr, %at: i64) -> f32 attributes {llvm.passthrough = ["noinline"]} {
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %slot : !llvm.ptr -> i8
    %number = func.call @strict_device_float8_number(%byte) : (i8) -> f32
    return %number : f32
  }
  // the dot product of count 32 bit floats of x from xStart with 8 bit floats of y (format 7) from yStart, the lanes
  // add it up (keys of `Attention.Scores` in 8 bits), kept out of line like strict_device_float8_read
  func.func private @strict_device_float8_dots(%x: !llvm.ptr, %xStart: i64, %y: !llvm.ptr, %yStart: i64, %count: i64, %lane: i64, %lanes: i64) -> f32 attributes {llvm.passthrough = ["noinline"]} {
    %zero = arith.constant 0.0 : f32
    cf.br ^loop(%lane, %zero : i64, f32)
  ^loop(%index: i64, %sum: f32):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %xAt = arith.addi %xStart, %index : i64
    %xSlot = llvm.getelementptr %x[%xAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %left = llvm.load %xSlot : !llvm.ptr -> f32
    %yAt = arith.addi %yStart, %index : i64
    %ySlot = llvm.getelementptr %y[%yAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %ySlot : !llvm.ptr -> i8
    %right = func.call @strict_device_float8_number(%byte) : (i8) -> f32
    %product = arith.mulf %left, %right fastmath<contract> : f32
    %next = arith.addf %sum, %product fastmath<contract> : f32
    %following = arith.addi %index, %lanes : i64
    cf.br ^loop(%following, %next : i64, f32)
  ^done:
    %reduced = func.call @strict_device_lanes_sums(%sum, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
  // a byte of a list in format 9 (Bytes), kept out of the element readers like strict_device_float8_read
  func.func private @strict_device_byte_read(%data: !llvm.ptr, %at: i64) -> f32 attributes {llvm.passthrough = ["noinline"]} {
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %slot : !llvm.ptr -> i8
    %number = arith.uitofp %byte : i8 to f32
    return %number : f32
  }
  // the byte of value (format 9): its whole part like a C cast, below 0 is 0, above 255 is 255
  func.func private @strict_device_byte(%value: f32) -> i8 attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0.0 : f32
    %largest = arith.constant 255.0 : f32
    %low = arith.maximumf %value, %zero : f32
    %clamped = arith.minimumf %low, %largest : f32
    %byte = arith.fptoui %clamped : f32 to i8
    return %byte : i8
  }
  // a number of a list in format 17 (bfloat16: the high half of a float)
  func.func private @strict_device_bfloat16_read(%data: !llvm.ptr, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brain = llvm.load %slot : !llvm.ptr -> bf16
    %number = arith.extf %brain : bf16 to f32
    return %number : f32
  }
  // bfloat16 of value (format 17): nearest, ties to even
  func.func private @strict_device_bfloat16(%value: f32) -> bf16 attributes {llvm.passthrough = ["alwaysinline"]} {
    %brain = arith.truncf %value : f32 to bf16
    return %brain : bf16
  }
  // Float8 of value (format 7, see strict_device_float8_number): nearest, ties to even, larger magnitudes become 448
  func.func private @strict_device_float8(%value: f32) -> i8 attributes {llvm.passthrough = ["alwaysinline"]} {
    %bits = arith.bitcast %value : f32 to i32
    %twentyFour = arith.constant 24 : i32
    %twentyThree = arith.constant 23 : i32
    %twenty = arith.constant 20 : i32
    %three = arith.constant 3 : i32
    %signBit = arith.constant 128 : i32
    %rebias = arith.constant 120 : i32
    %mantissaMask = arith.constant 8388607 : i32
    %belowHalf = arith.constant 524287 : i32
    %one = arith.constant 1 : i32
    %largest = arith.constant 448.0 : f32
    %smallestNormal = arith.constant 0.015625 : f32
    %steps = arith.constant 512.0 : f32
    %shiftedSign = arith.shrui %bits, %twentyFour : i32
    %sign = arith.andi %shiftedSign, %signBit : i32
    %magnitude = llvm.call_intrinsic "llvm.fabs.f32"(%value) : (f32) -> f32
    %clamped = arith.minnumf %magnitude, %largest : f32
    %scaled = arith.mulf %clamped, %steps : f32
    %roundedScaled = llvm.call_intrinsic "llvm.roundeven.f32"(%scaled) : (f32) -> f32
    %subnormal = arith.fptoui %roundedScaled : f32 to i32
    %clampedBits = arith.bitcast %clamped : f32 to i32
    %exponentField = arith.shrui %clampedBits, %twentyThree : i32
    %exponent = arith.subi %exponentField, %rebias : i32
    %mantissaField = arith.andi %clampedBits, %mantissaMask : i32
    %lastKept = arith.shrui %mantissaField, %twenty : i32
    %isOdd = arith.andi %lastKept, %one : i32
    %tieUp = arith.addi %belowHalf, %isOdd : i32
    %roundedField = arith.addi %mantissaField, %tieUp : i32
    %mantissa = arith.shrui %roundedField, %twenty : i32
    %exponentBits = arith.shli %exponent, %three : i32
    %normal = arith.addi %exponentBits, %mantissa : i32
    %isSubnormal = arith.cmpf olt, %clamped, %smallestNormal : f32
    %code = arith.select %isSubnormal, %subnormal, %normal : i32
    %signed = arith.ori %code, %sign : i32
    %byte = arith.trunci %signed : i32 to i8
    return %byte : i8
  }
  // Float8.from on the device: the number strict_device_float8 stores
  func.func private @strict_device_float8_rounded(%number: f64) -> f64 attributes {llvm.passthrough = ["alwaysinline"]} {
    %single = arith.truncf %number : f64 to f32
    %rounded = func.call @strict_device_float8_roundeds(%single) : (f32) -> f32
    %wide = arith.extf %rounded : f32 to f64
    return %wide : f64
  }
  func.func private @strict_device_float8_roundeds(%number: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %byte = func.call @strict_device_float8(%number) : (f32) -> i8
    %rounded = func.call @strict_device_float8_number(%byte) : (i8) -> f32
    return %rounded : f32
  }
  // mirrors strict_packed_data of the host runtime: the bytes of the numbers, after them come the scales
  func.func private @strict_device_data(%count: i64, %bits: i64) -> i64 {
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %spare = arith.constant 255 : i64
    %groupSpare = arith.constant 127 : i64
    %fourBlock = arith.constant 144 : i64
    %sixBlock = arith.constant 208 : i64
    %alignMask = arith.constant -4 : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %rounded = arith.addi %count, %spare : i64
    %blocks = arith.shrui %rounded, %eight : i64
    %fourBytes = arith.muli %blocks, %fourBlock : i64
    %sixBytes = arith.muli %blocks, %sixBlock : i64
    %groupRounded = arith.addi %count, %groupSpare : i64
    %groups = arith.shrui %groupRounded, %seven : i64
    %tritBlock = arith.constant 28 : i64
    %isThree = arith.cmpi eq, %bits, %three : i64
    %threeBytes = arith.muli %groups, %tritBlock : i64
    %padded = arith.addi %count, %three : i64
    %aligned = arith.andi %padded, %alignMask : i64
    %orThree = arith.select %isThree, %threeBytes, %aligned : i64
    %orSix = arith.select %isSix, %sixBytes, %orThree : i64
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
  // a row of a matrix, its width and the number of rows, behind the header of strict_gpu_rows (row stride and row
  // count in the first word, one width byte per row)
  func.func private @strict_device_row(%source: !llvm.ptr, %at: i64) -> (!llvm.ptr, i64, i64) {
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %alignMask = arith.constant -16 : i64
    %lowHalf = arith.constant 2147483647 : i64
    %thirtyTwo = arith.constant 32 : i64
    %word = llvm.load %source : !llvm.ptr -> i64
    %stride = arith.andi %word, %lowHalf : i64
    %count = arith.shrui %word, %thirtyTwo : i64
    %widthAt = arith.addi %at, %eight : i64
    %widthSlot = llvm.getelementptr %source[%widthAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %width = llvm.load %widthSlot : !llvm.ptr -> i8
    %bits = arith.extui %width : i8 to i64
    %lineSpare = arith.constant 127 : i64
    %lineMask = arith.constant -128 : i64
    %withTable = arith.addi %count, %eight : i64
    %paddedHeader = arith.addi %withTable, %lineSpare : i64
    %header = arith.andi %paddedHeader, %lineMask : i64
    %rowOffset = arith.muli %at, %stride : i64
    %offset = arith.addi %header, %rowOffset : i64
    %row = llvm.getelementptr %source[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    return %row, %bits, %count : !llvm.ptr, i64, i64
  }
  // the value of the lane whose number differs in the bit of offset, a step of a butterfly reduction over a warp
  func.func private @strict_device_shuffle(%value: f64, %offset: i32) -> f64 {
    %thirtyTwo = arith.constant 32 : i64
    %bits = arith.bitcast %value : f64 to i64
    %low = arith.trunci %bits : i64 to i32
    %shifted = arith.shrui %bits, %thirtyTwo : i64
    %high = arith.trunci %shifted : i64 to i32
    %otherLow = func.call @strict_device_swap(%low, %offset) : (i32, i32) -> i32
    %otherHigh = func.call @strict_device_swap(%high, %offset) : (i32, i32) -> i32
    %lowWide = arith.extui %otherLow : i32 to i64
    %highWide = arith.extui %otherHigh : i32 to i64
    %highPlaced = arith.shli %highWide, %thirtyTwo : i64
    %combined = arith.ori %highPlaced, %lowWide : i64
    %other = arith.bitcast %combined : i64 to f64
    return %other : f64
  }
  func.func private @strict_device_shuffles(%value: f32, %offset: i32) -> f32 {
    %bits = arith.bitcast %value : f32 to i32
    %otherBits = func.call @strict_device_swap(%bits, %offset) : (i32, i32) -> i32
    %other = arith.bitcast %otherBits : i32 to f32
    return %other : f32
  }
  func.func private @strict_device_sum(%value: f64) -> f64 {
    %sixteen = arith.constant 16 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    cf.br ^round(%value, %sixteen : f64, i32)
  ^round(%sum: f64, %offset: i32):
    %more = arith.cmpi sge, %offset, %one : i32
    cf.cond_br %more, ^shuffle, ^done
  ^shuffle:
    %other = func.call @strict_device_shuffle(%sum, %offset) : (f64, i32) -> f64
    %added = arith.addf %sum, %other : f64
    %half = arith.divui %offset, %two : i32
    cf.br ^round(%added, %half : f64, i32)
  ^done:
    return %sum : f64
  }
  llvm.mlir.global internal @strict_device_partials() {addr_space = 3 : i32} : !llvm.array<32 x f64>
  llvm.mlir.global internal @strict_device_partial_floats() {addr_space = 3 : i32} : !llvm.array<32 x f32>
  // the lanes strict_gpu_group_lanes launches per group: a warp up to 1024 numbers, wider groups 256 lanes
  func.func private @strict_device_group_lanes(%group: i64) -> i64 {
    %warp = arith.constant 32 : i64
    %block = arith.constant 256 : i64
    %warpLimit = arith.constant 1024 : i64
    %isWide = arith.cmpi sgt, %group, %warpLimit : i64
    %widest = arith.select %isWide, %block, %warp : i64
    %lanes = arith.minsi %group, %widest : i64
    return %lanes : i64
  }
  // the sum over the lanes of an element: one lane has it, up to 32 lanes are a warp, more lanes are a whole block of
  // warps (a group) that add their warp sums in shared memory, every lane gets the sum
  func.func private @strict_device_lanes_sum(%value: f64, %lane: i64, %lanes: i64) -> f64 {
    %one = arith.constant 1 : i64
    %warpSize = arith.constant 32 : i64
    %single = arith.cmpi sle, %lanes, %one : i64
    cf.cond_br %single, ^done(%value : f64), ^warp
  ^warp:
    %warped = func.call @strict_device_sum(%value) : (f64) -> f64
    %isWarp = arith.cmpi sle, %lanes, %warpSize : i64
    cf.cond_br %isWarp, ^done(%warped : f64), ^block
  ^block:
    %blocked = func.call @strict_device_block_sum(%warped, %lane, %lanes) : (f64, i64, i64) -> f64
    cf.br ^done(%blocked : f64)
  ^done(%sum: f64):
    return %sum : f64
  }
  func.func private @strict_device_lanes_sums(%value: f32, %lane: i64, %lanes: i64) -> f32 {
    %one = arith.constant 1 : i64
    %warpSize = arith.constant 32 : i64
    %single = arith.cmpi sle, %lanes, %one : i64
    cf.cond_br %single, ^done(%value : f32), ^warp
  ^warp:
    %warped = func.call @strict_device_sums(%value) : (f32) -> f32
    %isWarp = arith.cmpi sle, %lanes, %warpSize : i64
    cf.cond_br %isWarp, ^done(%warped : f32), ^block
  ^block:
    %blocked = func.call @strict_device_lanes_blocked(%warped, %lane, %lanes) : (f32, i64, i64) -> f32
    cf.br ^done(%blocked : f32)
  ^done(%sum: f32):
    return %sum : f32
  }
  func.func private @strict_device_lanes_blocked(%warped: f32, %lane: i64, %lanes: i64) -> f32 {
    %none = arith.constant 0 : i64
    %five = arith.constant 5 : i64
    %thirtyOne = arith.constant 31 : i64
    %zero = arith.constant 0.0 : f32
    %partials = llvm.mlir.addressof @strict_device_partial_floats : !llvm.ptr<3>
    %inWarp = arith.andi %lane, %thirtyOne : i64
    %warp = arith.shrui %lane, %five : i64
    %isFirst = arith.cmpi eq, %inWarp, %none : i64
    cf.cond_br %isFirst, ^share, ^shared
  ^share:
    %slot = llvm.getelementptr %partials[0, %warp] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<32 x f32>
    llvm.store %warped, %slot : f32, !llvm.ptr<3>
    cf.br ^shared
  ^shared:
    func.call @strict_device_barrier() : () -> ()
    %warps = arith.shrui %lanes, %five : i64
    %isPartial = arith.cmpi slt, %inWarp, %warps : i64
    %partialSlot = llvm.getelementptr %partials[0, %inWarp] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<32 x f32>
    %partial = llvm.load %partialSlot : !llvm.ptr<3> -> f32
    %mine = arith.select %isPartial, %partial, %zero : f32
    %total = func.call @strict_device_sums(%mine) : (f32) -> f32
    func.call @strict_device_barrier() : () -> ()
    return %total : f32
  }
  // the first lane of each warp shares its warp sum, every warp adds them up, the second barrier keeps the next sum
  // from overwriting them early
  func.func private @strict_device_block_sum(%warped: f64, %lane: i64, %lanes: i64) -> f64 {
    %none = arith.constant 0 : i64
    %five = arith.constant 5 : i64
    %thirtyOne = arith.constant 31 : i64
    %zero = arith.constant 0.0 : f64
    %partials = llvm.mlir.addressof @strict_device_partials : !llvm.ptr<3>
    %inWarp = arith.andi %lane, %thirtyOne : i64
    %warp = arith.shrui %lane, %five : i64
    %isFirst = arith.cmpi eq, %inWarp, %none : i64
    cf.cond_br %isFirst, ^share, ^shared
  ^share:
    %slot = llvm.getelementptr %partials[0, %warp] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<32 x f64>
    llvm.store %warped, %slot : f64, !llvm.ptr<3>
    cf.br ^shared
  ^shared:
    func.call @strict_device_barrier() : () -> ()
    %warps = arith.shrui %lanes, %five : i64
    %isPartial = arith.cmpi slt, %inWarp, %warps : i64
    %partialSlot = llvm.getelementptr %partials[0, %inWarp] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<32 x f64>
    %partial = llvm.load %partialSlot : !llvm.ptr<3> -> f64
    %mine = arith.select %isPartial, %partial, %zero : f64
    %total = func.call @strict_device_sum(%mine) : (f64) -> f64
    func.call @strict_device_barrier() : () -> ()
    return %total : f64
  }
  // the formats the guarded body of a 32 bit kernel reads with strict_device_floats: 32 bit floats, halves, bfloat16
  func.func private @strict_device_floating(%bits: i64) -> i1 attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %thirtyTwo = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    %paired = arith.ori %bits, %one : i64
    %isHalves = arith.cmpi eq, %paired, %brainBits : i64
    %floating = arith.ori %isSingle, %isHalves : i1
    return %floating : i1
  }
  func.func private @strict_device_floats(%data: !llvm.ptr, %bits: i64, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^halves
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %single = llvm.load %singleSlot : !llvm.ptr -> f32
    return %single : f32
  ^halves:
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %word = llvm.load %slot : !llvm.ptr -> i16
    %brain = arith.bitcast %word : i16 to bf16
    %half = arith.bitcast %word : i16 to f16
    %fromBrain = arith.extf %brain : bf16 to f32
    %fromHalf = arith.extf %half : f16 to f32
    %number = arith.select %isBrain, %fromBrain, %fromHalf : f32
    return %number : f32
  }
  // a number of a 32 bit kernel into its result in the format strict_gpu_run chose (17 bfloat16, else 32 bit floats)
  func.func private @strict_device_cached_four(%failed: !llvm.ptr, %data: !llvm.ptr, %bits: i64, %count: i64, %indexes: !llvm.ptr, %cached: !llvm.ptr, %active: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %singleBits = arith.constant 32 : i64
    %brainBits = arith.constant 17 : i64
    %halfBits = arith.constant 16 : i64
    %doubleBits = arith.constant 64 : i64
    %full = arith.cmpi eq, %active, %four : i64
    cf.cond_br %full, ^checked, ^scalarStart
  ^checked:
    %index0 = llvm.load %indexes : !llvm.ptr -> i64
    %pointer1 = llvm.getelementptr %indexes[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %pointer2 = llvm.getelementptr %indexes[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %pointer3 = llvm.getelementptr %indexes[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %index1 = llvm.load %pointer1 : !llvm.ptr -> i64
    %index2 = llvm.load %pointer2 : !llvm.ptr -> i64
    %index3 = llvm.load %pointer3 : !llvm.ptr -> i64
    %next1 = arith.addi %index0, %one : i64
    %next2 = arith.addi %index0, %two : i64
    %next3 = arith.addi %index0, %three : i64
    %same1 = arith.cmpi eq, %index1, %next1 : i64
    %same2 = arith.cmpi eq, %index2, %next2 : i64
    %same3 = arith.cmpi eq, %index3, %next3 : i64
    %firsts = arith.andi %same1, %same2 : i1
    %consecutive = arith.andi %firsts, %same3 : i1
    %rest = arith.andi %index0, %three : i64
    %aligned = arith.cmpi eq, %rest, %zero : i64
    %nonnegative = arith.cmpi sge, %index0, %zero : i64
    %inside = arith.cmpi ult, %next3, %count : i64
    %bounded = arith.andi %nonnegative, %inside : i1
    %whole = arith.andi %aligned, %full : i1
    %linear = arith.andi %consecutive, %whole : i1
    %usable = arith.andi %linear, %bounded : i1
    cf.cond_br %usable, ^vector, ^scalarStart
  ^vector:
    cf.switch %bits : i64, [
      default: ^scalarStart,
      32: ^single,
      17: ^brain,
      16: ^half,
      64: ^double
    ]
  ^single:
    %singlePointer = llvm.getelementptr %data[%index0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singles = llvm.load %singlePointer {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    cf.br ^stored(%singles : vector<4xf32>)
  ^brain:
    %brainPointer = llvm.getelementptr %data[%index0] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainPointer {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fromBrains = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    cf.br ^stored(%fromBrains : vector<4xf32>)
  ^half:
    %halfPointer = llvm.getelementptr %data[%index0] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halves = llvm.load %halfPointer {alignment = 8 : i64} : !llvm.ptr -> vector<4xf16>
    %fromHalves = arith.extf %halves : vector<4xf16> to vector<4xf32>
    cf.br ^stored(%fromHalves : vector<4xf32>)
  ^double:
    %doublePointer = llvm.getelementptr %data[%index0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %doubles = llvm.load %doublePointer {alignment = 16 : i64} : !llvm.ptr -> vector<4xf64>
    %fromDoubles = arith.truncf %doubles : vector<4xf64> to vector<4xf32>
    cf.br ^stored(%fromDoubles : vector<4xf32>)
  ^stored(%numbers: vector<4xf32>):
    llvm.store %numbers, %cached : vector<4xf32>, !llvm.ptr
    return
  ^scalarStart:
    cf.br ^scalar(%zero : i64)
  ^scalar(%lane: i64):
    %more = arith.cmpi ult, %lane, %four : i64
    cf.cond_br %more, ^live, ^done
  ^live:
    %isActive = arith.cmpi ult, %lane, %active : i64
    cf.cond_br %isActive, ^check, ^next
  ^check:
    %indexPointer = llvm.getelementptr %indexes[%lane] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %index = llvm.load %indexPointer : !llvm.ptr -> i64
    %inList = arith.cmpi ult, %index, %count : i64
    cf.cond_br %inList, ^read, ^failed
  ^read:
    %number = func.call @strict_device_elements(%data, %bits, %index, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    cf.br ^cached(%number : f32)
  ^failed:
    llvm.store %one, %failed : i64, !llvm.ptr
    %empty = arith.constant 0.0 : f32
    cf.br ^cached(%empty : f32)
  ^cached(%value: f32):
    %cachePointer = llvm.getelementptr %cached[%lane] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %cachePointer : f32, !llvm.ptr
    cf.br ^next
  ^next:
    %next = arith.addi %lane, %one : i64
    llvm.br ^scalar(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  func.func private @strict_device_cached_stored(%result: !llvm.ptr, %at: i64, %active: i64, %cached: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %brainBits = arith.constant 17 : i64
    %doubleBits = arith.constant 64 : i64
    %full = arith.cmpi eq, %active, %four : i64
    cf.cond_br %full, ^vector, ^scalarStart
  ^vector:
    %numbers = llvm.load %cached : !llvm.ptr -> vector<4xf32>
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^single
  ^brain:
    %brains = arith.truncf %numbers : vector<4xf32> to vector<4xbf16>
    %brainPointer = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %brains, %brainPointer {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    return
  ^single:
    %singlePointer = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %numbers, %singlePointer {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    return
  ^scalarStart:
    cf.br ^scalar(%zero : i64)
  ^scalar(%lane: i64):
    %more = arith.cmpi ult, %lane, %four : i64
    cf.cond_br %more, ^live, ^done
  ^live:
    %isActive = arith.cmpi ult, %lane, %active : i64
    cf.cond_br %isActive, ^write, ^next
  ^write:
    %pointer = llvm.getelementptr %cached[%lane] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %pointer : !llvm.ptr -> f32
    %index = arith.addi %at, %lane : i64
    func.call @strict_device_stored(%result, %index, %value, %bits) : (!llvm.ptr, i64, f32, i64) -> ()
    cf.br ^next
  ^next:
    %next = arith.addi %lane, %one : i64
    llvm.br ^scalar(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return
  }
  func.func private @strict_device_stored(%result: !llvm.ptr, %at: i64, %value: f32, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^single
  ^brain:
    %brain = func.call @strict_device_bfloat16(%value) : (f32) -> bf16
    %brainSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %brain, %brainSlot : bf16, !llvm.ptr
    return
  ^single:
    %singleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %singleSlot : f32, !llvm.ptr
    return
  }
  // value as number at of result: a double when wide (to the host), else in the format bits (strict_device_stored)
  func.func private @strict_device_result(%result: !llvm.ptr, %at: i64, %value: f32, %wide: i1, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    cf.cond_br %wide, ^double, ^single
  ^double:
    %double = arith.extf %value : f32 to f64
    %doubleSlot = llvm.getelementptr %result[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %double, %doubleSlot : f64, !llvm.ptr
    return
  ^single:
    func.call @strict_device_stored(%result, %at, %value, %bits) : (!llvm.ptr, i64, f32, i64) -> ()
    return
  }
  func.func private @strict_device_elements(%data: !llvm.ptr, %bits: i64, %at: i64, %count: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^narrower
  ^single:
    %singleSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singleValue = llvm.load %singleSlot : !llvm.ptr -> f32
    return %singleValue : f32
  ^narrower:
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^halfOrPacked
  ^brain:
    %brainNumber = func.call @strict_device_bfloat16_read(%data, %at) : (!llvm.ptr, i64) -> f32
    return %brainNumber : f32
  ^halfOrPacked:
    %sixteen = arith.constant 16 : i64
    %isHalf = arith.cmpi eq, %bits, %sixteen : i64
    cf.cond_br %isHalf, ^half, ^packed
  ^half:
    %halfSlot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %halfValue = llvm.load %halfSlot : !llvm.ptr -> f16
    %halfNumber = arith.extf %halfValue : f16 to f32
    return %halfNumber : f32
  ^packed:
    %packedNumber = func.call @strict_device_packed_elements(%data, %bits, %at, %count) : (!llvm.ptr, i64, i64, i64) -> f32
    return %packedNumber : f32
  }
  func.func private @strict_device_packed_elements(%data: !llvm.ptr, %bits: i64, %at: i64, %count: i64) -> f32 {
    %sixteen = arith.constant 16 : i64
    %two = arith.constant 2 : i64
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
    cf.cond_br %isSix, ^six, ^floatOrDouble
  ^floatOrDouble:
    %float8Code = arith.constant 7 : i64
    %isFloat8 = arith.cmpi eq, %bits, %float8Code : i64
    cf.cond_br %isFloat8, ^float8, ^bytesOrDouble
  ^bytesOrDouble:
    %bytesCode = arith.constant 9 : i64
    %isBytes = arith.cmpi eq, %bits, %bytesCode : i64
    cf.cond_br %isBytes, ^byte, ^tritsOrDouble
  ^tritsOrDouble:
    %tritsCode = arith.constant 3 : i64
    %isTrits = arith.cmpi eq, %bits, %tritsCode : i64
    cf.cond_br %isTrits, ^trits, ^double
  ^trits:
    %tritsNumber = func.call @strict_device_trits(%data, %at) : (!llvm.ptr, i64) -> f32
    return %tritsNumber : f32
  ^byte:
    %unsignedSingle = func.call @strict_device_byte_read(%data, %at) : (!llvm.ptr, i64) -> f32
    return %unsignedSingle : f32
  ^float8:
    %float8Number = func.call @strict_device_float8_read(%data, %at) : (!llvm.ptr, i64) -> f32
    return %float8Number : f32
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
  // mirrors strict_trits_number of the host runtime (PTQ1_0 blocks, 3 bits): (code - 1) times the half scale of its 128
  func.func private @strict_device_trits(%data: !llvm.ptr, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %sixteen = arith.constant 16 : i64
    %twentyFour = arith.constant 24 : i64
    %number = arith.constant 127 : i64
    %blockBytes = arith.constant 28 : i64
    %secondStart = arith.constant 80 : i64
    %thirdStart = arith.constant 120 : i64
    %offset = arith.constant 1 : i32
    %block = arith.shrui %at, %seven : i64
    %inBlock = arith.andi %at, %number : i64
    %blockStart = arith.muli %block, %blockBytes : i64
    %blockData = llvm.getelementptr %data[%blockStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %isFirst = arith.cmpi slt, %inBlock, %secondStart : i64
    %isThird = arith.cmpi sge, %inBlock, %thirdStart : i64
    %second = arith.subi %inBlock, %secondStart : i64
    %third = arith.subi %inBlock, %thirdStart : i64
    %firstByte = arith.andi %inBlock, %fifteen : i64
    %firstCode = arith.shrui %inBlock, %four : i64
    %secondInByte = arith.andi %second, %seven : i64
    %secondByte = arith.addi %secondInByte, %sixteen : i64
    %secondCode = arith.shrui %second, %three : i64
    %thirdInByte = arith.andi %third, %one : i64
    %thirdByte = arith.addi %thirdInByte, %twentyFour : i64
    %thirdCode = arith.shrui %third, %one : i64
    %laterByte = arith.select %isThird, %thirdByte, %secondByte : i64
    %laterCode = arith.select %isThird, %thirdCode, %secondCode : i64
    %byteAt = arith.select %isFirst, %firstByte, %laterByte : i64
    %codeAt = arith.select %isFirst, %firstCode, %laterCode : i64
    %codeIndex = arith.trunci %codeAt : i64 to i32
    %byteSlot = llvm.getelementptr %blockData[%byteAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %byteSlot : !llvm.ptr -> i8
    %wide = arith.extui %byte : i8 to i32
    %power = func.call @strict_device_trits_power(%codeIndex) : (i32) -> i32
    %signed = func.call @strict_device_trits_code(%wide, %power) : (i32, i32) -> i32
    %low = arith.trunci %signed : i32 to i8
    %level = arith.sitofp %low : i8 to f32
    %scaleSlot = llvm.getelementptr %blockData[26] : (!llvm.ptr) -> !llvm.ptr, i8
    %scaleHalf = llvm.load %scaleSlot : !llvm.ptr -> f16
    %scale = arith.extf %scaleHalf : f16 to f32
    %value = arith.mulf %level, %scale : f32
    return %value : f32
  }
  // 3^index for the code number index (0 to 4) of a byte of PTQ1_0 codes
  func.func private @strict_device_trits_power(%index: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c9 = arith.constant 9 : i32
    %c27 = arith.constant 27 : i32
    %c81 = arith.constant 81 : i32
    %isOne = arith.cmpi eq, %index, %c1 : i32
    %isTwo = arith.cmpi eq, %index, %c2 : i32
    %isThree = arith.cmpi eq, %index, %c3 : i32
    %isLow = arith.cmpi ult, %index, %c1 : i32
    %highPower = arith.select %isThree, %c27, %c81 : i32
    %middlePower = arith.select %isTwo, %c9, %highPower : i32
    %lowPower = arith.select %isOne, %c3, %middlePower : i32
    %power = arith.select %isLow, %c1, %lowPower : i32
    return %power : i32
  }
  // the 4 bytes of word spread over the 16 bit lanes of two words (bytes 0 and 1 in low, 2 and 3 in high, each in the low
  // byte of its lane), so one multiplication steps two codes without carries (the byte shuffles become prmt)
  func.func private @strict_device_trits_spread(%word: i32) -> (i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %zeros = llvm.mlir.constant(dense<0> : vector<4xi8>) : vector<4xi8>
    %bytes = llvm.bitcast %word : i32 to vector<4xi8>
    %lowBytes = llvm.shufflevector %bytes, %zeros [0, 4, 1, 4] : vector<4xi8>
    %highBytes = llvm.shufflevector %bytes, %zeros [2, 4, 3, 4] : vector<4xi8>
    %low = llvm.bitcast %lowBytes : vector<4xi8> to i32
    %high = llvm.bitcast %highBytes : vector<4xi8> to i32
    return %low, %high : i32, i32
  }
  // the signed weights c - 1 of the 4 codes c in the high bytes of the lanes of first and second, as (c + 127) ^ 128
  func.func private @strict_device_trits_signed(%first: i32, %second: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %lift = arith.constant 2139062143 : i32
    %sign = arith.constant -2139062144 : i32
    %firstBytes = llvm.bitcast %first : i32 to vector<4xi8>
    %secondBytes = llvm.bitcast %second : i32 to vector<4xi8>
    %codeBytes = llvm.shufflevector %firstBytes, %secondBytes [1, 3, 5, 7] : vector<4xi8>
    %codes = llvm.bitcast %codeBytes : vector<4xi8> to i32
    %lifted = arith.addi %codes, %lift : i32
    %signed = arith.xori %lifted, %sign : i32
    return %signed : i32
  }
  // the next code of the 4 bytes spread over low and high (strict.h: a byte holds its codes as a fraction of 243, times 3
  // its high byte is the code, the low byte what stays for the codes after it): its signed weights, the rest of both
  func.func private @strict_device_trits_step(%low: i32, %high: i32) -> (i32, i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c3 = arith.constant 3 : i32
    %lanes = arith.constant 16711935 : i32
    %lowTripled = arith.muli %low, %c3 : i32
    %highTripled = arith.muli %high, %c3 : i32
    %signed = func.call @strict_device_trits_signed(%lowTripled, %highTripled) : (i32, i32) -> i32
    %nextLow = arith.andi %lowTripled, %lanes : i32
    %nextHigh = arith.andi %highTripled, %lanes : i32
    return %signed, %nextLow, %nextHigh : i32, i32, i32
  }
  // code n of the 4 bytes of word as signed weights, power is 3^n
  func.func private @strict_device_trits_code(%word: i32, %power: i32) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %lanes = arith.constant 16711935 : i32
    %low, %high = func.call @strict_device_trits_spread(%word) : (i32) -> (i32, i32)
    %lowPowered = arith.muli %low, %power : i32
    %highPowered = arith.muli %high, %power : i32
    %lowRest = arith.andi %lowPowered, %lanes : i32
    %highRest = arith.andi %highPowered, %lanes : i32
    %signed, %nextLow, %nextHigh = func.call @strict_device_trits_step(%lowRest, %highRest) : (i32, i32) -> (i32, i32, i32)
    return %signed : i32
  }
  // the signed weights of numbers 120 to 123 and 124 to 127 of a PTQ1_0 block from its last word: bytes 24 and 25 hold 4
  // codes each, number 120 + 2n + b is code n of byte 24 + b, then the half scale
  func.func private @strict_device_trits_tail(%last: i32) -> (i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c3 = arith.constant 3 : i32
    %lanes = arith.constant 16711935 : i32
    %pair, %scaleLane = func.call @strict_device_trits_spread(%last) : (i32) -> (i32, i32)
    %first = arith.muli %pair, %c3 : i32
    %firstRest = arith.andi %first, %lanes : i32
    %second = arith.muli %firstRest, %c3 : i32
    %secondRest = arith.andi %second, %lanes : i32
    %third = arith.muli %secondRest, %c3 : i32
    %thirdRest = arith.andi %third, %lanes : i32
    %fourth = arith.muli %thirdRest, %c3 : i32
    %lower = func.call @strict_device_trits_signed(%first, %second) : (i32, i32) -> i32
    %upper = func.call @strict_device_trits_signed(%third, %fourth) : (i32, i32) -> i32
    return %lower, %upper : i32, i32
  }
  // the 128 signed weights of the PTQ1_0 block at block and its half scale (strict_device_trits_unpacked)
  func.func private @strict_device_trits_block(%block: !llvm.ptr) -> (vector<32xi32>, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %words = func.call @strict_device_trits_words(%block) : (!llvm.ptr) -> vector<8xi32>
    %weights, %scale = func.call @strict_device_trits_unpacked(%words) : (vector<8xi32>) -> (vector<32xi32>, f32)
    return %weights, %scale : vector<32xi32>, f32
  }
  // the 128 signed weights of the 7 words of a PTQ1_0 block in order, word k of the result holds numbers 4k to 4k + 3:
  // codes t of words 0 to 3 are numbers 16t to 16t + 15 (words 4t + w), those of words 4 and 5 numbers 80 + 8t to + 7
  // (20 + 2t + w - 4), the tail is 30 and 31; and the half scale of the block
  func.func private @strict_device_trits_unpacked(%words: vector<8xi32>) -> (vector<32xi32>, f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c16 = arith.constant 16 : i32
    %c30 = arith.constant 30 : i32
    %c31 = arith.constant 31 : i32
    %narrowStart = arith.constant 16 : i32
    %zeros = arith.constant dense<0> : vector<32xi32>
    cf.br ^word(%c0, %zeros : i32, vector<32xi32>)
  ^word(%index: i32, %weights: vector<32xi32>):
    %moreWords = arith.cmpi slt, %index, %c6 : i32
    cf.cond_br %moreWords, ^wordBody, ^tail
  ^wordBody:
    %word = llvm.extractelement %words[%index : i32] : vector<8xi32>
    %low, %high = func.call @strict_device_trits_spread(%word) : (i32) -> (i32, i32)
    %isWide = arith.cmpi slt, %index, %c4 : i32
    %narrowFirst = arith.addi %index, %narrowStart : i32
    %first = arith.select %isWide, %index, %narrowFirst : i32
    %stride = arith.select %isWide, %c4, %c2 : i32
    cf.br ^code(%c0, %low, %high, %weights : i32, i32, i32, vector<32xi32>)
  ^code(%code: i32, %codeLow: i32, %codeHigh: i32, %codeWeights: vector<32xi32>):
    %moreCodes = arith.cmpi slt, %code, %c5 : i32
    cf.cond_br %moreCodes, ^codeBody, ^wordNext(%codeWeights : vector<32xi32>)
  ^codeBody:
    %signed, %nextLow, %nextHigh = func.call @strict_device_trits_step(%codeLow, %codeHigh) : (i32, i32) -> (i32, i32, i32)
    %codeOffset = arith.muli %code, %stride : i32
    %position = arith.addi %first, %codeOffset : i32
    %placed = llvm.insertelement %signed, %codeWeights[%position : i32] : vector<32xi32>
    %nextCode = arith.addi %code, %c1 : i32
    llvm.br ^code(%nextCode, %nextLow, %nextHigh, %placed : i32, i32, i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^wordNext(%wordWeights: vector<32xi32>):
    %nextIndex = arith.addi %index, %c1 : i32
    llvm.br ^word(%nextIndex, %wordWeights : i32, vector<32xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^tail:
    %last = llvm.extractelement %words[%c6 : i32] : vector<8xi32>
    %lower, %upper = func.call @strict_device_trits_tail(%last) : (i32) -> (i32, i32)
    %withLower = llvm.insertelement %lower, %weights[%c30 : i32] : vector<32xi32>
    %withUpper = llvm.insertelement %upper, %withLower[%c31 : i32] : vector<32xi32>
    %scaleWord = arith.shrui %last, %c16 : i32
    %scaleShort = arith.trunci %scaleWord : i32 to i16
    %half = arith.bitcast %scaleShort : i16 to f16
    %scale = arith.extf %half : f16 to f32
    return %withUpper, %scale : vector<32xi32>, f32
  }
  // the 7 words of a PTQ1_0 block (4 byte aligned) and a zero
  func.func private @strict_device_trits_words(%block: !llvm.ptr) -> vector<8xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c7 = arith.constant 7 : i32
    %zeros = arith.constant dense<0> : vector<8xi32>
    cf.br ^word(%c0, %zeros : i32, vector<8xi32>)
  ^word(%index: i32, %words: vector<8xi32>):
    %more = arith.cmpi slt, %index, %c7 : i32
    cf.cond_br %more, ^load, ^done
  ^load:
    %slot = llvm.getelementptr %block[%index] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %word = llvm.load %slot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %placed = llvm.insertelement %word, %words[%index : i32] : vector<8xi32>
    %next = arith.addi %index, %c1 : i32
    llvm.br ^word(%next, %placed : i32, vector<8xi32>) {loop_annotation = #llvm.loop_annotation<unroll = #llvm.loop_unroll<full = true>>}
  ^done:
    return %words : vector<8xi32>
  }
  func.func private @strict_device_sums(%value: f32) -> f32 {
    %sixteen = arith.constant 16 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    cf.br ^round(%value, %sixteen : f32, i32)
  ^round(%sum: f32, %offset: i32):
    %more = arith.cmpi sge, %offset, %one : i32
    cf.cond_br %more, ^shuffle, ^done
  ^shuffle:
    %other = func.call @strict_device_shuffles(%sum, %offset) : (f32, i32) -> f32
    %added = arith.addf %sum, %other : f32
    %half = arith.divui %offset, %two : i32
    cf.br ^round(%added, %half : f32, i32)
  ^done:
    return %sum : f32
  }
  // exact double powers need libdevice, so the launch fails and the CPU runs the loop
  // like ^ a logarithm on the GPU only exists in 32 bits (lg2.approx), a 64 bit kernel goes back to the CPU
  func.func private @strict_device_logarithm(%failed: !llvm.ptr, %number: f64) -> f64 {
    %one = arith.constant 1 : i64
    llvm.store %one, %failed : i64, !llvm.ptr
    return %number : f64
  }
  func.func private @strict_device_logarithms(%failed: !llvm.ptr, %number: f32) -> f32 {
    %binary = func.call @strict_device_binary_logarithm(%number) : (f32) -> f32
    %natural = arith.constant 0.693147182 : f32
    %logarithm = arith.mulf %binary, %natural : f32
    return %logarithm : f32
  }
  // Sin and Cos of Degrees (32 bit numbers) also in 64 bit kernels: the degrees reduced to [-180, 180] (exact, 360 is a
  // float) go to sin.approx and cos.approx, which are within 2^-20 of the result there
  func.func private @strict_device_sine(%failed: !llvm.ptr, %degrees: f64) -> f64 {
    %narrow = arith.truncf %degrees : f64 to f32
    %sine = func.call @strict_device_sines(%failed, %narrow) : (!llvm.ptr, f32) -> f32
    %wide = arith.extf %sine : f32 to f64
    return %wide : f64
  }
  func.func private @strict_device_sines(%failed: !llvm.ptr, %degrees: f32) -> f32 {
    %radians = func.call @strict_device_radians(%degrees) : (f32) -> f32
    %sine = func.call @strict_device_radian_sine(%radians) : (f32) -> f32
    return %sine : f32
  }
  func.func private @strict_device_cosine(%failed: !llvm.ptr, %degrees: f64) -> f64 {
    %narrow = arith.truncf %degrees : f64 to f32
    %cosine = func.call @strict_device_cosines(%failed, %narrow) : (!llvm.ptr, f32) -> f32
    %wide = arith.extf %cosine : f32 to f64
    return %wide : f64
  }
  func.func private @strict_device_cosines(%failed: !llvm.ptr, %degrees: f32) -> f32 {
    %radians = func.call @strict_device_radians(%degrees) : (f32) -> f32
    %cosine = func.call @strict_device_radian_cosine(%radians) : (f32) -> f32
    return %cosine : f32
  }
  func.func private @strict_device_radians(%degrees: f32) -> f32 {
    %turn = arith.constant 360.0 : f32
    %perTurn = arith.constant 0.00277777785 : f32
    %perDegree = arith.constant 0.0174532924 : f32
    %turns = arith.mulf %degrees, %perTurn : f32
    %whole = llvm.call_intrinsic "llvm.roundeven.f32"(%turns) : (f32) -> f32
    %back = arith.negf %whole : f32
    %reduced = llvm.call_intrinsic "llvm.fma.f32"(%back, %turn, %degrees) : (f32, f32, f32) -> f32
    %radians = arith.mulf %reduced, %perDegree : f32
    return %radians : f32
  }
  func.func private @strict_device_power(%failed: !llvm.ptr, %left: f64, %right: f64) -> f64 {
    %one = arith.constant 1 : i64
    llvm.store %one, %failed : i64, !llvm.ptr
    return %left : f64
  }
  // 32 bit powers: powers of e with the approximate exponential of base 2 first (Exponential is E ^ value, the check
  // folds away and no exponent is tested), square roots and small whole exponents of other bases exactly, the rest
  // with the approximate exponential and logarithm of base 2 (ex2.approx and lg2.approx, about 2 units in the last place)
  func.func private @strict_device_powers(%failed: !llvm.ptr, %left: f32, %right: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %e = arith.constant 2.71828175 : f32
    %log2e = arith.constant 1.44269504 : f32
    %isE = arith.cmpf oeq, %left, %e : f32
    cf.cond_br %isE, ^exponential(%log2e, %one : f32, f32), ^other
  ^other:
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
    %size = llvm.intr.fabs(%left) : (f32) -> f32
    %logarithm = func.call @strict_device_binary_logarithm(%size) : (f32) -> f32
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
    %isQuantizedOther = arith.cmpi eq, %otherBits, %eightBits : i64
    cf.cond_br %isQuantizedOther, ^quantized, ^plain
  ^quantized:
    %eightFits = func.call @strict_device_fits8(%row, %rowBits, %count, %other) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %eightFits, ^eights, ^generic(%lane, %zero : i64, f32)
  ^eights:
    %otherData = func.call @strict_device_data(%otherCount, %eightBits) : (i64, i64) -> i64
    %otherScales = llvm.getelementptr %other[%otherData] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %eightDot = func.call @strict_device_dots8(%row, %rowBits, %count, %other, %otherScales, %lane, %lanes) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> f32
    return %eightDot : f32
  ^plain:
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
    %fourOne = arith.constant 1 : i32
    %fourTwo = arith.constant 2 : i32
    %fourThree = arith.constant 3 : i32
    %fourFour = arith.constant 4 : i32
    %fourFive = arith.constant 5 : i32
    %fourSix = arith.constant 6 : i32
    %fourEight = arith.constant 8 : i32
    %fourFifteen = arith.constant 15 : i32
    %fourSixteen = arith.constant 16 : i32
    %fourMask = arith.constant 63 : i32
    %fourTop = arith.constant 48 : i32
    %fourBytes = arith.constant 255 : i32
    %fourBlockBytes = arith.constant 144 : i32
    %fourMagic = arith.constant 1258291200 : i32
    %fourMagicFloat = arith.constant 8388608.0 : f32
    %fourUnitsWide = arith.shrui %count, %five : i64
    %fourUnits = arith.trunci %fourUnitsWide : i64 to i32
    %lane32 = arith.trunci %lane : i64 to i32
    %lanes32 = arith.trunci %lanes : i64 to i32
    %rowGlobal = llvm.addrspacecast %row : !llvm.ptr to !llvm.ptr<1>
    %otherGlobal = llvm.addrspacecast %other : !llvm.ptr to !llvm.ptr<1>
    %rowAddress = llvm.ptrtoint %row : !llvm.ptr to i64
    %alignmentMask = arith.constant 15 : i64
    %rowRest = arith.andi %rowAddress, %alignmentMask : i64
    %rowAligned = arith.cmpi eq, %rowRest, %none : i64
    %notFourRow = arith.xori %isFourRow, %true : i1
    %rowFits = arith.ori %notFourRow, %rowAligned : i1
    %fastRow = arith.andi %fast, %rowFits : i1
    cf.cond_br %fastRow, ^dispatch, ^generic(%lane, %zero : i64, f32)
  ^dispatch:
    cf.switch %rowBits : i64, [
      default: ^generic(%lane, %zero : i64, f32),
      4: ^four(%lane32, %zero : i32, f32),
      6: ^sixes(%lane, %zero : i64, f32),
      8: ^bytes(%lane, %zero : i64, f32),
      16: ^halves(%lane, %zero : i64, f32),
      32: ^singles(%lane, %zero : i64, f32)
    ]
  ^four(%fourUnit: i32, %fourSum: f32):
    %fourMore = arith.cmpi slt, %fourUnit, %fourUnits : i32
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
    %fourABlock = arith.shrui %fourUnit, %fourThree : i32
    %fourAParts = arith.shrui %fourUnit, %fourOne : i32
    %fourAPart = arith.andi %fourAParts, %fourThree : i32
    %fourAHalf = arith.andi %fourUnit, %fourOne : i32
    %fourAOffset = arith.muli %fourABlock, %fourBlockBytes : i32
    %fourAOffsetWide = arith.extui %fourAOffset : i32 to i64
    %fourAStart = llvm.getelementptr %rowGlobal[%fourAOffsetWide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %fourAHeader = llvm.load %fourAStart invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    %fourAPartBytes = arith.shli %fourAPart, %fourFive : i32
    %fourAHalfBytes = arith.shli %fourAHalf, %fourFour : i32
    %fourAInBlock = arith.addi %fourAPartBytes, %fourAHalfBytes : i32
    %fourANibbleAt = arith.addi %fourAInBlock, %fourSixteen : i32
    %fourANibbleWide = arith.extui %fourANibbleAt : i32 to i64
    %fourANibbleSlot = llvm.getelementptr %fourAStart[%fourANibbleWide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %fourANibbles = llvm.load %fourANibbleSlot invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    %fourABlockFirst = arith.shli %fourABlock, %fourEight : i32
    %fourAPartFirst = arith.shli %fourAPart, %fourSix : i32
    %fourABlockPart = arith.addi %fourABlockFirst, %fourAPartFirst : i32
    %fourAFirst = arith.addi %fourABlockPart, %fourAHalfBytes : i32
    %fourAFirstWide = arith.extui %fourAFirst : i32 to i64
    %fourALowSlot = llvm.getelementptr %otherGlobal[%fourAFirstWide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %fourALowInputSlot0 = llvm.getelementptr %fourALowSlot[0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourALowInputs0 = llvm.load %fourALowInputSlot0 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourALowInputSlot1 = llvm.getelementptr %fourALowSlot[4] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourALowInputs1 = llvm.load %fourALowInputSlot1 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourALowInputSlot2 = llvm.getelementptr %fourALowSlot[8] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourALowInputs2 = llvm.load %fourALowInputSlot2 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourALowInputSlot3 = llvm.getelementptr %fourALowSlot[12] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourALowInputs3 = llvm.load %fourALowInputSlot3 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourAHighInputSlot0 = llvm.getelementptr %fourALowSlot[32] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourAHighInputs0 = llvm.load %fourAHighInputSlot0 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourAHighInputSlot1 = llvm.getelementptr %fourALowSlot[36] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourAHighInputs1 = llvm.load %fourAHighInputSlot1 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourAHighInputSlot2 = llvm.getelementptr %fourALowSlot[40] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourAHighInputs2 = llvm.load %fourAHighInputSlot2 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourAHighInputSlot3 = llvm.getelementptr %fourALowSlot[44] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
    %fourAHighInputs3 = llvm.load %fourAHighInputSlot3 invariant {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %fourAScaleWord = llvm.extractelement %fourAHeader[%c0 : i32] : vector<4xi32>
    %fourAHalves = llvm.bitcast %fourAScaleWord : i32 to vector<2xf16>
    %fourAHalfScale = llvm.extractelement %fourAHalves[%c0 : i32] : vector<2xf16>
    %fourAHalfMinimum = llvm.extractelement %fourAHalves[%c1 : i32] : vector<2xf16>
    %fourAScale = arith.extf %fourAHalfScale : f16 to f32
    %fourAMinimum = arith.extf %fourAHalfMinimum : f16 to f32
    %fourAFirstBytes = llvm.extractelement %fourAHeader[%c1 : i32] : vector<4xi32>
    %fourASecondBytes = llvm.extractelement %fourAHeader[%c2 : i32] : vector<4xi32>
    %fourAThirdBytes = llvm.extractelement %fourAHeader[%c3 : i32] : vector<4xi32>
    %fourAOdd = arith.andi %fourAPart, %fourOne : i32
    %fourAShift = arith.shli %fourAOdd, %fourFour : i32
    %fourAShiftHigh = arith.addi %fourAShift, %fourEight : i32
    %fourAIsHigh = arith.cmpi uge, %fourAPart, %fourTwo : i32
    %fourALowaShifted = arith.shrui %fourAFirstBytes, %fourAShift : i32
    %fourALowa = arith.andi %fourALowaShifted, %fourBytes : i32
    %fourALowbShifted = arith.shrui %fourASecondBytes, %fourAShift : i32
    %fourALowb = arith.andi %fourALowbShifted, %fourBytes : i32
    %fourALowcShifted = arith.shrui %fourAThirdBytes, %fourAShift : i32
    %fourALowc = arith.andi %fourALowcShifted, %fourBytes : i32
    %fourALowPlainScale = arith.andi %fourALowa, %fourMask : i32
    %fourALowPlainMinimum = arith.andi %fourALowb, %fourMask : i32
    %fourALowcLow = arith.andi %fourALowc, %fourFifteen : i32
    %fourALowcHigh = arith.shrui %fourALowc, %fourFour : i32
    %fourALowaTop = arith.shrui %fourALowa, %fourTwo : i32
    %fourALowaBits = arith.andi %fourALowaTop, %fourTop : i32
    %fourALowbTop = arith.shrui %fourALowb, %fourTwo : i32
    %fourALowbBits = arith.andi %fourALowbTop, %fourTop : i32
    %fourALowSplitScale = arith.ori %fourALowcLow, %fourALowaBits : i32
    %fourALowSplitMinimum = arith.ori %fourALowcHigh, %fourALowbBits : i32
    %fourALowScaleBits = arith.select %fourAIsHigh, %fourALowSplitScale, %fourALowPlainScale : i32
    %fourALowMinimumBits = arith.select %fourAIsHigh, %fourALowSplitMinimum, %fourALowPlainMinimum : i32
    %fourALowScaleMagic = arith.ori %fourALowScaleBits, %fourMagic : i32
    %fourALowScaleFloat = arith.bitcast %fourALowScaleMagic : i32 to f32
    %fourALowScaleFactor = arith.subf %fourALowScaleFloat, %fourMagicFloat : f32
    %fourALowMinimumMagic = arith.ori %fourALowMinimumBits, %fourMagic : i32
    %fourALowMinimumFloat = arith.bitcast %fourALowMinimumMagic : i32 to f32
    %fourALowMinimumFactor = arith.subf %fourALowMinimumFloat, %fourMagicFloat : f32
    %fourAHighaShifted = arith.shrui %fourAFirstBytes, %fourAShiftHigh : i32
    %fourAHigha = arith.andi %fourAHighaShifted, %fourBytes : i32
    %fourAHighbShifted = arith.shrui %fourASecondBytes, %fourAShiftHigh : i32
    %fourAHighb = arith.andi %fourAHighbShifted, %fourBytes : i32
    %fourAHighcShifted = arith.shrui %fourAThirdBytes, %fourAShiftHigh : i32
    %fourAHighc = arith.andi %fourAHighcShifted, %fourBytes : i32
    %fourAHighPlainScale = arith.andi %fourAHigha, %fourMask : i32
    %fourAHighPlainMinimum = arith.andi %fourAHighb, %fourMask : i32
    %fourAHighcLow = arith.andi %fourAHighc, %fourFifteen : i32
    %fourAHighcHigh = arith.shrui %fourAHighc, %fourFour : i32
    %fourAHighaTop = arith.shrui %fourAHigha, %fourTwo : i32
    %fourAHighaBits = arith.andi %fourAHighaTop, %fourTop : i32
    %fourAHighbTop = arith.shrui %fourAHighb, %fourTwo : i32
    %fourAHighbBits = arith.andi %fourAHighbTop, %fourTop : i32
    %fourAHighSplitScale = arith.ori %fourAHighcLow, %fourAHighaBits : i32
    %fourAHighSplitMinimum = arith.ori %fourAHighcHigh, %fourAHighbBits : i32
    %fourAHighScaleBits = arith.select %fourAIsHigh, %fourAHighSplitScale, %fourAHighPlainScale : i32
    %fourAHighMinimumBits = arith.select %fourAIsHigh, %fourAHighSplitMinimum, %fourAHighPlainMinimum : i32
    %fourAHighScaleMagic = arith.ori %fourAHighScaleBits, %fourMagic : i32
    %fourAHighScaleFloat = arith.bitcast %fourAHighScaleMagic : i32 to f32
    %fourAHighScaleFactor = arith.subf %fourAHighScaleFloat, %fourMagicFloat : f32
    %fourAHighMinimumMagic = arith.ori %fourAHighMinimumBits, %fourMagic : i32
    %fourAHighMinimumFloat = arith.bitcast %fourAHighMinimumMagic : i32 to f32
    %fourAHighMinimumFactor = arith.subf %fourAHighMinimumFloat, %fourMagicFloat : f32
    %fourAWord0 = llvm.extractelement %fourANibbles[%c0 : i32] : vector<4xi32>
    %fourAWord1 = llvm.extractelement %fourANibbles[%c1 : i32] : vector<4xi32>
    %fourAWord2 = llvm.extractelement %fourANibbles[%c2 : i32] : vector<4xi32>
    %fourAWord3 = llvm.extractelement %fourANibbles[%c3 : i32] : vector<4xi32>
    %fourALowq0Shift = arith.constant 0 : i32
    %fourALowq0Shifted = arith.shrui %fourAWord0, %fourALowq0Shift : i32
    %fourALowq0Nibble = arith.andi %fourALowq0Shifted, %fourFifteen : i32
    %fourALowq0Magic = arith.ori %fourALowq0Nibble, %fourMagic : i32
    %fourALowq0Float = arith.bitcast %fourALowq0Magic : i32 to f32
    %fourALowq0 = arith.subf %fourALowq0Float, %fourMagicFloat : f32
    %fourALowx0 = llvm.extractelement %fourALowInputs0[%c0 : i32] : vector<4xf32>
    %fourALowq1Shift = arith.constant 8 : i32
    %fourALowq1Shifted = arith.shrui %fourAWord0, %fourALowq1Shift : i32
    %fourALowq1Nibble = arith.andi %fourALowq1Shifted, %fourFifteen : i32
    %fourALowq1Magic = arith.ori %fourALowq1Nibble, %fourMagic : i32
    %fourALowq1Float = arith.bitcast %fourALowq1Magic : i32 to f32
    %fourALowq1 = arith.subf %fourALowq1Float, %fourMagicFloat : f32
    %fourALowx1 = llvm.extractelement %fourALowInputs0[%c1 : i32] : vector<4xf32>
    %fourALowq2Shift = arith.constant 16 : i32
    %fourALowq2Shifted = arith.shrui %fourAWord0, %fourALowq2Shift : i32
    %fourALowq2Nibble = arith.andi %fourALowq2Shifted, %fourFifteen : i32
    %fourALowq2Magic = arith.ori %fourALowq2Nibble, %fourMagic : i32
    %fourALowq2Float = arith.bitcast %fourALowq2Magic : i32 to f32
    %fourALowq2 = arith.subf %fourALowq2Float, %fourMagicFloat : f32
    %fourALowx2 = llvm.extractelement %fourALowInputs0[%c2 : i32] : vector<4xf32>
    %fourALowq3Shift = arith.constant 24 : i32
    %fourALowq3Shifted = arith.shrui %fourAWord0, %fourALowq3Shift : i32
    %fourALowq3Nibble = arith.andi %fourALowq3Shifted, %fourFifteen : i32
    %fourALowq3Magic = arith.ori %fourALowq3Nibble, %fourMagic : i32
    %fourALowq3Float = arith.bitcast %fourALowq3Magic : i32 to f32
    %fourALowq3 = arith.subf %fourALowq3Float, %fourMagicFloat : f32
    %fourALowx3 = llvm.extractelement %fourALowInputs0[%c3 : i32] : vector<4xf32>
    %fourALowq4Shift = arith.constant 0 : i32
    %fourALowq4Shifted = arith.shrui %fourAWord1, %fourALowq4Shift : i32
    %fourALowq4Nibble = arith.andi %fourALowq4Shifted, %fourFifteen : i32
    %fourALowq4Magic = arith.ori %fourALowq4Nibble, %fourMagic : i32
    %fourALowq4Float = arith.bitcast %fourALowq4Magic : i32 to f32
    %fourALowq4 = arith.subf %fourALowq4Float, %fourMagicFloat : f32
    %fourALowx4 = llvm.extractelement %fourALowInputs1[%c0 : i32] : vector<4xf32>
    %fourALowq5Shift = arith.constant 8 : i32
    %fourALowq5Shifted = arith.shrui %fourAWord1, %fourALowq5Shift : i32
    %fourALowq5Nibble = arith.andi %fourALowq5Shifted, %fourFifteen : i32
    %fourALowq5Magic = arith.ori %fourALowq5Nibble, %fourMagic : i32
    %fourALowq5Float = arith.bitcast %fourALowq5Magic : i32 to f32
    %fourALowq5 = arith.subf %fourALowq5Float, %fourMagicFloat : f32
    %fourALowx5 = llvm.extractelement %fourALowInputs1[%c1 : i32] : vector<4xf32>
    %fourALowq6Shift = arith.constant 16 : i32
    %fourALowq6Shifted = arith.shrui %fourAWord1, %fourALowq6Shift : i32
    %fourALowq6Nibble = arith.andi %fourALowq6Shifted, %fourFifteen : i32
    %fourALowq6Magic = arith.ori %fourALowq6Nibble, %fourMagic : i32
    %fourALowq6Float = arith.bitcast %fourALowq6Magic : i32 to f32
    %fourALowq6 = arith.subf %fourALowq6Float, %fourMagicFloat : f32
    %fourALowx6 = llvm.extractelement %fourALowInputs1[%c2 : i32] : vector<4xf32>
    %fourALowq7Shift = arith.constant 24 : i32
    %fourALowq7Shifted = arith.shrui %fourAWord1, %fourALowq7Shift : i32
    %fourALowq7Nibble = arith.andi %fourALowq7Shifted, %fourFifteen : i32
    %fourALowq7Magic = arith.ori %fourALowq7Nibble, %fourMagic : i32
    %fourALowq7Float = arith.bitcast %fourALowq7Magic : i32 to f32
    %fourALowq7 = arith.subf %fourALowq7Float, %fourMagicFloat : f32
    %fourALowx7 = llvm.extractelement %fourALowInputs1[%c3 : i32] : vector<4xf32>
    %fourALowq8Shift = arith.constant 0 : i32
    %fourALowq8Shifted = arith.shrui %fourAWord2, %fourALowq8Shift : i32
    %fourALowq8Nibble = arith.andi %fourALowq8Shifted, %fourFifteen : i32
    %fourALowq8Magic = arith.ori %fourALowq8Nibble, %fourMagic : i32
    %fourALowq8Float = arith.bitcast %fourALowq8Magic : i32 to f32
    %fourALowq8 = arith.subf %fourALowq8Float, %fourMagicFloat : f32
    %fourALowx8 = llvm.extractelement %fourALowInputs2[%c0 : i32] : vector<4xf32>
    %fourALowq9Shift = arith.constant 8 : i32
    %fourALowq9Shifted = arith.shrui %fourAWord2, %fourALowq9Shift : i32
    %fourALowq9Nibble = arith.andi %fourALowq9Shifted, %fourFifteen : i32
    %fourALowq9Magic = arith.ori %fourALowq9Nibble, %fourMagic : i32
    %fourALowq9Float = arith.bitcast %fourALowq9Magic : i32 to f32
    %fourALowq9 = arith.subf %fourALowq9Float, %fourMagicFloat : f32
    %fourALowx9 = llvm.extractelement %fourALowInputs2[%c1 : i32] : vector<4xf32>
    %fourALowq10Shift = arith.constant 16 : i32
    %fourALowq10Shifted = arith.shrui %fourAWord2, %fourALowq10Shift : i32
    %fourALowq10Nibble = arith.andi %fourALowq10Shifted, %fourFifteen : i32
    %fourALowq10Magic = arith.ori %fourALowq10Nibble, %fourMagic : i32
    %fourALowq10Float = arith.bitcast %fourALowq10Magic : i32 to f32
    %fourALowq10 = arith.subf %fourALowq10Float, %fourMagicFloat : f32
    %fourALowx10 = llvm.extractelement %fourALowInputs2[%c2 : i32] : vector<4xf32>
    %fourALowq11Shift = arith.constant 24 : i32
    %fourALowq11Shifted = arith.shrui %fourAWord2, %fourALowq11Shift : i32
    %fourALowq11Nibble = arith.andi %fourALowq11Shifted, %fourFifteen : i32
    %fourALowq11Magic = arith.ori %fourALowq11Nibble, %fourMagic : i32
    %fourALowq11Float = arith.bitcast %fourALowq11Magic : i32 to f32
    %fourALowq11 = arith.subf %fourALowq11Float, %fourMagicFloat : f32
    %fourALowx11 = llvm.extractelement %fourALowInputs2[%c3 : i32] : vector<4xf32>
    %fourALowq12Shift = arith.constant 0 : i32
    %fourALowq12Shifted = arith.shrui %fourAWord3, %fourALowq12Shift : i32
    %fourALowq12Nibble = arith.andi %fourALowq12Shifted, %fourFifteen : i32
    %fourALowq12Magic = arith.ori %fourALowq12Nibble, %fourMagic : i32
    %fourALowq12Float = arith.bitcast %fourALowq12Magic : i32 to f32
    %fourALowq12 = arith.subf %fourALowq12Float, %fourMagicFloat : f32
    %fourALowx12 = llvm.extractelement %fourALowInputs3[%c0 : i32] : vector<4xf32>
    %fourALowq13Shift = arith.constant 8 : i32
    %fourALowq13Shifted = arith.shrui %fourAWord3, %fourALowq13Shift : i32
    %fourALowq13Nibble = arith.andi %fourALowq13Shifted, %fourFifteen : i32
    %fourALowq13Magic = arith.ori %fourALowq13Nibble, %fourMagic : i32
    %fourALowq13Float = arith.bitcast %fourALowq13Magic : i32 to f32
    %fourALowq13 = arith.subf %fourALowq13Float, %fourMagicFloat : f32
    %fourALowx13 = llvm.extractelement %fourALowInputs3[%c1 : i32] : vector<4xf32>
    %fourALowq14Shift = arith.constant 16 : i32
    %fourALowq14Shifted = arith.shrui %fourAWord3, %fourALowq14Shift : i32
    %fourALowq14Nibble = arith.andi %fourALowq14Shifted, %fourFifteen : i32
    %fourALowq14Magic = arith.ori %fourALowq14Nibble, %fourMagic : i32
    %fourALowq14Float = arith.bitcast %fourALowq14Magic : i32 to f32
    %fourALowq14 = arith.subf %fourALowq14Float, %fourMagicFloat : f32
    %fourALowx14 = llvm.extractelement %fourALowInputs3[%c2 : i32] : vector<4xf32>
    %fourALowq15Shift = arith.constant 24 : i32
    %fourALowq15Shifted = arith.shrui %fourAWord3, %fourALowq15Shift : i32
    %fourALowq15Nibble = arith.andi %fourALowq15Shifted, %fourFifteen : i32
    %fourALowq15Magic = arith.ori %fourALowq15Nibble, %fourMagic : i32
    %fourALowq15Float = arith.bitcast %fourALowq15Magic : i32 to f32
    %fourALowq15 = arith.subf %fourALowq15Float, %fourMagicFloat : f32
    %fourALowx15 = llvm.extractelement %fourALowInputs3[%c3 : i32] : vector<4xf32>
    %fourALowDot0p0 = arith.mulf %fourALowq0, %fourALowx0 fastmath<contract> : f32
    %fourALowDot0m1 = arith.mulf %fourALowq4, %fourALowx4 fastmath<contract> : f32
    %fourALowDot0p1 = arith.addf %fourALowDot0p0, %fourALowDot0m1 fastmath<contract> : f32
    %fourALowDot0m2 = arith.mulf %fourALowq8, %fourALowx8 fastmath<contract> : f32
    %fourALowDot0p2 = arith.addf %fourALowDot0p1, %fourALowDot0m2 fastmath<contract> : f32
    %fourALowDot0m3 = arith.mulf %fourALowq12, %fourALowx12 fastmath<contract> : f32
    %fourALowDot0p3 = arith.addf %fourALowDot0p2, %fourALowDot0m3 fastmath<contract> : f32
    %fourALowSum0a = arith.addf %fourALowx0, %fourALowx1 : f32
    %fourALowSum0b = arith.addf %fourALowx2, %fourALowx3 : f32
    %fourALowSum0 = arith.addf %fourALowSum0a, %fourALowSum0b : f32
    %fourALowDot1p0 = arith.mulf %fourALowq1, %fourALowx1 fastmath<contract> : f32
    %fourALowDot1m1 = arith.mulf %fourALowq5, %fourALowx5 fastmath<contract> : f32
    %fourALowDot1p1 = arith.addf %fourALowDot1p0, %fourALowDot1m1 fastmath<contract> : f32
    %fourALowDot1m2 = arith.mulf %fourALowq9, %fourALowx9 fastmath<contract> : f32
    %fourALowDot1p2 = arith.addf %fourALowDot1p1, %fourALowDot1m2 fastmath<contract> : f32
    %fourALowDot1m3 = arith.mulf %fourALowq13, %fourALowx13 fastmath<contract> : f32
    %fourALowDot1p3 = arith.addf %fourALowDot1p2, %fourALowDot1m3 fastmath<contract> : f32
    %fourALowSum1a = arith.addf %fourALowx4, %fourALowx5 : f32
    %fourALowSum1b = arith.addf %fourALowx6, %fourALowx7 : f32
    %fourALowSum1 = arith.addf %fourALowSum1a, %fourALowSum1b : f32
    %fourALowDot2p0 = arith.mulf %fourALowq2, %fourALowx2 fastmath<contract> : f32
    %fourALowDot2m1 = arith.mulf %fourALowq6, %fourALowx6 fastmath<contract> : f32
    %fourALowDot2p1 = arith.addf %fourALowDot2p0, %fourALowDot2m1 fastmath<contract> : f32
    %fourALowDot2m2 = arith.mulf %fourALowq10, %fourALowx10 fastmath<contract> : f32
    %fourALowDot2p2 = arith.addf %fourALowDot2p1, %fourALowDot2m2 fastmath<contract> : f32
    %fourALowDot2m3 = arith.mulf %fourALowq14, %fourALowx14 fastmath<contract> : f32
    %fourALowDot2p3 = arith.addf %fourALowDot2p2, %fourALowDot2m3 fastmath<contract> : f32
    %fourALowSum2a = arith.addf %fourALowx8, %fourALowx9 : f32
    %fourALowSum2b = arith.addf %fourALowx10, %fourALowx11 : f32
    %fourALowSum2 = arith.addf %fourALowSum2a, %fourALowSum2b : f32
    %fourALowDot3p0 = arith.mulf %fourALowq3, %fourALowx3 fastmath<contract> : f32
    %fourALowDot3m1 = arith.mulf %fourALowq7, %fourALowx7 fastmath<contract> : f32
    %fourALowDot3p1 = arith.addf %fourALowDot3p0, %fourALowDot3m1 fastmath<contract> : f32
    %fourALowDot3m2 = arith.mulf %fourALowq11, %fourALowx11 fastmath<contract> : f32
    %fourALowDot3p2 = arith.addf %fourALowDot3p1, %fourALowDot3m2 fastmath<contract> : f32
    %fourALowDot3m3 = arith.mulf %fourALowq15, %fourALowx15 fastmath<contract> : f32
    %fourALowDot3p3 = arith.addf %fourALowDot3p2, %fourALowDot3m3 fastmath<contract> : f32
    %fourALowSum3a = arith.addf %fourALowx12, %fourALowx13 : f32
    %fourALowSum3b = arith.addf %fourALowx14, %fourALowx15 : f32
    %fourALowSum3 = arith.addf %fourALowSum3a, %fourALowSum3b : f32
    %fourALowDotPair0 = arith.addf %fourALowDot0p3, %fourALowDot1p3 : f32
    %fourALowDotPair1 = arith.addf %fourALowDot2p3, %fourALowDot3p3 : f32
    %fourALowDot = arith.addf %fourALowDotPair0, %fourALowDotPair1 : f32
    %fourALowSumPair0 = arith.addf %fourALowSum0, %fourALowSum1 : f32
    %fourALowSumPair1 = arith.addf %fourALowSum2, %fourALowSum3 : f32
    %fourALowSum = arith.addf %fourALowSumPair0, %fourALowSumPair1 : f32
    %fourAHighq0Shift = arith.constant 4 : i32
    %fourAHighq0Shifted = arith.shrui %fourAWord0, %fourAHighq0Shift : i32
    %fourAHighq0Nibble = arith.andi %fourAHighq0Shifted, %fourFifteen : i32
    %fourAHighq0Magic = arith.ori %fourAHighq0Nibble, %fourMagic : i32
    %fourAHighq0Float = arith.bitcast %fourAHighq0Magic : i32 to f32
    %fourAHighq0 = arith.subf %fourAHighq0Float, %fourMagicFloat : f32
    %fourAHighx0 = llvm.extractelement %fourAHighInputs0[%c0 : i32] : vector<4xf32>
    %fourAHighq1Shift = arith.constant 12 : i32
    %fourAHighq1Shifted = arith.shrui %fourAWord0, %fourAHighq1Shift : i32
    %fourAHighq1Nibble = arith.andi %fourAHighq1Shifted, %fourFifteen : i32
    %fourAHighq1Magic = arith.ori %fourAHighq1Nibble, %fourMagic : i32
    %fourAHighq1Float = arith.bitcast %fourAHighq1Magic : i32 to f32
    %fourAHighq1 = arith.subf %fourAHighq1Float, %fourMagicFloat : f32
    %fourAHighx1 = llvm.extractelement %fourAHighInputs0[%c1 : i32] : vector<4xf32>
    %fourAHighq2Shift = arith.constant 20 : i32
    %fourAHighq2Shifted = arith.shrui %fourAWord0, %fourAHighq2Shift : i32
    %fourAHighq2Nibble = arith.andi %fourAHighq2Shifted, %fourFifteen : i32
    %fourAHighq2Magic = arith.ori %fourAHighq2Nibble, %fourMagic : i32
    %fourAHighq2Float = arith.bitcast %fourAHighq2Magic : i32 to f32
    %fourAHighq2 = arith.subf %fourAHighq2Float, %fourMagicFloat : f32
    %fourAHighx2 = llvm.extractelement %fourAHighInputs0[%c2 : i32] : vector<4xf32>
    %fourAHighq3Shift = arith.constant 28 : i32
    %fourAHighq3Shifted = arith.shrui %fourAWord0, %fourAHighq3Shift : i32
    %fourAHighq3Nibble = arith.andi %fourAHighq3Shifted, %fourFifteen : i32
    %fourAHighq3Magic = arith.ori %fourAHighq3Nibble, %fourMagic : i32
    %fourAHighq3Float = arith.bitcast %fourAHighq3Magic : i32 to f32
    %fourAHighq3 = arith.subf %fourAHighq3Float, %fourMagicFloat : f32
    %fourAHighx3 = llvm.extractelement %fourAHighInputs0[%c3 : i32] : vector<4xf32>
    %fourAHighq4Shift = arith.constant 4 : i32
    %fourAHighq4Shifted = arith.shrui %fourAWord1, %fourAHighq4Shift : i32
    %fourAHighq4Nibble = arith.andi %fourAHighq4Shifted, %fourFifteen : i32
    %fourAHighq4Magic = arith.ori %fourAHighq4Nibble, %fourMagic : i32
    %fourAHighq4Float = arith.bitcast %fourAHighq4Magic : i32 to f32
    %fourAHighq4 = arith.subf %fourAHighq4Float, %fourMagicFloat : f32
    %fourAHighx4 = llvm.extractelement %fourAHighInputs1[%c0 : i32] : vector<4xf32>
    %fourAHighq5Shift = arith.constant 12 : i32
    %fourAHighq5Shifted = arith.shrui %fourAWord1, %fourAHighq5Shift : i32
    %fourAHighq5Nibble = arith.andi %fourAHighq5Shifted, %fourFifteen : i32
    %fourAHighq5Magic = arith.ori %fourAHighq5Nibble, %fourMagic : i32
    %fourAHighq5Float = arith.bitcast %fourAHighq5Magic : i32 to f32
    %fourAHighq5 = arith.subf %fourAHighq5Float, %fourMagicFloat : f32
    %fourAHighx5 = llvm.extractelement %fourAHighInputs1[%c1 : i32] : vector<4xf32>
    %fourAHighq6Shift = arith.constant 20 : i32
    %fourAHighq6Shifted = arith.shrui %fourAWord1, %fourAHighq6Shift : i32
    %fourAHighq6Nibble = arith.andi %fourAHighq6Shifted, %fourFifteen : i32
    %fourAHighq6Magic = arith.ori %fourAHighq6Nibble, %fourMagic : i32
    %fourAHighq6Float = arith.bitcast %fourAHighq6Magic : i32 to f32
    %fourAHighq6 = arith.subf %fourAHighq6Float, %fourMagicFloat : f32
    %fourAHighx6 = llvm.extractelement %fourAHighInputs1[%c2 : i32] : vector<4xf32>
    %fourAHighq7Shift = arith.constant 28 : i32
    %fourAHighq7Shifted = arith.shrui %fourAWord1, %fourAHighq7Shift : i32
    %fourAHighq7Nibble = arith.andi %fourAHighq7Shifted, %fourFifteen : i32
    %fourAHighq7Magic = arith.ori %fourAHighq7Nibble, %fourMagic : i32
    %fourAHighq7Float = arith.bitcast %fourAHighq7Magic : i32 to f32
    %fourAHighq7 = arith.subf %fourAHighq7Float, %fourMagicFloat : f32
    %fourAHighx7 = llvm.extractelement %fourAHighInputs1[%c3 : i32] : vector<4xf32>
    %fourAHighq8Shift = arith.constant 4 : i32
    %fourAHighq8Shifted = arith.shrui %fourAWord2, %fourAHighq8Shift : i32
    %fourAHighq8Nibble = arith.andi %fourAHighq8Shifted, %fourFifteen : i32
    %fourAHighq8Magic = arith.ori %fourAHighq8Nibble, %fourMagic : i32
    %fourAHighq8Float = arith.bitcast %fourAHighq8Magic : i32 to f32
    %fourAHighq8 = arith.subf %fourAHighq8Float, %fourMagicFloat : f32
    %fourAHighx8 = llvm.extractelement %fourAHighInputs2[%c0 : i32] : vector<4xf32>
    %fourAHighq9Shift = arith.constant 12 : i32
    %fourAHighq9Shifted = arith.shrui %fourAWord2, %fourAHighq9Shift : i32
    %fourAHighq9Nibble = arith.andi %fourAHighq9Shifted, %fourFifteen : i32
    %fourAHighq9Magic = arith.ori %fourAHighq9Nibble, %fourMagic : i32
    %fourAHighq9Float = arith.bitcast %fourAHighq9Magic : i32 to f32
    %fourAHighq9 = arith.subf %fourAHighq9Float, %fourMagicFloat : f32
    %fourAHighx9 = llvm.extractelement %fourAHighInputs2[%c1 : i32] : vector<4xf32>
    %fourAHighq10Shift = arith.constant 20 : i32
    %fourAHighq10Shifted = arith.shrui %fourAWord2, %fourAHighq10Shift : i32
    %fourAHighq10Nibble = arith.andi %fourAHighq10Shifted, %fourFifteen : i32
    %fourAHighq10Magic = arith.ori %fourAHighq10Nibble, %fourMagic : i32
    %fourAHighq10Float = arith.bitcast %fourAHighq10Magic : i32 to f32
    %fourAHighq10 = arith.subf %fourAHighq10Float, %fourMagicFloat : f32
    %fourAHighx10 = llvm.extractelement %fourAHighInputs2[%c2 : i32] : vector<4xf32>
    %fourAHighq11Shift = arith.constant 28 : i32
    %fourAHighq11Shifted = arith.shrui %fourAWord2, %fourAHighq11Shift : i32
    %fourAHighq11Nibble = arith.andi %fourAHighq11Shifted, %fourFifteen : i32
    %fourAHighq11Magic = arith.ori %fourAHighq11Nibble, %fourMagic : i32
    %fourAHighq11Float = arith.bitcast %fourAHighq11Magic : i32 to f32
    %fourAHighq11 = arith.subf %fourAHighq11Float, %fourMagicFloat : f32
    %fourAHighx11 = llvm.extractelement %fourAHighInputs2[%c3 : i32] : vector<4xf32>
    %fourAHighq12Shift = arith.constant 4 : i32
    %fourAHighq12Shifted = arith.shrui %fourAWord3, %fourAHighq12Shift : i32
    %fourAHighq12Nibble = arith.andi %fourAHighq12Shifted, %fourFifteen : i32
    %fourAHighq12Magic = arith.ori %fourAHighq12Nibble, %fourMagic : i32
    %fourAHighq12Float = arith.bitcast %fourAHighq12Magic : i32 to f32
    %fourAHighq12 = arith.subf %fourAHighq12Float, %fourMagicFloat : f32
    %fourAHighx12 = llvm.extractelement %fourAHighInputs3[%c0 : i32] : vector<4xf32>
    %fourAHighq13Shift = arith.constant 12 : i32
    %fourAHighq13Shifted = arith.shrui %fourAWord3, %fourAHighq13Shift : i32
    %fourAHighq13Nibble = arith.andi %fourAHighq13Shifted, %fourFifteen : i32
    %fourAHighq13Magic = arith.ori %fourAHighq13Nibble, %fourMagic : i32
    %fourAHighq13Float = arith.bitcast %fourAHighq13Magic : i32 to f32
    %fourAHighq13 = arith.subf %fourAHighq13Float, %fourMagicFloat : f32
    %fourAHighx13 = llvm.extractelement %fourAHighInputs3[%c1 : i32] : vector<4xf32>
    %fourAHighq14Shift = arith.constant 20 : i32
    %fourAHighq14Shifted = arith.shrui %fourAWord3, %fourAHighq14Shift : i32
    %fourAHighq14Nibble = arith.andi %fourAHighq14Shifted, %fourFifteen : i32
    %fourAHighq14Magic = arith.ori %fourAHighq14Nibble, %fourMagic : i32
    %fourAHighq14Float = arith.bitcast %fourAHighq14Magic : i32 to f32
    %fourAHighq14 = arith.subf %fourAHighq14Float, %fourMagicFloat : f32
    %fourAHighx14 = llvm.extractelement %fourAHighInputs3[%c2 : i32] : vector<4xf32>
    %fourAHighq15Shift = arith.constant 28 : i32
    %fourAHighq15Shifted = arith.shrui %fourAWord3, %fourAHighq15Shift : i32
    %fourAHighq15Nibble = arith.andi %fourAHighq15Shifted, %fourFifteen : i32
    %fourAHighq15Magic = arith.ori %fourAHighq15Nibble, %fourMagic : i32
    %fourAHighq15Float = arith.bitcast %fourAHighq15Magic : i32 to f32
    %fourAHighq15 = arith.subf %fourAHighq15Float, %fourMagicFloat : f32
    %fourAHighx15 = llvm.extractelement %fourAHighInputs3[%c3 : i32] : vector<4xf32>
    %fourAHighDot0p0 = arith.mulf %fourAHighq0, %fourAHighx0 fastmath<contract> : f32
    %fourAHighDot0m1 = arith.mulf %fourAHighq4, %fourAHighx4 fastmath<contract> : f32
    %fourAHighDot0p1 = arith.addf %fourAHighDot0p0, %fourAHighDot0m1 fastmath<contract> : f32
    %fourAHighDot0m2 = arith.mulf %fourAHighq8, %fourAHighx8 fastmath<contract> : f32
    %fourAHighDot0p2 = arith.addf %fourAHighDot0p1, %fourAHighDot0m2 fastmath<contract> : f32
    %fourAHighDot0m3 = arith.mulf %fourAHighq12, %fourAHighx12 fastmath<contract> : f32
    %fourAHighDot0p3 = arith.addf %fourAHighDot0p2, %fourAHighDot0m3 fastmath<contract> : f32
    %fourAHighSum0a = arith.addf %fourAHighx0, %fourAHighx1 : f32
    %fourAHighSum0b = arith.addf %fourAHighx2, %fourAHighx3 : f32
    %fourAHighSum0 = arith.addf %fourAHighSum0a, %fourAHighSum0b : f32
    %fourAHighDot1p0 = arith.mulf %fourAHighq1, %fourAHighx1 fastmath<contract> : f32
    %fourAHighDot1m1 = arith.mulf %fourAHighq5, %fourAHighx5 fastmath<contract> : f32
    %fourAHighDot1p1 = arith.addf %fourAHighDot1p0, %fourAHighDot1m1 fastmath<contract> : f32
    %fourAHighDot1m2 = arith.mulf %fourAHighq9, %fourAHighx9 fastmath<contract> : f32
    %fourAHighDot1p2 = arith.addf %fourAHighDot1p1, %fourAHighDot1m2 fastmath<contract> : f32
    %fourAHighDot1m3 = arith.mulf %fourAHighq13, %fourAHighx13 fastmath<contract> : f32
    %fourAHighDot1p3 = arith.addf %fourAHighDot1p2, %fourAHighDot1m3 fastmath<contract> : f32
    %fourAHighSum1a = arith.addf %fourAHighx4, %fourAHighx5 : f32
    %fourAHighSum1b = arith.addf %fourAHighx6, %fourAHighx7 : f32
    %fourAHighSum1 = arith.addf %fourAHighSum1a, %fourAHighSum1b : f32
    %fourAHighDot2p0 = arith.mulf %fourAHighq2, %fourAHighx2 fastmath<contract> : f32
    %fourAHighDot2m1 = arith.mulf %fourAHighq6, %fourAHighx6 fastmath<contract> : f32
    %fourAHighDot2p1 = arith.addf %fourAHighDot2p0, %fourAHighDot2m1 fastmath<contract> : f32
    %fourAHighDot2m2 = arith.mulf %fourAHighq10, %fourAHighx10 fastmath<contract> : f32
    %fourAHighDot2p2 = arith.addf %fourAHighDot2p1, %fourAHighDot2m2 fastmath<contract> : f32
    %fourAHighDot2m3 = arith.mulf %fourAHighq14, %fourAHighx14 fastmath<contract> : f32
    %fourAHighDot2p3 = arith.addf %fourAHighDot2p2, %fourAHighDot2m3 fastmath<contract> : f32
    %fourAHighSum2a = arith.addf %fourAHighx8, %fourAHighx9 : f32
    %fourAHighSum2b = arith.addf %fourAHighx10, %fourAHighx11 : f32
    %fourAHighSum2 = arith.addf %fourAHighSum2a, %fourAHighSum2b : f32
    %fourAHighDot3p0 = arith.mulf %fourAHighq3, %fourAHighx3 fastmath<contract> : f32
    %fourAHighDot3m1 = arith.mulf %fourAHighq7, %fourAHighx7 fastmath<contract> : f32
    %fourAHighDot3p1 = arith.addf %fourAHighDot3p0, %fourAHighDot3m1 fastmath<contract> : f32
    %fourAHighDot3m2 = arith.mulf %fourAHighq11, %fourAHighx11 fastmath<contract> : f32
    %fourAHighDot3p2 = arith.addf %fourAHighDot3p1, %fourAHighDot3m2 fastmath<contract> : f32
    %fourAHighDot3m3 = arith.mulf %fourAHighq15, %fourAHighx15 fastmath<contract> : f32
    %fourAHighDot3p3 = arith.addf %fourAHighDot3p2, %fourAHighDot3m3 fastmath<contract> : f32
    %fourAHighSum3a = arith.addf %fourAHighx12, %fourAHighx13 : f32
    %fourAHighSum3b = arith.addf %fourAHighx14, %fourAHighx15 : f32
    %fourAHighSum3 = arith.addf %fourAHighSum3a, %fourAHighSum3b : f32
    %fourAHighDotPair0 = arith.addf %fourAHighDot0p3, %fourAHighDot1p3 : f32
    %fourAHighDotPair1 = arith.addf %fourAHighDot2p3, %fourAHighDot3p3 : f32
    %fourAHighDot = arith.addf %fourAHighDotPair0, %fourAHighDotPair1 : f32
    %fourAHighSumPair0 = arith.addf %fourAHighSum0, %fourAHighSum1 : f32
    %fourAHighSumPair1 = arith.addf %fourAHighSum2, %fourAHighSum3 : f32
    %fourAHighSum = arith.addf %fourAHighSumPair0, %fourAHighSumPair1 : f32
    %fourALowScaled = arith.mulf %fourALowScaleFactor, %fourALowDot fastmath<contract> : f32
    %fourAHighScaled = arith.mulf %fourAHighScaleFactor, %fourAHighDot fastmath<contract> : f32
    %fourALevels = arith.addf %fourALowScaled, %fourAHighScaled fastmath<contract> : f32
    %fourALowShifted = arith.mulf %fourALowMinimumFactor, %fourALowSum fastmath<contract> : f32
    %fourAHighShifted = arith.mulf %fourAHighMinimumFactor, %fourAHighSum fastmath<contract> : f32
    %fourAShifts = arith.addf %fourALowShifted, %fourAHighShifted fastmath<contract> : f32
    %fourAScaled = arith.mulf %fourAScale, %fourALevels fastmath<contract> : f32
    %fourAShifted = arith.mulf %fourAMinimum, %fourAShifts fastmath<contract> : f32
    %fourATerm = arith.subf %fourAScaled, %fourAShifted fastmath<contract> : f32
    %fourNext = arith.addf %fourSum, %fourATerm : f32
    %fourFollowing = arith.addi %fourUnit, %lanes32 : i32
    cf.br ^four(%fourFollowing, %fourNext : i32, f32)
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
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
  // the 8 bit code of a number scaled to the 127 steps of its scale, with residual the code of what that code leaves of
  // it in steps of 1 / 254 (the second pass of strict_gpu_tokens for rows that keep their tokens)
  func.func private @strict_device_code(%scaled: f32, %residual: i1) -> i8 attributes {llvm.passthrough = ["alwaysinline"]} {
    %rounded = func.call @strict_device_rounded(%scaled) : (f32) -> i32
    %whole = arith.sitofp %rounded : i32 to f32
    %left = arith.subf %scaled, %whole : f32
    %steps = arith.constant 254.0 : f32
    %finer = arith.mulf %left, %steps : f32
    %fine = func.call @strict_device_rounded(%finer) : (f32) -> i32
    %code = arith.select %residual, %fine, %rounded : i32
    %byte = arith.trunci %code : i32 to i8
    return %byte : i8
  }
  // the scale of the codes of strict_device_code
  func.func private @strict_device_code_scale(%scale: f32, %residual: i1) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %steps = arith.constant 254.0 : f32
    %finer = arith.divf %scale, %steps : f32
    %chosen = arith.select %residual, %finer, %scale : f32
    return %chosen : f32
  }
  // whether the bits of a quantize kernel ask for residual codes (1024 added, strict_gpu_quantized) and the format of
  // its source numbers
  func.func private @strict_device_code_format(%bits: i64) -> (i1, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %flag = arith.constant 1024 : i64
    %formats = arith.constant 1023 : i64
    %none = arith.constant 0 : i64
    %flagged = arith.andi %bits, %flag : i64
    %residual = arith.cmpi ne, %flagged, %none : i64
    %format = arith.andi %bits, %formats : i64
    return %residual, %format : i1, i64
  }
  // 32 or 16 bit numbers (bits as in strict_device_floats) to 8 bit numbers with a scale per 32 (a warp shares the
  // largest magnitude), the numbers that dot products with packed rows read through dp4a; width is not used, the
  // parameters are those of strict_device_quantize_tokens
  func.func private @strict_device_quantize_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %residual, %format = func.call @strict_device_code_format(%bits) : (i64) -> (i1, i64)
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %blockSize = func.call @strict_device_block_size() : () -> i32
    %start = arith.muli %block, %blockSize : i32
    %sum = arith.addi %start, %thread : i32
    %at = arith.extui %sum : i32 to i64
    %inside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %inside, ^body, ^exit
  ^body:
    %value = func.call @strict_device_floats(%source, %format, %at) : (!llvm.ptr, i64, i64) -> f32
    %magnitude = llvm.call_intrinsic "llvm.fabs.f32"(%value) : (f32) -> f32
    %largest = func.call @strict_device_largest(%magnitude) : (f32) -> f32
    %most = arith.constant 127.0 : f32
    %zero = arith.constant 0.0 : f32
    %largestScale = arith.divf %largest, %most : f32
    %scale = func.call @strict_device_code_scale(%largestScale, %residual) : (f32, i1) -> f32
    %isEmpty = arith.cmpf oeq, %largest, %zero : f32
    %inverse = arith.divf %most, %largest : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    %scaled = arith.mulf %value, %factor : f32
    %byte = func.call @strict_device_code(%scaled, %residual) : (f32, i1) -> i8
    %byteSlot = llvm.getelementptr %target[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %byte, %byteSlot : i8, !llvm.ptr
    %thirtyOne = arith.constant 31 : i64
    %lane = arith.andi %at, %thirtyOne : i64
    %none = arith.constant 0 : i64
    %isFirst = arith.cmpi eq, %lane, %none : i64
    cf.cond_br %isFirst, ^scale, ^exit
  ^scale:
    %scales = llvm.getelementptr %target[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %five = arith.constant 5 : i64
    %scaleIndex = arith.shrui %at, %five : i64
    %scaleSlot = llvm.getelementptr %scales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %scaleSlot : f32, !llvm.ptr
    cf.br ^exit
  ^exit:
    return
  }
  // the doubles of a list computed on the GPU as 32 bit floats (strict_gpu_narrow), rounded like arith.truncf on the CPU
  llvm.func @strict_device_singles(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {nvvm.kernel} {
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %size = func.call @strict_device_block_size() : () -> i32
    %thread64 = arith.extui %thread : i32 to i64
    %block64 = arith.extui %block : i32 to i64
    %size64 = arith.extui %size : i32 to i64
    %start = arith.muli %block64, %size64 : i64
    %at = arith.addi %start, %thread64 : i64
    %inside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %inside, ^convert, ^exit
  ^convert:
    %doubleSlot = llvm.getelementptr %source[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %double = llvm.load %doubleSlot : !llvm.ptr -> f64
    %single = arith.truncf %double : f64 to f32
    %singleSlot = llvm.getelementptr %target[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %single, %singleSlot : f32, !llvm.ptr
    cf.br ^exit
  ^exit:
    llvm.return
  }
  llvm.func @strict_device_quantize(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {nvvm.kernel} {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    cf.cond_br %isBrain, ^brain, ^other
  ^brain:
    func.call @strict_device_quantize_run(%source, %count, %width, %target, %brainBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^other:
    func.call @strict_device_quantize_run(%source, %count, %width, %target, %bits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  }
  // 32 or 16 bit numbers to 8 bit numbers with one scale per token of width numbers (a block of 256 threads per token, a
  // warp per token of at most 1024 numbers, its scales per 32 all are the scale of the token, like the activations of
  // ComfyUI's int8 layers) for strict_device_eights
  llvm.mlir.global internal @strict_device_quantize_largest() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8 x f32>
  func.func private @strict_device_quantize_tokens_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %residual, %format = func.call @strict_device_code_format(%bits) : (i64) -> (i1, i64)
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %c8 = arith.constant 8 : i32
    %c31 = arith.constant 31 : i32
    %five = arith.constant 5 : i64
    %threads = arith.constant 256 : i64
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %thread64 = arith.extui %thread : i32 to i64
    %token = arith.extui %block : i32 to i64
    %start = arith.muli %token, %width : i64
    cf.br ^find(%thread64, %zero : i64, f32)
  ^find(%at: i64, %largest: f32):
    %moreFound = arith.cmpi slt, %at, %width : i64
    cf.cond_br %moreFound, ^findBody, ^reduce
  ^findBody:
    %valueAt = arith.addi %start, %at : i64
    %value = func.call @strict_device_floats(%source, %format, %valueAt) : (!llvm.ptr, i64, i64) -> f32
    %magnitude = llvm.call_intrinsic "llvm.fabs.f32"(%value) : (f32) -> f32
    %larger = arith.maxnumf %largest, %magnitude : f32
    %atNext = arith.addi %at, %threads : i64
    cf.br ^find(%atNext, %larger : i64, f32)
  ^reduce:
    %warpLargest = func.call @strict_device_largest(%largest) : (f32) -> f32
    %shared = llvm.mlir.addressof @strict_device_quantize_largest : !llvm.ptr<3>
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %isFirst = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %isFirst, ^share, ^shared
  ^share:
    %warp64 = arith.extui %warp : i32 to i64
    %sharedSlot = llvm.getelementptr %shared[%warp64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %warpLargest, %sharedSlot : f32, !llvm.ptr<3>
    cf.br ^shared
  ^shared:
    func.call @strict_device_barrier() : () -> ()
    cf.br ^gather(%c0, %zero : i32, f32)
  ^gather(%other: i32, %gathered: f32):
    %moreOthers = arith.cmpi slt, %other, %c8 : i32
    cf.cond_br %moreOthers, ^gatherBody, ^scaled
  ^gatherBody:
    %other64 = arith.extui %other : i32 to i64
    %otherSlot = llvm.getelementptr %shared[%other64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %otherLargest = llvm.load %otherSlot : !llvm.ptr<3> -> f32
    %gatheredNext = arith.maxnumf %gathered, %otherLargest : f32
    %otherNext = arith.addi %other, %c1 : i32
    cf.br ^gather(%otherNext, %gatheredNext : i32, f32)
  ^scaled:
    %largestScale = arith.divf %gathered, %most : f32
    %scale = func.call @strict_device_code_scale(%largestScale, %residual) : (f32, i1) -> f32
    %isEmpty = arith.cmpf oeq, %gathered, %zero : f32
    %inverse = arith.divf %most, %gathered : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    %bytes = llvm.getelementptr %target[%start] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.br ^write(%thread64 : i64)
  ^write(%writeAt: i64):
    %moreWritten = arith.cmpi slt, %writeAt, %width : i64
    cf.cond_br %moreWritten, ^writeBody, ^scales
  ^writeBody:
    %writtenAt = arith.addi %start, %writeAt : i64
    %written = func.call @strict_device_floats(%source, %format, %writtenAt) : (!llvm.ptr, i64, i64) -> f32
    %multiplied = arith.mulf %written, %factor : f32
    %byte = func.call @strict_device_code(%multiplied, %residual) : (f32, i1) -> i8
    %byteSlot = llvm.getelementptr %bytes[%writeAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %byte, %byteSlot : i8, !llvm.ptr
    %writeNext = arith.addi %writeAt, %threads : i64
    cf.br ^write(%writeNext : i64)
  ^scales:
    %scaleStart = llvm.getelementptr %target[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %tokenParts = arith.shrui %start, %five : i64
    %parts = arith.shrui %width, %five : i64
    cf.br ^scale(%thread64 : i64)
  ^scale(%part: i64):
    %moreParts = arith.cmpi slt, %part, %parts : i64
    cf.cond_br %moreParts, ^scaleBody, ^exit
  ^scaleBody:
    %partIndex = arith.addi %tokenParts, %part : i64
    %scaleSlot = llvm.getelementptr %scaleStart[%partIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %scaleSlot : f32, !llvm.ptr
    %partNext = arith.addi %part, %threads : i64
    cf.br ^scale(%partNext : i64)
  ^exit:
    return
  }
  // strict_device_quantize_tokens_run of tokens of at most 1024 numbers, a warp per token
  func.func private @strict_device_quantize_rows_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %residual, %format = func.call @strict_device_code_format(%bits) : (i64) -> (i1, i64)
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %c31 = arith.constant 31 : i32
    %five = arith.constant 5 : i64
    %lanes = arith.constant 32 : i64
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %blockRows = arith.shli %block, %c3 : i32
    %row = arith.addi %blockRows, %warp : i32
    %row64 = arith.extui %row : i32 to i64
    %lane64 = arith.extui %lane : i32 to i64
    %rows = arith.divui %count, %width : i64
    %inside = arith.cmpi slt, %row64, %rows : i64
    cf.cond_br %inside, ^start, ^exit
  ^start:
    %start = arith.muli %row64, %width : i64
    cf.br ^find(%lane64, %zero : i64, f32)
  ^find(%at: i64, %largest: f32):
    %moreFound = arith.cmpi slt, %at, %width : i64
    cf.cond_br %moreFound, ^findBody, ^scaled
  ^findBody:
    %valueAt = arith.addi %start, %at : i64
    %value = func.call @strict_device_floats(%source, %format, %valueAt) : (!llvm.ptr, i64, i64) -> f32
    %magnitude = llvm.call_intrinsic "llvm.fabs.f32"(%value) : (f32) -> f32
    %larger = arith.maxnumf %largest, %magnitude : f32
    %atNext = arith.addi %at, %lanes : i64
    cf.br ^find(%atNext, %larger : i64, f32)
  ^scaled:
    %rowLargest = func.call @strict_device_largest(%largest) : (f32) -> f32
    %largestScale = arith.divf %rowLargest, %most : f32
    %scale = func.call @strict_device_code_scale(%largestScale, %residual) : (f32, i1) -> f32
    %isEmpty = arith.cmpf oeq, %rowLargest, %zero : f32
    %inverse = arith.divf %most, %rowLargest : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    %bytes = llvm.getelementptr %target[%start] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    cf.br ^write(%lane64 : i64)
  ^write(%writeAt: i64):
    %moreWritten = arith.cmpi slt, %writeAt, %width : i64
    cf.cond_br %moreWritten, ^writeBody, ^scales
  ^writeBody:
    %writtenAt = arith.addi %start, %writeAt : i64
    %written = func.call @strict_device_floats(%source, %format, %writtenAt) : (!llvm.ptr, i64, i64) -> f32
    %multiplied = arith.mulf %written, %factor : f32
    %byte = func.call @strict_device_code(%multiplied, %residual) : (f32, i1) -> i8
    %byteSlot = llvm.getelementptr %bytes[%writeAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %byte, %byteSlot : i8, !llvm.ptr
    %writeNext = arith.addi %writeAt, %lanes : i64
    cf.br ^write(%writeNext : i64)
  ^scales:
    %scaleStart = llvm.getelementptr %target[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %tokenParts = arith.shrui %start, %five : i64
    %parts = arith.shrui %width, %five : i64
    %isScaling = arith.cmpi slt, %lane64, %parts : i64
    cf.cond_br %isScaling, ^scale, ^exit
  ^scale:
    %partIndex = arith.addi %tokenParts, %lane64 : i64
    %scaleSlot = llvm.getelementptr %scaleStart[%partIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %scaleSlot : f32, !llvm.ptr
    cf.br ^exit
  ^exit:
    return
  }
  llvm.func @strict_device_quantize_tokens(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>} {
    %brainBits = arith.constant 17 : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    %warpWidth = arith.constant 1024 : i64
    %isNarrow = arith.cmpi sle, %width, %warpWidth : i64
    cf.cond_br %isNarrow, ^rows, ^tokens
  ^rows:
    cf.cond_br %isBrain, ^brainRows, ^otherRows
  ^brainRows:
    func.call @strict_device_quantize_rows_run(%source, %count, %width, %target, %brainBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^otherRows:
    func.call @strict_device_quantize_rows_run(%source, %count, %width, %target, %bits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^tokens:
    cf.cond_br %isBrain, ^brain, ^other
  ^brain:
    func.call @strict_device_quantize_tokens_run(%source, %count, %width, %target, %brainBits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  ^other:
    func.call @strict_device_quantize_tokens_run(%source, %count, %width, %target, %bits) : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> ()
    llvm.return
  }
  // strict_gpu_tuned: a block of 256 threads per 16 packed rows of 8 bit numbers (rows, one every stride bytes, the scales
  // per 32 at scaleOffset) adds ups (rank numbers per row, scaled) times downs (rank x width) to every row whose flag is
  // not 0, then requantizes it to its largest magnitude / 127 with stochastic rounding like TensorFile_Tuned. A thread sums
  // the 16 rows of its columns at once, so downs are read once per 16 rows (the ups of the rows by rank part in shared)
  llvm.mlir.global internal @strict_device_tuned_ups() {addr_space = 3 : i32, alignment = 64 : i64} : !llvm.array<8192 x f32>
  // the scales, flags and largest magnitudes of the 16 rows, then the largest of every warp (8 times 16)
  llvm.mlir.global internal @strict_device_tuned_rows() {addr_space = 3 : i32, alignment = 64 : i64} : !llvm.array<176 x f32>
  llvm.func @strict_device_tuned(%bytes: !llvm.ptr, %stride: i64, %scaleOffset: i64, %downs: !llvm.ptr, %ups: !llvm.ptr, %flags: !llvm.ptr, %width: i64, %rank: i64, %seed: i64, %height: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32} {
    func.call @strict_device_tuned_run(%bytes, %stride, %scaleOffset, %downs, %ups, %flags, %width, %rank, %seed, %height) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> ()
    llvm.return
  }
  func.func private @strict_device_tuned_run(%bytes: !llvm.ptr, %stride: i64, %scaleOffset: i64, %downs: !llvm.ptr, %ups: !llvm.ptr, %flags: !llvm.ptr, %width: i64, %rank: i64, %seed: i64, %height: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c31 = arith.constant 31 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %thirtyTwo = arith.constant 32 : i64
    %fortyEight = arith.constant 48 : i64
    %threads = arith.constant 256 : i64
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %mosts = arith.constant dense<127.0> : vector<16xf32>
    %iota = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi64>
    %thread = func.call @strict_device_thread_index() : () -> i32
    %block = func.call @strict_device_block_index() : () -> i32
    %thread64 = arith.extui %thread : i32 to i64
    %block64 = arith.extui %block : i32 to i64
    %first = arith.muli %block64, %sixteen : i64
    %transposed = llvm.mlir.addressof @strict_device_tuned_ups : !llvm.ptr<3>
    %shared = llvm.mlir.addressof @strict_device_tuned_rows : !llvm.ptr<3>
    %upCount = arith.muli %rank, %sixteen : i64
    cf.br ^load(%thread64 : i64)
  ^load(%at: i64):
    %moreUps = arith.cmpi slt, %at, %upCount : i64
    cf.cond_br %moreUps, ^loadBody, ^rowLoad
  ^loadBody:
    %part = arith.remui %at, %rank : i64
    %index = arith.divui %at, %rank : i64
    %row = arith.addi %first, %index : i64
    %isRow = arith.cmpi slt, %row, %height : i64
    %rowStart = arith.muli %row, %rank : i64
    %upAt = arith.addi %rowStart, %part : i64
    %safeAt = arith.select %isRow, %upAt, %none : i64
    %upSlot = llvm.getelementptr %ups[%safeAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %up = llvm.load %upSlot : !llvm.ptr -> f32
    %kept = arith.select %isRow, %up, %zero : f32
    %partStart = arith.muli %part, %sixteen : i64
    %sharedAt = arith.addi %partStart, %index : i64
    %sharedSlot = llvm.getelementptr %transposed[%sharedAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %kept, %sharedSlot : f32, !llvm.ptr<3>
    %atNext = arith.addi %at, %threads : i64
    cf.br ^load(%atNext : i64)
  ^rowLoad:
    %isRowThread = arith.cmpi slt, %thread64, %sixteen : i64
    cf.cond_br %isRowThread, ^rowValues, ^loaded
  ^rowValues:
    %ownRow = arith.addi %first, %thread64 : i64
    %ownInside = arith.cmpi slt, %ownRow, %height : i64
    %safeRow = arith.select %ownInside, %ownRow, %first : i64
    %ownStart = arith.muli %safeRow, %stride : i64
    %scale = func.call @strict_device_tuned_scale(%bytes, %ownStart, %scaleOffset) : (!llvm.ptr, i64, i64) -> f32
    %flagSlot = llvm.getelementptr %flags[%safeRow] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %flag = llvm.load %flagSlot : !llvm.ptr -> f32
    %ownScale = arith.select %ownInside, %scale, %zero : f32
    %ownFlag = arith.select %ownInside, %flag, %zero : f32
    %scaleSlot = llvm.getelementptr %shared[%thread64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %ownScale, %scaleSlot : f32, !llvm.ptr<3>
    %flagAt = arith.addi %thread64, %sixteen : i64
    %ownFlagSlot = llvm.getelementptr %shared[%flagAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %ownFlag, %ownFlagSlot : f32, !llvm.ptr<3>
    cf.br ^loaded
  ^loaded:
    func.call @strict_device_barrier() : () -> ()
    %scales = llvm.load %shared {alignment = 64 : i64} : !llvm.ptr<3> -> vector<16xf32>
    %flagsSlot = llvm.getelementptr %shared[%sixteen] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %rowFlags = llvm.load %flagsSlot {alignment = 64 : i64} : !llvm.ptr<3> -> vector<16xf32>
    %firsts = func.call @strict_device_tuned_splat(%first) : (i64) -> vector<16xi64>
    %heights = func.call @strict_device_tuned_splat(%height) : (i64) -> vector<16xi64>
    %strides = func.call @strict_device_tuned_splat(%stride) : (i64) -> vector<16xi64>
    %rows = arith.addi %firsts, %iota : vector<16xi64>
    %inside = arith.cmpi slt, %rows, %heights : vector<16xi64>
    %rowBase = arith.muli %first, %stride : i64
    %wideOffsets = arith.muli %iota, %strides : vector<16xi64>
    %offsets = arith.trunci %wideOffsets : vector<16xi64> to vector<16xi32>
    cf.br ^measure(%thread64, %zeros : i64, vector<16xf32>)
  ^measure(%column: i64, %maxima: vector<16xf32>):
    %moreColumns = arith.cmpi slt, %column, %width : i64
    cf.cond_br %moreColumns, ^measureBody, ^reduce
  ^measureBody:
    %sums = func.call @strict_device_tuned_sums(%downs, %column, %width, %rank) : (!llvm.ptr, i64, i64, i64) -> vector<16xf32>
    %place = arith.addi %rowBase, %column : i64
    %values = func.call @strict_device_tuned_values(%bytes, %place, %offsets, %inside, %scales, %sums) : (!llvm.ptr, i64, vector<16xi32>, vector<16xi1>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %magnitudes = llvm.intr.fabs(%values) : (vector<16xf32>) -> vector<16xf32>
    %maximaNext = llvm.intr.maxnum(%maxima, %magnitudes) : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %columnNext = arith.addi %column, %threads : i64
    cf.br ^measure(%columnNext, %maximaNext : i64, vector<16xf32>)
  ^reduce:
    %warp = arith.shrui %thread, %c5 : i32
    %lane = arith.andi %thread, %c31 : i32
    %isFirst = arith.cmpi eq, %lane, %c0 : i32
    %warp64 = arith.extui %warp : i32 to i64
    %warpStart = arith.muli %warp64, %sixteen : i64
    %warpOffset = arith.addi %warpStart, %fortyEight : i64
    cf.br ^warpRows(%none : i64)
  ^warpRows(%warpRow: i64):
    %moreWarpRows = arith.cmpi slt, %warpRow, %sixteen : i64
    cf.cond_br %moreWarpRows, ^warpRow, ^warpsDone
  ^warpRow:
    %rowMaximum = llvm.extractelement %maxima[%warpRow : i64] : vector<16xf32>
    %warpLargest = func.call @strict_device_largest(%rowMaximum) : (f32) -> f32
    %warpRowNext = arith.addi %warpRow, %one : i64
    cf.cond_br %isFirst, ^warpStore, ^warpRows(%warpRowNext : i64)
  ^warpStore:
    %warpAt = arith.addi %warpOffset, %warpRow : i64
    %warpSlot = llvm.getelementptr %shared[%warpAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %warpLargest, %warpSlot : f32, !llvm.ptr<3>
    cf.br ^warpRows(%warpRowNext : i64)
  ^warpsDone:
    func.call @strict_device_barrier() : () -> ()
    cf.cond_br %isRowThread, ^gather(%none, %zero : i64, f32), ^gathered
  ^gather(%gatherWarp: i64, %gatheredSoFar: f32):
    %moreWarps = arith.cmpi slt, %gatherWarp, %eight : i64
    cf.cond_br %moreWarps, ^gatherBody, ^gatherStore
  ^gatherBody:
    %gatherStart = arith.muli %gatherWarp, %sixteen : i64
    %gatherOffset = arith.addi %gatherStart, %fortyEight : i64
    %gatherAt = arith.addi %gatherOffset, %thread64 : i64
    %gatherSlot = llvm.getelementptr %shared[%gatherAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %other = llvm.load %gatherSlot : !llvm.ptr<3> -> f32
    %larger = arith.maxnumf %gatheredSoFar, %other : f32
    %gatherWarpNext = arith.addi %gatherWarp, %one : i64
    cf.br ^gather(%gatherWarpNext, %larger : i64, f32)
  ^gatherStore:
    %largestAt = arith.addi %thread64, %thirtyTwo : i64
    %largestSlot = llvm.getelementptr %shared[%largestAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %gatheredSoFar, %largestSlot : f32, !llvm.ptr<3>
    cf.br ^gathered
  ^gathered:
    func.call @strict_device_barrier() : () -> ()
    %largestsSlot = llvm.getelementptr %shared[%thirtyTwo] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %largests = llvm.load %largestsSlot {alignment = 64 : i64} : !llvm.ptr<3> -> vector<16xf32>
    %isEmpty = arith.cmpf oeq, %largests, %zeros : vector<16xf32>
    %inverses = arith.divf %mosts, %largests : vector<16xf32>
    %factors = arith.select %isEmpty, %zeros, %inverses : vector<16xi1>, vector<16xf32>
    %flagged = arith.cmpf une, %rowFlags, %zeros : vector<16xf32>
    %written = arith.andi %inside, %flagged : vector<16xi1>
    cf.br ^write(%thread64 : i64)
  ^write(%writeColumn: i64):
    %moreWritten = arith.cmpi slt, %writeColumn, %width : i64
    cf.cond_br %moreWritten, ^writeBody, ^scales
  ^writeBody:
    %again = func.call @strict_device_tuned_sums(%downs, %writeColumn, %width, %rank) : (!llvm.ptr, i64, i64, i64) -> vector<16xf32>
    %writePlace = arith.addi %rowBase, %writeColumn : i64
    %updated = func.call @strict_device_tuned_values(%bytes, %writePlace, %offsets, %inside, %scales, %again) : (!llvm.ptr, i64, vector<16xi32>, vector<16xi1>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    func.call @strict_device_tuned_stored(%bytes, %writePlace, %offsets, %updated, %factors, %written, %seed) : (!llvm.ptr, i64, vector<16xi32>, vector<16xf32>, vector<16xf32>, vector<16xi1>, i64) -> ()
    %writeColumnNext = arith.addi %writeColumn, %threads : i64
    cf.br ^write(%writeColumnNext : i64)
  ^scales:
    cf.cond_br %isRowThread, ^rescale, ^done
  ^rescale:
    %rescaledRow = arith.addi %first, %thread64 : i64
    %rescaledInside = arith.cmpi slt, %rescaledRow, %height : i64
    %rescaledSafe = arith.select %rescaledInside, %rescaledRow, %first : i64
    %rescaledStart = arith.muli %rescaledSafe, %stride : i64
    %ownLargestAt = arith.addi %thread64, %thirtyTwo : i64
    %ownLargestSlot = llvm.getelementptr %shared[%ownLargestAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %ownLargest = llvm.load %ownLargestSlot : !llvm.ptr<3> -> f32
    %rescaledFlagAt = arith.addi %thread64, %sixteen : i64
    %rescaledFlagSlot = llvm.getelementptr %shared[%rescaledFlagAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %rescaledFlag = llvm.load %rescaledFlagSlot : !llvm.ptr<3> -> f32
    func.call @strict_device_tuned_rescaled(%bytes, %rescaledStart, %scaleOffset, %width, %ownLargest, %rescaledFlag) : (!llvm.ptr, i64, i64, i64, f32, f32) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_device_tuned_splat(%number: i64) -> vector<16xi64> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %undefined = llvm.mlir.undef : vector<16xi64>
    %inserted = llvm.insertelement %number, %undefined[%c0 : i32] : vector<16xi64>
    %splat = llvm.shufflevector %inserted, %undefined [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi64>
    return %splat : vector<16xi64>
  }
  // the updates of the 16 rows at column: the sums of their ups (in shared memory) times the column of downs
  func.func private @strict_device_tuned_sums(%downs: !llvm.ptr, %column: i64, %width: i64, %rank: i64) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %undefined = llvm.mlir.undef : vector<16xf32>
    %transposed = llvm.mlir.addressof @strict_device_tuned_ups : !llvm.ptr<3>
    cf.br ^sum(%none, %zeros : i64, vector<16xf32>)
  ^sum(%part: i64, %sums: vector<16xf32>):
    %more = arith.cmpi slt, %part, %rank : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %downStart = arith.muli %part, %width : i64
    %downAt = arith.addi %downStart, %column : i64
    %downSlot = llvm.getelementptr %downs[%downAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %down = llvm.load %downSlot : !llvm.ptr -> f32
    %inserted = llvm.insertelement %down, %undefined[%c0 : i32] : vector<16xf32>
    %downs16 = llvm.shufflevector %inserted, %undefined [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %upAt = arith.muli %part, %sixteen : i64
    %upSlot = llvm.getelementptr %transposed[%upAt] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    %ups16 = llvm.load %upSlot {alignment = 64 : i64} : !llvm.ptr<3> -> vector<16xf32>
    %next = llvm.intr.fma(%ups16, %downs16, %sums) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %partNext = arith.addi %part, %one : i64
    cf.br ^sum(%partNext, %next : i64, vector<16xf32>)
  ^done:
    return %sums : vector<16xf32>
  }
  // the numbers of the 16 rows (offsets after the first, inside the matrix) at place plus their updates
  func.func private @strict_device_tuned_values(%bytes: !llvm.ptr, %place: i64, %offsets: vector<16xi32>, %inside: vector<16xi1>, %scales: vector<16xf32>, %sums: vector<16xf32>) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %nothing = arith.constant dense<0> : vector<16xi8>
    %first = llvm.getelementptr %bytes[%place] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %pointers = llvm.getelementptr %first[%offsets] : (!llvm.ptr, vector<16xi32>) -> vector<16x!llvm.ptr>, i8
    %numbers = llvm.intr.masked.gather %pointers, %inside, %nothing {alignment = 1 : i32} : (vector<16x!llvm.ptr>, vector<16xi1>, vector<16xi8>) -> vector<16xi8>
    %floats = arith.sitofp %numbers : vector<16xi8> to vector<16xf32>
    %values = llvm.intr.fma(%floats, %scales, %sums) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    return %values : vector<16xf32>
  }
  // values times factors (127 / largest magnitude) with stochastic rounding into the bytes of the written rows (offsets
  // after the first) at place
  func.func private @strict_device_tuned_stored(%bytes: !llvm.ptr, %place: i64, %offsets: vector<16xi32>, %values: vector<16xf32>, %factors: vector<16xf32>, %written: vector<16xi1>, %seed: i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %ones = arith.constant dense<1.0> : vector<16xf32>
    %mosts = arith.constant dense<127.0> : vector<16xf32>
    %leasts = arith.constant dense<-127.0> : vector<16xf32>
    %firstPlaces = func.call @strict_device_tuned_splat(%place) : (i64) -> vector<16xi64>
    %wideOffsets = arith.extui %offsets : vector<16xi32> to vector<16xi64>
    %places = arith.addi %firstPlaces, %wideOffsets : vector<16xi64>
    %scaled = arith.mulf %values, %factors : vector<16xf32>
    %lower = llvm.intr.floor(%scaled) : (vector<16xf32>) -> vector<16xf32>
    %uniforms = func.call @strict_device_uniforms(%seed, %places) : (i64, vector<16xi64>) -> vector<16xf32>
    %fraction = arith.subf %scaled, %lower : vector<16xf32>
    %isUp = arith.cmpf ogt, %fraction, %uniforms : vector<16xf32>
    %step = arith.select %isUp, %ones, %zeros : vector<16xi1>, vector<16xf32>
    %rounded = arith.addf %lower, %step : vector<16xf32>
    %capped = arith.minnumf %rounded, %mosts : vector<16xf32>
    %clamped = arith.maxnumf %capped, %leasts : vector<16xf32>
    %whole = arith.fptosi %clamped : vector<16xf32> to vector<16xi32>
    %narrow = arith.trunci %whole : vector<16xi32> to vector<16xi8>
    %first = llvm.getelementptr %bytes[%place] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %pointers = llvm.getelementptr %first[%offsets] : (!llvm.ptr, vector<16xi32>) -> vector<16x!llvm.ptr>, i8
    llvm.intr.masked.scatter %narrow, %pointers, %written {alignment = 1 : i32} : vector<16xi8>, vector<16xi1> into vector<16x!llvm.ptr>
    return
  }
  // the scale of the row starting at start (its first scale per 32)
  func.func private @strict_device_tuned_scale(%bytes: !llvm.ptr, %start: i64, %scaleOffset: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %at = arith.addi %start, %scaleOffset : i64
    %slot = llvm.getelementptr %bytes[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %scale = llvm.load %slot : !llvm.ptr -> f32
    return %scale : f32
  }
  // the new scale of a flagged row into every one of its scales per 32
  func.func private @strict_device_tuned_rescaled(%bytes: !llvm.ptr, %start: i64, %scaleOffset: i64, %width: i64, %largest: f32, %flag: f32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %thirtyOne = arith.constant 31 : i64
    %zero = arith.constant 0.0 : f32
    %most = arith.constant 127.0 : f32
    %isFlagged = arith.cmpf une, %flag, %zero : f32
    cf.cond_br %isFlagged, ^store, ^done
  ^store:
    %scale = arith.divf %largest, %most : f32
    %at = arith.addi %start, %scaleOffset : i64
    %scales = llvm.getelementptr %bytes[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %widened = arith.addi %width, %thirtyOne : i64
    %parts = arith.shrui %widened, %five : i64
    cf.br ^part(%none : i64)
  ^part(%part: i64):
    %more = arith.cmpi slt, %part, %parts : i64
    cf.cond_br %more, ^partBody, ^done
  ^partBody:
    %slot = llvm.getelementptr %scales[%part] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %scale, %slot : f32, !llvm.ptr
    %next = arith.addi %part, %one : i64
    cf.br ^part(%next : i64)
  ^done:
    return
  }
  // numbers in [0, 1) for places in the stream of seed (splitmix64, like Uniform in TensorFile.c)
  func.func private @strict_device_uniforms(%seed: i64, %places: vector<16xi64>) -> vector<16xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %ones = arith.constant dense<1> : vector<16xi64>
    %golden = arith.constant dense<-7046029254386353131> : vector<16xi64>
    %first = arith.constant dense<-4658895280553007687> : vector<16xi64>
    %second = arith.constant dense<-7723592293110705685> : vector<16xi64>
    %c27 = arith.constant dense<27> : vector<16xi64>
    %c30 = arith.constant dense<30> : vector<16xi64>
    %c31 = arith.constant dense<31> : vector<16xi64>
    %c40 = arith.constant dense<40> : vector<16xi64>
    %unit = arith.constant dense<16777216.0> : vector<16xf32>
    %seeds = func.call @strict_device_tuned_splat(%seed) : (i64) -> vector<16xi64>
    %counted = arith.addi %places, %ones : vector<16xi64>
    %stepped = arith.muli %counted, %golden : vector<16xi64>
    %mixed = arith.addi %seeds, %stepped : vector<16xi64>
    %shifted1 = arith.shrui %mixed, %c30 : vector<16xi64>
    %xored1 = arith.xori %mixed, %shifted1 : vector<16xi64>
    %mixed1 = arith.muli %xored1, %first : vector<16xi64>
    %shifted2 = arith.shrui %mixed1, %c27 : vector<16xi64>
    %xored2 = arith.xori %mixed1, %shifted2 : vector<16xi64>
    %mixed2 = arith.muli %xored2, %second : vector<16xi64>
    %shifted3 = arith.shrui %mixed2, %c31 : vector<16xi64>
    %xored3 = arith.xori %mixed2, %shifted3 : vector<16xi64>
    %bits = arith.shrui %xored3, %c40 : vector<16xi64>
    %numbers = arith.uitofp %bits : vector<16xi64> to vector<16xf32>
    %uniform = arith.divf %numbers, %unit : vector<16xf32>
    return %uniform : vector<16xf32>
  }
  func.func private @strict_device_largest(%value: f32) -> f32 {
    %sixteen = arith.constant 16 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    cf.br ^round(%value, %sixteen : f32, i32)
  ^round(%largest: f32, %offset: i32):
    %more = arith.cmpi sge, %offset, %one : i32
    cf.cond_br %more, ^shuffle, ^done
  ^shuffle:
    %other = func.call @strict_device_shuffles(%largest, %offset) : (f32, i32) -> f32
    %larger = arith.maxnumf %largest, %other : f32
    %half = arith.divui %offset, %two : i32
    cf.br ^round(%larger, %half : f32, i32)
  ^done:
    return %largest : f32
  }
  // m16n8k16 of 16 bit floats as two m16n8k8 (numbers 0 to 7, then 8 to 15)
  func.func private @strict_device_halves_mma8x2(%a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>, %b0: vector<2xf16>, %b1: vector<2xf16>, %c0: f32, %c1: f32, %c2: f32, %c3: f32) -> !llvm.struct<(f32, f32, f32, f32)> attributes {llvm.passthrough = ["alwaysinline"]} {
    %first = func.call @strict_device_halves_mma(%a0, %a1, %b0, %c0, %c1, %c2, %c3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %first0 = llvm.extractvalue %first[0] : !llvm.struct<(f32, f32, f32, f32)>
    %first1 = llvm.extractvalue %first[1] : !llvm.struct<(f32, f32, f32, f32)>
    %first2 = llvm.extractvalue %first[2] : !llvm.struct<(f32, f32, f32, f32)>
    %first3 = llvm.extractvalue %first[3] : !llvm.struct<(f32, f32, f32, f32)>
    %second = func.call @strict_device_halves_mma(%a2, %a3, %b1, %first0, %first1, %first2, %first3) : (vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    return %second : !llvm.struct<(f32, f32, f32, f32)>
  }
  func.func private @strict_device_words(%first: i32, %second: i32, %third: i32, %fourth: i32) -> vector<4xi32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %empty = llvm.mlir.poison : vector<4xi32>
    %one = llvm.insertelement %first, %empty[%c0 : i32] : vector<4xi32>
    %two = llvm.insertelement %second, %one[%c1 : i32] : vector<4xi32>
    %three = llvm.insertelement %third, %two[%c2 : i32] : vector<4xi32>
    %four = llvm.insertelement %fourth, %three[%c3 : i32] : vector<4xi32>
    return %four : vector<4xi32>
  }
  // the sum of the products of 4 pairs of 4 bytes
  func.func private @strict_device_dp4as(%left: vector<4xi32>, %right: vector<4xi32>) -> i32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %nothing = arith.constant 0 : i32
    %left0 = llvm.extractelement %left[%c0 : i32] : vector<4xi32>
    %left1 = llvm.extractelement %left[%c1 : i32] : vector<4xi32>
    %left2 = llvm.extractelement %left[%c2 : i32] : vector<4xi32>
    %left3 = llvm.extractelement %left[%c3 : i32] : vector<4xi32>
    %right0 = llvm.extractelement %right[%c0 : i32] : vector<4xi32>
    %right1 = llvm.extractelement %right[%c1 : i32] : vector<4xi32>
    %right2 = llvm.extractelement %right[%c2 : i32] : vector<4xi32>
    %right3 = llvm.extractelement %right[%c3 : i32] : vector<4xi32>
    %sum0 = func.call @strict_device_dp4a(%left0, %right0, %nothing) : (i32, i32, i32) -> i32
    %sum1 = func.call @strict_device_dp4a(%left1, %right1, %sum0) : (i32, i32, i32) -> i32
    %sum2 = func.call @strict_device_dp4a(%left2, %right2, %sum1) : (i32, i32, i32) -> i32
    %sum3 = func.call @strict_device_dp4a(%left3, %right3, %sum2) : (i32, i32, i32) -> i32
    return %sum3 : i32
  }
  // the 6 bit scale and minimum of part j (32 numbers) of a Q4_K block, like get_scale_min_k4 of ggml: the bytes j & 3 of
  // the three scale words are all it needs (constant indices keep the header in registers)
  func.func private @strict_device_four_scale(%header: vector<4xi32>, %j: i32) -> (i32, i32) attributes {llvm.passthrough = ["alwaysinline"]} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %three = arith.constant 3 : i32
    %four = arith.constant 4 : i32
    %six = arith.constant 6 : i32
    %fifteen = arith.constant 15 : i32
    %sixtyThree = arith.constant 63 : i32
    %byteMask = arith.constant 255 : i32
    %isLow = arith.cmpi ult, %j, %four : i32
    %inWord = arith.andi %j, %three : i32
    %shift = arith.shli %inWord, %three : i32
    %firstWord = llvm.extractelement %header[%c1 : i32] : vector<4xi32>
    %secondWord = llvm.extractelement %header[%c2 : i32] : vector<4xi32>
    %thirdWord = llvm.extractelement %header[%c3 : i32] : vector<4xi32>
    %firstShifted = arith.shrui %firstWord, %shift : i32
    %first = arith.andi %firstShifted, %byteMask : i32
    %secondShifted = arith.shrui %secondWord, %shift : i32
    %second = arith.andi %secondShifted, %byteMask : i32
    %thirdShifted = arith.shrui %thirdWord, %shift : i32
    %third = arith.andi %thirdShifted, %byteMask : i32
    %lowScale = arith.andi %first, %sixtyThree : i32
    %lowMinimum = arith.andi %second, %sixtyThree : i32
    %thirdLow = arith.andi %third, %fifteen : i32
    %firstTop = arith.shrui %first, %six : i32
    %firstUp = arith.shli %firstTop, %four : i32
    %highScale = arith.ori %thirdLow, %firstUp : i32
    %thirdHigh = arith.shrui %third, %four : i32
    %secondTop = arith.shrui %second, %six : i32
    %secondUp = arith.shli %secondTop, %four : i32
    %highMinimum = arith.ori %thirdHigh, %secondUp : i32
    %scale = arith.select %isLow, %lowScale, %highScale : i32
    %minimum = arith.select %isLow, %lowMinimum, %highMinimum : i32
    return %scale, %minimum : i32, i32
  }
  // the dot product of a packed row (PTQ1_0, Q4_K, Q6_K or 8 bits) with 8 bit numbers (bytes, then a scale per 32),
  // the whole warp gets the sum
  func.func private @strict_device_dots8(%row: !llvm.ptr, %rowBits: i64, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %lane: i64, %lanes: i64) -> f32 attributes {llvm.passthrough = ["noinline"]} {
    %zero = arith.constant 0.0 : f32
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    cf.switch %rowBits : i64, [
      default: ^bytes(%lane, %zero : i64, f32),
      3: ^three(%lane, %zero : i64, f32),
      4: ^four(%lane, %zero : i64, f32),
      6: ^six(%lane, %zero : i64, f32)
    ]
  ^three(%threeUnit: i64, %threeSum: f32):
    %unitShift = arith.constant 6 : i64
    %threeUnits = arith.shrui %count, %unitShift : i64
    %threeMore = arith.cmpi slt, %threeUnit, %threeUnits : i64
    cf.cond_br %threeMore, ^threeBody, ^reduce(%threeSum : f32)
  ^threeBody:
    %threeWeights:7 = func.call @strict_device_trits_weights(%row, %count, %threeUnit) : (!llvm.ptr, i64, i64) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, f32, i64, i64)
    %threeNext = func.call @strict_device_trits_token(%threeWeights#0, %threeWeights#1, %threeWeights#2, %threeWeights#3, %threeWeights#4, %threeWeights#5, %threeWeights#6, %bytes, %scales, %threeSum) : (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %threeFollowing = arith.addi %threeUnit, %lanes : i64
    cf.br ^three(%threeFollowing, %threeNext : i64, f32)
  ^four(%unit: i64, %fourSum: f32):
    %units = arith.shrui %count, %five : i64
    %fourMore = arith.cmpi slt, %unit, %units : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
    %fourWeights:10 = func.call @strict_device_four_weights(%row, %unit) : (!llvm.ptr, i64) -> (vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64)
    %fourNext = func.call @strict_device_four_token(%fourWeights#0, %fourWeights#1, %fourWeights#2, %fourWeights#3, %fourWeights#4, %fourWeights#5, %fourWeights#6, %fourWeights#7, %fourWeights#8, %fourWeights#9, %bytes, %scales, %fourSum) : (vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %fourFollowing = arith.addi %unit, %lanes : i64
    cf.br ^four(%fourFollowing, %fourNext : i64, f32)
  ^six(%chunk: i64, %sixSum: f32):
    %chunks = arith.shrui %count, %four : i64
    %sixMore = arith.cmpi slt, %chunk, %chunks : i64
    cf.cond_br %sixMore, ^sixBody, ^reduce(%sixSum : f32)
  ^sixBody:
    %sixWeights:8 = func.call @strict_device_six_weights(%row, %count, %chunk) : (!llvm.ptr, i64, i64) -> (vector<4xi32>, f32, f32, f32, f32, f32, i64, i64)
    %sixNext = func.call @strict_device_six_token(%sixWeights#0, %sixWeights#1, %sixWeights#2, %sixWeights#3, %sixWeights#4, %sixWeights#5, %sixWeights#6, %sixWeights#7, %bytes, %scales, %sixSum) : (vector<4xi32>, f32, f32, f32, f32, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %sixFollowing = arith.addi %chunk, %lanes : i64
    cf.br ^six(%sixFollowing, %sixNext : i64, f32)
  ^bytes(%byteUnit: i64, %byteSum: f32):
    %byteUnits = arith.shrui %count, %four : i64
    %byteMore = arith.cmpi slt, %byteUnit, %byteUnits : i64
    cf.cond_br %byteMore, ^bytesBody, ^reduce(%byteSum : f32)
  ^bytesBody:
    %byteWeights:4 = func.call @strict_device_byte_weights(%row, %count, %byteUnit) : (!llvm.ptr, i64, i64) -> (vector<4xi32>, f32, i64, i64)
    %byteNext = func.call @strict_device_byte_token(%byteWeights#0, %byteWeights#1, %byteWeights#2, %byteWeights#3, %bytes, %scales, %byteSum) : (vector<4xi32>, f32, i64, i64, !llvm.ptr, !llvm.ptr, f32) -> f32
    %byteFollowing = arith.addi %byteUnit, %lanes : i64
    cf.br ^bytes(%byteFollowing, %byteNext : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
  // a lane's part of a Q4_K block for dots8: 16 numbers of two parts with their scales and minimums, where its 8 bit
  // numbers start and their first scale
  func.func private @strict_device_four_weights(%row: !llvm.ptr, %unit: i64) -> (vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %nibbleMasks = llvm.mlir.constant(dense<252645135> : vector<4xi32>) : vector<4xi32>
    %fourShifts = llvm.mlir.constant(dense<4> : vector<4xi32>) : vector<4xi32>
    %fourBlockBytes = arith.constant 144 : i64
    %sixteenBytes = arith.constant 16 : i64
    %fourBlock = arith.shrui %unit, %three : i64
    %fourParts = arith.shrui %unit, %one : i64
    %fourPart = arith.andi %fourParts, %three : i64
    %fourHalf = arith.andi %unit, %one : i64
    %fourOffset = arith.muli %fourBlock, %fourBlockBytes : i64
    %fourStart = llvm.getelementptr %row[%fourOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %header = llvm.load %fourStart invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %partBytes = arith.shli %fourPart, %five : i64
    %halfBytes = arith.shli %fourHalf, %four : i64
    %inBlock = arith.addi %partBytes, %halfBytes : i64
    %nibbleAt = arith.addi %inBlock, %sixteenBytes : i64
    %nibbleSlot = llvm.getelementptr %fourStart[%nibbleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %nibbles = llvm.load %nibbleSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %blockFirst = arith.shli %fourBlock, %eight : i64
    %partFirst = arith.shli %fourPart, %six : i64
    %blockPart = arith.addi %blockFirst, %partFirst : i64
    %first = arith.addi %blockPart, %halfBytes : i64
    %scaleBlock = arith.shli %fourBlock, %three : i64
    %scalePart = arith.shli %fourPart, %one : i64
    %scaleIndex = arith.addi %scaleBlock, %scalePart : i64
    %headerWord = llvm.extractelement %header[%c0 : i32] : vector<4xi32>
    %halves = llvm.bitcast %headerWord : i32 to vector<2xf16>
    %halfScale = llvm.extractelement %halves[%c0 : i32] : vector<2xf16>
    %halfMinimum = llvm.extractelement %halves[%c1 : i32] : vector<2xf16>
    %blockScale = arith.extf %halfScale : f16 to f32
    %blockMinimum = arith.extf %halfMinimum : f16 to f32
    %part32 = arith.trunci %fourPart : i64 to i32
    %lowPart = arith.shli %part32, %c1 : i32
    %highPart = arith.addi %lowPart, %c1 : i32
    %lowScaleBits, %lowMinimumBits = func.call @strict_device_four_scale(%header, %lowPart) : (vector<4xi32>, i32) -> (i32, i32)
    %highScaleBits, %highMinimumBits = func.call @strict_device_four_scale(%header, %highPart) : (vector<4xi32>, i32) -> (i32, i32)
    %lowNibbles = arith.andi %nibbles, %nibbleMasks : vector<4xi32>
    %highShifted = arith.shrui %nibbles, %fourShifts : vector<4xi32>
    %highNibbles = arith.andi %highShifted, %nibbleMasks : vector<4xi32>
    %lowScaleFloat = arith.uitofp %lowScaleBits : i32 to f32
    %lowMinimumFloat = arith.uitofp %lowMinimumBits : i32 to f32
    %highScaleFloat = arith.uitofp %highScaleBits : i32 to f32
    %highMinimumFloat = arith.uitofp %highMinimumBits : i32 to f32
    return %lowNibbles, %highNibbles, %lowScaleFloat, %lowMinimumFloat, %highScaleFloat, %highMinimumFloat, %blockScale, %blockMinimum, %first, %scaleIndex : vector<4xi32>, vector<4xi32>, f32, f32, f32, f32, f32, f32, i64, i64
  }
  // the decoded Q4_K part of strict_device_four_weights times the 8 bit numbers of one token, added to sum
  func.func private @strict_device_four_token(%lowNibbles: vector<4xi32>, %highNibbles: vector<4xi32>, %lowScaleFloat: f32, %lowMinimumFloat: f32, %highScaleFloat: f32, %highMinimumFloat: f32, %blockScale: f32, %blockMinimum: f32, %first: i64, %scaleIndex: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %fourSum: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %onesVector = llvm.mlir.constant(dense<16843009> : vector<4xi32>) : vector<4xi32>
    %lowInputSlot = llvm.getelementptr %bytes[%first] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowInputs = llvm.load %lowInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %highInputSlot = llvm.getelementptr %lowInputSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %highInputs = llvm.load %highInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %lowScaleSlot = llvm.getelementptr %scales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lowInputScale = llvm.load %lowScaleSlot invariant : !llvm.ptr -> f32
    %highScaleSlot = llvm.getelementptr %lowScaleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %highInputScale = llvm.load %highScaleSlot invariant : !llvm.ptr -> f32
    %lowDot = func.call @strict_device_dp4as(%lowNibbles, %lowInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %highDot = func.call @strict_device_dp4as(%highNibbles, %highInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %lowTotal = func.call @strict_device_dp4as(%onesVector, %lowInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %highTotal = func.call @strict_device_dp4as(%onesVector, %highInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %lowDotFloat = arith.sitofp %lowDot : i32 to f32
    %highDotFloat = arith.sitofp %highDot : i32 to f32
    %lowTotalFloat = arith.sitofp %lowTotal : i32 to f32
    %highTotalFloat = arith.sitofp %highTotal : i32 to f32
    %lowWeighted = arith.mulf %lowScaleFloat, %lowDotFloat : f32
    %lowShifted = arith.mulf %lowMinimumFloat, %lowTotalFloat : f32
    %highWeighted = arith.mulf %highScaleFloat, %highDotFloat : f32
    %highShiftedTotal = arith.mulf %highMinimumFloat, %highTotalFloat : f32
    %lowScaled = arith.mulf %lowWeighted, %blockScale : f32
    %lowMinimum = arith.mulf %lowShifted, %blockMinimum : f32
    %lowPartSum = arith.subf %lowScaled, %lowMinimum : f32
    %highScaled = arith.mulf %highWeighted, %blockScale : f32
    %highMinimum = arith.mulf %highShiftedTotal, %blockMinimum : f32
    %highPartSum = arith.subf %highScaled, %highMinimum : f32
    %lowContribution = arith.mulf %lowPartSum, %lowInputScale : f32
    %fourAdded = arith.addf %fourSum, %lowContribution : f32
    %highContribution = arith.mulf %highPartSum, %highInputScale : f32
    %fourNext = arith.addf %fourAdded, %highContribution : f32
    return %fourNext : f32
  }
  // a lane's 16 numbers of a Q6_K block for dots8 (centered 6 bit numbers), their 4 scales, the block scale, where the
  // 8 bit numbers start and their first scale
  func.func private @strict_device_six_weights(%row: !llvm.ptr, %count: i64, %chunk: i64) -> (vector<4xi32>, f32, f32, f32, f32, f32, i64, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %fourBits = arith.constant 4 : i32
    %sixBlockBytes = arith.constant 208 : i64
    %fifteen = arith.constant 15 : i64
    %sixBlock = arith.shrui %chunk, %four : i64
    %sixWord = arith.andi %chunk, %fifteen : i64
    %sixHalf = arith.shrui %sixWord, %three : i64
    %sixInHalf = arith.andi %sixWord, %seven : i64
    %sixFirst = arith.shli %sixInHalf, %two : i64
    %sixOffset = arith.muli %sixBlock, %sixBlockBytes : i64
    %sixStart = llvm.getelementptr %row[%sixOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixLowHalf = arith.shli %sixHalf, %six : i64
    %sixLowAt = arith.addi %sixLowHalf, %sixFirst : i64
    %sixLowSlot = llvm.getelementptr %sixStart[%sixLowAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixLow = llvm.load %sixLowSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixSecondSlot = llvm.getelementptr %sixLowSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixSecond = llvm.load %sixSecondSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixHighHalf = arith.shli %sixHalf, %five : i64
    %sixHighStart = arith.constant 128 : i64
    %sixHighBase = arith.addi %sixHighStart, %sixHighHalf : i64
    %sixHighAt = arith.addi %sixHighBase, %sixFirst : i64
    %sixHighSlot = llvm.getelementptr %sixStart[%sixHighAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixHigh = llvm.load %sixHighSlot invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixScaleStart = arith.constant 192 : i64
    %sixScaleHalf = arith.shli %sixHalf, %three : i64
    %sixSixteenth = arith.shrui %sixInHalf, %two : i64
    %sixScaleBase = arith.addi %sixScaleStart, %sixScaleHalf : i64
    %sixScaleAt = arith.addi %sixScaleBase, %sixSixteenth : i64
    %sixScaleSlot0 = llvm.getelementptr %sixStart[%sixScaleAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixScale0 = llvm.load %sixScaleSlot0 invariant : !llvm.ptr -> i8
    %sixScaleSlot1 = llvm.getelementptr %sixScaleSlot0[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixScale1 = llvm.load %sixScaleSlot1 invariant : !llvm.ptr -> i8
    %sixScaleSlot2 = llvm.getelementptr %sixScaleSlot0[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixScale2 = llvm.load %sixScaleSlot2 invariant : !llvm.ptr -> i8
    %sixScaleSlot3 = llvm.getelementptr %sixScaleSlot0[6] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixScale3 = llvm.load %sixScaleSlot3 invariant : !llvm.ptr -> i8
    %sixDataBytes = func.call @strict_device_data(%count, %six) : (i64, i64) -> i64
    %sixHalves = llvm.getelementptr %row[%sixDataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixHalfSlot = llvm.getelementptr %sixHalves[%sixBlock] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %sixHalfScale = llvm.load %sixHalfSlot invariant : !llvm.ptr -> f16
    %sixBlockScale = arith.extf %sixHalfScale : f16 to f32
    %lowNibble = arith.constant 252645135 : i32
    %topPair = arith.constant 808464432 : i32
    %twoBits = arith.constant 2 : i32
    %sixFirstLow = arith.andi %sixLow, %lowNibble : i32
    %sixHighUp4 = arith.shli %sixHigh, %fourBits : i32
    %sixHighTop0 = arith.andi %sixHighUp4, %topPair : i32
    %sixQ0 = arith.ori %sixFirstLow, %sixHighTop0 : i32
    %sixSecondLow = arith.andi %sixSecond, %lowNibble : i32
    %sixHighUp2 = arith.shli %sixHigh, %twoBits : i32
    %sixHighTop1 = arith.andi %sixHighUp2, %topPair : i32
    %sixQ1 = arith.ori %sixSecondLow, %sixHighTop1 : i32
    %sixLowDown = arith.shrui %sixLow, %fourBits : i32
    %sixFirstHigh = arith.andi %sixLowDown, %lowNibble : i32
    %sixHighTop2 = arith.andi %sixHigh, %topPair : i32
    %sixQ2 = arith.ori %sixFirstHigh, %sixHighTop2 : i32
    %sixSecondDown = arith.shrui %sixSecond, %fourBits : i32
    %sixSecondHigh = arith.andi %sixSecondDown, %lowNibble : i32
    %sixHighDown2 = arith.shrui %sixHigh, %twoBits : i32
    %sixHighTop3 = arith.andi %sixHighDown2, %topPair : i32
    %sixQ3 = arith.ori %sixSecondHigh, %sixHighTop3 : i32
    %sixWords = func.call @strict_device_words(%sixQ0, %sixQ1, %sixQ2, %sixQ3) : (i32, i32, i32, i32) -> vector<4xi32>
    %signs = llvm.mlir.constant(dense<-2139062144> : vector<4xi32>) : vector<4xi32>
    %centers = llvm.mlir.constant(dense<538976288> : vector<4xi32>) : vector<4xi32>
    %sixMarked = arith.ori %sixWords, %signs : vector<4xi32>
    %sixLowered = arith.subi %sixMarked, %centers : vector<4xi32>
    %sixCentered = arith.xori %sixLowered, %signs : vector<4xi32>
    %sixBlockFirst = arith.shli %sixBlock, %eight : i64
    %sixHalfFirst = arith.shli %sixHalf, %seven : i64
    %sixPartStart = arith.addi %sixBlockFirst, %sixHalfFirst : i64
    %sixInputAt = arith.addi %sixPartStart, %sixFirst : i64
    %sixScaleIndex = arith.shrui %sixPartStart, %five : i64
    %sixScaleFloat0 = arith.sitofp %sixScale0 : i8 to f32
    %sixScaleFloat1 = arith.sitofp %sixScale1 : i8 to f32
    %sixScaleFloat2 = arith.sitofp %sixScale2 : i8 to f32
    %sixScaleFloat3 = arith.sitofp %sixScale3 : i8 to f32
    return %sixCentered, %sixScaleFloat0, %sixScaleFloat1, %sixScaleFloat2, %sixScaleFloat3, %sixBlockScale, %sixInputAt, %sixScaleIndex : vector<4xi32>, f32, f32, f32, f32, f32, i64, i64
  }
  // the decoded Q6_K numbers of strict_device_six_weights times the 8 bit numbers of one token, added to sum
  func.func private @strict_device_six_token(%sixCentered: vector<4xi32>, %sixScaleFloat0: f32, %sixScaleFloat1: f32, %sixScaleFloat2: f32, %sixScaleFloat3: f32, %sixBlockScale: f32, %sixInputAt: i64, %sixScaleIndex: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %sixSum: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %nothing = arith.constant 0 : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %sixInputSlot0 = llvm.getelementptr %bytes[%sixInputAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixInput0 = llvm.load %sixInputSlot0 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot1 = llvm.getelementptr %sixInputSlot0[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput1 = llvm.load %sixInputSlot1 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot2 = llvm.getelementptr %sixInputSlot0[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput2 = llvm.load %sixInputSlot2 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot3 = llvm.getelementptr %sixInputSlot0[96] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput3 = llvm.load %sixInputSlot3 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputScaleSlot0 = llvm.getelementptr %scales[%sixScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sixInputScale0 = llvm.load %sixInputScaleSlot0 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot1 = llvm.getelementptr %sixInputScaleSlot0[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale1 = llvm.load %sixInputScaleSlot1 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot2 = llvm.getelementptr %sixInputScaleSlot0[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale2 = llvm.load %sixInputScaleSlot2 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot3 = llvm.getelementptr %sixInputScaleSlot0[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale3 = llvm.load %sixInputScaleSlot3 invariant : !llvm.ptr -> f32
    %sixC0 = llvm.extractelement %sixCentered[%c0 : i32] : vector<4xi32>
    %sixC1 = llvm.extractelement %sixCentered[%c1 : i32] : vector<4xi32>
    %sixC2 = llvm.extractelement %sixCentered[%c2 : i32] : vector<4xi32>
    %sixC3 = llvm.extractelement %sixCentered[%c3 : i32] : vector<4xi32>
    %sixDot0 = func.call @strict_device_dp4a(%sixC0, %sixInput0, %nothing) : (i32, i32, i32) -> i32
    %sixDot1 = func.call @strict_device_dp4a(%sixC1, %sixInput1, %nothing) : (i32, i32, i32) -> i32
    %sixDot2 = func.call @strict_device_dp4a(%sixC2, %sixInput2, %nothing) : (i32, i32, i32) -> i32
    %sixDot3 = func.call @strict_device_dp4a(%sixC3, %sixInput3, %nothing) : (i32, i32, i32) -> i32
    %sixDotFloat0 = arith.sitofp %sixDot0 : i32 to f32
    %sixDotFloat1 = arith.sitofp %sixDot1 : i32 to f32
    %sixDotFloat2 = arith.sitofp %sixDot2 : i32 to f32
    %sixDotFloat3 = arith.sitofp %sixDot3 : i32 to f32
    %sixWeight0 = arith.mulf %sixScaleFloat0, %sixInputScale0 : f32
    %sixWeight1 = arith.mulf %sixScaleFloat1, %sixInputScale1 : f32
    %sixWeight2 = arith.mulf %sixScaleFloat2, %sixInputScale2 : f32
    %sixWeight3 = arith.mulf %sixScaleFloat3, %sixInputScale3 : f32
    %sixPart0 = arith.mulf %sixWeight0, %sixDotFloat0 : f32
    %sixPart1 = arith.mulf %sixWeight1, %sixDotFloat1 : f32
    %sixPart2 = arith.mulf %sixWeight2, %sixDotFloat2 : f32
    %sixPart3 = arith.mulf %sixWeight3, %sixDotFloat3 : f32
    %sixPairs0 = arith.addf %sixPart0, %sixPart1 : f32
    %sixPairs1 = arith.addf %sixPart2, %sixPart3 : f32
    %sixParts = arith.addf %sixPairs0, %sixPairs1 : f32
    %sixScaled = arith.mulf %sixParts, %sixBlockScale : f32
    %sixNext = arith.addf %sixSum, %sixScaled : f32
    return %sixNext : f32
  }
  // a lane's 16 numbers of an 8 bit row (Q8_0) for dots8 with their scale, where they start and their scale index
  func.func private @strict_device_byte_weights(%row: !llvm.ptr, %count: i64, %byteUnit: i64) -> (vector<4xi32>, f32, i64, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %byteFirst = arith.shli %byteUnit, %four : i64
    %rowSlot0 = llvm.getelementptr %row[%byteFirst] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowWord0 = llvm.load %rowSlot0 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %rowSlot1 = llvm.getelementptr %rowSlot0[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %rowWord1 = llvm.load %rowSlot1 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %rowSlot2 = llvm.getelementptr %rowSlot0[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %rowWord2 = llvm.load %rowSlot2 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %rowSlot3 = llvm.getelementptr %rowSlot0[12] : (!llvm.ptr) -> !llvm.ptr, i8
    %rowWord3 = llvm.load %rowSlot3 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %rowWords = func.call @strict_device_words(%rowWord0, %rowWord1, %rowWord2, %rowWord3) : (i32, i32, i32, i32) -> vector<4xi32>
    %rowDataBytes = func.call @strict_device_data(%count, %eight) : (i64, i64) -> i64
    %rowScales = llvm.getelementptr %row[%rowDataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byteScaleIndex = arith.shrui %byteFirst, %five : i64
    %rowScaleSlot = llvm.getelementptr %rowScales[%byteScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %rowScale = llvm.load %rowScaleSlot invariant : !llvm.ptr -> f32
    return %rowWords, %rowScale, %byteFirst, %byteScaleIndex : vector<4xi32>, f32, i64, i64
  }
  // the 8 bit row numbers of strict_device_byte_weights times the 8 bit numbers of one token, added to sum
  func.func private @strict_device_byte_token(%rowWords: vector<4xi32>, %rowScale: f32, %byteFirst: i64, %byteScaleIndex: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %byteSum: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %byteInputSlot = llvm.getelementptr %bytes[%byteFirst] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byteInputs = llvm.load %byteInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %byteDot = func.call @strict_device_dp4as(%rowWords, %byteInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %byteDotFloat = arith.sitofp %byteDot : i32 to f32
    %byteInputScaleSlot = llvm.getelementptr %scales[%byteScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %byteInputScale = llvm.load %byteInputScaleSlot invariant : !llvm.ptr -> f32
    %byteScale = arith.mulf %rowScale, %byteInputScale : f32
    %byteScaled = arith.mulf %byteDotFloat, %byteScale : f32
    %byteNext = arith.addf %byteSum, %byteScaled : f32
    return %byteNext : f32
  }
  // the 64 ternary weights of unit (half a PTQ1_0 block) of a 3 bit row as signed bytes for dp4a (the two lanes of a
  // block each decode all of it), its half scale, where its 8 bit numbers start and their first scale
  func.func private @strict_device_trits_weights(%row: !llvm.ptr, %count: i64, %unit: i64) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, f32, i64, i64) attributes {llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %blockBytes = arith.constant 28 : i64
    %block = arith.shrui %unit, %one : i64
    %halfBit = arith.andi %unit, %one : i64
    %isLater = arith.cmpi eq, %halfBit, %one : i64
    %blockStart = arith.muli %block, %blockBytes : i64
    %blockData = llvm.getelementptr %row[%blockStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %all, %scale = func.call @strict_device_trits_block(%blockData) : (!llvm.ptr) -> (vector<32xi32>, f32)
    %earlier = llvm.shufflevector %all, %all [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi32>
    %later = llvm.shufflevector %all, %all [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi32>
    %half = arith.select %isLater, %later, %earlier : vector<16xi32>
    %weights0 = llvm.shufflevector %half, %half [0, 1, 2, 3] : vector<16xi32>
    %weights1 = llvm.shufflevector %half, %half [4, 5, 6, 7] : vector<16xi32>
    %weights2 = llvm.shufflevector %half, %half [8, 9, 10, 11] : vector<16xi32>
    %weights3 = llvm.shufflevector %half, %half [12, 13, 14, 15] : vector<16xi32>
    %first = arith.shli %unit, %six : i64
    %scaleIndex = arith.shrui %first, %five : i64
    return %weights0, %weights1, %weights2, %weights3, %scale, %first, %scaleIndex : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, f32, i64, i64
  }
  // the ternary weights of strict_device_trits_weights times the 8 bit numbers of one token (two scales of 32), added to sum
  func.func private @strict_device_trits_token(%weights0: vector<4xi32>, %weights1: vector<4xi32>, %weights2: vector<4xi32>, %weights3: vector<4xi32>, %scale: f32, %first: i64, %scaleIndex: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %sum: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %inputSlot0 = llvm.getelementptr %bytes[%first] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %inputs0 = llvm.load %inputSlot0 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %inputSlot1 = llvm.getelementptr %inputSlot0[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %inputs1 = llvm.load %inputSlot1 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %inputSlot2 = llvm.getelementptr %inputSlot0[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %inputs2 = llvm.load %inputSlot2 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %inputSlot3 = llvm.getelementptr %inputSlot0[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %inputs3 = llvm.load %inputSlot3 invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %lowScaleSlot = llvm.getelementptr %scales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lowInputScale = llvm.load %lowScaleSlot invariant : !llvm.ptr -> f32
    %highScaleSlot = llvm.getelementptr %lowScaleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %highInputScale = llvm.load %highScaleSlot invariant : !llvm.ptr -> f32
    %dot0 = func.call @strict_device_dp4as(%weights0, %inputs0) : (vector<4xi32>, vector<4xi32>) -> i32
    %dot1 = func.call @strict_device_dp4as(%weights1, %inputs1) : (vector<4xi32>, vector<4xi32>) -> i32
    %dot2 = func.call @strict_device_dp4as(%weights2, %inputs2) : (vector<4xi32>, vector<4xi32>) -> i32
    %dot3 = func.call @strict_device_dp4as(%weights3, %inputs3) : (vector<4xi32>, vector<4xi32>) -> i32
    %lowDot = arith.addi %dot0, %dot1 : i32
    %highDot = arith.addi %dot2, %dot3 : i32
    %lowDotFloat = arith.sitofp %lowDot : i32 to f32
    %highDotFloat = arith.sitofp %highDot : i32 to f32
    %lowPart = arith.mulf %lowDotFloat, %lowInputScale : f32
    %highPart = arith.mulf %highDotFloat, %highInputScale : f32
    %parts = arith.addf %lowPart, %highPart : f32
    %scaled = arith.mulf %parts, %scale : f32
    %next = arith.addf %sum, %scaled : f32
    return %next : f32
  }
  // dots8 needs whole blocks of the row (256 numbers for 4 and 6 bits, 128 for 3, 32 for 8) and 16 byte aligned
  // vector loads (3 bit rows are read in words)
  func.func private @strict_device_fits8(%row: !llvm.ptr, %rowBits: i64, %count: i64, %bytes: !llvm.ptr) -> i1 {
    %none = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %thirtyOne = arith.constant 31 : i64
    %blockMask = arith.constant 255 : i64
    %true = arith.constant true
    %groupMask = arith.constant 127 : i64
    %three = arith.constant 3 : i64
    %isThree = arith.cmpi eq, %rowBits, %three : i64
    %isFour = arith.cmpi eq, %rowBits, %four : i64
    %isSix = arith.cmpi eq, %rowBits, %six : i64
    %isEight = arith.cmpi eq, %rowBits, %eight : i64
    %isBlocked = arith.ori %isFour, %isSix : i1
    %blockRest = arith.andi %count, %blockMask : i64
    %wholeBlocks = arith.cmpi eq, %blockRest, %none : i64
    %wordRest = arith.andi %count, %thirtyOne : i64
    %wholeWords = arith.cmpi eq, %wordRest, %none : i64
    %groupRest = arith.andi %count, %groupMask : i64
    %wholeGroups = arith.cmpi eq, %groupRest, %none : i64
    %blockedFits = arith.andi %isBlocked, %wholeBlocks : i1
    %eightFits = arith.andi %isEight, %wholeWords : i1
    %threeFits = arith.andi %isThree, %wholeGroups : i1
    %wordFits = arith.ori %blockedFits, %eightFits : i1
    %sizeFits = arith.ori %wordFits, %threeFits : i1
    %rowAddress = llvm.ptrtoint %row : !llvm.ptr to i64
    %rowRest = arith.andi %rowAddress, %fifteen : i64
    %rowAligned = arith.cmpi eq, %rowRest, %none : i64
    %notVectorRow = arith.xori %isFour, %true : i1
    %rowFits = arith.ori %notVectorRow, %rowAligned : i1
    %bytesAddress = llvm.ptrtoint %bytes : !llvm.ptr to i64
    %bytesRest = arith.andi %bytesAddress, %fifteen : i64
    %bytesAligned = arith.cmpi eq, %bytesRest, %none : i64
    %placed = arith.andi %rowFits, %bytesAligned : i1
    %fits = arith.andi %sizeFits, %placed : i1
    return %fits : i1
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
    %eightBits = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %yQuantized = arith.cmpi eq, %yBits, %eightBits : i64
    %yStartRest = arith.andi %yStart, %thirtyOne : i64
    %yBlockAligned = arith.cmpi eq, %yStartRest, %none : i64
    %xFromZero = arith.cmpi eq, %xStart, %none : i64
    %xToCount = arith.cmpi eq, %count, %xCount : i64
    %xComplete = arith.andi %xFromZero, %xToCount : i1
    %yAlignedQuantized = arith.andi %yQuantized, %yBlockAligned : i1
    %quantizable = arith.andi %yAlignedQuantized, %xComplete : i1
    %float8Code = arith.constant 7 : i64
    %yFloat8 = arith.cmpi eq, %yBits, %float8Code : i64
    %xSingles = arith.cmpi eq, %xBits, %thirtyTwo : i64
    %singleEights = arith.andi %xSingles, %yFloat8 : i1
    cf.cond_br %singleEights, ^float8s, ^scaled
  ^float8s:
    %float8Dot = func.call @strict_device_float8_dots(%x, %xStart, %y, %yStart, %count, %lane, %lanes) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
    return %float8Dot : f32
  ^scaled:
    cf.cond_br %quantizable, ^quantized, ^plain
  ^quantized:
    %yBytes = llvm.getelementptr %y[%yStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %eightFits = func.call @strict_device_fits8(%x, %xBits, %count, %yBytes) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %eightFits, ^eights, ^generic(%lane, %zero : i64, f32)
  ^eights:
    %yData = func.call @strict_device_data(%yCount, %eightBits) : (i64, i64) -> i64
    %yScales = llvm.getelementptr %y[%yData] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %five = arith.constant 5 : i64
    %yScaleStart = arith.shrui %yStart, %five : i64
    %yFirstScales = llvm.getelementptr %yScales[%yScaleStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %eightDot = func.call @strict_device_dots8(%x, %xBits, %count, %yBytes, %yFirstScales, %lane, %lanes) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> f32
    return %eightDot : f32
  ^plain:
    %xFloats = arith.cmpi eq, %xBits, %thirtyTwo : i64
    %yFloats = arith.cmpi eq, %yBits, %thirtyTwo : i64
    %xFirst = llvm.getelementptr %x[%xStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %yFirst = llvm.getelementptr %y[%yStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %xAddress = llvm.ptrtoint %xFirst : !llvm.ptr to i64
    %yAddress = llvm.ptrtoint %yFirst : !llvm.ptr to i64
    %xMisaligned = arith.andi %xAddress, %fifteen : i64
    %yMisaligned = arith.andi %yAddress, %fifteen : i64
    %xAligned = arith.cmpi eq, %xMisaligned, %none : i64
    %yAligned = arith.cmpi eq, %yMisaligned, %none : i64
    %xFromStart = arith.cmpi eq, %xStart, %none : i64
    %xToEnd = arith.cmpi eq, %count, %xCount : i64
    %xWhole = arith.andi %xFromStart, %xToEnd : i1
    %xAlignedFloats = arith.andi %xFloats, %xAligned : i1
    %xReady = arith.ori %xAlignedFloats, %xWhole : i1
    %yReady = arith.andi %yFloats, %yAligned : i1
    %rest = arith.andi %count, %seven : i64
    %whole = arith.cmpi eq, %rest, %none : i64
    %ready = arith.andi %xReady, %yReady : i1
    %fast = arith.andi %ready, %whole : i1
    %xFirstBits = arith.select %xWhole, %xBits, %thirtyTwo : i64
    %xDots = llvm.select %xWhole, %x, %xFirst : i1, !llvm.ptr
    cf.cond_br %fast, ^dots, ^generic(%lane, %zero : i64, f32)
  ^dots:
    %dot = func.call @strict_device_dots(%failed, %xDots, %xFirstBits, %count, %yFirst, %thirtyTwo, %count, %lane, %lanes) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> f32
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
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
  // the 4 floats or bfloat16 numbers at at (16 or 8 byte aligned) as floats, and sum plus their squares in order
  func.func private @strict_device_four_singles(%data: !llvm.ptr, %at: i64) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %four = llvm.load %slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %four : vector<4xf32>
  }
  func.func private @strict_device_four_brains(%data: !llvm.ptr, %at: i64) -> vector<4xf32> attributes {llvm.passthrough = ["alwaysinline"]} {
    %slot = llvm.getelementptr %data[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %slot {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %four = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    return %four : vector<4xf32>
  }
  func.func private @strict_device_four_squared(%four: vector<4xf32>, %sum: f32) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %squared = arith.mulf %four, %four fastmath<contract> : vector<4xf32>
    %s0 = llvm.extractelement %squared[%c0 : i32] : vector<4xf32>
    %s1 = llvm.extractelement %squared[%c1 : i32] : vector<4xf32>
    %s2 = llvm.extractelement %squared[%c2 : i32] : vector<4xf32>
    %s3 = llvm.extractelement %squared[%c3 : i32] : vector<4xf32>
    %added0 = arith.addf %sum, %s0 fastmath<contract> : f32
    %added1 = arith.addf %added0, %s1 fastmath<contract> : f32
    %added2 = arith.addf %added1, %s2 fastmath<contract> : f32
    %added3 = arith.addf %added2, %s3 fastmath<contract> : f32
    return %added3 : f32
  }
  // the sum of squares of count numbers of one list from its start: each number is read once, 32 bit floats four at a
  // time when the start is 16 byte aligned, four of those loads at once (adding in the same order) while they fit
  func.func private @strict_device_squares(%failed: !llvm.ptr, %data: !llvm.ptr, %bits: i64, %dataCount: i64, %count: i64, %start: i64, %lane: i64, %lanes: i64) -> f32 {
    %zero = arith.constant 0.0 : f32
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %end = arith.addi %start, %count : i64
    %low = arith.cmpi slt, %start, %none : i64
    %high = arith.cmpi sgt, %end, %dataCount : i64
    %outside = arith.ori %low, %high : i1
    cf.cond_br %outside, ^fail, ^check
  ^fail:
    %failure = arith.constant 1 : i64
    llvm.store %failure, %failed : i64, !llvm.ptr
    return %zero : f32
  ^check:
    %brainBits = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    %isBrain = arith.cmpi eq, %bits, %brainBits : i64
    %isVectorBits = arith.ori %isSingle, %isBrain : i1
    %rest = arith.andi %count, %three : i64
    %whole = arith.cmpi eq, %rest, %none : i64
    %startRest = arith.andi %start, %three : i64
    %startAligned = arith.cmpi eq, %startRest, %none : i64
    %vectors = arith.andi %isVectorBits, %whole : i1
    %fours = arith.andi %vectors, %startAligned : i1
    cf.cond_br %fours, ^fourStart, ^genericStart
  ^fourStart:
    %fourFirst = arith.shli %lane, %two : i64
    %fourStep = arith.shli %lanes, %two : i64
    %quadSpan = arith.muli %fourStep, %three : i64
    %quadStride = arith.shli %fourStep, %two : i64
    cf.cond_br %isBrain, ^brainQuads(%fourFirst, %zero : i64, f32), ^singleQuads(%fourFirst, %zero : i64, f32)
  ^singleQuads(%singleIndex: i64, %singleSum: f32):
    %singleLast = arith.addi %singleIndex, %quadSpan : i64
    %singleMore = arith.cmpi slt, %singleLast, %count : i64
    cf.cond_br %singleMore, ^singleQuad, ^fours(%singleIndex, %singleSum : i64, f32)
  ^singleQuad:
    %singleAt0 = arith.addi %start, %singleIndex : i64
    %singleAt1 = arith.addi %singleAt0, %fourStep : i64
    %singleAt2 = arith.addi %singleAt1, %fourStep : i64
    %singleAt3 = arith.addi %singleAt2, %fourStep : i64
    %single0 = func.call @strict_device_four_singles(%data, %singleAt0) : (!llvm.ptr, i64) -> vector<4xf32>
    %single1 = func.call @strict_device_four_singles(%data, %singleAt1) : (!llvm.ptr, i64) -> vector<4xf32>
    %single2 = func.call @strict_device_four_singles(%data, %singleAt2) : (!llvm.ptr, i64) -> vector<4xf32>
    %single3 = func.call @strict_device_four_singles(%data, %singleAt3) : (!llvm.ptr, i64) -> vector<4xf32>
    %singleSum1 = func.call @strict_device_four_squared(%single0, %singleSum) : (vector<4xf32>, f32) -> f32
    %singleSum2 = func.call @strict_device_four_squared(%single1, %singleSum1) : (vector<4xf32>, f32) -> f32
    %singleSum3 = func.call @strict_device_four_squared(%single2, %singleSum2) : (vector<4xf32>, f32) -> f32
    %singleSum4 = func.call @strict_device_four_squared(%single3, %singleSum3) : (vector<4xf32>, f32) -> f32
    %singleNext = arith.addi %singleIndex, %quadStride : i64
    cf.br ^singleQuads(%singleNext, %singleSum4 : i64, f32)
  ^brainQuads(%brainIndex: i64, %brainSum: f32):
    %brainLast = arith.addi %brainIndex, %quadSpan : i64
    %brainMore = arith.cmpi slt, %brainLast, %count : i64
    cf.cond_br %brainMore, ^brainQuad, ^fours(%brainIndex, %brainSum : i64, f32)
  ^brainQuad:
    %brainAt0 = arith.addi %start, %brainIndex : i64
    %brainAt1 = arith.addi %brainAt0, %fourStep : i64
    %brainAt2 = arith.addi %brainAt1, %fourStep : i64
    %brainAt3 = arith.addi %brainAt2, %fourStep : i64
    %brain0 = func.call @strict_device_four_brains(%data, %brainAt0) : (!llvm.ptr, i64) -> vector<4xf32>
    %brain1 = func.call @strict_device_four_brains(%data, %brainAt1) : (!llvm.ptr, i64) -> vector<4xf32>
    %brain2 = func.call @strict_device_four_brains(%data, %brainAt2) : (!llvm.ptr, i64) -> vector<4xf32>
    %brain3 = func.call @strict_device_four_brains(%data, %brainAt3) : (!llvm.ptr, i64) -> vector<4xf32>
    %brainSum1 = func.call @strict_device_four_squared(%brain0, %brainSum) : (vector<4xf32>, f32) -> f32
    %brainSum2 = func.call @strict_device_four_squared(%brain1, %brainSum1) : (vector<4xf32>, f32) -> f32
    %brainSum3 = func.call @strict_device_four_squared(%brain2, %brainSum2) : (vector<4xf32>, f32) -> f32
    %brainSum4 = func.call @strict_device_four_squared(%brain3, %brainSum3) : (vector<4xf32>, f32) -> f32
    %brainNext = arith.addi %brainIndex, %quadStride : i64
    cf.br ^brainQuads(%brainNext, %brainSum4 : i64, f32)
  ^fours(%fourIndex: i64, %fourSum: f32):
    %fourMore = arith.cmpi slt, %fourIndex, %count : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
    %fourAt = arith.addi %start, %fourIndex : i64
    cf.cond_br %isBrain, ^brainFour, ^singleFour
  ^brainFour:
    %fromBrains = func.call @strict_device_four_brains(%data, %fourAt) : (!llvm.ptr, i64) -> vector<4xf32>
    cf.br ^fourLoaded(%fromBrains : vector<4xf32>)
  ^singleFour:
    %fromSingles = func.call @strict_device_four_singles(%data, %fourAt) : (!llvm.ptr, i64) -> vector<4xf32>
    cf.br ^fourLoaded(%fromSingles : vector<4xf32>)
  ^fourLoaded(%four: vector<4xf32>):
    %added = func.call @strict_device_four_squared(%four, %fourSum) : (vector<4xf32>, f32) -> f32
    %fourFollowing = arith.addi %fourIndex, %fourStep : i64
    cf.br ^fours(%fourFollowing, %added : i64, f32)
  ^genericStart:
    cf.br ^generic(%lane, %zero : i64, f32)
  ^generic(%index: i64, %sum: f32):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^genericBody, ^reduce(%sum : f32)
  ^genericBody:
    %at = arith.addi %start, %index : i64
    %value = func.call @strict_device_elements(%data, %bits, %at, %dataCount) : (!llvm.ptr, i64, i64, i64) -> f32
    %product = arith.mulf %value, %value fastmath<contract> : f32
    %next = arith.addf %sum, %product fastmath<contract> : f32
    %following = arith.addi %index, %lanes : i64
    cf.br ^generic(%following, %next : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
