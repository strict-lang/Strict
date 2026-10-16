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
  // a row of a matrix, its width and the number of rows, behind the header of strict_gpu_rows (row stride and row
  // count in the first word, one width byte per row)
  func.func private @strict_device_row(%source: !llvm.ptr, %at: i64) -> (!llvm.ptr, i64, i64) {
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %alignMask = arith.constant -16 : i64
    %lowHalf = arith.constant 4294967295 : i64
    %thirtyTwo = arith.constant 32 : i64
    %word = llvm.load %source : !llvm.ptr -> i64
    %stride = arith.andi %word, %lowHalf : i64
    %count = arith.shrui %word, %thirtyTwo : i64
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
    return %row, %bits, %count : !llvm.ptr, i64, i64
  }
  // the value of the lane whose number differs in the bit of offset, a step of a butterfly reduction over a warp
  func.func private @strict_device_shuffle(%value: f64, %offset: i32) -> f64 {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %thirtyTwo = arith.constant 32 : i64
    %bits = arith.bitcast %value : f64 to i64
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
    return %other : f64
  }
  func.func private @strict_device_shuffles(%value: f32, %offset: i32) -> f32 {
    %all = arith.constant -1 : i32
    %clamp = arith.constant 31 : i32
    %bits = arith.bitcast %value : f32 to i32
    %otherBits = nvvm.shfl.sync bfly %all, %bits, %offset, %clamp : i32 -> i32
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
    %wide = arith.extf %warped : f32 to f64
    %blocked = func.call @strict_device_block_sum(%wide, %lane, %lanes) : (f64, i64, i64) -> f64
    %narrow = arith.truncf %blocked : f64 to f32
    cf.br ^done(%narrow : f32)
  ^done(%sum: f32):
    return %sum : f32
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
    nvvm.barrier0
    %warps = arith.shrui %lanes, %five : i64
    %isPartial = arith.cmpi slt, %inWarp, %warps : i64
    %partialSlot = llvm.getelementptr %partials[0, %inWarp] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<32 x f64>
    %partial = llvm.load %partialSlot : !llvm.ptr<3> -> f64
    %mine = arith.select %isPartial, %partial, %zero : f64
    %total = func.call @strict_device_sum(%mine) : (f64) -> f64
    nvvm.barrier0
    return %total : f64
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
  // 32 bit numbers to 8 bit numbers with a scale per 32 (a warp shares the largest magnitude), the numbers that dot
  // products with packed rows read through dp4a
  llvm.func @strict_device_quantize(%source: !llvm.ptr, %count: i64, %target: !llvm.ptr) attributes {nvvm.kernel} {
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    %width = nvvm.read.ptx.sreg.ntid.x : i32
    %start = arith.muli %block, %width : i32
    %sum = arith.addi %start, %thread : i32
    %at = arith.extui %sum : i32 to i64
    %inside = arith.cmpi slt, %at, %count : i64
    cf.cond_br %inside, ^body, ^exit
  ^body:
    %slot = llvm.getelementptr %source[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %slot : !llvm.ptr -> f32
    %magnitude = llvm.call_intrinsic "llvm.fabs.f32"(%value) : (f32) -> f32
    %largest = func.call @strict_device_largest(%magnitude) : (f32) -> f32
    %most = arith.constant 127.0 : f32
    %zero = arith.constant 0.0 : f32
    %scale = arith.divf %largest, %most : f32
    %isEmpty = arith.cmpf oeq, %largest, %zero : f32
    %inverse = arith.divf %most, %largest : f32
    %factor = arith.select %isEmpty, %zero, %inverse : f32
    %scaled = arith.mulf %value, %factor : f32
    %rounded = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%scaled) : (f32) -> i32
    %byte = arith.trunci %rounded : i32 to i8
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
    llvm.return
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
  func.func private @strict_device_dp4a(%left: i32, %right: i32, %sum: i32) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %result = llvm.call_intrinsic "llvm.nvvm.idp4a.s.s"(%left, %right, %sum) : (i32, i32, i32) -> i32
    return %result : i32
  }
  func.func private @strict_device_words(%first: i32, %second: i32, %third: i32, %fourth: i32) -> vector<4xi32> attributes {passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_dp4as(%left: vector<4xi32>, %right: vector<4xi32>) -> i32 attributes {passthrough = ["alwaysinline"]} {
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
  func.func private @strict_device_four_scale(%header: vector<4xi32>, %j: i32) -> (i32, i32) attributes {passthrough = ["alwaysinline"]} {
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
  // the dot product of a packed row (Q4_K, Q6_K or 8 bits) with 8 bit numbers (bytes, then a scale per 32), the whole
  // warp gets the sum
  func.func private @strict_device_dots8(%row: !llvm.ptr, %rowBits: i64, %count: i64, %bytes: !llvm.ptr, %scales: !llvm.ptr, %lane: i64, %lanes: i64) -> f32 attributes {passthrough = ["noinline"]} {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %nothing = arith.constant 0 : i32
    %onesVector = llvm.mlir.constant(dense<16843009> : vector<4xi32>) : vector<4xi32>
    %nibbleMasks = llvm.mlir.constant(dense<252645135> : vector<4xi32>) : vector<4xi32>
    %fourBits = arith.constant 4 : i32
    %fourShifts = llvm.mlir.constant(dense<4> : vector<4xi32>) : vector<4xi32>
    cf.switch %rowBits : i64, [
      default: ^bytes(%lane, %zero : i64, f32),
      4: ^four(%lane, %zero : i64, f32),
      6: ^six(%lane, %zero : i64, f32)
    ]
  ^four(%unit: i64, %fourSum: f32):
    %units = arith.shrui %count, %five : i64
    %fourMore = arith.cmpi slt, %unit, %units : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
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
    %lowInputSlot = llvm.getelementptr %bytes[%first] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowInputs = llvm.load %lowInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %highInputSlot = llvm.getelementptr %lowInputSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %highInputs = llvm.load %highInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %scaleBlock = arith.shli %fourBlock, %three : i64
    %scalePart = arith.shli %fourPart, %one : i64
    %scaleIndex = arith.addi %scaleBlock, %scalePart : i64
    %lowScaleSlot = llvm.getelementptr %scales[%scaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lowInputScale = llvm.load %lowScaleSlot invariant : !llvm.ptr -> f32
    %highScaleSlot = llvm.getelementptr %lowScaleSlot[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %highInputScale = llvm.load %highScaleSlot invariant : !llvm.ptr -> f32
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
    %lowDot = func.call @strict_device_dp4as(%lowNibbles, %lowInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %highDot = func.call @strict_device_dp4as(%highNibbles, %highInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %lowTotal = func.call @strict_device_dp4as(%onesVector, %lowInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %highTotal = func.call @strict_device_dp4as(%onesVector, %highInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %lowDotFloat = arith.sitofp %lowDot : i32 to f32
    %highDotFloat = arith.sitofp %highDot : i32 to f32
    %lowTotalFloat = arith.sitofp %lowTotal : i32 to f32
    %highTotalFloat = arith.sitofp %highTotal : i32 to f32
    %lowScaleFloat = arith.uitofp %lowScaleBits : i32 to f32
    %lowMinimumFloat = arith.uitofp %lowMinimumBits : i32 to f32
    %highScaleFloat = arith.uitofp %highScaleBits : i32 to f32
    %highMinimumFloat = arith.uitofp %highMinimumBits : i32 to f32
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
    %fourFollowing = arith.addi %unit, %lanes : i64
    cf.br ^four(%fourFollowing, %fourNext : i64, f32)
  ^six(%chunk: i64, %sixSum: f32):
    %chunks = arith.shrui %count, %four : i64
    %sixMore = arith.cmpi slt, %chunk, %chunks : i64
    cf.cond_br %sixMore, ^sixBody, ^reduce(%sixSum : f32)
  ^sixBody:
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
    %sixInputSlot0 = llvm.getelementptr %bytes[%sixInputAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sixInput0 = llvm.load %sixInputSlot0 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot1 = llvm.getelementptr %sixInputSlot0[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput1 = llvm.load %sixInputSlot1 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot2 = llvm.getelementptr %sixInputSlot0[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput2 = llvm.load %sixInputSlot2 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixInputSlot3 = llvm.getelementptr %sixInputSlot0[96] : (!llvm.ptr) -> !llvm.ptr, i8
    %sixInput3 = llvm.load %sixInputSlot3 invariant {alignment = 4 : i64} : !llvm.ptr -> i32
    %sixScaleIndex = arith.shrui %sixPartStart, %five : i64
    %sixInputScaleSlot0 = llvm.getelementptr %scales[%sixScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sixInputScale0 = llvm.load %sixInputScaleSlot0 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot1 = llvm.getelementptr %sixInputScaleSlot0[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale1 = llvm.load %sixInputScaleSlot1 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot2 = llvm.getelementptr %sixInputScaleSlot0[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale2 = llvm.load %sixInputScaleSlot2 invariant : !llvm.ptr -> f32
    %sixInputScaleSlot3 = llvm.getelementptr %sixInputScaleSlot0[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %sixInputScale3 = llvm.load %sixInputScaleSlot3 invariant : !llvm.ptr -> f32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
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
    %sixScaleFloat0 = arith.sitofp %sixScale0 : i8 to f32
    %sixScaleFloat1 = arith.sitofp %sixScale1 : i8 to f32
    %sixScaleFloat2 = arith.sitofp %sixScale2 : i8 to f32
    %sixScaleFloat3 = arith.sitofp %sixScale3 : i8 to f32
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
    %sixFollowing = arith.addi %chunk, %lanes : i64
    cf.br ^six(%sixFollowing, %sixNext : i64, f32)
  ^bytes(%byteUnit: i64, %byteSum: f32):
    %byteUnits = arith.shrui %count, %four : i64
    %byteMore = arith.cmpi slt, %byteUnit, %byteUnits : i64
    cf.cond_br %byteMore, ^bytesBody, ^reduce(%byteSum : f32)
  ^bytesBody:
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
    %byteInputSlot = llvm.getelementptr %bytes[%byteFirst] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byteInputs = llvm.load %byteInputSlot invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xi32>
    %byteDot = func.call @strict_device_dp4as(%rowWords, %byteInputs) : (vector<4xi32>, vector<4xi32>) -> i32
    %byteDotFloat = arith.sitofp %byteDot : i32 to f32
    %rowDataBytes = func.call @strict_device_data(%count, %eight) : (i64, i64) -> i64
    %rowScales = llvm.getelementptr %row[%rowDataBytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byteScaleIndex = arith.shrui %byteFirst, %five : i64
    %rowScaleSlot = llvm.getelementptr %rowScales[%byteScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %rowScale = llvm.load %rowScaleSlot invariant : !llvm.ptr -> f32
    %byteInputScaleSlot = llvm.getelementptr %scales[%byteScaleIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %byteInputScale = llvm.load %byteInputScaleSlot invariant : !llvm.ptr -> f32
    %byteScale = arith.mulf %rowScale, %byteInputScale : f32
    %byteScaled = arith.mulf %byteDotFloat, %byteScale : f32
    %byteNext = arith.addf %byteSum, %byteScaled : f32
    %byteFollowing = arith.addi %byteUnit, %lanes : i64
    cf.br ^bytes(%byteFollowing, %byteNext : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
  // dots8 needs whole blocks of the row (256 numbers for 4 and 6 bits, 32 for 8) and 16 byte aligned vector loads
  func.func private @strict_device_fits8(%row: !llvm.ptr, %rowBits: i64, %count: i64, %bytes: !llvm.ptr) -> i1 {
    %none = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %fifteen = arith.constant 15 : i64
    %thirtyOne = arith.constant 31 : i64
    %blockMask = arith.constant 255 : i64
    %true = arith.constant true
    %isFour = arith.cmpi eq, %rowBits, %four : i64
    %isSix = arith.cmpi eq, %rowBits, %six : i64
    %isEight = arith.cmpi eq, %rowBits, %eight : i64
    %isBlocked = arith.ori %isFour, %isSix : i1
    %blockRest = arith.andi %count, %blockMask : i64
    %wholeBlocks = arith.cmpi eq, %blockRest, %none : i64
    %wordRest = arith.andi %count, %thirtyOne : i64
    %wholeWords = arith.cmpi eq, %wordRest, %none : i64
    %blockedFits = arith.andi %isBlocked, %wholeBlocks : i1
    %eightFits = arith.andi %isEight, %wholeWords : i1
    %sizeFits = arith.ori %blockedFits, %eightFits : i1
    %rowAddress = llvm.ptrtoint %row : !llvm.ptr to i64
    %rowRest = arith.andi %rowAddress, %fifteen : i64
    %rowAligned = arith.cmpi eq, %rowRest, %none : i64
    %notFour = arith.xori %isFour, %true : i1
    %rowFits = arith.ori %notFour, %rowAligned : i1
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
