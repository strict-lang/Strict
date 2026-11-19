  // `RopeRotated.Rotated`: the RMS normalization of every head of 128 or 64 numbers and its rotation in one pass like
  // strict_device_head_rotated, for a rotation that pairs number i with i + half below 2 half and leaves the numbers
  // after them to themselves (H3: half 48 of 128, its table has cos 1 and sin 0 there, its decoder 24 of 64), into
  // heads of wide numbers with zeros after the rotated ones (the decoder pads its heads to 128). The lanes of a token
  // (a block of 256 above 1024 numbers of the result, else a warp, strict_gpu_group_lanes) read their 4 weights and
  // rotations once, then every warp takes every head (or pair of heads of 64) of its token that is its number modulo
  // the warps: of the 4 numbers of a lane the first half / 4 lanes of a head hold the first numbers of the pairs, the
  // lanes half of the head after them their partners (one shuffle brings them), the other lanes of both sides the
  // unpaired numbers. The lists are 32 bit floats, bfloat16 or doubles, strict_gpu_roperotated checked their lengths
  // and formats, so no thread reads outside of them. No thread converts a double (64 bit floats are slow on most
  // cards): the whole numbers come once more as integers after the result bits (strict_gpu_exacts), the epsilon as
  // the bits of its 32 bit float
  llvm.func @strict_device_rope_rotated(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %combined: !llvm.ptr, %combinedCount: i64, %combinedBits: i64, %startNumber: f64, %widthNumber: f64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %rotation: !llvm.ptr, %rotationCount: i64, %rotationBits: i64, %halfNumber: f64, %wideNumber: f64, %epsilonNumber: f64, %epsilonBitsNumber: f64, %resultBits: i64, %start: i64, %startMagic: i64, %width: i64, %widthMagic: i64, %half: i64, %halfMagic: i64, %wide: i64, %wideMagic: i64, %epsilonExact: i64, %epsilonMagic: i64, %epsilonBits: i64, %epsilonBitsMagic: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c31 = arith.constant 31 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %seventeen = arith.constant 17 : i64
    %bigHead = arith.constant 128 : i64
    %warpLimit = arith.constant 1024 : i64
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %tid = func.call @strict_device_thread_index() : () -> i32
    %bid = func.call @strict_device_block_index() : () -> i32
    %bsize = func.call @strict_device_block_size() : () -> i32
    %blockBase = arith.muli %bid, %bsize : i32
    %thread = arith.addi %blockBase, %tid : i32
    %size = arith.addi %weightsCount, %none : i64
    %isBig = arith.cmpi eq, %size, %bigHead : i64
    %sizeShift = arith.select %isBig, %seven, %six : i64
    %heads = arith.shrui %width, %sizeShift : i64
    %outWidth = arith.muli %heads, %wide : i64
    %isWide = arith.cmpi sgt, %outWidth, %warpLimit : i64
    %tokenShift = arith.select %isWide, %c8, %c5 : i32
    %token32 = arith.shrui %thread, %tokenShift : i32
    %token = arith.extui %token32 : i32 to i64
    %outToken = arith.muli %token, %outWidth : i64
    %inBounds = arith.cmpi slt, %outToken, %count : i64
    cf.cond_br %inBounds, ^work, ^exit
  ^work:
    %tokenFirst = arith.shli %token32, %tokenShift : i32
    %inToken = arith.subi %thread, %tokenFirst : i32
    %warp32 = arith.shrui %inToken, %c5 : i32
    %warps32 = arith.select %isWide, %c8, %c1 : i32
    %lane32 = arith.andi %thread, %c31 : i32
    %lane = arith.extui %lane32 : i32 to i64
    %heads32 = arith.trunci %heads : i64 to i32
    %headLanes = arith.shrui %size, %two : i64
    %headLaneMask = arith.subi %headLanes, %one : i64
    %sideLanes = arith.shrui %size, %three : i64
    %sideMask = arith.subi %sideLanes, %one : i64
    %laneInHead = arith.andi %lane, %headLaneMask : i64
    %laneShift = arith.subi %sizeShift, %two : i64
    %headInWarp = arith.shrui %lane, %laneShift : i64
    %headInWarp32 = arith.trunci %headInWarp : i64 to i32
    %headsPerWarp32 = arith.select %isBig, %c1, %c2 : i32
    %swapDistance = arith.trunci %sideLanes : i64 to i32
    %halfLanes = arith.shrui %half, %two : i64
    %side = arith.andi %laneInHead, %sideMask : i64
    %isLowSide = arith.cmpi ult, %laneInHead, %sideLanes : i64
    %isPaired = arith.cmpi ult, %side, %halfLanes : i64
    %sideOffset = arith.shli %side, %two : i64
    %highOffset = arith.addi %sideOffset, %half : i64
    %pairedOffset = arith.select %isLowSide, %sideOffset, %highOffset : i64
    %tailStart = arith.shli %half, %one : i64
    %tailIndex = arith.subi %side, %halfLanes : i64
    %tailLanes = arith.subi %sideLanes, %halfLanes : i64
    %tailHigh = arith.addi %tailIndex, %tailLanes : i64
    %tailLane = arith.select %isLowSide, %tailIndex, %tailHigh : i64
    %tailLaneOffset = arith.shli %tailLane, %two : i64
    %tailOffset = arith.addi %tailStart, %tailLaneOffset : i64
    %laneOffset = arith.select %isPaired, %pairedOffset, %tailOffset : i64
    %wVec = func.call @strict_device_rope_load(%weights, %weightsBits, %laneOffset) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %turnShift = arith.addi %sizeShift, %one : i64
    %rotBase = arith.shli %token, %turnShift : i64
    %cosIndex = arith.addi %rotBase, %laneOffset : i64
    %sinIndex = arith.addi %cosIndex, %size : i64
    %cosVec = func.call @strict_device_rope_load(%rotation, %rotationBits, %cosIndex) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %sinVec = func.call @strict_device_rope_load(%rotation, %rotationBits, %sinIndex) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %epsilonBits32 = arith.trunci %epsilonBits : i64 to i32
    %eps = arith.bitcast %epsilonBits32 : i32 to f32
    %tripleWidth = arith.muli %width, %three : i64
    %tokenBase = arith.muli %token, %tripleWidth : i64
    %tokenStart = arith.addi %tokenBase, %start : i64
    %inLane = arith.addi %tokenStart, %laneOffset : i64
    %zeroOffset = arith.shli %laneInHead, %two : i64
    %zeroLane = arith.addi %zeroOffset, %size : i64
    %isPadded = arith.cmpi sgt, %wide, %size : i64
    %isBrainResult = arith.cmpi eq, %resultBits, %seventeen : i64
    %invBig = arith.constant 0.0078125 : f32
    %invSmall = arith.constant 0.015625 : f32
    %invSize = arith.select %isBig, %invBig, %invSmall : f32
    %zeros = arith.constant dense<0.0> : vector<4xf32>
    cf.br ^head(%warp32 : i32)
  ^head(%group32: i32):
    %firstHead32 = arith.muli %group32, %headsPerWarp32 : i32
    %more = arith.cmpi ult, %firstHead32, %heads32 : i32
    cf.cond_br %more, ^rotate, ^exit
  ^rotate:
    %head32 = arith.addi %firstHead32, %headInWarp32 : i32
    %head = arith.extui %head32 : i32 to i64
    %headOffset = arith.shli %head, %sizeShift : i64
    %inIndex = arith.addi %inLane, %headOffset : i64
    %xVec = func.call @strict_device_rope_load(%combined, %combinedBits, %inIndex) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %sqVec = arith.mulf %xVec, %xVec : vector<4xf32>
    %sq0 = llvm.extractelement %sqVec[%i0 : i32] : vector<4xf32>
    %sq1 = llvm.extractelement %sqVec[%i1 : i32] : vector<4xf32>
    %sq2 = llvm.extractelement %sqVec[%i2 : i32] : vector<4xf32>
    %sq3 = llvm.extractelement %sqVec[%i3 : i32] : vector<4xf32>
    %sq01 = arith.addf %sq0, %sq1 : f32
    %sq23 = arith.addf %sq2, %sq3 : f32
    %sq = arith.addf %sq01, %sq23 : f32
    %sqBits16 = arith.bitcast %sq : f32 to i32
    %sqSwapped16 = func.call @strict_device_swap(%sqBits16, %c16) : (i32, i32) -> i32
    %sqOther16 = arith.bitcast %sqSwapped16 : i32 to f32
    %sqBoth16 = arith.addf %sq, %sqOther16 : f32
    %sqSum16 = arith.select %isBig, %sqBoth16, %sq : f32
    %sqBits8 = arith.bitcast %sqSum16 : f32 to i32
    %sqSwapped8 = func.call @strict_device_swap(%sqBits8, %c8) : (i32, i32) -> i32
    %sqOther8 = arith.bitcast %sqSwapped8 : i32 to f32
    %sqSum8 = arith.addf %sqSum16, %sqOther8 : f32
    %sqBits4 = arith.bitcast %sqSum8 : f32 to i32
    %sqSwapped4 = func.call @strict_device_swap(%sqBits4, %c4) : (i32, i32) -> i32
    %sqOther4 = arith.bitcast %sqSwapped4 : i32 to f32
    %sqSum4 = arith.addf %sqSum8, %sqOther4 : f32
    %sqBits2 = arith.bitcast %sqSum4 : f32 to i32
    %sqSwapped2 = func.call @strict_device_swap(%sqBits2, %c2) : (i32, i32) -> i32
    %sqOther2 = arith.bitcast %sqSwapped2 : i32 to f32
    %sqSum2 = arith.addf %sqSum4, %sqOther2 : f32
    %sqBits1 = arith.bitcast %sqSum2 : f32 to i32
    %sqSwapped1 = func.call @strict_device_swap(%sqBits1, %c1) : (i32, i32) -> i32
    %sqOther1 = arith.bitcast %sqSwapped1 : i32 to f32
    %sqSum1 = arith.addf %sqSum2, %sqOther1 : f32
    %meanSq = arith.mulf %sqSum1, %invSize : f32
    %var = arith.addf %meanSq, %eps : f32
    %scale = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%var) : (f32) -> f32
    %wxVec = arith.mulf %xVec, %wVec : vector<4xf32>
    %wx0 = llvm.extractelement %wxVec[%i0 : i32] : vector<4xf32>
    %wx1 = llvm.extractelement %wxVec[%i1 : i32] : vector<4xf32>
    %wx2 = llvm.extractelement %wxVec[%i2 : i32] : vector<4xf32>
    %wx3 = llvm.extractelement %wxVec[%i3 : i32] : vector<4xf32>
    %wx0Bits = arith.bitcast %wx0 : f32 to i32
    %wx1Bits = arith.bitcast %wx1 : f32 to i32
    %wx2Bits = arith.bitcast %wx2 : f32 to i32
    %wx3Bits = arith.bitcast %wx3 : f32 to i32
    %p0Bits = func.call @strict_device_swap(%wx0Bits, %swapDistance) : (i32, i32) -> i32
    %p1Bits = func.call @strict_device_swap(%wx1Bits, %swapDistance) : (i32, i32) -> i32
    %p2Bits = func.call @strict_device_swap(%wx2Bits, %swapDistance) : (i32, i32) -> i32
    %p3Bits = func.call @strict_device_swap(%wx3Bits, %swapDistance) : (i32, i32) -> i32
    %p0 = arith.bitcast %p0Bits : i32 to f32
    %p1 = arith.bitcast %p1Bits : i32 to f32
    %p2 = arith.bitcast %p2Bits : i32 to f32
    %p3 = arith.bitcast %p3Bits : i32 to f32
    %undef = llvm.mlir.undef : vector<4xf32>
    %pv0 = llvm.insertelement %p0, %undef[%i0 : i32] : vector<4xf32>
    %pv1 = llvm.insertelement %p1, %pv0[%i1 : i32] : vector<4xf32>
    %pv2 = llvm.insertelement %p2, %pv1[%i2 : i32] : vector<4xf32>
    %partnerVec = llvm.insertelement %p3, %pv2[%i3 : i32] : vector<4xf32>
    %otherVec = arith.select %isPaired, %partnerVec, %wxVec : vector<4xf32>
    %ownVec = arith.mulf %wxVec, %cosVec : vector<4xf32>
    %sumVec = llvm.intr.fma(%otherVec, %sinVec, %ownVec) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %scaleVecUndefined = llvm.mlir.undef : vector<4xf32>
    %scaleVec0 = llvm.insertelement %scale, %scaleVecUndefined[%i0 : i32] : vector<4xf32>
    %scaleVec1 = llvm.insertelement %scale, %scaleVec0[%i1 : i32] : vector<4xf32>
    %scaleVec2 = llvm.insertelement %scale, %scaleVec1[%i2 : i32] : vector<4xf32>
    %scaleVec = llvm.insertelement %scale, %scaleVec2[%i3 : i32] : vector<4xf32>
    %outVec = arith.mulf %sumVec, %scaleVec : vector<4xf32>
    %outHeadOffset = arith.muli %head, %wide : i64
    %outHead = arith.addi %outToken, %outHeadOffset : i64
    %outIndex = arith.addi %outHead, %laneOffset : i64
    %zeroIndex = arith.addi %outHead, %zeroLane : i64
    cf.cond_br %isBrainResult, ^storeBrainOut, ^storeSingleOut
  ^storeBrainOut:
    %outBrain = arith.truncf %outVec : vector<4xf32> to vector<4xbf16>
    %outBrainPtr = llvm.getelementptr %result[%outIndex] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %outBrain, %outBrainPtr {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    cf.cond_br %isPadded, ^padBrain, ^next
  ^padBrain:
    %zeroBrain = arith.truncf %zeros : vector<4xf32> to vector<4xbf16>
    %zeroBrainPtr = llvm.getelementptr %result[%zeroIndex] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %zeroBrain, %zeroBrainPtr {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    cf.br ^next
  ^storeSingleOut:
    %outSinglePtr = llvm.getelementptr %result[%outIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %outVec, %outSinglePtr {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    cf.cond_br %isPadded, ^padSingle, ^next
  ^padSingle:
    %zeroSinglePtr = llvm.getelementptr %result[%zeroIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %zeros, %zeroSinglePtr {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    cf.br ^next
  ^next:
    %following = arith.addi %group32, %warps32 : i32
    cf.br ^head(%following : i32)
  ^exit:
    llvm.return
  }
  // 4 numbers from index of a list of 32 bit floats, bfloat16 (17) or doubles as 32 bit floats
  func.func private @strict_device_rope_load(%ptr: !llvm.ptr, %bits: i64, %index: i64) -> vector<4xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %seventeen = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^other
  ^single:
    %singlePtr = llvm.getelementptr %ptr[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %singles = llvm.load %singlePtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %singles : vector<4xf32>
  ^other:
    %isBrain = arith.cmpi eq, %bits, %seventeen : i64
    cf.cond_br %isBrain, ^brain, ^wide
  ^brain:
    %brainPtr = llvm.getelementptr %ptr[%index] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %brains = llvm.load %brainPtr invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fromBrains = arith.extf %brains : vector<4xbf16> to vector<4xf32>
    return %fromBrains : vector<4xf32>
  ^wide:
    %widePtr = llvm.getelementptr %ptr[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %wides = llvm.load %widePtr invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xf64>
    %fromWides = arith.truncf %wides : vector<4xf64> to vector<4xf32>
    return %fromWides : vector<4xf32>
  }
