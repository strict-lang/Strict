  // Fused head RMS-normalization + RoPE rotation for 128-element heads: one warp of 32 lanes per head (4 elements per lane),
  // computing the RMS scale in warp registers and exchanging partner elements (within + 64) % 128 via lane ^ 16 shuffles.
  llvm.func @strict_device_head_rotated(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %combined: !llvm.ptr, %combinedCount: i64, %combinedBits: i64, %startNumber: f64, %widthNumber: f64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %rotation: !llvm.ptr, %rotationCount: i64, %rotationBits: i64, %epsilonNumber: f64, %resultBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %c31 = arith.constant 31 : i32
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %seventeen = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %headSize = arith.constant 128 : i64
    %headMask = arith.constant 127 : i64
    %width = arith.fptosi %widthNumber : f64 to i64
    %start = arith.fptosi %startNumber : f64 to i64
    %isSize128 = arith.cmpi eq, %weightsCount, %headSize : i64
    %hasCount = arith.cmpi sgt, %count, %none : i64
    %tripleCount = arith.muli %count, %three : i64
    %isTriple = arith.cmpi eq, %tripleCount, %combinedCount : i64
    %widthRem = arith.andi %width, %headMask : i64
    %isWidthAligned = arith.cmpi eq, %widthRem, %none : i64
    %hasWidth = arith.cmpi sge, %width, %headSize : i64
    %isBrainCombined = arith.cmpi eq, %combinedBits, %seventeen : i64
    %isSingleCombined = arith.cmpi eq, %combinedBits, %thirtyTwo : i64
    %okCombined = arith.ori %isBrainCombined, %isSingleCombined : i1
    %isSingleWeights = arith.cmpi eq, %weightsBits, %thirtyTwo : i64
    %isBrainWeights = arith.cmpi eq, %weightsBits, %seventeen : i64
    %okWeights = arith.ori %isSingleWeights, %isBrainWeights : i1
    %okRotation = arith.cmpi eq, %rotationBits, %thirtyTwo : i64
    %isBrainResult = arith.cmpi eq, %resultBits, %seventeen : i64
    %isSingleResult = arith.cmpi eq, %resultBits, %thirtyTwo : i64
    %okResult = arith.ori %isBrainResult, %isSingleResult : i1
    %ok0 = arith.andi %isSize128, %hasCount : i1
    %ok1 = arith.andi %isTriple, %isWidthAligned : i1
    %ok2 = arith.andi %hasWidth, %okCombined : i1
    %ok3 = arith.andi %okWeights, %okRotation : i1
    %ok01 = arith.andi %ok0, %ok1 : i1
    %ok23 = arith.andi %ok2, %ok3 : i1
    %okAll = arith.andi %ok01, %ok23 : i1
    %usable = arith.andi %okAll, %okResult : i1
    cf.cond_br %usable, ^run, ^fail
  ^fail:
    %localTid = func.call @strict_device_thread_index() : () -> i32
    %localBid = func.call @strict_device_block_index() : () -> i32
    %isZeroTid = arith.cmpi eq, %localTid, %c0 : i32
    %isZeroBid = arith.cmpi eq, %localBid, %c0 : i32
    %isFirstThread = arith.andi %isZeroTid, %isZeroBid : i1
    cf.cond_br %isFirstThread, ^markFail, ^exit
  ^markFail:
    llvm.store %one, %failed : i64, !llvm.ptr
    cf.br ^exit
  ^run:
    %tid = func.call @strict_device_thread_index() : () -> i32
    %bid = func.call @strict_device_block_index() : () -> i32
    %bsize = func.call @strict_device_block_size() : () -> i32
    %blockBase = arith.muli %bid, %bsize : i32
    %thread = arith.addi %blockBase, %tid : i32
    %warp32 = arith.shrui %thread, %c5 : i32
    %lane32 = arith.andi %thread, %c31 : i32
    %warp = arith.extui %warp32 : i32 to i64
    %lane = arith.extui %lane32 : i32 to i64
    %seven = arith.constant 7 : i64
    %outOffset = arith.shli %warp, %seven : i64
    %inBounds = arith.cmpi slt, %outOffset, %count : i64
    cf.cond_br %inBounds, ^work, ^exit
  ^work:
    %heads = arith.shrui %width, %seven : i64
    %token = arith.divui %warp, %heads : i64
    %head = arith.remui %warp, %heads : i64
    %tripleWidth = arith.muli %width, %three : i64
    %tokenBase = arith.muli %token, %tripleWidth : i64
    %tokenStart = arith.addi %tokenBase, %start : i64
    %headOffset = arith.shli %head, %seven : i64
    %headBase = arith.addi %tokenStart, %headOffset : i64
    %two = arith.constant 2 : i64
    %laneOffset = arith.shli %lane, %two : i64
    %inIndex = arith.addi %headBase, %laneOffset : i64
    cf.cond_br %isBrainCombined, ^loadBrainX, ^loadSingleX
  ^loadBrainX:
    %inBrainPtr = llvm.getelementptr %combined[%inIndex] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %inBrain = llvm.load %inBrainPtr invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %xFromBrain = arith.extf %inBrain : vector<4xbf16> to vector<4xf32>
    cf.br ^loadedX(%xFromBrain : vector<4xf32>)
  ^loadSingleX:
    %inSinglePtr = llvm.getelementptr %combined[%inIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %xFromSingle = llvm.load %inSinglePtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    cf.br ^loadedX(%xFromSingle : vector<4xf32>)
  ^loadedX(%xVec: vector<4xf32>):
    cf.cond_br %isSingleWeights, ^loadSingleW, ^loadBrainW
  ^loadSingleW:
    %wSinglePtr = llvm.getelementptr %weights[%laneOffset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wFromSingle = llvm.load %wSinglePtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    cf.br ^loadedW(%wFromSingle : vector<4xf32>)
  ^loadBrainW:
    %wBrainPtr = llvm.getelementptr %weights[%laneOffset] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %wBrain = llvm.load %wBrainPtr invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %wFromBrain = arith.extf %wBrain : vector<4xbf16> to vector<4xf32>
    cf.br ^loadedW(%wFromBrain : vector<4xf32>)
  ^loadedW(%wVec: vector<4xf32>):
    %eight = arith.constant 8 : i64
    %rotBase = arith.shli %token, %eight : i64
    %cosIndex = arith.addi %rotBase, %laneOffset : i64
    %sinIndex = arith.addi %cosIndex, %headSize : i64
    %cosPtr = llvm.getelementptr %rotation[%cosIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sinPtr = llvm.getelementptr %rotation[%sinIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %cosVec = llvm.load %cosPtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sinVec = llvm.load %sinPtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %sqVec = arith.mulf %xVec, %xVec : vector<4xf32>
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %sq0 = llvm.extractelement %sqVec[%i0 : i32] : vector<4xf32>
    %sq1 = llvm.extractelement %sqVec[%i1 : i32] : vector<4xf32>
    %sq2 = llvm.extractelement %sqVec[%i2 : i32] : vector<4xf32>
    %sq3 = llvm.extractelement %sqVec[%i3 : i32] : vector<4xf32>
    %sq01 = arith.addf %sq0, %sq1 : f32
    %sq23 = arith.addf %sq2, %sq3 : f32
    %sq = arith.addf %sq01, %sq23 : f32
    %c8 = arith.constant 8 : i32
    %c4 = arith.constant 4 : i32
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %sqBits0 = arith.bitcast %sq : f32 to i32
    %s16Bits = func.call @strict_device_swap(%sqBits0, %c16) : (i32, i32) -> i32
    %s16 = arith.bitcast %s16Bits : i32 to f32
    %sum16 = arith.addf %sq, %s16 : f32
    %sqBits1 = arith.bitcast %sum16 : f32 to i32
    %s8Bits = func.call @strict_device_swap(%sqBits1, %c8) : (i32, i32) -> i32
    %s8 = arith.bitcast %s8Bits : i32 to f32
    %sum8 = arith.addf %sum16, %s8 : f32
    %sqBits2 = arith.bitcast %sum8 : f32 to i32
    %s4Bits = func.call @strict_device_swap(%sqBits2, %c4) : (i32, i32) -> i32
    %s4 = arith.bitcast %s4Bits : i32 to f32
    %sum4 = arith.addf %sum8, %s4 : f32
    %sqBits3 = arith.bitcast %sum4 : f32 to i32
    %s2Bits = func.call @strict_device_swap(%sqBits3, %c2) : (i32, i32) -> i32
    %s2 = arith.bitcast %s2Bits : i32 to f32
    %sum2 = arith.addf %sum4, %s2 : f32
    %sqBits4 = arith.bitcast %sum2 : f32 to i32
    %s1Bits = func.call @strict_device_swap(%sqBits4, %c1) : (i32, i32) -> i32
    %s1 = arith.bitcast %s1Bits : i32 to f32
    %totalSq = arith.addf %sum2, %s1 : f32
    %inv128 = arith.constant 0.0078125 : f32
    %meanSq = arith.mulf %totalSq, %inv128 : f32
    %eps = arith.truncf %epsilonNumber : f64 to f32
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
    %p0Bits = func.call @strict_device_swap(%wx0Bits, %c16) : (i32, i32) -> i32
    %p1Bits = func.call @strict_device_swap(%wx1Bits, %c16) : (i32, i32) -> i32
    %p2Bits = func.call @strict_device_swap(%wx2Bits, %c16) : (i32, i32) -> i32
    %p3Bits = func.call @strict_device_swap(%wx3Bits, %c16) : (i32, i32) -> i32
    %p0 = arith.bitcast %p0Bits : i32 to f32
    %p1 = arith.bitcast %p1Bits : i32 to f32
    %p2 = arith.bitcast %p2Bits : i32 to f32
    %p3 = arith.bitcast %p3Bits : i32 to f32
    %undef = llvm.mlir.undef : vector<4xf32>
    %pv0 = llvm.insertelement %p0, %undef[%i0 : i32] : vector<4xf32>
    %pv1 = llvm.insertelement %p1, %pv0[%i1 : i32] : vector<4xf32>
    %pv2 = llvm.insertelement %p2, %pv1[%i2 : i32] : vector<4xf32>
    %pVec = llvm.insertelement %p3, %pv2[%i3 : i32] : vector<4xf32>
    %ownVec = arith.mulf %wxVec, %cosVec : vector<4xf32>
    %sumVec = llvm.intr.fma(%pVec, %sinVec, %ownVec) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %s0 = llvm.insertelement %scale, %undef[%i0 : i32] : vector<4xf32>
    %s1v = llvm.insertelement %scale, %s0[%i1 : i32] : vector<4xf32>
    %s2v = llvm.insertelement %scale, %s1v[%i2 : i32] : vector<4xf32>
    %scaleVec = llvm.insertelement %scale, %s2v[%i3 : i32] : vector<4xf32>
    %outVec = arith.mulf %sumVec, %scaleVec : vector<4xf32>
    %outIndex = arith.addi %outOffset, %laneOffset : i64
    cf.cond_br %isBrainResult, ^storeBrainOut, ^storeSingleOut
  ^storeBrainOut:
    %outBrain = arith.truncf %outVec : vector<4xf32> to vector<4xbf16>
    %outBrainPtr = llvm.getelementptr %result[%outIndex] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %outBrain, %outBrainPtr {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    cf.br ^exit
  ^storeSingleOut:
    %outSinglePtr = llvm.getelementptr %result[%outIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %outVec, %outSinglePtr {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    cf.br ^exit
  ^exit:
    llvm.return
  }
