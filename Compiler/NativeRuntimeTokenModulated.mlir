  llvm.mlir.global internal @strict_device_token_modulated_shared() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8 x f32>
  // Fused RMS-normalization + AdaLN modulation for token widths that are multiples of 1024 (e.g. 3072):
  // 256 threads per token (up to 4 chunks of 1024 = 4096 width), holding input in registers across the reduction.
  llvm.func @strict_device_token_modulated(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %modulation: !llvm.ptr, %modulationCount: i64, %modulationBits: i64, %partNumber: f64, %epsilonNumber: f64, %resultBits: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
    %c0 = arith.constant 0 : i32
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
    %seventeen = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %chunkWidth = arith.constant 1024 : i64
    %chunkMask = arith.constant 1023 : i64
    %maxChunks = arith.constant 4096 : i64
    %width = arith.addi %weightsCount, %none : i64
    %hasCount = arith.cmpi sgt, %count, %none : i64
    %sameCount = arith.cmpi eq, %count, %inputCount : i64
    %widthRem = arith.andi %width, %chunkMask : i64
    %isChunkAligned = arith.cmpi eq, %widthRem, %none : i64
    %hasWidth = arith.cmpi sge, %width, %chunkWidth : i64
    %fitsChunks = arith.cmpi sle, %width, %maxChunks : i64
    %isBrainInput = arith.cmpi eq, %inputBits, %seventeen : i64
    %isSingleInput = arith.cmpi eq, %inputBits, %thirtyTwo : i64
    %okInput = arith.ori %isBrainInput, %isSingleInput : i1
    %isSingleWeights = arith.cmpi eq, %weightsBits, %thirtyTwo : i64
    %isBrainWeights = arith.cmpi eq, %weightsBits, %seventeen : i64
    %okWeights = arith.ori %isSingleWeights, %isBrainWeights : i1
    %isSingleMod = arith.cmpi eq, %modulationBits, %thirtyTwo : i64
    %isBrainMod = arith.cmpi eq, %modulationBits, %seventeen : i64
    %okMod = arith.ori %isSingleMod, %isBrainMod : i1
    %isBrainResult = arith.cmpi eq, %resultBits, %seventeen : i64
    %isSingleResult = arith.cmpi eq, %resultBits, %thirtyTwo : i64
    %okResult = arith.ori %isBrainResult, %isSingleResult : i1
    %ok0 = arith.andi %hasCount, %sameCount : i1
    %ok1 = arith.andi %isChunkAligned, %hasWidth : i1
    %ok2 = arith.andi %fitsChunks, %okInput : i1
    %ok3 = arith.andi %okWeights, %okMod : i1
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
    %token = arith.extui %bid : i32 to i64
    %tokenStart = arith.muli %token, %width : i64
    %inBounds = arith.cmpi slt, %tokenStart, %count : i64
    cf.cond_br %inBounds, ^work, ^exit
  ^work:
    %tid64 = arith.extui %tid : i32 to i64
    %colBase = arith.shli %tid64, %two : i64
    %ten = arith.constant 10 : i64
    %numChunks = arith.shrui %width, %ten : i64
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : vector<4xf32>
    %c3 = arith.constant 3 : i64
    %has1 = arith.cmpi sgt, %numChunks, %one : i64
    %has2 = arith.cmpi sgt, %numChunks, %two : i64
    %has3 = arith.cmpi sgt, %numChunks, %c3 : i64
    %x0 = func.call @strict_device_token_mod_load(%input, %isBrainInput, %tokenStart, %colBase) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    %col1 = arith.addi %colBase, %chunkWidth : i64
    cf.cond_br %has1, ^load1, ^skip1
  ^load1:
    %lx1 = func.call @strict_device_token_mod_load(%input, %isBrainInput, %tokenStart, %col1) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    cf.br ^after1(%lx1 : vector<4xf32>)
  ^skip1:
    cf.br ^after1(%zeros : vector<4xf32>)
  ^after1(%x1: vector<4xf32>):
    %twoK = arith.constant 2048 : i64
    %col2 = arith.addi %colBase, %twoK : i64
    cf.cond_br %has2, ^load2, ^skip2
  ^load2:
    %lx2 = func.call @strict_device_token_mod_load(%input, %isBrainInput, %tokenStart, %col2) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    cf.br ^after2(%lx2 : vector<4xf32>)
  ^skip2:
    cf.br ^after2(%zeros : vector<4xf32>)
  ^after2(%x2: vector<4xf32>):
    %threeK = arith.constant 3072 : i64
    %col3 = arith.addi %colBase, %threeK : i64
    cf.cond_br %has3, ^load3, ^skip3
  ^load3:
    %lx3 = func.call @strict_device_token_mod_load(%input, %isBrainInput, %tokenStart, %col3) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    cf.br ^after3(%lx3 : vector<4xf32>)
  ^skip3:
    cf.br ^after3(%zeros : vector<4xf32>)
  ^after3(%x3: vector<4xf32>):
    %sq0 = arith.mulf %x0, %x0 : vector<4xf32>
    %sq1 = arith.mulf %x1, %x1 : vector<4xf32>
    %sq2 = arith.mulf %x2, %x2 : vector<4xf32>
    %sq3 = arith.mulf %x3, %x3 : vector<4xf32>
    %sq01 = arith.addf %sq0, %sq1 : vector<4xf32>
    %sq23 = arith.addf %sq2, %sq3 : vector<4xf32>
    %sqVec = arith.addf %sq01, %sq23 : vector<4xf32>
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %e0 = llvm.extractelement %sqVec[%i0 : i32] : vector<4xf32>
    %e1 = llvm.extractelement %sqVec[%i1 : i32] : vector<4xf32>
    %e2 = llvm.extractelement %sqVec[%i2 : i32] : vector<4xf32>
    %e3 = llvm.extractelement %sqVec[%i3 : i32] : vector<4xf32>
    %e01 = arith.addf %e0, %e1 : f32
    %e23 = arith.addf %e2, %e3 : f32
    %sq = arith.addf %e01, %e23 : f32
    %b0 = arith.bitcast %sq : f32 to i32
    %s16Bits = func.call @strict_device_swap(%b0, %c16) : (i32, i32) -> i32
    %s16 = arith.bitcast %s16Bits : i32 to f32
    %sum16 = arith.addf %sq, %s16 : f32
    %b1 = arith.bitcast %sum16 : f32 to i32
    %s8Bits = func.call @strict_device_swap(%b1, %c8) : (i32, i32) -> i32
    %s8 = arith.bitcast %s8Bits : i32 to f32
    %sum8 = arith.addf %sum16, %s8 : f32
    %b2 = arith.bitcast %sum8 : f32 to i32
    %s4Bits = func.call @strict_device_swap(%b2, %c4) : (i32, i32) -> i32
    %s4 = arith.bitcast %s4Bits : i32 to f32
    %sum4 = arith.addf %sum8, %s4 : f32
    %b3 = arith.bitcast %sum4 : f32 to i32
    %s2Bits = func.call @strict_device_swap(%b3, %c2) : (i32, i32) -> i32
    %s2 = arith.bitcast %s2Bits : i32 to f32
    %sum2 = arith.addf %sum4, %s2 : f32
    %b4 = arith.bitcast %sum2 : f32 to i32
    %s1Bits = func.call @strict_device_swap(%b4, %c1) : (i32, i32) -> i32
    %s1 = arith.bitcast %s1Bits : i32 to f32
    %warpSum = arith.addf %sum2, %s1 : f32
    %lane = arith.andi %tid, %c31 : i32
    %warp = arith.shrui %tid, %c5 : i32
    %shared = llvm.mlir.addressof @strict_device_token_modulated_shared : !llvm.ptr<3>
    %isLane0 = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %isLane0, ^storeWarp, ^afterStoreWarp
  ^storeWarp:
    %warpPtr = llvm.getelementptr %shared[%warp] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %warpSum, %warpPtr : f32, !llvm.ptr<3>
    cf.br ^afterStoreWarp
  ^afterStoreWarp:
    func.call @strict_device_barrier() : () -> ()
    %sp0 = llvm.getelementptr %shared[0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp1 = llvm.getelementptr %shared[1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp2 = llvm.getelementptr %shared[2] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp3 = llvm.getelementptr %shared[3] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp4 = llvm.getelementptr %shared[4] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp5 = llvm.getelementptr %shared[5] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp6 = llvm.getelementptr %shared[6] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %sp7 = llvm.getelementptr %shared[7] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
    %w0 = llvm.load %sp0 : !llvm.ptr<3> -> f32
    %w1 = llvm.load %sp1 : !llvm.ptr<3> -> f32
    %w2 = llvm.load %sp2 : !llvm.ptr<3> -> f32
    %w3 = llvm.load %sp3 : !llvm.ptr<3> -> f32
    %w4 = llvm.load %sp4 : !llvm.ptr<3> -> f32
    %w5 = llvm.load %sp5 : !llvm.ptr<3> -> f32
    %w6 = llvm.load %sp6 : !llvm.ptr<3> -> f32
    %w7 = llvm.load %sp7 : !llvm.ptr<3> -> f32
    %w01 = arith.addf %w0, %w1 : f32
    %w23 = arith.addf %w2, %w3 : f32
    %w45 = arith.addf %w4, %w5 : f32
    %w67 = arith.addf %w6, %w7 : f32
    %w03 = arith.addf %w01, %w23 : f32
    %w47 = arith.addf %w45, %w67 : f32
    %totalSq = arith.addf %w03, %w47 : f32
    %widthFloat = arith.sitofp %width : i64 to f32
    %meanSq = arith.divf %totalSq, %widthFloat : f32
    %eps = arith.truncf %epsilonNumber : f64 to f32
    %var = arith.addf %meanSq, %eps : f32
    %scale = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%var) : (f32) -> f32
    %undef = llvm.mlir.undef : vector<4xf32>
    %sv0 = llvm.insertelement %scale, %undef[%i0 : i32] : vector<4xf32>
    %sv1 = llvm.insertelement %scale, %sv0[%i1 : i32] : vector<4xf32>
    %sv2 = llvm.insertelement %scale, %sv1[%i2 : i32] : vector<4xf32>
    %scaleVec = llvm.insertelement %scale, %sv2[%i3 : i32] : vector<4xf32>
    %part = arith.fptosi %partNumber : f64 to i64
    %shifts = arith.muli %part, %width : i64
    %scaling = arith.addi %shifts, %width : i64
    func.call @strict_device_token_mod_store(%result, %tokenStart, %colBase, %x0, %scaleVec, %weights, %isSingleWeights, %modulation, %isSingleMod, %shifts, %scaling, %isBrainResult) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i1, !llvm.ptr, i1, i64, i64, i1) -> ()
    cf.cond_br %has1, ^store1, ^done1
  ^store1:
    func.call @strict_device_token_mod_store(%result, %tokenStart, %col1, %x1, %scaleVec, %weights, %isSingleWeights, %modulation, %isSingleMod, %shifts, %scaling, %isBrainResult) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i1, !llvm.ptr, i1, i64, i64, i1) -> ()
    cf.br ^done1
  ^done1:
    cf.cond_br %has2, ^store2, ^done2
  ^store2:
    func.call @strict_device_token_mod_store(%result, %tokenStart, %col2, %x2, %scaleVec, %weights, %isSingleWeights, %modulation, %isSingleMod, %shifts, %scaling, %isBrainResult) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i1, !llvm.ptr, i1, i64, i64, i1) -> ()
    cf.br ^done2
  ^done2:
    cf.cond_br %has3, ^store3, ^exit
  ^store3:
    func.call @strict_device_token_mod_store(%result, %tokenStart, %col3, %x3, %scaleVec, %weights, %isSingleWeights, %modulation, %isSingleMod, %shifts, %scaling, %isBrainResult) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i1, !llvm.ptr, i1, i64, i64, i1) -> ()
    cf.br ^exit
  ^exit:
    llvm.return
  }
  func.func private @strict_device_token_mod_load(%ptr: !llvm.ptr, %isBrain: i1, %base: i64, %col: i64) -> vector<4xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %idx = arith.addi %base, %col : i64
    cf.cond_br %isBrain, ^brain, ^single
  ^brain:
    %bPtr = llvm.getelementptr %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bVec = llvm.load %bPtr invariant {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %fVec = arith.extf %bVec : vector<4xbf16> to vector<4xf32>
    return %fVec : vector<4xf32>
  ^single:
    %sPtr = llvm.getelementptr %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sVec = llvm.load %sPtr invariant {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    return %sVec : vector<4xf32>
  }
  func.func private @strict_device_token_mod_store(%result: !llvm.ptr, %tokenStart: i64, %col: i64, %x: vector<4xf32>, %scaleVec: vector<4xf32>, %weights: !llvm.ptr, %isSingleWeights: i1, %modulation: !llvm.ptr, %isSingleMod: i1, %shifts: i64, %scaling: i64, %isBrainResult: i1) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %isBrainW = arith.xori %isSingleWeights, %true : i1
    %isBrainM = arith.xori %isSingleMod, %true : i1
    %wVec = func.call @strict_device_token_mod_load(%weights, %isBrainW, %zero, %col) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    %mScale = func.call @strict_device_token_mod_load(%modulation, %isBrainM, %scaling, %col) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    %mShift = func.call @strict_device_token_mod_load(%modulation, %isBrainM, %shifts, %col) : (!llvm.ptr, i1, i64, i64) -> vector<4xf32>
    %ones = arith.constant dense<1.0> : vector<4xf32>
    %onePlus = arith.addf %ones, %mScale : vector<4xf32>
    %xw = arith.mulf %x, %wVec : vector<4xf32>
    %xs = arith.mulf %xw, %scaleVec : vector<4xf32>
    %out = llvm.intr.fma(%xs, %onePlus, %mShift) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %outIdx = arith.addi %tokenStart, %col : i64
    cf.cond_br %isBrainResult, ^brainOut, ^singleOut
  ^brainOut:
    %outBrain = arith.truncf %out : vector<4xf32> to vector<4xbf16>
    %outBrainPtr = llvm.getelementptr %result[%outIdx] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %outBrain, %outBrainPtr {alignment = 8 : i64} : vector<4xbf16>, !llvm.ptr
    return
  ^singleOut:
    %outSinglePtr = llvm.getelementptr %result[%outIdx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %out, %outSinglePtr {alignment = 16 : i64} : vector<4xf32>, !llvm.ptr
    return
  }
