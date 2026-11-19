  llvm.mlir.global internal @strict_device_row_modulated_shared() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<8 x f32>
  // `RowModulated.Modulated`: the RMS normalization of every token and its AdaLN modulation in one pass like
  // strict_device_token_modulated, for any width that is a multiple of 256 from 1280 to 8192 (H3: 5376) and a row of
  // the modulation per token (starts): a block of 256 threads per token, a thread holds 4 numbers of each of the up
  // to 8 chunks of 1024 in registers across the sum of the squares. The lists are 32 bit floats, bfloat16 or
  // doubles (strict_gpu_rowmodulated checked their formats and lengths), a row outside of the modulation fails,
  // without a modulation (an empty list) the tokens are only normalized and no starts are read.
  // The part comes once more as an integer after the result bits (strict_gpu_exacts) and the epsilon as the bits of
  // its 32 bit float, so no thread converts a double but the one of its start
  llvm.func @strict_device_row_modulated(%count: i64, %result: !llvm.ptr, %failed: !llvm.ptr, %source: !llvm.ptr, %columns: i64, %bits: i64, %input: !llvm.ptr, %inputCount: i64, %inputBits: i64, %weights: !llvm.ptr, %weightsCount: i64, %weightsBits: i64, %modulation: !llvm.ptr, %modulationCount: i64, %modulationBits: i64, %starts: !llvm.ptr, %startsCount: i64, %startsBits: i64, %partNumber: f64, %epsilonNumber: f64, %epsilonBitsNumber: f64, %resultBits: i64, %part: i64, %partMagic: i64, %epsilonExact: i64, %epsilonMagic: i64, %epsilonBits: i64, %epsilonBitsMagic: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 1 : i32} {
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
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %width = arith.addi %weightsCount, %none : i64
    %tid = func.call @strict_device_thread_index() : () -> i32
    %bid = func.call @strict_device_block_index() : () -> i32
    %token = arith.extui %bid : i32 to i64
    %tokenStart = arith.muli %token, %width : i64
    %inBounds = arith.cmpi slt, %tokenStart, %count : i64
    cf.cond_br %inBounds, ^work, ^exit
  ^work:
    %tid64 = arith.extui %tid : i32 to i64
    %col0 = arith.shli %tid64, %two : i64
    %chunk1 = arith.constant 1024 : i64
    %col1 = arith.addi %col0, %chunk1 : i64
    %chunk2 = arith.constant 2048 : i64
    %col2 = arith.addi %col0, %chunk2 : i64
    %chunk3 = arith.constant 3072 : i64
    %col3 = arith.addi %col0, %chunk3 : i64
    %chunk4 = arith.constant 4096 : i64
    %col4 = arith.addi %col0, %chunk4 : i64
    %chunk5 = arith.constant 5120 : i64
    %col5 = arith.addi %col0, %chunk5 : i64
    %chunk6 = arith.constant 6144 : i64
    %col6 = arith.addi %col0, %chunk6 : i64
    %chunk7 = arith.constant 7168 : i64
    %col7 = arith.addi %col0, %chunk7 : i64
    %x0 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col0, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x1 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col1, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x2 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col2, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x3 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col3, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x4 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col4, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x5 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col5, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x6 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col6, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %x7 = func.call @strict_device_row_mod_load(%input, %inputBits, %tokenStart, %col7, %width) : (!llvm.ptr, i64, i64, i64, i64) -> vector<4xf32>
    %sq0 = arith.mulf %x0, %x0 : vector<4xf32>
    %sq1 = arith.mulf %x1, %x1 : vector<4xf32>
    %sq2 = arith.mulf %x2, %x2 : vector<4xf32>
    %sq3 = arith.mulf %x3, %x3 : vector<4xf32>
    %sq4 = arith.mulf %x4, %x4 : vector<4xf32>
    %sq5 = arith.mulf %x5, %x5 : vector<4xf32>
    %sq6 = arith.mulf %x6, %x6 : vector<4xf32>
    %sq7 = arith.mulf %x7, %x7 : vector<4xf32>
    %sqA = arith.addf %sq0, %sq1 : vector<4xf32>
    %sqB = arith.addf %sq2, %sq3 : vector<4xf32>
    %sqC = arith.addf %sq4, %sq5 : vector<4xf32>
    %sqD = arith.addf %sq6, %sq7 : vector<4xf32>
    %sqAB = arith.addf %sqA, %sqB : vector<4xf32>
    %sqCD = arith.addf %sqC, %sqD : vector<4xf32>
    %sqVec = arith.addf %sqAB, %sqCD : vector<4xf32>
    %own0 = llvm.extractelement %sqVec[%i0 : i32] : vector<4xf32>
    %own1 = llvm.extractelement %sqVec[%i1 : i32] : vector<4xf32>
    %own2 = llvm.extractelement %sqVec[%i2 : i32] : vector<4xf32>
    %own3 = llvm.extractelement %sqVec[%i3 : i32] : vector<4xf32>
    %own01 = arith.addf %own0, %own1 : f32
    %own23 = arith.addf %own2, %own3 : f32
    %own = arith.addf %own01, %own23 : f32
    %ownBits16 = arith.bitcast %own : f32 to i32
    %ownSwapped16 = func.call @strict_device_swap(%ownBits16, %c16) : (i32, i32) -> i32
    %ownOther16 = arith.bitcast %ownSwapped16 : i32 to f32
    %ownSum16 = arith.addf %own, %ownOther16 : f32
    %ownBits8 = arith.bitcast %ownSum16 : f32 to i32
    %ownSwapped8 = func.call @strict_device_swap(%ownBits8, %c8) : (i32, i32) -> i32
    %ownOther8 = arith.bitcast %ownSwapped8 : i32 to f32
    %ownSum8 = arith.addf %ownSum16, %ownOther8 : f32
    %ownBits4 = arith.bitcast %ownSum8 : f32 to i32
    %ownSwapped4 = func.call @strict_device_swap(%ownBits4, %c4) : (i32, i32) -> i32
    %ownOther4 = arith.bitcast %ownSwapped4 : i32 to f32
    %ownSum4 = arith.addf %ownSum8, %ownOther4 : f32
    %ownBits2 = arith.bitcast %ownSum4 : f32 to i32
    %ownSwapped2 = func.call @strict_device_swap(%ownBits2, %c2) : (i32, i32) -> i32
    %ownOther2 = arith.bitcast %ownSwapped2 : i32 to f32
    %ownSum2 = arith.addf %ownSum4, %ownOther2 : f32
    %ownBits1 = arith.bitcast %ownSum2 : f32 to i32
    %ownSwapped1 = func.call @strict_device_swap(%ownBits1, %c1) : (i32, i32) -> i32
    %ownOther1 = arith.bitcast %ownSwapped1 : i32 to f32
    %ownSum1 = arith.addf %ownSum2, %ownOther1 : f32
    %lane = arith.andi %tid, %c31 : i32
    %warp = arith.shrui %tid, %c5 : i32
    %shared = llvm.mlir.addressof @strict_device_row_modulated_shared : !llvm.ptr<3>
    %isLane0 = arith.cmpi eq, %lane, %c0 : i32
    cf.cond_br %isLane0, ^storeWarp, ^afterStoreWarp
  ^storeWarp:
    %warpPtr = llvm.getelementptr %shared[%warp] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %ownSum1, %warpPtr : f32, !llvm.ptr<3>
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
    %width32 = arith.trunci %width : i64 to i32
    %widthFloat = arith.sitofp %width32 : i32 to f32
    %meanSq = arith.divf %totalSq, %widthFloat : f32
    %epsilonBits32 = arith.trunci %epsilonBits : i64 to i32
    %eps = arith.bitcast %epsilonBits32 : i32 to f32
    %var = arith.addf %meanSq, %eps : f32
    %scale = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%var) : (f32) -> f32
    %scaleVecUndefined = llvm.mlir.undef : vector<4xf32>
    %scaleVec0 = llvm.insertelement %scale, %scaleVecUndefined[%i0 : i32] : vector<4xf32>
    %scaleVec1 = llvm.insertelement %scale, %scaleVec0[%i1 : i32] : vector<4xf32>
    %scaleVec2 = llvm.insertelement %scale, %scaleVec1[%i2 : i32] : vector<4xf32>
    %scaleVec = llvm.insertelement %scale, %scaleVec2[%i3 : i32] : vector<4xf32>
    %isModulated = arith.cmpi sgt, %modulationCount, %none : i64
    cf.cond_br %isModulated, ^rowed, ^store(%none, %none : i64, i64)
  ^rowed:
    %row = func.call @strict_device_row_mod_start(%starts, %startsCount, %startsBits, %token) : (!llvm.ptr, i64, i64, i64) -> i64
    %shifted = arith.addi %row, %part : i64
    %rowShifts = arith.muli %shifted, %width : i64
    %rowScaling = arith.addi %rowShifts, %width : i64
    %scalingEnd = arith.addi %rowScaling, %width : i64
    %rowFits = arith.cmpi sle, %scalingEnd, %modulationCount : i64
    %rowStarts = arith.cmpi sge, %shifted, %none : i64
    %rowInside = arith.andi %rowFits, %rowStarts : i1
    cf.cond_br %rowInside, ^store(%rowShifts, %rowScaling : i64, i64), ^outside
  ^outside:
    %isFirstThread = arith.cmpi eq, %tid, %c0 : i32
    cf.cond_br %isFirstThread, ^markFail, ^exit
  ^markFail:
    llvm.store %one, %failed : i64, !llvm.ptr
    cf.br ^exit
  ^store(%shifts: i64, %scaling: i64):
    %isBrainResult = arith.cmpi eq, %resultBits, %seventeen : i64
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col0, %x0, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col1, %x1, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col2, %x2, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col3, %x3, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col4, %x4, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col5, %x5, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col6, %x6, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    func.call @strict_device_row_mod_store(%result, %tokenStart, %col7, %x7, %scaleVec, %weights, %weightsBits, %modulation, %modulationBits, %isModulated, %shifts, %scaling, %isBrainResult, %width) : (!llvm.ptr, i64, i64, vector<4xf32>, vector<4xf32>, !llvm.ptr, i64, !llvm.ptr, i64, i1, i64, i64, i1, i64) -> ()
    cf.br ^exit
  ^exit:
    llvm.return
  }
  // the 4 numbers from base + col of a token, zeros after its width
  func.func private @strict_device_row_mod_load(%ptr: !llvm.ptr, %bits: i64, %base: i64, %col: i64, %width: i64) -> vector<4xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %inside = arith.cmpi slt, %col, %width : i64
    cf.cond_br %inside, ^read, ^outside
  ^outside:
    %zeros = arith.constant dense<0.0> : vector<4xf32>
    return %zeros : vector<4xf32>
  ^read:
    %index = arith.addi %base, %col : i64
    %numbers = func.call @strict_device_row_mod_read(%ptr, %bits, %index) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    return %numbers : vector<4xf32>
  }
  // the row of the modulation of token: its number in starts (32 bit floats, bfloat16 or doubles)
  func.func private @strict_device_row_mod_start(%starts: !llvm.ptr, %count: i64, %bits: i64, %token: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %one = arith.constant 1 : i64
    %seventeen = arith.constant 17 : i64
    %thirtyTwo = arith.constant 32 : i64
    %last = arith.subi %count, %one : i64
    %at = arith.minsi %token, %last : i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    cf.cond_br %isSingle, ^single, ^other
  ^single:
    %sPtr = llvm.getelementptr %starts[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %sValue = llvm.load %sPtr : !llvm.ptr -> f32
    %sRow = arith.fptosi %sValue : f32 to i64
    return %sRow : i64
  ^other:
    %isBrain = arith.cmpi eq, %bits, %seventeen : i64
    cf.cond_br %isBrain, ^brain, ^wide
  ^brain:
    %bPtr = llvm.getelementptr %starts[%at] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %bValue = llvm.load %bPtr : !llvm.ptr -> bf16
    %bFloat = arith.extf %bValue : bf16 to f32
    %bRow = arith.fptosi %bFloat : f32 to i64
    return %bRow : i64
  ^wide:
    %wPtr = llvm.getelementptr %starts[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %wValue = llvm.load %wPtr : !llvm.ptr -> f64
    %wRow = arith.fptosi %wValue : f64 to i64
    return %wRow : i64
  }
  // 4 normalized and modulated numbers of a token into the result (32 bit floats or bfloat16), none after its width
  func.func private @strict_device_row_mod_store(%result: !llvm.ptr, %tokenStart: i64, %col: i64, %x: vector<4xf32>, %scaleVec: vector<4xf32>, %weights: !llvm.ptr, %weightsBits: i64, %modulation: !llvm.ptr, %modulationBits: i64, %isModulated: i1, %shifts: i64, %scaling: i64, %isBrainResult: i1, %width: i64) attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
    %inside = arith.cmpi slt, %col, %width : i64
    cf.cond_br %inside, ^compute, ^outside
  ^outside:
    return
  ^compute:
    %wVec = func.call @strict_device_row_mod_read(%weights, %weightsBits, %col) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %xw = arith.mulf %x, %wVec : vector<4xf32>
    %xs = arith.mulf %xw, %scaleVec : vector<4xf32>
    %outIdx = arith.addi %tokenStart, %col : i64
    cf.cond_br %isModulated, ^modulated, ^write(%xs : vector<4xf32>)
  ^modulated:
    %scaleIndex = arith.addi %scaling, %col : i64
    %shiftIndex = arith.addi %shifts, %col : i64
    %mScale = func.call @strict_device_row_mod_read(%modulation, %modulationBits, %scaleIndex) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %mShift = func.call @strict_device_row_mod_read(%modulation, %modulationBits, %shiftIndex) : (!llvm.ptr, i64, i64) -> vector<4xf32>
    %ones = arith.constant dense<1.0> : vector<4xf32>
    %onePlus = arith.addf %ones, %mScale : vector<4xf32>
    %shiftedOut = llvm.intr.fma(%xs, %onePlus, %mShift) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    cf.br ^write(%shiftedOut : vector<4xf32>)
  ^write(%out: vector<4xf32>):
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
  // 4 numbers from index of a list of 32 bit floats, bfloat16 (17) or doubles as 32 bit floats
  func.func private @strict_device_row_mod_read(%ptr: !llvm.ptr, %bits: i64, %index: i64) -> vector<4xf32> attributes {llvm.linkage = #llvm.linkage<internal>, llvm.passthrough = ["alwaysinline"]} {
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
