  llvm.func @llvm.eh.unwind.init()
  llvm.func @exit(i32)
  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
  llvm.func @sin(f64) -> f64
  llvm.func @cos(f64) -> f64
  llvm.func @tan(f64) -> f64
  llvm.func @asin(f64) -> f64
  llvm.func @acos(f64) -> f64
  llvm.func @atan(f64) -> f64
  llvm.func @atan2(f64, f64) -> f64
  llvm.func @log(f64) -> f64
  llvm.func @strftime(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i64
  llvm.func @remove(!llvm.ptr) -> i32
  llvm.func @rename(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @fgetc(!llvm.ptr) -> i32
  llvm.func @fputc(i32, !llvm.ptr) -> i32
  llvm.func @fflush(!llvm.ptr) -> i32
  func.func private @strict_flush() {
    %null = llvm.mlir.zero : !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    return
  }
  func.func private @strict_file_delete(%path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %removed = llvm.call @remove(%name) : (!llvm.ptr) -> i32
    return %zero : f64
  }
  llvm.mlir.global internal constant @strict_time_format("%Y-%m-%d %H:%M:%S\00") : !llvm.array<18 x i8>
  func.func private @strict_time_text(%local: !llvm.ptr) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %invalid = llvm.icmp "eq" %local, %null : !llvm.ptr
    cf.cond_br %invalid, ^empty, ^format
  ^format:
    %size = arith.constant 32 : i64
    %text = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %pattern = llvm.mlir.addressof @strict_time_format : !llvm.ptr
    %written = llvm.call @strftime(%text, %size, %pattern, %local) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i64
    %end = llvm.getelementptr %text[%written] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%text) : (!llvm.ptr) -> f64
    return %result : f64
  ^empty:
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %nothing = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    return %nothing : f64
  }
  llvm.mlir.global internal @strict_gc_base(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_top(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_limit(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_committed(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_table(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_stack(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_allocated(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_threshold(16777216 : i64) : i64
  llvm.mlir.global internal @strict_gc_marked(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_pool(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_pooled(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_spare(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_spare_count(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_spare_capacity(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_large(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_work(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_work_size(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_work_capacity(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_cursors(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_ends(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_spans(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_pages(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_sweeps(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_fulls(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_tails(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @strict_gc_cards(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_dirty(dense<0> : tensor<3xi64>) : !llvm.array<3 x i64>
  llvm.mlir.global internal @strict_gc_young(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_young_limit(67108864 : i64) : i64
  llvm.mlir.global internal constant @strict_gc_young_switch("STRICT_YOUNG\00") : !llvm.array<13 x i8>
  llvm.func @atoll(!llvm.ptr) -> i64
  llvm.mlir.global internal constant @strict_gc_sizes(dense<[16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 12288, 16384]> : tensor<32xi64>) : !llvm.array<32 x i64>
  llvm.mlir.global internal constant @strict_gc_classes(dense<[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19]> : tensor<65xi8>) : !llvm.array<65 x i8>
  llvm.mlir.global internal constant @strict_gc_out_of_memory("Strict: out of memory\0A\00") : !llvm.array<23 x i8>
  llvm.mlir.global internal constant @strict_dimensions("listsHaveDifferentDimensions\00") : !llvm.array<29 x i8>
  llvm.mlir.global internal constant @strict_downcast_prefix("Cannot downcast Text to Number for list: Text \22\00") : !llvm.array<48 x i8>
  llvm.mlir.global internal constant @strict_quote("\22\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_key_not_found("KeyNotFound\00") : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @strict_index_out_of_range("IndexOutOfRange\00") : !llvm.array<16 x i8>
  llvm.mlir.global internal constant @strict_duplicate_key("DuplicateKey\00") : !llvm.array<13 x i8>
  func.func private @strict_remainder(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %limit = arith.constant 9007199254740992.0 : f64
    %leftSize = llvm.intr.fabs(%left) : (f64) -> f64
    %rightSize = llvm.intr.fabs(%right) : (f64) -> f64
    %leftSmall = arith.cmpf olt, %leftSize, %limit : f64
    %rightSmall = arith.cmpf olt, %rightSize, %limit : f64
    %rightSet = arith.cmpf one, %right, %zero : f64
    %small = arith.andi %leftSmall, %rightSmall : i1
    %usable = arith.andi %small, %rightSet : i1
    cf.cond_br %usable, ^convert, ^fractional
  ^convert:
    %leftWhole = arith.fptosi %left : f64 to i64
    %rightWhole = arith.fptosi %right : f64 to i64
    %leftBack = arith.sitofp %leftWhole : i64 to f64
    %rightBack = arith.sitofp %rightWhole : i64 to f64
    %leftExact = arith.cmpf oeq, %leftBack, %left : f64
    %rightExact = arith.cmpf oeq, %rightBack, %right : f64
    %whole = arith.andi %leftExact, %rightExact : i1
    cf.cond_br %whole, ^integral, ^fractional
  ^integral:
    %rest = arith.remsi %leftWhole, %rightWhole : i64
    %restNumber = arith.sitofp %rest : i64 to f64
    %signed = llvm.intr.copysign(%restNumber, %left) : (f64, f64) -> f64
    return %signed : f64
  ^fractional:
    %fraction = arith.remf %left, %right : f64
    return %fraction : f64
  }
  llvm.mlir.global internal @strict_started(0.0 : f64) : f64
  func.func private @strict_process_elapsed(%self: f64) -> f64 {
    %now = func.call @strict_clock() : () -> f64
    %startedSlot = llvm.mlir.addressof @strict_started : !llvm.ptr
    %started = llvm.load %startedSlot : !llvm.ptr -> f64
    %elapsed = arith.subf %now, %started : f64
    return %elapsed : f64
  }
  llvm.mlir.global internal constant @strict_line_format("Line %s executed %.0f times in %.2f ms (self %.2f ms)\0A\00") : !llvm.array<55 x i8>
  llvm.mlir.global internal @strict_line_callees(0.0 : f64) : f64
  func.func private @strict_line_enter(%state: !llvm.ptr) {
    %none = arith.constant -1.0 : f64
    llvm.store %none, %state : f64, !llvm.ptr
    %now = func.call @strict_clock() : () -> f64
    %enteredSlot = llvm.getelementptr %state[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %now, %enteredSlot : f64, !llvm.ptr
    %calleesSlot = llvm.mlir.addressof @strict_line_callees : !llvm.ptr
    %callees = llvm.load %calleesSlot : !llvm.ptr -> f64
    %calleesAtEntrySlot = llvm.getelementptr %state[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %callees, %calleesAtEntrySlot : f64, !llvm.ptr
    return
  }
  func.func private @strict_line(%table: !llvm.ptr, %index: i64, %state: !llvm.ptr) {
    func.call @strict_line_stop(%table, %state) : (!llvm.ptr, !llvm.ptr) -> ()
    %now = func.call @strict_clock() : () -> f64
    %three = arith.constant 3 : i64
    %countIndex = arith.muli %index, %three : i64
    %countSlot = llvm.getelementptr %table[%countIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %count = llvm.load %countSlot : !llvm.ptr -> f64
    %one = arith.constant 1.0 : f64
    %counted = arith.addf %count, %one : f64
    llvm.store %counted, %countSlot : f64, !llvm.ptr
    %indexNumber = arith.sitofp %index : i64 to f64
    llvm.store %indexNumber, %state : f64, !llvm.ptr
    %startedSlot = llvm.getelementptr %state[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %now, %startedSlot : f64, !llvm.ptr
    %calleesSlot = llvm.mlir.addressof @strict_line_callees : !llvm.ptr
    %callees = llvm.load %calleesSlot : !llvm.ptr -> f64
    %calleesAtStartSlot = llvm.getelementptr %state[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %callees, %calleesAtStartSlot : f64, !llvm.ptr
    return
  }
  func.func private @strict_line_stop(%table: !llvm.ptr, %state: !llvm.ptr) {
    %previous = llvm.load %state : !llvm.ptr -> f64
    %zero = arith.constant 0.0 : f64
    %isCounted = arith.cmpf oge, %previous, %zero : f64
    cf.cond_br %isCounted, ^add, ^done
  ^add:
    %now = func.call @strict_clock() : () -> f64
    %startedSlot = llvm.getelementptr %state[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %started = llvm.load %startedSlot : !llvm.ptr -> f64
    %spent = arith.subf %now, %started : f64
    %calleesSlot = llvm.mlir.addressof @strict_line_callees : !llvm.ptr
    %callees = llvm.load %calleesSlot : !llvm.ptr -> f64
    %calleesAtStartSlot = llvm.getelementptr %state[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %calleesAtStart = llvm.load %calleesAtStartSlot : !llvm.ptr -> f64
    %inCallees = arith.subf %callees, %calleesAtStart : f64
    %own = arith.subf %spent, %inCallees : f64
    %previousIndex = arith.fptosi %previous : f64 to i64
    %three = arith.constant 3 : i64
    %countIndex = arith.muli %previousIndex, %three : i64
    %one = arith.constant 1 : i64
    %spentIndex = arith.addi %countIndex, %one : i64
    %spentSlot = llvm.getelementptr %table[%spentIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %spentBefore = llvm.load %spentSlot : !llvm.ptr -> f64
    %spentTotal = arith.addf %spentBefore, %spent : f64
    llvm.store %spentTotal, %spentSlot : f64, !llvm.ptr
    %ownIndex = arith.addi %spentIndex, %one : i64
    %ownSlot = llvm.getelementptr %table[%ownIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %ownBefore = llvm.load %ownSlot : !llvm.ptr -> f64
    %ownTotal = arith.addf %ownBefore, %own : f64
    llvm.store %ownTotal, %ownSlot : f64, !llvm.ptr
    %none = arith.constant -1.0 : f64
    llvm.store %none, %state : f64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_line_leave(%table: !llvm.ptr, %state: !llvm.ptr) {
    func.call @strict_line_stop(%table, %state) : (!llvm.ptr, !llvm.ptr) -> ()
    %now = func.call @strict_clock() : () -> f64
    %enteredSlot = llvm.getelementptr %state[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %entered = llvm.load %enteredSlot : !llvm.ptr -> f64
    %duration = arith.subf %now, %entered : f64
    %calleesAtEntrySlot = llvm.getelementptr %state[4] : (!llvm.ptr) -> !llvm.ptr, f64
    %calleesAtEntry = llvm.load %calleesAtEntrySlot : !llvm.ptr -> f64
    %callees = arith.addf %calleesAtEntry, %duration : f64
    %calleesSlot = llvm.mlir.addressof @strict_line_callees : !llvm.ptr
    llvm.store %callees, %calleesSlot : f64, !llvm.ptr
    return
  }
  func.func private @strict_line_print(%table: !llvm.ptr, %index: i64, %place: !llvm.ptr) {
    %three = arith.constant 3 : i64
    %countIndex = arith.muli %index, %three : i64
    %countSlot = llvm.getelementptr %table[%countIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %count = llvm.load %countSlot : !llvm.ptr -> f64
    %one = arith.constant 1 : i64
    %spentIndex = arith.addi %countIndex, %one : i64
    %spentSlot = llvm.getelementptr %table[%spentIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %spent = llvm.load %spentSlot : !llvm.ptr -> f64
    %ownIndex = arith.addi %spentIndex, %one : i64
    %ownSlot = llvm.getelementptr %table[%ownIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %own = llvm.load %ownSlot : !llvm.ptr -> f64
    %thousand = arith.constant 1000.0 : f64
    %spentMilliseconds = arith.mulf %spent, %thousand : f64
    %ownMilliseconds = arith.mulf %own, %thousand : f64
    %format = llvm.mlir.addressof @strict_line_format : !llvm.ptr
    %printed = llvm.call @printf(%format, %place, %count, %spentMilliseconds, %ownMilliseconds) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64, f64, f64) -> i32
    return
  }
  func.func private @strict_list_flat(%list: f64) -> f64 {
    %lengthNumber = func.call @strict_list_length(%list) : (f64) -> f64
    %length = arith.fptosi %lengthNumber : f64 to i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %slots = arith.addi %length, %one : i64
    %bytes = arith.muli %slots, %eight : i64
    %flat = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    llvm.store %lengthNumber, %flat : f64, !llvm.ptr
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %target = llvm.getelementptr %flat[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %itemBytes = arith.muli %length, %eight : i64
    %copied = llvm.call @memcpy(%target, %items, %itemBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    func.call @strict_gc_written_range(%target, %itemBytes) : (!llvm.ptr, i64) -> ()
    %result = func.call @strict_bits(%flat) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_foreign_failed(%result: f64) -> i1 {
    %bits = arith.bitcast %result : f64 to i64
    %shift = arith.constant 48 : i64
    %top = arith.shrui %bits, %shift : i64
    %failTag = arith.constant 32764 : i64
    %failed = arith.cmpi eq, %top, %failTag : i64
    return %failed : i1
  }
  func.func private @strict_foreign_error(%result: f64) -> f64 {
    %name = func.call @strict_ptr(%result) : (f64) -> !llvm.ptr
    %error = func.call @strict_error_named(%name) : (!llvm.ptr) -> f64
    return %error : f64
  }
  func.func private @strict_foreign_number(%result: f64) -> f64 {
    %failed = func.call @strict_foreign_failed(%result) : (f64) -> i1
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    %error = func.call @strict_foreign_error(%result) : (f64) -> f64
    return %error : f64
  ^done:
    %number = func.call @strict_canonical(%result) : (f64) -> f64
    return %number : f64
  }
  func.func private @strict_canonical(%value: f64) -> f64 attributes {passthrough = ["alwaysinline"]} {
    %isTagged = func.call @strict_is_failure(%value) : (f64) -> i1
    %canonical = arith.constant 0x7FF8000000000000 : f64
    %result = arith.select %isTagged, %canonical, %value : f64
    return %result : f64
  }
  func.func private @strict_foreign_text(%result: f64) -> f64 {
    %failed = func.call @strict_foreign_failed(%result) : (f64) -> i1
    cf.cond_br %failed, ^fail, ^copy
  ^fail:
    %error = func.call @strict_foreign_error(%result) : (f64) -> f64
    return %error : f64
  ^copy:
    %copied = func.call @strict_text_copy(%result) : (f64) -> f64
    return %copied : f64
  }
  func.func private @strict_foreign_numbers(%result: f64) -> f64 {
    %failed = func.call @strict_foreign_failed(%result) : (f64) -> i1
    cf.cond_br %failed, ^fail, ^copy
  ^fail:
    %error = func.call @strict_foreign_error(%result) : (f64) -> f64
    return %error : f64
  ^copy:
    %flat = func.call @strict_ptr(%result) : (f64) -> !llvm.ptr
    %resultBits = arith.bitcast %result : f64 to i64
    %tagShift = arith.constant 48 : i64
    %tag = arith.shrui %resultBits, %tagShift : i64
    %packedTag = arith.constant 32761 : i64
    %isPacked = arith.cmpi eq, %tag, %packedTag : i64
    cf.cond_br %isPacked, ^packed, ^plain
  ^packed:
    %packedList = func.call @strict_packed_list(%flat) : (!llvm.ptr) -> f64
    return %packedList : f64
  ^plain:
    %countNumber = llvm.load %flat : !llvm.ptr -> f64
    %list = func.call @strict_list_new(%countNumber, %countNumber) : (f64, f64) -> f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %source = llvm.getelementptr %flat[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %count = arith.fptosi %countNumber : f64 to i64
    %eight = arith.constant 8 : i64
    %bytes = arith.muli %count, %eight : i64
    %copied = llvm.call @memcpy(%items, %source, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^canonical, ^done
  ^canonical:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %number = func.call @strict_canonical(%value) : (f64) -> f64
    llvm.store %number, %slot : f64, !llvm.ptr
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next : i64)
  ^done:
    return %list : f64
  }
  func.func private @strict_packed_list(%packed: !llvm.ptr) -> f64 {
    %none = arith.constant 0 : i64
    %eight = arith.constant 8.0 : f64
    %count = llvm.load %packed : !llvm.ptr -> f64
    %bitsSlot = llvm.getelementptr %packed[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %bits = llvm.load %bitsSlot : !llvm.ptr -> f64
    %elements = arith.fptosi %count : f64 to i64
    %wholeBits = arith.fptosi %bits : f64 to i64
    %list = func.call @strict_packed_new(%elements, %wholeBits) : (i64, i64) -> f64
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    %bytes = func.call @strict_packed_bytes(%elements, %wholeBits) : (i64, i64) -> i64
    %source = llvm.getelementptr %packed[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %copied = llvm.call @memcpy(%data, %source, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %half = arith.constant 16.0 : f64
    %isHalf = arith.cmpf oeq, %bits, %half : f64
    %isQuantized = arith.cmpf ole, %bits, %eight : f64
    %floats = arith.select %isQuantized, %none, %elements : i64
    func.call @strict_packed_canonical(%data, %floats, %isHalf) : (!llvm.ptr, i64, i1) -> ()
    return %list : f64
  }
  func.func private @strict_packed_new(%elements: i64, %wholeBits: i64) -> f64 {
    %bytes = func.call @strict_packed_bytes(%elements, %wholeBits) : (i64, i64) -> i64
    %data = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %dataBits = func.call @strict_bits(%data) : (!llvm.ptr) -> f64
    %zero = arith.constant 0.0 : f64
    %list = func.call @strict_packed_view(%elements, %wholeBits, %dataBits, %zero) : (i64, i64, f64, f64) -> f64
    return %list : f64
  }
  // a packed list keeps its data in holder[3], a list computed on the GPU its device address in holder[4] (writes of
  // growable lists), its data is 0 until the CPU reads it (strict_list_data downloads it then)
  func.func private @strict_packed_view(%elements: i64, %wholeBits: i64, %dataBits: f64, %device: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %blockSize = arith.constant 56 : i64
    %count = arith.sitofp %elements : i64 to f64
    %bits = arith.sitofp %wholeBits : i64 to f64
    %view = func.call @strict_alloc(%blockSize) : (i64) -> !llvm.ptr
    %holder = llvm.getelementptr %view[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %zero, %holder : f64, !llvm.ptr
    %used = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %count, %used : f64, !llvm.ptr
    %negative = arith.negf %bits : f64
    %capacity = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %negative, %capacity : f64, !llvm.ptr
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %dataBits, %dataSlot : f64, !llvm.ptr
    %writes = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %device, %writes : f64, !llvm.ptr
    llvm.store %count, %view : f64, !llvm.ptr
    %holderBits = func.call @strict_bits(%holder) : (!llvm.ptr) -> f64
    %viewHolder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderBits, %viewHolder : f64, !llvm.ptr
    %result = func.call @strict_bits(%view) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_packed_data(%count: i64, %bits: i64) -> i64 attributes {passthrough = ["alwaysinline"]} {
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
  func.func private @strict_packed_bytes(%count: i64, %bits: i64) -> i64 attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %six = arith.constant 6 : i64
    %four = arith.constant 4 : i64
    %thirtyOne = arith.constant 31 : i64
    %five = arith.constant 5 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %aligned = func.call @strict_packed_data(%count, %bits) : (i64, i64) -> i64
    %rounded = arith.addi %count, %thirtyOne : i64
    %blocks = arith.shrui %rounded, %five : i64
    %scaleBytes = arith.shli %blocks, %two : i64
    %quantizedBytes = arith.addi %aligned, %scaleBytes : i64
    %spare = arith.constant 255 : i64
    %roundedBlocks = arith.addi %count, %spare : i64
    %sixBlocks = arith.shrui %roundedBlocks, %eight : i64
    %halfBytes = arith.shli %sixBlocks, %one : i64
    %paddedHalves = arith.addi %halfBytes, %three : i64
    %alignMask = arith.constant -4 : i64
    %halvesBytes = arith.andi %paddedHalves, %alignMask : i64
    %sixBytes = arith.addi %aligned, %halvesBytes : i64
    %fourBytes = arith.addi %aligned, %none : i64
    %cells = arith.muli %count, %bits : i64
    %plainBytes = arith.divui %cells, %eight : i64
    %isQuantized = arith.cmpi eq, %bits, %eight : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %orFour = arith.select %isFour, %fourBytes, %plainBytes : i64
    %orSix = arith.select %isSix, %sixBytes, %orFour : i64
    %bytes = arith.select %isQuantized, %quantizedBytes, %orSix : i64
    return %bytes : i64
  }
  func.func private @strict_packed_scales(%data: !llvm.ptr, %count: i64, %bits: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %aligned = func.call @strict_packed_data(%count, %bits) : (i64, i64) -> i64
    %scales = llvm.getelementptr %data[%aligned] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    return %scales : !llvm.ptr
  }
  func.func private @strict_packed_canonical(%data: !llvm.ptr, %count: i64, %isHalf: i1) {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %singleExponent = arith.constant 2139095040 : i32
    %singleNaN = arith.constant 2143289344 : i32
    %halfExponent = arith.constant 31744 : i16
    %halfNaN = arith.constant 32256 : i16
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^element, ^done
  ^element:
    cf.cond_br %isHalf, ^half, ^single
  ^half:
    %halfSlot = llvm.getelementptr %data[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %halfBits = llvm.load %halfSlot : !llvm.ptr -> i16
    %halfSigned = arith.constant 32767 : i16
    %halfMagnitude = arith.andi %halfBits, %halfSigned : i16
    %halfIsNaN = arith.cmpi ugt, %halfMagnitude, %halfExponent : i16
    %halfFixed = arith.select %halfIsNaN, %halfNaN, %halfBits : i16
    llvm.store %halfFixed, %halfSlot : i16, !llvm.ptr
    cf.br ^next
  ^single:
    %singleSlot = llvm.getelementptr %data[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %singleBits = llvm.load %singleSlot : !llvm.ptr -> i32
    %singleSigned = arith.constant 2147483647 : i32
    %singleMagnitude = arith.andi %singleBits, %singleSigned : i32
    %singleIsNaN = arith.cmpi ugt, %singleMagnitude, %singleExponent : i32
    %singleFixed = arith.select %singleIsNaN, %singleNaN, %singleBits : i32
    llvm.store %singleFixed, %singleSlot : i32, !llvm.ptr
    cf.br ^next
  ^next:
    %following = arith.addi %index, %step : i64
    cf.br ^check(%following : i64)
  ^done:
    return
  }
  func.func private @strict_foreign_rows(%result: f64) -> f64 {
    %failed = func.call @strict_foreign_failed(%result) : (f64) -> i1
    cf.cond_br %failed, ^fail, ^rows
  ^fail:
    %error = func.call @strict_foreign_error(%result) : (f64) -> f64
    return %error : f64
  ^rows:
    %flat = func.call @strict_ptr(%result) : (f64) -> !llvm.ptr
    %countNumber = llvm.load %flat : !llvm.ptr -> f64
    %list = func.call @strict_list_new(%countNumber, %countNumber) : (f64, f64) -> f64
    %count = arith.fptosi %countNumber : f64 to i64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^row, ^done
  ^row:
    %position = arith.addi %index, %step : i64
    %slot = llvm.getelementptr %flat[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %rowBits = llvm.load %slot : !llvm.ptr -> f64
    %row = func.call @strict_foreign_numbers(%rowBits) : (f64) -> f64
    %isError = func.call @strict_is_failure(%row) : (f64) -> i1
    cf.cond_br %isError, ^failedRow, ^stored
  ^failedRow:
    return %row : f64
  ^stored:
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %target = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %row, %target : f64, !llvm.ptr
    func.call @strict_gc_written(%target) : (!llvm.ptr) -> ()
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next : i64)
  ^done:
    return %list : f64
  }
  llvm.mlir.global internal constant @strict_library_no_failure("\00") : !llvm.array<1 x i8>
  func.func private @strict_library_textslot(%library: f64, %text: f64) -> f64 {
    return %text : f64
  }
  func.func private @strict_library_numbersslot(%library: f64, %numbers: f64) -> f64 {
    %flat = func.call @strict_list_flat(%numbers) : (f64) -> f64
    return %flat : f64
  }
  func.func private @strict_library_textresult(%library: f64, %bits: f64) -> f64 {
    %copied = func.call @strict_foreign_text(%bits) : (f64) -> f64
    return %copied : f64
  }
  func.func private @strict_library_numbersresult(%library: f64, %bits: f64) -> f64 {
    %copied = func.call @strict_foreign_numbers(%bits) : (f64) -> f64
    return %copied : f64
  }
  func.func private @strict_library_failure(%library: f64, %bits: f64) -> f64 {
    %failed = func.call @strict_foreign_failed(%bits) : (f64) -> i1
    cf.cond_br %failed, ^name, ^none
  ^name:
    %name = func.call @strict_text_copy(%bits) : (f64) -> f64
    return %name : f64
  ^none:
    %empty = llvm.mlir.addressof @strict_library_no_failure : !llvm.ptr
    %emptyBits = func.call @strict_bits(%empty) : (!llvm.ptr) -> f64
    %copied = func.call @strict_text_copy(%emptyBits) : (f64) -> f64
    return %copied : f64
  }
  llvm.mlir.global internal constant @strict_library_missing("LibraryNotFound\00") : !llvm.array<16 x i8>
  llvm.mlir.global internal constant @strict_library_symbol_missing("LibrarySymbolNotFound\00") : !llvm.array<22 x i8>
  llvm.mlir.global internal constant @strict_library_arguments("LibraryCallNeedsOneToFiveNumbers\00") : !llvm.array<33 x i8>
  func.func private @strict_library_invoke(%function: !llvm.ptr, %arguments: f64) -> f64 {
    %items = func.call @strict_list_items(%arguments) : (f64) -> !llvm.ptr
    %countNumber = func.call @strict_list_length(%arguments) : (f64) -> f64
    %count = arith.fptosi %countNumber : f64 to i64
    cf.switch %count : i64, [
      default: ^unsupported,
      1: ^one,
      2: ^two,
      3: ^three,
      4: ^four,
      5: ^five
    ]
  ^one:
    %a10 = llvm.load %items : !llvm.ptr -> f64
    %r1 = llvm.call %function(%a10) : !llvm.ptr, (f64) -> f64
    return %r1 : f64
  ^two:
    %a20 = llvm.load %items : !llvm.ptr -> f64
    %p21 = llvm.getelementptr %items[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %a21 = llvm.load %p21 : !llvm.ptr -> f64
    %r2 = llvm.call %function(%a20, %a21) : !llvm.ptr, (f64, f64) -> f64
    return %r2 : f64
  ^three:
    %a30 = llvm.load %items : !llvm.ptr -> f64
    %p31 = llvm.getelementptr %items[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %a31 = llvm.load %p31 : !llvm.ptr -> f64
    %p32 = llvm.getelementptr %items[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %a32 = llvm.load %p32 : !llvm.ptr -> f64
    %r3 = llvm.call %function(%a30, %a31, %a32) : !llvm.ptr, (f64, f64, f64) -> f64
    return %r3 : f64
  ^four:
    %a40 = llvm.load %items : !llvm.ptr -> f64
    %p41 = llvm.getelementptr %items[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %a41 = llvm.load %p41 : !llvm.ptr -> f64
    %p42 = llvm.getelementptr %items[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %a42 = llvm.load %p42 : !llvm.ptr -> f64
    %p43 = llvm.getelementptr %items[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %a43 = llvm.load %p43 : !llvm.ptr -> f64
    %r4 = llvm.call %function(%a40, %a41, %a42, %a43) : !llvm.ptr, (f64, f64, f64, f64) -> f64
    return %r4 : f64
  ^five:
    %a50 = llvm.load %items : !llvm.ptr -> f64
    %p51 = llvm.getelementptr %items[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %a51 = llvm.load %p51 : !llvm.ptr -> f64
    %p52 = llvm.getelementptr %items[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %a52 = llvm.load %p52 : !llvm.ptr -> f64
    %p53 = llvm.getelementptr %items[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %a53 = llvm.load %p53 : !llvm.ptr -> f64
    %p54 = llvm.getelementptr %items[4] : (!llvm.ptr) -> !llvm.ptr, f64
    %a54 = llvm.load %p54 : !llvm.ptr -> f64
    %r5 = llvm.call %function(%a50, %a51, %a52, %a53, %a54) : !llvm.ptr, (f64, f64, f64, f64, f64) -> f64
    return %r5 : f64
  ^unsupported:
    %argumentsMessage = llvm.mlir.addressof @strict_library_arguments : !llvm.ptr
    %argumentsError = func.call @strict_error_named(%argumentsMessage) : (!llvm.ptr) -> f64
    return %argumentsError : f64
  }
  func.func private @strict_gc_init(%stack: !llvm.ptr) {
    func.call @strict_guard() : () -> ()
    %startedNow = func.call @strict_clock() : () -> f64
    %startedSlot = llvm.mlir.addressof @strict_started : !llvm.ptr
    llvm.store %startedNow, %startedSlot : f64, !llvm.ptr
    %tableBytes = arith.constant 8388608 : i64
    %cardBytes = arith.constant 134217728 : i64
    %heapBytes = arith.constant 68719476736 : i64
    %pageSize = arith.constant 65536 : i64
    %pageRest = arith.constant 65535 : i64
    %pageMask = arith.constant -65536 : i64
    %tables = arith.addi %tableBytes, %cardBytes : i64
    %slack = arith.addi %tables, %pageSize : i64
    %total = arith.addi %heapBytes, %slack : i64
    %reserved = func.call @strict_reserve(%total) : (i64) -> !llvm.ptr
    %start = llvm.ptrtoint %reserved : !llvm.ptr to i64
    %padded = arith.addi %start, %pageRest : i64
    %table = arith.andi %padded, %pageMask : i64
    %tablePointer = llvm.inttoptr %table : i64 to !llvm.ptr
    func.call @strict_commit(%tablePointer, %tableBytes) : (!llvm.ptr, i64) -> ()
    %cards = arith.addi %table, %tableBytes : i64
    %cardsSlot = llvm.mlir.addressof @strict_gc_cards : !llvm.ptr
    llvm.store %cards, %cardsSlot : i64, !llvm.ptr
    %base = arith.addi %cards, %cardBytes : i64
    %limit = arith.addi %base, %heapBytes : i64
    %tableSlot = llvm.mlir.addressof @strict_gc_table : !llvm.ptr
    llvm.store %table, %tableSlot : i64, !llvm.ptr
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    llvm.store %base, %baseSlot : i64, !llvm.ptr
    %topSlot = llvm.mlir.addressof @strict_gc_top : !llvm.ptr
    llvm.store %base, %topSlot : i64, !llvm.ptr
    %committedSlot = llvm.mlir.addressof @strict_gc_committed : !llvm.ptr
    llvm.store %base, %committedSlot : i64, !llvm.ptr
    %limitSlot = llvm.mlir.addressof @strict_gc_limit : !llvm.ptr
    llvm.store %limit, %limitSlot : i64, !llvm.ptr
    %emptyHolder = llvm.mlir.addressof @strict_empty_holder : !llvm.ptr
    %emptyHolderBits = func.call @strict_bits(%emptyHolder) : (!llvm.ptr) -> f64
    llvm.store %emptyHolderBits, %emptyHolder : f64, !llvm.ptr
    %emptyView = llvm.mlir.addressof @strict_empty_view : !llvm.ptr
    %emptyViewHolder = llvm.getelementptr %emptyView[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %emptyHolderBits, %emptyViewHolder : f64, !llvm.ptr
    func.call @strict_characters_init() : () -> ()
    %stackBits = llvm.ptrtoint %stack : !llvm.ptr to i64
    %stackSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    llvm.store %stackBits, %stackSlot : i64, !llvm.ptr
    %youngName = llvm.mlir.addressof @strict_gc_young_switch : !llvm.ptr
    %youngSetting = llvm.call @getenv(%youngName) : (!llvm.ptr) -> !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %youngUnset = llvm.icmp "eq" %youngSetting, %unset : !llvm.ptr
    cf.cond_br %youngUnset, ^done, ^given
  ^given:
    %firstDigit = llvm.load %youngSetting : !llvm.ptr -> i8
    %terminator = arith.constant 0 : i8
    %youngEmpty = arith.cmpi eq, %firstDigit, %terminator : i8
    cf.cond_br %youngEmpty, ^done, ^young
  ^young:
    %megabytes = llvm.call @atoll(%youngSetting) : (!llvm.ptr) -> i64
    %megabyteShift = arith.constant 20 : i64
    %youngLimit = arith.shli %megabytes, %megabyteShift : i64
    %youngLimitSlot = llvm.mlir.addressof @strict_gc_young_limit : !llvm.ptr
    llvm.store %youngLimit, %youngLimitSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_fail() {
    %message = llvm.mlir.addressof @strict_gc_out_of_memory : !llvm.ptr
    %printed = llvm.call @printf(%message) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %code = arith.constant 1 : i32
    llvm.call @exit(%code) : (i32) -> ()
    return
  }
  func.func private @strict_alloc(%size: i64) -> !llvm.ptr {
    %scanned = arith.constant 0 : i64
    %object = func.call @strict_gc_allocate(%size, %scanned) : (i64, i64) -> !llvm.ptr
    return %object : !llvm.ptr
  }
  func.func private @strict_alloc_text(%size: i64) -> !llvm.ptr {
    %unscanned = arith.constant 32 : i64
    %object = func.call @strict_gc_allocate(%size, %unscanned) : (i64, i64) -> !llvm.ptr
    return %object : !llvm.ptr
  }
  llvm.mlir.global internal @strict_pool_busy(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_spawned(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_workers(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_job(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_body(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_context(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_count(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_chunk(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_next(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_done(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_failure(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_sleepers(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_mutex(0 : i64) : i64
  func.func private @strict_parallel(%body: (!llvm.ptr, i64, i64) -> f64, %context: !llvm.ptr, %count: i64) -> f64 {
    %function = builtin.unrealized_conversion_cast %body : (!llvm.ptr, i64, i64) -> f64 to !llvm.ptr
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %isBusy = arith.cmpi ne, %busy, %zero : i64
    cf.cond_br %isBusy, ^serial, ^pool
  ^pool:
    %workers = func.call @strict_pool_started() : () -> i64
    %alone = arith.cmpi eq, %workers, %zero : i64
    cf.cond_br %alone, ^serial, ^shared
  ^serial:
    %serialResult = llvm.call %function(%context, %zero, %count) : !llvm.ptr, (!llvm.ptr, i64, i64) -> f64
    return %serialResult : f64
  ^shared:
    %bodyBits = llvm.ptrtoint %function : !llvm.ptr to i64
    %bodySlot = llvm.mlir.addressof @strict_pool_body : !llvm.ptr
    llvm.store %bodyBits, %bodySlot : i64, !llvm.ptr
    %contextBits = llvm.ptrtoint %context : !llvm.ptr to i64
    %contextSlot = llvm.mlir.addressof @strict_pool_context : !llvm.ptr
    llvm.store %contextBits, %contextSlot : i64, !llvm.ptr
    %countSlot = llvm.mlir.addressof @strict_pool_count : !llvm.ptr
    llvm.store %count, %countSlot : i64, !llvm.ptr
    %threads = arith.addi %workers, %one : i64
    %four = arith.constant 4 : i64
    %pieces = arith.muli %threads, %four : i64
    %share = arith.divsi %count, %pieces : i64
    %chunk = arith.maxsi %share, %one : i64
    %chunkSlot = llvm.mlir.addressof @strict_pool_chunk : !llvm.ptr
    llvm.store %chunk, %chunkSlot : i64, !llvm.ptr
    %doneSlot = llvm.mlir.addressof @strict_pool_done : !llvm.ptr
    llvm.store %zero, %doneSlot : i64, !llvm.ptr
    %failureSlot = llvm.mlir.addressof @strict_pool_failure : !llvm.ptr
    llvm.store %zero, %failureSlot : i64, !llvm.ptr
    %jobSlot = llvm.mlir.addressof @strict_pool_job : !llvm.ptr
    %previous = llvm.load %jobSlot : !llvm.ptr -> i64
    %job = arith.addi %previous, %one : i64
    %mask = arith.constant 4294967295 : i64
    %tag = arith.andi %job, %mask : i64
    %shift = arith.constant 32 : i64
    %first = arith.shli %tag, %shift : i64
    %nextSlot = llvm.mlir.addressof @strict_pool_next : !llvm.ptr
    llvm.store %first, %nextSlot atomic release {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %one, %busySlot atomic release {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %job, %jobSlot atomic seq_cst {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.call @strict_pool_wake() : () -> ()
    llvm.call @strict_pool_run(%job) : (i64) -> ()
    cf.br ^wait
  ^wait:
    %done = llvm.load %doneSlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %finished = arith.cmpi sge, %done, %count : i64
    cf.cond_br %finished, ^finish, ^spin
  ^spin:
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    cf.br ^wait
  ^finish:
    llvm.store %zero, %busySlot atomic release {alignment = 8 : i64} : i64, !llvm.ptr
    %failureBits = llvm.load %failureSlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %failure = arith.bitcast %failureBits : i64 to f64
    return %failure : f64
  }
  func.func private @strict_pool_started() -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %spawnedSlot = llvm.mlir.addressof @strict_pool_spawned : !llvm.ptr
    %workersSlot = llvm.mlir.addressof @strict_pool_workers : !llvm.ptr
    %spawned = llvm.load %spawnedSlot : !llvm.ptr -> i64
    %fresh = arith.cmpi eq, %spawned, %zero : i64
    cf.cond_br %fresh, ^spawn, ^known
  ^spawn:
    %count = func.call @strict_pool_spawn() : () -> i64
    llvm.store %count, %workersSlot : i64, !llvm.ptr
    llvm.store %one, %spawnedSlot : i64, !llvm.ptr
    return %count : i64
  ^known:
    %workers = llvm.load %workersSlot : !llvm.ptr -> i64
    return %workers : i64
  }
  llvm.func @strict_pool_run(%job: i64) {
    %nextSlot = llvm.mlir.addressof @strict_pool_next : !llvm.ptr
    %countSlot = llvm.mlir.addressof @strict_pool_count : !llvm.ptr
    %chunkSlot = llvm.mlir.addressof @strict_pool_chunk : !llvm.ptr
    %doneSlot = llvm.mlir.addressof @strict_pool_done : !llvm.ptr
    %mask = llvm.mlir.constant(4294967295 : i64) : i64
    %shift = llvm.mlir.constant(32 : i64) : i64
    %tag = llvm.and %job, %mask : i64
    llvm.br ^claim
  ^claim:
    %packed = llvm.load %nextSlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %owner = llvm.lshr %packed, %shift : i64
    %isCurrent = llvm.icmp "eq" %owner, %tag : i64
    llvm.cond_br %isCurrent, ^check, ^leave
  ^check:
    %start = llvm.and %packed, %mask : i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %more = llvm.icmp "slt" %start, %count : i64
    llvm.cond_br %more, ^try, ^leave
  ^try:
    %chunk = llvm.load %chunkSlot : !llvm.ptr -> i64
    %proposed = llvm.add %start, %chunk : i64
    %over = llvm.icmp "sgt" %proposed, %count : i64
    %end = llvm.select %over, %count, %proposed : i1, i64
    %size = llvm.sub %end, %start : i64
    %claimed = llvm.add %packed, %size : i64
    %pair = llvm.cmpxchg %nextSlot, %packed, %claimed acq_rel monotonic : !llvm.ptr, i64
    %won = llvm.extractvalue %pair[1] : !llvm.struct<(i64, i1)>
    llvm.cond_br %won, ^work, ^claim
  ^work:
    %result = llvm.call @strict_pool_chunk_run(%start, %end) : (i64, i64) -> i1
    %finished = llvm.atomicrmw add %doneSlot, %size acq_rel : !llvm.ptr, i64
    llvm.cond_br %result, ^close, ^claim
  ^close:
    %open = llvm.load %nextSlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %openOwner = llvm.lshr %open, %shift : i64
    %stillCurrent = llvm.icmp "eq" %openOwner, %tag : i64
    %index = llvm.and %open, %mask : i64
    %total = llvm.load %countSlot : !llvm.ptr -> i64
    %remaining = llvm.icmp "slt" %index, %total : i64
    %closable = llvm.and %stillCurrent, %remaining : i1
    llvm.cond_br %closable, ^skip, ^leave
  ^skip:
    %rest = llvm.sub %total, %index : i64
    %closed = llvm.add %open, %rest : i64
    %closing = llvm.cmpxchg %nextSlot, %open, %closed acq_rel monotonic : !llvm.ptr, i64
    %skipped = llvm.extractvalue %closing[1] : !llvm.struct<(i64, i1)>
    llvm.cond_br %skipped, ^skippedRest, ^close
  ^skippedRest:
    %counted = llvm.atomicrmw add %doneSlot, %rest acq_rel : !llvm.ptr, i64
    llvm.return
  ^leave:
    llvm.return
  }
  llvm.func @strict_pool_chunk_run(%start: i64, %end: i64) -> i1 {
    %bodySlot = llvm.mlir.addressof @strict_pool_body : !llvm.ptr
    %bodyBits = llvm.load %bodySlot : !llvm.ptr -> i64
    %function = llvm.inttoptr %bodyBits : i64 to !llvm.ptr
    %contextSlot = llvm.mlir.addressof @strict_pool_context : !llvm.ptr
    %contextBits = llvm.load %contextSlot : !llvm.ptr -> i64
    %context = llvm.inttoptr %contextBits : i64 to !llvm.ptr
    %result = llvm.call %function(%context, %start, %end) : !llvm.ptr, (!llvm.ptr, i64, i64) -> f64
    %bits = llvm.bitcast %result : f64 to i64
    %shift = llvm.mlir.constant(48 : i64) : i64
    %errorTag = llvm.mlir.constant(32762 : i64) : i64
    %valueTag = llvm.mlir.constant(32763 : i64) : i64
    %tag = llvm.lshr %bits, %shift : i64
    %isError = llvm.icmp "eq" %tag, %errorTag : i64
    %isValue = llvm.icmp "eq" %tag, %valueTag : i64
    %failed = llvm.or %isError, %isValue : i1
    llvm.cond_br %failed, ^failed, ^passed
  ^failed:
    %failureSlot = llvm.mlir.addressof @strict_pool_failure : !llvm.ptr
    %none = llvm.mlir.constant(0 : i64) : i64
    %pair = llvm.cmpxchg %failureSlot, %none, %bits acq_rel monotonic : !llvm.ptr, i64
    llvm.return %failed : i1
  ^passed:
    llvm.return %failed : i1
  }
  llvm.func @strict_pool_serve() {
    %jobSlot = llvm.mlir.addressof @strict_pool_job : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %limit = llvm.mlir.constant(200000 : i64) : i64
    llvm.br ^wait(%zero, %zero : i64, i64)
  ^wait(%seen: i64, %spins: i64):
    %current = llvm.load %jobSlot atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
    %changed = llvm.icmp "ne" %current, %seen : i64
    llvm.cond_br %changed, ^work(%current : i64), ^idle
  ^idle:
    %tired = llvm.icmp "sge" %spins, %limit : i64
    llvm.cond_br %tired, ^sleep, ^spin
  ^spin:
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    %counted = llvm.add %spins, %one : i64
    llvm.br ^wait(%seen, %counted : i64, i64)
  ^sleep:
    llvm.call @strict_pool_sleep(%seen) : (i64) -> ()
    llvm.br ^wait(%seen, %zero : i64, i64)
  ^work(%job: i64):
    llvm.call @strict_pool_run(%job) : (i64) -> ()
    llvm.br ^wait(%job, %zero : i64, i64)
  }
  llvm.func @strict_gc_lock() {
    %slot = llvm.mlir.addressof @strict_gc_mutex : !llvm.ptr
    llvm.call @strict_spin_lock(%slot) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @strict_spin_lock(%slot: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^try
  ^try:
    %pair = llvm.cmpxchg %slot, %zero, %one acquire monotonic : !llvm.ptr, i64
    %taken = llvm.extractvalue %pair[1] : !llvm.struct<(i64, i1)>
    llvm.cond_br %taken, ^done, ^spin
  ^spin:
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    llvm.br ^try
  ^done:
    llvm.return
  }
  llvm.func @strict_gc_unlock() {
    %slot = llvm.mlir.addressof @strict_gc_mutex : !llvm.ptr
    llvm.call @strict_spin_unlock(%slot) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @strict_spin_unlock(%slot: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    llvm.store %zero, %slot atomic release {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.return
  }
  func.func private @strict_gc_allocate(%size: i64, %list: i64) -> !llvm.ptr {
    %zero = arith.constant 0 : i64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %shared = arith.cmpi ne, %busy, %zero : i64
    cf.cond_br %shared, ^locked, ^alone
  ^alone:
    %object = func.call @strict_gc_allocate_now(%size, %list) : (i64, i64) -> !llvm.ptr
    return %object : !llvm.ptr
  ^locked:
    llvm.call @strict_gc_lock() : () -> ()
    %lockedObject = func.call @strict_gc_allocate_now(%size, %list) : (i64, i64) -> !llvm.ptr
    llvm.call @strict_gc_unlock() : () -> ()
    return %lockedObject : !llvm.ptr
  }
  func.func private @strict_gc_allocate_now(%size: i64, %list: i64) -> !llvm.ptr {
    %smallLimit = arith.constant 1024 : i64
    %isTable = arith.cmpi ule, %size, %smallLimit : i64
    cf.cond_br %isTable, ^table, ^larger
  ^table:
    %fifteen = arith.constant 15 : i64
    %four = arith.constant 4 : i64
    %padded = arith.addi %size, %fifteen : i64
    %slot = arith.shrui %padded, %four : i64
    %classes = llvm.mlir.addressof @strict_gc_classes : !llvm.ptr
    %entry = llvm.getelementptr %classes[0, %slot] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<65 x i8>
    %classByte = llvm.load %entry : !llvm.ptr -> i8
    %tableClass = arith.extui %classByte : i8 to i64
    %class = arith.addi %tableClass, %list : i64
    %object = func.call @strict_gc_small(%class) : (i64) -> !llvm.ptr
    return %object : !llvm.ptr
  ^larger:
    %maxSmall = arith.constant 16384 : i64
    %isLarge = arith.cmpi ugt, %size, %maxSmall : i64
    %firstLarger = arith.constant 20 : i64
    cf.cond_br %isLarge, ^large, ^search(%firstLarger : i64)
  ^search(%candidate: i64):
    %sizes = llvm.mlir.addressof @strict_gc_sizes : !llvm.ptr
    %sizeSlot = llvm.getelementptr %sizes[0, %candidate] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<32 x i64>
    %candidateSize = llvm.load %sizeSlot : !llvm.ptr -> i64
    %fits = arith.cmpi uge, %candidateSize, %size : i64
    %one = arith.constant 1 : i64
    %following = arith.addi %candidate, %one : i64
    cf.cond_br %fits, ^found(%candidate : i64), ^search(%following : i64)
  ^found(%chosen: i64):
    %chosenList = arith.addi %chosen, %list : i64
    %medium = func.call @strict_gc_small(%chosenList) : (i64) -> !llvm.ptr
    return %medium : !llvm.ptr
  ^large:
    %big = func.call @strict_gc_large_alloc(%size, %list) : (i64, i64) -> !llvm.ptr
    return %big : !llvm.ptr
  }
  func.func private @strict_gc_count(%bytes: i64) {
    func.call @strict_gc_add(%bytes) : (i64) -> ()
    %due = func.call @strict_gc_due() : () -> i1
    cf.cond_br %due, ^collect, ^done
  ^collect:
    func.call @strict_gc_collect() : () -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_add(%bytes: i64) {
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    %allocated = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %counted = arith.addi %allocated, %bytes : i64
    llvm.store %counted, %allocatedSlot : i64, !llvm.ptr
    %youngSlot = llvm.mlir.addressof @strict_gc_young : !llvm.ptr
    %young = llvm.load %youngSlot : !llvm.ptr -> i64
    %grown = arith.addi %young, %bytes : i64
    llvm.store %grown, %youngSlot : i64, !llvm.ptr
    return
  }
  // a full collection is due when 8 times the bytes live after the last one were allocated since, a young one after
  // the young limit (64 MB, STRICT_YOUNG sets it in MB, 0 has none)
  func.func private @strict_gc_due() -> i1 {
    %none = arith.constant 0 : i64
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    %allocated = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %thresholdSlot = llvm.mlir.addressof @strict_gc_threshold : !llvm.ptr
    %threshold = llvm.load %thresholdSlot : !llvm.ptr -> i64
    %full = arith.cmpi ugt, %allocated, %threshold : i64
    %youngSlot = llvm.mlir.addressof @strict_gc_young : !llvm.ptr
    %young = llvm.load %youngSlot : !llvm.ptr -> i64
    %limitSlot = llvm.mlir.addressof @strict_gc_young_limit : !llvm.ptr
    %limit = llvm.load %limitSlot : !llvm.ptr -> i64
    %crowded = arith.cmpi ugt, %young, %limit : i64
    %enabled = arith.cmpi ne, %limit, %none : i64
    %minor = arith.andi %crowded, %enabled : i1
    %due = arith.ori %full, %minor : i1
    return %due : i1
  }
  func.func private @strict_gc_small(%class: i64) -> !llvm.ptr {
    %sizeMask = arith.constant 31 : i64
    %sizeClass = arith.andi %class, %sizeMask : i64
    %sizes = llvm.mlir.addressof @strict_gc_sizes : !llvm.ptr
    %sizeSlot = llvm.getelementptr %sizes[0, %sizeClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<32 x i64>
    %objectSize = llvm.load %sizeSlot : !llvm.ptr -> i64
    %cursors = llvm.mlir.addressof @strict_gc_cursors : !llvm.ptr
    %cursorSlot = llvm.getelementptr %cursors[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %ends = llvm.mlir.addressof @strict_gc_ends : !llvm.ptr
    %endSlot = llvm.getelementptr %ends[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    cf.br ^try
  ^try:
    %cursor = llvm.load %cursorSlot : !llvm.ptr -> i64
    %end = llvm.load %endSlot : !llvm.ptr -> i64
    %next = arith.addi %cursor, %objectSize : i64
    %fits = arith.cmpi ule, %next, %end : i64
    cf.cond_br %fits, ^take, ^refill
  ^take:
    llvm.store %next, %cursorSlot : i64, !llvm.ptr
    %pointer = llvm.inttoptr %cursor : i64 to !llvm.ptr
    return %pointer : !llvm.ptr
  ^refill:
    func.call @strict_gc_next_span(%class) : (i64) -> ()
    cf.br ^try
  }
  func.func private @strict_gc_link(%object: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %flag = arith.constant -9223372036854775808 : i64
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %offset = arith.subi %object, %base : i64
    %shifted = arith.addi %offset, %one : i64
    %encoded = arith.ori %shifted, %flag : i64
    %isNull = arith.cmpi eq, %object, %zero : i64
    %result = arith.select %isNull, %zero, %encoded : i64
    return %result : i64
  }
  func.func private @strict_gc_unlink(%link: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 9223372036854775807 : i64
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %offset = arith.andi %link, %mask : i64
    %address = arith.addi %base, %offset : i64
    %decoded = arith.subi %address, %one : i64
    %isNull = arith.cmpi eq, %link, %zero : i64
    %result = arith.select %isNull, %zero, %decoded : i64
    return %result : i64
  }
  func.func private @strict_gc_next_span(%class: i64) {
    %zero = arith.constant 0 : i64
    %clear = arith.constant 0 : i32
    %due = func.call @strict_gc_due() : () -> i1
    %spans = llvm.mlir.addressof @strict_gc_spans : !llvm.ptr
    %spanSlot = llvm.getelementptr %spans[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %sweeps = llvm.mlir.addressof @strict_gc_sweeps : !llvm.ptr
    %sweepSlot = llvm.getelementptr %sweeps[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    cf.cond_br %due, ^collect, ^search
  ^collect:
    func.call @strict_gc_collect() : () -> ()
    cf.br ^search
  ^search:
    %span = llvm.load %spanSlot : !llvm.ptr -> i64
    %hasSpan = arith.cmpi ne, %span, %zero : i64
    cf.cond_br %hasSpan, ^pop, ^sweep
  ^pop:
    %startPointer = llvm.inttoptr %span : i64 to !llvm.ptr
    %link = llvm.load %startPointer : !llvm.ptr -> i64
    %endWordSlot = llvm.getelementptr %startPointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %endWord = llvm.load %endWordSlot : !llvm.ptr -> i64
    %following = func.call @strict_gc_unlink(%link) : (i64) -> i64
    llvm.store %following, %spanSlot : i64, !llvm.ptr
    %spanEnd = func.call @strict_gc_unlink(%endWord) : (i64) -> i64
    %bytes = arith.subi %spanEnd, %span : i64
    %cleared = llvm.call @memset(%startPointer, %clear, %bytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %cursors = llvm.mlir.addressof @strict_gc_cursors : !llvm.ptr
    %cursorSlot = llvm.getelementptr %cursors[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %span, %cursorSlot : i64, !llvm.ptr
    %ends = llvm.mlir.addressof @strict_gc_ends : !llvm.ptr
    %endSlot = llvm.getelementptr %ends[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %spanEnd, %endSlot : i64, !llvm.ptr
    func.call @strict_gc_add(%bytes) : (i64) -> ()
    return
  ^sweep:
    %page = llvm.load %sweepSlot : !llvm.ptr -> i64
    %hasPage = arith.cmpi ne, %page, %zero : i64
    cf.cond_br %hasPage, ^sweepPage, ^fresh
  ^sweepPage:
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %nextPage = llvm.load %nextSlot : !llvm.ptr -> i64
    llvm.store %nextPage, %sweepSlot : i64, !llvm.ptr
    func.call @strict_gc_sweep_page(%page, %class) : (i64, i64) -> ()
    cf.br ^search
  ^fresh:
    func.call @strict_gc_new_small_page(%class) : (i64) -> ()
    cf.br ^search
  }
  func.func private @strict_gc_add_span(%class: i64, %start: i64, %end: i64) {
    %spans = llvm.mlir.addressof @strict_gc_spans : !llvm.ptr
    %spanSlot = llvm.getelementptr %spans[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %head = llvm.load %spanSlot : !llvm.ptr -> i64
    %link = func.call @strict_gc_link(%head) : (i64) -> i64
    %endLink = func.call @strict_gc_link(%end) : (i64) -> i64
    %startPointer = llvm.inttoptr %start : i64 to !llvm.ptr
    llvm.store %link, %startPointer : i64, !llvm.ptr
    %endSlot = llvm.getelementptr %startPointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %endLink, %endSlot : i64, !llvm.ptr
    llvm.store %start, %spanSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_add_run(%page: i64, %first: i64, %objectSize: i64, %class: i64, %from: i64, %to: i64) {
    %fromOffset = arith.muli %from, %objectSize : i64
    %toOffset = arith.muli %to, %objectSize : i64
    %pageFirst = arith.addi %page, %first : i64
    %start = arith.addi %pageFirst, %fromOffset : i64
    %end = arith.addi %pageFirst, %toOffset : i64
    func.call @strict_gc_add_span(%class, %start, %end) : (i64, i64, i64) -> ()
    return
  }
  func.func private @strict_gc_keep_page(%page: i64, %class: i64) {
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %pages = llvm.mlir.addressof @strict_gc_pages : !llvm.ptr
    %pagesSlot = llvm.getelementptr %pages[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %oldFirst = llvm.load %pagesSlot : !llvm.ptr -> i64
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %oldFirst, %nextSlot : i64, !llvm.ptr
    llvm.store %page, %pagesSlot : i64, !llvm.ptr
    %none = arith.constant 0 : i64
    %wasEmpty = arith.cmpi eq, %oldFirst, %none : i64
    %tails = llvm.mlir.addressof @strict_gc_tails : !llvm.ptr
    %tailSlot = llvm.getelementptr %tails[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %tail = llvm.load %tailSlot : !llvm.ptr -> i64
    %newTail = arith.select %wasEmpty, %page, %tail : i64
    llvm.store %newTail, %tailSlot : i64, !llvm.ptr
    return
  }
  // a page without free objects waits for the next full collection, only old objects are left in it
  func.func private @strict_gc_keep_full(%page: i64, %class: i64) {
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %fulls = llvm.mlir.addressof @strict_gc_fulls : !llvm.ptr
    %fullSlot = llvm.getelementptr %fulls[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %oldFirst = llvm.load %fullSlot : !llvm.ptr -> i64
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %oldFirst, %nextSlot : i64, !llvm.ptr
    llvm.store %page, %fullSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_resweep(%class: i64) {
    %none = arith.constant 0 : i64
    %pages = llvm.mlir.addressof @strict_gc_pages : !llvm.ptr
    %pagesSlot = llvm.getelementptr %pages[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %first = llvm.load %pagesSlot : !llvm.ptr -> i64
    %isEmpty = arith.cmpi eq, %first, %none : i64
    cf.cond_br %isEmpty, ^done, ^join
  ^join:
    %tails = llvm.mlir.addressof @strict_gc_tails : !llvm.ptr
    %tailSlot = llvm.getelementptr %tails[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %tail = llvm.load %tailSlot : !llvm.ptr -> i64
    %tailPointer = llvm.inttoptr %tail : i64 to !llvm.ptr
    %sweeps = llvm.mlir.addressof @strict_gc_sweeps : !llvm.ptr
    %sweepSlot = llvm.getelementptr %sweeps[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %waiting = llvm.load %sweepSlot : !llvm.ptr -> i64
    %nextSlot = llvm.getelementptr %tailPointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %waiting, %nextSlot : i64, !llvm.ptr
    llvm.store %first, %sweepSlot : i64, !llvm.ptr
    llvm.store %none, %pagesSlot : i64, !llvm.ptr
    llvm.store %none, %tailSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_sweep_page(%page: i64, %class: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %none = arith.constant -1 : i64
    %unmarked = arith.constant 0 : i8
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %objectSize = llvm.load %pagePointer : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %firstSlot = llvm.getelementptr %pagePointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %first = llvm.load %firstSlot : !llvm.ptr -> i64
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %spans = llvm.mlir.addressof @strict_gc_spans : !llvm.ptr
    %spanSlot = llvm.getelementptr %spans[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %savedHead = llvm.load %spanSlot : !llvm.ptr -> i64
    cf.br ^scan(%zero, %none, %zero : i64, i64, i64)
  ^scan(%index: i64, %runStart: i64, %live: i64):
    %more = arith.cmpi ult, %index, %count : i64
    cf.cond_br %more, ^fast, ^finish
  ^fast:
    %plusEight = arith.addi %index, %eight : i64
    %roomForEight = arith.cmpi ule, %plusEight, %count : i64
    cf.cond_br %roomForEight, ^wide, ^byte
  ^wide:
    %wordSlot = llvm.getelementptr %marks[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %word = llvm.load %wordSlot {alignment = 1 : i64} : !llvm.ptr -> i64
    %allDead = arith.cmpi eq, %word, %zero : i64
    %noRun = arith.cmpi eq, %runStart, %none : i64
    %openedEight = arith.select %noRun, %index, %runStart : i64
    cf.cond_br %allDead, ^scan(%plusEight, %openedEight, %live : i64, i64, i64), ^byte
  ^byte:
    %markSlot = llvm.getelementptr %marks[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %mark = llvm.load %markSlot : !llvm.ptr -> i8
    %isLive = arith.cmpi ne, %mark, %unmarked : i8
    %nextIndex = arith.addi %index, %one : i64
    %hasNoRun = arith.cmpi eq, %runStart, %none : i64
    %opened = arith.select %hasNoRun, %index, %runStart : i64
    cf.cond_br %isLive, ^liveObject, ^scan(%nextIndex, %opened, %live : i64, i64, i64)
  ^liveObject:
    %moreLive = arith.addi %live, %one : i64
    cf.cond_br %hasNoRun, ^scan(%nextIndex, %none, %moreLive : i64, i64, i64), ^closeRun
  ^closeRun:
    func.call @strict_gc_add_run(%page, %first, %objectSize, %class, %runStart, %index) : (i64, i64, i64, i64, i64, i64) -> ()
    cf.br ^scan(%nextIndex, %none, %moreLive : i64, i64, i64)
  ^finish:
    %hasOpenRun = arith.cmpi ne, %runStart, %none : i64
    %isEmpty = arith.cmpi eq, %live, %zero : i64
    cf.cond_br %isEmpty, ^release, ^closeLast
  ^release:
    llvm.store %savedHead, %spanSlot : i64, !llvm.ptr
    func.call @strict_gc_release_page(%page) : (i64) -> ()
    return
  ^closeLast:
    cf.cond_br %hasOpenRun, ^addLast, ^keep
  ^addLast:
    func.call @strict_gc_add_run(%page, %first, %objectSize, %class, %runStart, %count) : (i64, i64, i64, i64, i64, i64) -> ()
    cf.br ^keep
  ^keep:
    %isFull = arith.cmpi eq, %live, %count : i64
    cf.cond_br %isFull, ^full, ^partial
  ^full:
    func.call @strict_gc_keep_full(%page, %class) : (i64, i64) -> ()
    return
  ^partial:
    func.call @strict_gc_keep_page(%page, %class) : (i64, i64) -> ()
    return
  }
  func.func private @strict_gc_page_kind(%page: i64, %kind: i64) {
    %shift = arith.constant 16 : i64
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %tableSlot = llvm.mlir.addressof @strict_gc_table : !llvm.ptr
    %table = llvm.load %tableSlot : !llvm.ptr -> i64
    %offset = arith.subi %page, %base : i64
    %index = arith.shrui %offset, %shift : i64
    %tablePointer = llvm.inttoptr %table : i64 to !llvm.ptr
    %entry = llvm.getelementptr %tablePointer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %kind, %entry : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_new_small_page(%class: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %usable = arith.constant 65456 : i64
    %header = arith.constant 64 : i64
    %fifteen = arith.constant 15 : i64
    %alignMask = arith.constant -16 : i64
    %page = func.call @strict_gc_take_pages(%one) : (i64) -> i64
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %sizeMask = arith.constant 31 : i64
    %sizeClass = arith.andi %class, %sizeMask : i64
    %sizes = llvm.mlir.addressof @strict_gc_sizes : !llvm.ptr
    %sizeSlot = llvm.getelementptr %sizes[0, %sizeClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<32 x i64>
    %objectSize = llvm.load %sizeSlot : !llvm.ptr -> i64
    %perObject = arith.addi %objectSize, %one : i64
    %count = arith.divui %usable, %perObject : i64
    %withMarks = arith.addi %header, %count : i64
    %paddedMarks = arith.addi %withMarks, %fifteen : i64
    %first = arith.andi %paddedMarks, %alignMask : i64
    llvm.store %objectSize, %pagePointer : i64, !llvm.ptr
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %count, %countSlot : i64, !llvm.ptr
    %firstSlot = llvm.getelementptr %pagePointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %first, %firstSlot : i64, !llvm.ptr
    %classSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %class, %classSlot : i64, !llvm.ptr
    %scale = arith.constant 4294967296 : i64
    %sizeRest = arith.subi %objectSize, %one : i64
    %scaled = arith.addi %scale, %sizeRest : i64
    %reciprocal = arith.divui %scaled, %objectSize : i64
    %reciprocalSlot = llvm.getelementptr %pagePointer[5] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %reciprocal, %reciprocalSlot : i64, !llvm.ptr
    func.call @strict_gc_page_kind(%page, %one) : (i64, i64) -> ()
    func.call @strict_gc_keep_page(%page, %class) : (i64, i64) -> ()
    %objectBytes = arith.muli %count, %objectSize : i64
    %start = arith.addi %page, %first : i64
    %end = arith.addi %start, %objectBytes : i64
    func.call @strict_gc_add_span(%class, %start, %end) : (i64, i64, i64) -> ()
    return
  }
  func.func private @strict_gc_take_pages(%count: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %pageSize = arith.constant 65536 : i64
    %clear = arith.constant 0 : i32
    %single = arith.cmpi eq, %count, %one : i64
    %poolSlot = llvm.mlir.addressof @strict_gc_pool : !llvm.ptr
    %pooledSlot = llvm.mlir.addressof @strict_gc_pooled : !llvm.ptr
    %spareSlot = llvm.mlir.addressof @strict_gc_spare : !llvm.ptr
    %spareCountSlot = llvm.mlir.addressof @strict_gc_spare_count : !llvm.ptr
    cf.cond_br %single, ^pool, ^grow
  ^pool:
    %pooled = llvm.load %poolSlot : !llvm.ptr -> i64
    %hasPooled = arith.cmpi ne, %pooled, %zero : i64
    cf.cond_br %hasPooled, ^reuse, ^spare
  ^reuse:
    %pooledPointer = llvm.inttoptr %pooled : i64 to !llvm.ptr
    %nextPooled = llvm.load %pooledPointer : !llvm.ptr -> i64
    llvm.store %nextPooled, %poolSlot : i64, !llvm.ptr
    %poolCount = llvm.load %pooledSlot : !llvm.ptr -> i64
    %fewer = arith.subi %poolCount, %one : i64
    llvm.store %fewer, %pooledSlot : i64, !llvm.ptr
    %headerAndMarks = arith.constant 4096 : i64
    %zeroed = llvm.call @memset(%pooledPointer, %clear, %headerAndMarks) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    return %pooled : i64
  ^spare:
    %spareCount = llvm.load %spareCountSlot : !llvm.ptr -> i64
    %hasSpare = arith.cmpi ne, %spareCount, %zero : i64
    cf.cond_br %hasSpare, ^recommit, ^grow
  ^recommit:
    %last = arith.subi %spareCount, %one : i64
    llvm.store %last, %spareCountSlot : i64, !llvm.ptr
    %spare = llvm.load %spareSlot : !llvm.ptr -> i64
    %sparePointer = llvm.inttoptr %spare : i64 to !llvm.ptr
    %spareEntry = llvm.getelementptr %sparePointer[%last] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %sparePage = llvm.load %spareEntry : !llvm.ptr -> i64
    %sparePagePointer = llvm.inttoptr %sparePage : i64 to !llvm.ptr
    func.call @strict_commit(%sparePagePointer, %pageSize) : (!llvm.ptr, i64) -> ()
    return %sparePage : i64
  ^grow:
    %topSlot = llvm.mlir.addressof @strict_gc_top : !llvm.ptr
    %top = llvm.load %topSlot : !llvm.ptr -> i64
    %bytes = arith.muli %count, %pageSize : i64
    %newTop = arith.addi %top, %bytes : i64
    %limitSlot = llvm.mlir.addressof @strict_gc_limit : !llvm.ptr
    %limit = llvm.load %limitSlot : !llvm.ptr -> i64
    %exceeds = arith.cmpi ugt, %newTop, %limit : i64
    cf.cond_br %exceeds, ^fail, ^commit
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    return %zero : i64
  ^commit:
    llvm.store %newTop, %topSlot : i64, !llvm.ptr
    %committedSlot = llvm.mlir.addressof @strict_gc_committed : !llvm.ptr
    %committed = llvm.load %committedSlot : !llvm.ptr -> i64
    %needsCommit = arith.cmpi ugt, %newTop, %committed : i64
    cf.cond_br %needsCommit, ^ahead, ^ready
  ^ahead:
    %stepRest = arith.constant 1048575 : i64
    %stepMask = arith.constant -1048576 : i64
    %paddedTop = arith.addi %newTop, %stepRest : i64
    %commitEnd = arith.andi %paddedTop, %stepMask : i64
    %fitsLimit = arith.cmpi ule, %commitEnd, %limit : i64
    %ending = arith.select %fitsLimit, %commitEnd, %limit : i64
    %commitBytes = arith.subi %ending, %committed : i64
    %committedPointer = llvm.inttoptr %committed : i64 to !llvm.ptr
    func.call @strict_commit(%committedPointer, %commitBytes) : (!llvm.ptr, i64) -> ()
    llvm.store %ending, %committedSlot : i64, !llvm.ptr
    %cardShift = arith.constant 9 : i64
    %cardsSlot = llvm.mlir.addressof @strict_gc_cards : !llvm.ptr
    %cards = llvm.load %cardsSlot : !llvm.ptr -> i64
    %heapBaseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %heapBase = llvm.load %heapBaseSlot : !llvm.ptr -> i64
    %committedOffset = arith.subi %committed, %heapBase : i64
    %cardOffset = arith.shrui %committedOffset, %cardShift : i64
    %cardStart = arith.addi %cards, %cardOffset : i64
    %cardStartPointer = llvm.inttoptr %cardStart : i64 to !llvm.ptr
    %cardRest = arith.constant 511 : i64
    %cardBytes = arith.addi %commitBytes, %cardRest : i64
    %cardCount = arith.shrui %cardBytes, %cardShift : i64
    func.call @strict_commit(%cardStartPointer, %cardCount) : (!llvm.ptr, i64) -> ()
    cf.br ^ready
  ^ready:
    return %top : i64
  }
  func.func private @strict_gc_release_page(%page: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    func.call @strict_gc_page_kind(%page, %zero) : (i64, i64) -> ()
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %poolSlot = llvm.mlir.addressof @strict_gc_pool : !llvm.ptr
    %pooledSlot = llvm.mlir.addressof @strict_gc_pooled : !llvm.ptr
    %poolCount = llvm.load %pooledSlot : !llvm.ptr -> i64
    %pooled = llvm.load %poolSlot : !llvm.ptr -> i64
    llvm.store %pooled, %pagePointer : i64, !llvm.ptr
    llvm.store %page, %poolSlot : i64, !llvm.ptr
    %more = arith.addi %poolCount, %one : i64
    llvm.store %more, %pooledSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_trim_pool(%keep: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %pageSize = arith.constant 65536 : i64
    %poolSlot = llvm.mlir.addressof @strict_gc_pool : !llvm.ptr
    %pooledSlot = llvm.mlir.addressof @strict_gc_pooled : !llvm.ptr
    %spareSlot = llvm.mlir.addressof @strict_gc_spare : !llvm.ptr
    %spareCountSlot = llvm.mlir.addressof @strict_gc_spare_count : !llvm.ptr
    %spareCapacitySlot = llvm.mlir.addressof @strict_gc_spare_capacity : !llvm.ptr
    cf.br ^check
  ^check:
    %poolCount = llvm.load %pooledSlot : !llvm.ptr -> i64
    %tooMany = arith.cmpi ugt, %poolCount, %keep : i64
    cf.cond_br %tooMany, ^decommit, ^done
  ^decommit:
    %page = llvm.load %poolSlot : !llvm.ptr -> i64
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %nextPooled = llvm.load %pagePointer : !llvm.ptr -> i64
    llvm.store %nextPooled, %poolSlot : i64, !llvm.ptr
    %fewer = arith.subi %poolCount, %one : i64
    llvm.store %fewer, %pooledSlot : i64, !llvm.ptr
    func.call @strict_decommit(%pagePointer, %pageSize) : (!llvm.ptr, i64) -> ()
    %spare = llvm.load %spareSlot : !llvm.ptr -> i64
    %spareCount = llvm.load %spareCountSlot : !llvm.ptr -> i64
    %spareCapacity = llvm.load %spareCapacitySlot : !llvm.ptr -> i64
    %full = arith.cmpi uge, %spareCount, %spareCapacity : i64
    cf.cond_br %full, ^grow, ^store(%spare : i64)
  ^grow:
    %initial = arith.constant 1024 : i64
    %doubled = arith.addi %spareCapacity, %spareCapacity : i64
    %isFirst = arith.cmpi eq, %spareCapacity, %zero : i64
    %capacity = arith.select %isFirst, %initial, %doubled : i64
    %bytes = arith.muli %capacity, %eight : i64
    %sparePointer = llvm.inttoptr %spare : i64 to !llvm.ptr
    %grown = llvm.call @realloc(%sparePointer, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %grownBits = llvm.ptrtoint %grown : !llvm.ptr to i64
    llvm.store %grownBits, %spareSlot : i64, !llvm.ptr
    llvm.store %capacity, %spareCapacitySlot : i64, !llvm.ptr
    cf.br ^store(%grownBits : i64)
  ^store(%buffer: i64):
    %bufferPointer = llvm.inttoptr %buffer : i64 to !llvm.ptr
    %count = llvm.load %spareCountSlot : !llvm.ptr -> i64
    %entry = llvm.getelementptr %bufferPointer[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %page, %entry : i64, !llvm.ptr
    %increased = arith.addi %count, %one : i64
    llvm.store %increased, %spareCountSlot : i64, !llvm.ptr
    cf.br ^check
  ^done:
    return
  }
  // a single page may come from the pool, which only clears its first 4 KB: old words in the payload would keep
  // dead objects alive and point from old objects to young ones no card knows about
  func.func private @strict_gc_large_alloc(%size: i64, %list: i64) -> !llvm.ptr {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %shift = arith.constant 16 : i64
    %header = arith.constant 64 : i64
    %pageRest = arith.constant 65535 : i64
    func.call @strict_gc_count(%size) : (i64) -> ()
    %total = arith.addi %size, %header : i64
    %padded = arith.addi %total, %pageRest : i64
    %pages = arith.shrui %padded, %shift : i64
    %page = func.call @strict_gc_take_pages(%pages) : (i64) -> i64
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    llvm.store %size, %pagePointer : i64, !llvm.ptr
    %pagesSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %pages, %pagesSlot : i64, !llvm.ptr
    %listSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %list, %listSlot : i64, !llvm.ptr
    %largeSlot = llvm.mlir.addressof @strict_gc_large : !llvm.ptr
    %previous = llvm.load %largeSlot : !llvm.ptr -> i64
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %previous, %nextSlot : i64, !llvm.ptr
    llvm.store %page, %largeSlot : i64, !llvm.ptr
    func.call @strict_gc_page_kind(%page, %two) : (i64, i64) -> ()
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %offset = arith.subi %page, %base : i64
    %firstIndex = arith.shrui %offset, %shift : i64
    %firstShifted = arith.shli %firstIndex, %eight : i64
    %continuation = arith.ori %firstShifted, %three : i64
    cf.br ^mark(%one : i64)
  ^mark(%index: i64):
    %more = arith.cmpi ult, %index, %pages : i64
    cf.cond_br %more, ^continue, ^done
  ^continue:
    %pageSize = arith.constant 65536 : i64
    %pageOffset = arith.muli %index, %pageSize : i64
    %continued = arith.addi %page, %pageOffset : i64
    func.call @strict_gc_page_kind(%continued, %continuation) : (i64, i64) -> ()
    %nextIndex = arith.addi %index, %one : i64
    cf.br ^mark(%nextIndex : i64)
  ^done:
    %payload = arith.addi %page, %header : i64
    %result = llvm.inttoptr %payload : i64 to !llvm.ptr
    %reused = arith.cmpi eq, %pages, %one : i64
    cf.cond_br %reused, ^zero, ^zeroed
  ^zero:
    %clear = arith.constant 0 : i32
    %cleared = llvm.call @memset(%result, %clear, %size) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    cf.br ^zeroed
  ^zeroed:
    return %result : !llvm.ptr
  }
  func.func private @strict_gc_push(%object: i64, %size: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %sixteen = arith.constant 16 : i64
    %workSlot = llvm.mlir.addressof @strict_gc_work : !llvm.ptr
    %usedSlot = llvm.mlir.addressof @strict_gc_work_size : !llvm.ptr
    %capacitySlot = llvm.mlir.addressof @strict_gc_work_capacity : !llvm.ptr
    %work = llvm.load %workSlot : !llvm.ptr -> i64
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %full = arith.cmpi uge, %used, %capacity : i64
    cf.cond_br %full, ^grow, ^store(%work : i64)
  ^grow:
    %initial = arith.constant 65536 : i64
    %doubled = arith.addi %capacity, %capacity : i64
    %isFirst = arith.cmpi eq, %capacity, %zero : i64
    %newCapacity = arith.select %isFirst, %initial, %doubled : i64
    %bytes = arith.muli %newCapacity, %sixteen : i64
    %workPointer = llvm.inttoptr %work : i64 to !llvm.ptr
    %grown = llvm.call @realloc(%workPointer, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %grownBits = llvm.ptrtoint %grown : !llvm.ptr to i64
    %failed = arith.cmpi eq, %grownBits, %zero : i64
    cf.cond_br %failed, ^fail, ^grown
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    return
  ^grown:
    llvm.store %grownBits, %workSlot : i64, !llvm.ptr
    llvm.store %newCapacity, %capacitySlot : i64, !llvm.ptr
    cf.br ^store(%grownBits : i64)
  ^store(%buffer: i64):
    %bufferPointer = llvm.inttoptr %buffer : i64 to !llvm.ptr
    %position = arith.muli %used, %two : i64
    %entry = llvm.getelementptr %bufferPointer[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %object, %entry : i64, !llvm.ptr
    %sizeEntry = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %size, %sizeEntry : i64, !llvm.ptr
    %increased = arith.addi %used, %one : i64
    llvm.store %increased, %usedSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_locate(%tagged: i64) -> (!llvm.ptr, i64, i64, i1, i1) attributes {passthrough = ["alwaysinline"]} {
    %address = arith.constant 281474976710655 : i64
    %word = arith.andi %tagged, %address : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %shift = arith.constant 16 : i64
    %eight = arith.constant 8 : i64
    %kindMask = arith.constant 255 : i64
    %textLists = arith.constant 32 : i64
    %unset = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false
    %true = arith.constant true
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %topSlot = llvm.mlir.addressof @strict_gc_top : !llvm.ptr
    %top = llvm.load %topSlot : !llvm.ptr -> i64
    %aboveBase = arith.cmpi uge, %word, %base : i64
    %belowTop = arith.cmpi ult, %word, %top : i64
    %inHeap = arith.andi %aboveBase, %belowTop : i1
    cf.cond_br %inHeap, ^page, ^none
  ^page:
    %offset = arith.subi %word, %base : i64
    %index = arith.shrui %offset, %shift : i64
    %tableSlot = llvm.mlir.addressof @strict_gc_table : !llvm.ptr
    %table = llvm.load %tableSlot : !llvm.ptr -> i64
    %tablePointer = llvm.inttoptr %table : i64 to !llvm.ptr
    %entrySlot = llvm.getelementptr %tablePointer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %entry = llvm.load %entrySlot : !llvm.ptr -> i64
    %kind = arith.andi %entry, %kindMask : i64
    %isSmall = arith.cmpi eq, %kind, %one : i64
    cf.cond_br %isSmall, ^small, ^other
  ^small:
    %pageStart = arith.shli %index, %shift : i64
    %page = arith.addi %base, %pageStart : i64
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %objectSize = llvm.load %pagePointer : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %firstSlot = llvm.getelementptr %pagePointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %first = llvm.load %firstSlot : !llvm.ptr -> i64
    %within = arith.subi %word, %page : i64
    %pastHeader = arith.cmpi uge, %within, %first : i64
    cf.cond_br %pastHeader, ^object, ^none
  ^object:
    %relative = arith.subi %within, %first : i64
    %reciprocalSlot = llvm.getelementptr %pagePointer[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %reciprocal = llvm.load %reciprocalSlot : !llvm.ptr -> i64
    %product = arith.muli %relative, %reciprocal : i64
    %thirtyTwo = arith.constant 32 : i64
    %objectIndex = arith.shrui %product, %thirtyTwo : i64
    %valid = arith.cmpi ult, %objectIndex, %count : i64
    cf.cond_br %valid, ^found, ^none
  ^found:
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %markSlot = llvm.getelementptr %marks[%objectIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %objectOffset = arith.muli %objectIndex, %objectSize : i64
    %objectStart = arith.addi %first, %objectOffset : i64
    %objectAddress = arith.addi %page, %objectStart : i64
    %pageListSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %pageList = llvm.load %pageListSlot : !llvm.ptr -> i64
    %isScanned = arith.cmpi ult, %pageList, %textLists : i64
    return %markSlot, %objectAddress, %objectSize, %false, %isScanned : !llvm.ptr, i64, i64, i1, i1
  ^other:
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %isFirst = arith.cmpi eq, %kind, %two : i64
    %isContinued = arith.cmpi eq, %kind, %three : i64
    %isLarge = arith.ori %isFirst, %isContinued : i1
    %continuedFrom = arith.shrui %entry, %eight : i64
    %firstIndex = arith.select %isContinued, %continuedFrom, %index : i64
    cf.cond_br %isLarge, ^large(%firstIndex : i64), ^none
  ^large(%startIndex: i64):
    %largeStart = arith.shli %startIndex, %shift : i64
    %header = arith.addi %base, %largeStart : i64
    %headerPointer = llvm.inttoptr %header : i64 to !llvm.ptr
    %payloadSize = llvm.load %headerPointer : !llvm.ptr -> i64
    %largeMarkSlot = llvm.getelementptr %headerPointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %payloadOffset = arith.constant 64 : i64
    %payload = arith.addi %header, %payloadOffset : i64
    %largeListSlot = llvm.getelementptr %headerPointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %largeList = llvm.load %largeListSlot : !llvm.ptr -> i64
    %isLargeScanned = arith.cmpi ult, %largeList, %textLists : i64
    return %largeMarkSlot, %payload, %payloadSize, %true, %isLargeScanned : !llvm.ptr, i64, i64, i1, i1
  ^none:
    return %unset, %zero, %zero, %false, %false : !llvm.ptr, i64, i64, i1, i1
  }
  func.func private @strict_gc_mark(%tagged: i64) {
    %located:5 = func.call @strict_gc_locate(%tagged) : (i64) -> (!llvm.ptr, i64, i64, i1, i1)
    %unset = llvm.mlir.zero : !llvm.ptr
    %found = llvm.icmp "ne" %located#0, %unset : !llvm.ptr
    cf.cond_br %found, ^width, ^done
  ^width:
    cf.cond_br %located#3, ^large, ^small
  ^small:
    %mark = llvm.load %located#0 : !llvm.ptr -> i8
    %unmarked = arith.constant 0 : i8
    %isUnmarked = arith.cmpi eq, %mark, %unmarked : i8
    cf.cond_br %isUnmarked, ^setSmall, ^done
  ^setSmall:
    %marked = arith.constant 1 : i8
    llvm.store %marked, %located#0 : i8, !llvm.ptr
    cf.br ^counted
  ^large:
    %largeMark = llvm.load %located#0 : !llvm.ptr -> i64
    %zero = arith.constant 0 : i64
    %isLargeUnmarked = arith.cmpi eq, %largeMark, %zero : i64
    cf.cond_br %isLargeUnmarked, ^setLarge, ^done
  ^setLarge:
    %one = arith.constant 1 : i64
    llvm.store %one, %located#0 : i64, !llvm.ptr
    cf.br ^counted
  ^counted:
    func.call @strict_gc_add_marked(%located#2) : (i64) -> ()
    cf.cond_br %located#4, ^scan, ^done
  ^scan:
    func.call @strict_gc_push(%located#1, %located#2) : (i64, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_is_marked(%tagged: i64) -> i1 {
    %located:5 = func.call @strict_gc_locate(%tagged) : (i64) -> (!llvm.ptr, i64, i64, i1, i1)
    %unset = llvm.mlir.zero : !llvm.ptr
    %found = llvm.icmp "ne" %located#0, %unset : !llvm.ptr
    %true = arith.constant true
    cf.cond_br %found, ^width, ^alive
  ^alive:
    return %true : i1
  ^width:
    cf.cond_br %located#3, ^large, ^small
  ^small:
    %mark = llvm.load %located#0 : !llvm.ptr -> i8
    %unmarked = arith.constant 0 : i8
    %isMarked = arith.cmpi ne, %mark, %unmarked : i8
    return %isMarked : i1
  ^large:
    %largeMark = llvm.load %located#0 : !llvm.ptr -> i64
    %zero = arith.constant 0 : i64
    %isLargeMarked = arith.cmpi ne, %largeMark, %zero : i64
    return %isLargeMarked : i1
  }
  func.func private @strict_gc_add_marked(%bytes: i64) {
    %markedSlot = llvm.mlir.addressof @strict_gc_marked : !llvm.ptr
    %marked = llvm.load %markedSlot : !llvm.ptr -> i64
    %more = arith.addi %marked, %bytes : i64
    llvm.store %more, %markedSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_mark_range(%from: i64, %to: i64) {
    %eight = arith.constant 8 : i64
    %alignMask = arith.constant -8 : i64
    %aligned = arith.andi %from, %alignMask : i64
    cf.br ^loop(%aligned : i64)
  ^loop(%at: i64):
    %more = arith.cmpi ult, %at, %to : i64
    cf.cond_br %more, ^word, ^done
  ^word:
    %pointer = llvm.inttoptr %at : i64 to !llvm.ptr
    %value = llvm.load %pointer : !llvm.ptr -> i64
    func.call @strict_gc_mark(%value) : (i64) -> ()
    %next = arith.addi %at, %eight : i64
    cf.br ^loop(%next : i64)
  ^done:
    return
  }
  func.func private @strict_gc_drain() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %workSlot = llvm.mlir.addressof @strict_gc_work : !llvm.ptr
    %usedSlot = llvm.mlir.addressof @strict_gc_work_size : !llvm.ptr
    cf.br ^loop
  ^loop:
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    %empty = arith.cmpi eq, %used, %zero : i64
    cf.cond_br %empty, ^done, ^pop
  ^pop:
    %last = arith.subi %used, %one : i64
    llvm.store %last, %usedSlot : i64, !llvm.ptr
    %work = llvm.load %workSlot : !llvm.ptr -> i64
    %workPointer = llvm.inttoptr %work : i64 to !llvm.ptr
    %position = arith.muli %last, %two : i64
    %entry = llvm.getelementptr %workPointer[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %object = llvm.load %entry : !llvm.ptr -> i64
    %sizeEntry = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %size = llvm.load %sizeEntry : !llvm.ptr -> i64
    %end = arith.addi %object, %size : i64
    func.call @strict_gc_mark_range(%object, %end) : (i64, i64) -> ()
    cf.br ^loop
  ^done:
    return
  }
  func.func private @strict_gc_unsweep(%class: i64) {
    %zero = arith.constant 0 : i64
    %sweeps = llvm.mlir.addressof @strict_gc_sweeps : !llvm.ptr
    %sweepSlot = llvm.getelementptr %sweeps[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %pages = llvm.mlir.addressof @strict_gc_pages : !llvm.ptr
    %pagesSlot = llvm.getelementptr %pages[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %fulls = llvm.mlir.addressof @strict_gc_fulls : !llvm.ptr
    %fullSlot = llvm.getelementptr %fulls[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %tails = llvm.mlir.addressof @strict_gc_tails : !llvm.ptr
    %tailSlot = llvm.getelementptr %tails[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %waiting = llvm.load %sweepSlot : !llvm.ptr -> i64
    %swept = llvm.load %pagesSlot : !llvm.ptr -> i64
    %full = llvm.load %fullSlot : !llvm.ptr -> i64
    llvm.store %zero, %sweepSlot : i64, !llvm.ptr
    llvm.store %zero, %pagesSlot : i64, !llvm.ptr
    llvm.store %zero, %fullSlot : i64, !llvm.ptr
    llvm.store %zero, %tailSlot : i64, !llvm.ptr
    func.call @strict_gc_unmark(%waiting) : (i64) -> ()
    func.call @strict_gc_unmark(%swept) : (i64) -> ()
    func.call @strict_gc_unmark(%full) : (i64) -> ()
    return
  }
  func.func private @strict_gc_unmark(%first: i64) {
    %zero = arith.constant 0 : i64
    %clear = arith.constant 0 : i32
    cf.br ^next(%first : i64)
  ^next(%page: i64):
    %done = arith.cmpi eq, %page, %zero : i64
    cf.cond_br %done, ^finished, ^clear
  ^clear:
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %following = llvm.load %nextSlot : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %cleared = llvm.call @memset(%marks, %clear, %count) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %classSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %pageClass = llvm.load %classSlot : !llvm.ptr -> i64
    func.call @strict_gc_keep_page(%page, %pageClass) : (i64, i64) -> ()
    cf.br ^next(%following : i64)
  ^finished:
    return
  }
  func.func private @strict_gc_unmark_large() {
    %zero = arith.constant 0 : i64
    %largeSlot = llvm.mlir.addressof @strict_gc_large : !llvm.ptr
    %first = llvm.load %largeSlot : !llvm.ptr -> i64
    cf.br ^next(%first : i64)
  ^next(%current: i64):
    %done = arith.cmpi eq, %current, %zero : i64
    cf.cond_br %done, ^finished, ^clear
  ^clear:
    %headerPointer = llvm.inttoptr %current : i64 to !llvm.ptr
    %markSlot = llvm.getelementptr %headerPointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %zero, %markSlot : i64, !llvm.ptr
    %nextSlot = llvm.getelementptr %headerPointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %following = llvm.load %nextSlot : !llvm.ptr -> i64
    cf.br ^next(%following : i64)
  ^finished:
    return
  }
  // a write of a value into an object that may be old marks the card (512 bytes) of its slot, a young collection
  // scans the old objects of marked cards: they are the only old objects that can point to young ones
  func.func private @strict_gc_written(%slot: !llvm.ptr) attributes {passthrough = ["alwaysinline"]} {
    %address = llvm.ptrtoint %slot : !llvm.ptr to i64
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %topSlot = llvm.mlir.addressof @strict_gc_top : !llvm.ptr
    %top = llvm.load %topSlot : !llvm.ptr -> i64
    %offset = arith.subi %address, %base : i64
    %size = arith.subi %top, %base : i64
    %inHeap = arith.cmpi ult, %offset, %size : i64
    cf.cond_br %inHeap, ^card, ^done
  ^card:
    %nine = arith.constant 9 : i64
    %index = arith.shrui %offset, %nine : i64
    %cardsSlot = llvm.mlir.addressof @strict_gc_cards : !llvm.ptr
    %cardsBits = llvm.load %cardsSlot : !llvm.ptr -> i64
    %cards = llvm.inttoptr %cardsBits : i64 to !llvm.ptr
    %card = llvm.getelementptr %cards[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %state = llvm.load %card : !llvm.ptr -> i8
    %clean = arith.constant 0 : i8
    %isClean = arith.cmpi eq, %state, %clean : i8
    cf.cond_br %isClean, ^dirty, ^done
  ^dirty:
    %marked = arith.constant 1 : i8
    llvm.store %marked, %card : i8, !llvm.ptr
    func.call @strict_gc_remember(%index) : (i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_written_range(%start: !llvm.ptr, %bytes: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %card = arith.constant 512 : i64
    %hasBytes = arith.cmpi sgt, %bytes, %zero : i64
    cf.cond_br %hasBytes, ^loop(%zero : i64), ^done
  ^loop(%offset: i64):
    %slot = llvm.getelementptr %start[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    %next = arith.addi %offset, %card : i64
    %more = arith.cmpi slt, %next, %bytes : i64
    cf.cond_br %more, ^loop(%next : i64), ^last
  ^last:
    %end = arith.subi %bytes, %one : i64
    %lastSlot = llvm.getelementptr %start[%end] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    func.call @strict_gc_written(%lastSlot) : (!llvm.ptr) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_remember(%index: i64) {
    %zero = arith.constant 0 : i64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %shared = arith.cmpi ne, %busy, %zero : i64
    cf.cond_br %shared, ^locked, ^alone
  ^alone:
    func.call @strict_gc_dirty_card(%index) : (i64) -> ()
    return
  ^locked:
    llvm.call @strict_gc_lock() : () -> ()
    func.call @strict_gc_dirty_card(%index) : (i64) -> ()
    llvm.call @strict_gc_unlock() : () -> ()
    return
  }
  func.func private @strict_gc_dirty_card(%index: i64) {
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %dirty = llvm.mlir.addressof @strict_gc_dirty : !llvm.ptr
    %countSlot = llvm.getelementptr %dirty[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %capacitySlot = llvm.getelementptr %dirty[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %full = arith.cmpi sge, %count, %capacity : i64
    cf.cond_br %full, ^grow, ^store
  ^grow:
    %oldBits = llvm.load %dirty : !llvm.ptr -> i64
    %old = llvm.inttoptr %oldBits : i64 to !llvm.ptr
    %doubled = arith.shli %capacity, %one : i64
    %smallest = arith.constant 1024 : i64
    %grown = arith.maxui %doubled, %smallest : i64
    %bytes = arith.muli %grown, %eight : i64
    %fresh = llvm.call @realloc(%old, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %freshBits = llvm.ptrtoint %fresh : !llvm.ptr to i64
    %zero = arith.constant 0 : i64
    %failed = arith.cmpi eq, %freshBits, %zero : i64
    cf.cond_br %failed, ^fail, ^grown
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    return
  ^grown:
    llvm.store %freshBits, %dirty : i64, !llvm.ptr
    llvm.store %grown, %capacitySlot : i64, !llvm.ptr
    cf.br ^store
  ^store:
    %itemsBits = llvm.load %dirty : !llvm.ptr -> i64
    %items = llvm.inttoptr %itemsBits : i64 to !llvm.ptr
    %slot = llvm.getelementptr %items[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %index, %slot : i64, !llvm.ptr
    %more = arith.addi %count, %one : i64
    llvm.store %more, %countSlot : i64, !llvm.ptr
    return
  }
  // cleans all marked cards, a young collection first marks what the old objects in them point to
  func.func private @strict_gc_cards_done(%scan: i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %clean = arith.constant 0 : i8
    %dirty = llvm.mlir.addressof @strict_gc_dirty : !llvm.ptr
    %countSlot = llvm.getelementptr %dirty[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %itemsBits = llvm.load %dirty : !llvm.ptr -> i64
    %items = llvm.inttoptr %itemsBits : i64 to !llvm.ptr
    %cardsSlot = llvm.mlir.addressof @strict_gc_cards : !llvm.ptr
    %cardsBits = llvm.load %cardsSlot : !llvm.ptr -> i64
    %cards = llvm.inttoptr %cardsBits : i64 to !llvm.ptr
    cf.br ^loop(%zero : i64)
  ^loop(%at: i64):
    %more = arith.cmpi slt, %at, %count : i64
    cf.cond_br %more, ^card, ^done
  ^card:
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %index = llvm.load %slot : !llvm.ptr -> i64
    %card = llvm.getelementptr %cards[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %clean, %card : i8, !llvm.ptr
    %next = arith.addi %at, %one : i64
    cf.cond_br %scan, ^scan, ^loop(%next : i64)
  ^scan:
    func.call @strict_gc_scan_card(%index) : (i64) -> ()
    cf.br ^loop(%next : i64)
  ^done:
    llvm.store %zero, %countSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gc_scan_card(%index: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %nine = arith.constant 9 : i64
    %sixteen = arith.constant 16 : i64
    %card = arith.constant 512 : i64
    %kindMask = arith.constant 255 : i64
    %textLists = arith.constant 32 : i64
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %offset = arith.shli %index, %nine : i64
    %start = arith.addi %base, %offset : i64
    %pageIndex = arith.shrui %offset, %sixteen : i64
    %tableSlot = llvm.mlir.addressof @strict_gc_table : !llvm.ptr
    %table = llvm.load %tableSlot : !llvm.ptr -> i64
    %tablePointer = llvm.inttoptr %table : i64 to !llvm.ptr
    %entrySlot = llvm.getelementptr %tablePointer[%pageIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %entry = llvm.load %entrySlot : !llvm.ptr -> i64
    %kind = arith.andi %entry, %kindMask : i64
    %isSmall = arith.cmpi eq, %kind, %one : i64
    cf.cond_br %isSmall, ^small, ^other
  ^small:
    %pageStart = arith.shli %pageIndex, %sixteen : i64
    %page = arith.addi %base, %pageStart : i64
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %listSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %list = llvm.load %listSlot : !llvm.ptr -> i64
    %isScanned = arith.cmpi ult, %list, %textLists : i64
    %objectSize = llvm.load %pagePointer : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %firstSlot = llvm.getelementptr %pagePointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %first = llvm.load %firstSlot : !llvm.ptr -> i64
    %reciprocalSlot = llvm.getelementptr %pagePointer[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %reciprocal = llvm.load %reciprocalSlot : !llvm.ptr -> i64
    %within = arith.subi %start, %page : i64
    %end = arith.addi %within, %card : i64
    %reaches = arith.cmpi ugt, %end, %first : i64
    %wanted = arith.andi %isScanned, %reaches : i1
    %from = arith.maxui %within, %first : i64
    %relative = arith.subi %from, %first : i64
    %product = arith.muli %relative, %reciprocal : i64
    %thirtyTwo = arith.constant 32 : i64
    %firstObject = arith.shrui %product, %thirtyTwo : i64
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.cond_br %wanted, ^object(%firstObject : i64), ^done
  ^object(%objectIndex: i64):
    %objectOffset = arith.muli %objectIndex, %objectSize : i64
    %objectStart = arith.addi %first, %objectOffset : i64
    %inPage = arith.cmpi ult, %objectIndex, %count : i64
    %inCard = arith.cmpi ult, %objectStart, %end : i64
    %inside = arith.andi %inPage, %inCard : i1
    cf.cond_br %inside, ^check, ^done
  ^check:
    %markSlot = llvm.getelementptr %marks[%objectIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %mark = llvm.load %markSlot : !llvm.ptr -> i8
    %unmarked = arith.constant 0 : i8
    %isOld = arith.cmpi ne, %mark, %unmarked : i8
    %nextObject = arith.addi %objectIndex, %one : i64
    cf.cond_br %isOld, ^scan, ^object(%nextObject : i64)
  ^scan:
    %objectEnd = arith.addi %objectStart, %objectSize : i64
    %low = arith.maxui %objectStart, %within : i64
    %high = arith.minui %objectEnd, %end : i64
    %lowAddress = arith.addi %page, %low : i64
    %highAddress = arith.addi %page, %high : i64
    func.call @strict_gc_mark_range(%lowAddress, %highAddress) : (i64, i64) -> ()
    cf.br ^object(%nextObject : i64)
  ^other:
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %isFirst = arith.cmpi eq, %kind, %two : i64
    %isContinued = arith.cmpi eq, %kind, %three : i64
    %isLarge = arith.ori %isFirst, %isContinued : i1
    %continuedFrom = arith.shrui %entry, %eight : i64
    %firstIndex = arith.select %isContinued, %continuedFrom, %pageIndex : i64
    cf.cond_br %isLarge, ^large(%firstIndex : i64), ^done
  ^large(%startIndex: i64):
    %largeStart = arith.shli %startIndex, %sixteen : i64
    %header = arith.addi %base, %largeStart : i64
    %headerPointer = llvm.inttoptr %header : i64 to !llvm.ptr
    %payloadSize = llvm.load %headerPointer : !llvm.ptr -> i64
    %largeMarkSlot = llvm.getelementptr %headerPointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %largeMark = llvm.load %largeMarkSlot : !llvm.ptr -> i64
    %largeListSlot = llvm.getelementptr %headerPointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %largeList = llvm.load %largeListSlot : !llvm.ptr -> i64
    %largeScanned = arith.cmpi ult, %largeList, %textLists : i64
    %largeOld = arith.cmpi ne, %largeMark, %zero : i64
    %payloadOffset = arith.constant 64 : i64
    %payload = arith.addi %header, %payloadOffset : i64
    %payloadEnd = arith.addi %payload, %payloadSize : i64
    %cardEnd = arith.addi %start, %card : i64
    %largeLow = arith.maxui %start, %payload : i64
    %largeHigh = arith.minui %cardEnd, %payloadEnd : i64
    %overlaps = arith.cmpi ult, %largeLow, %largeHigh : i64
    %oldScanned = arith.andi %largeScanned, %largeOld : i1
    %scanLarge = arith.andi %oldScanned, %overlaps : i1
    cf.cond_br %scanLarge, ^scanLarge, ^done
  ^scanLarge:
    func.call @strict_gc_mark_range(%largeLow, %largeHigh) : (i64, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gc_sweep_large() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %pageSize = arith.constant 65536 : i64
    %largeSlot = llvm.mlir.addressof @strict_gc_large : !llvm.ptr
    cf.br ^visit(%largeSlot : !llvm.ptr)
  ^visit(%link: !llvm.ptr):
    %current = llvm.load %link : !llvm.ptr -> i64
    %end = arith.cmpi eq, %current, %zero : i64
    cf.cond_br %end, ^done, ^check
  ^check:
    %headerPointer = llvm.inttoptr %current : i64 to !llvm.ptr
    %markSlot = llvm.getelementptr %headerPointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %mark = llvm.load %markSlot : !llvm.ptr -> i64
    %nextSlot = llvm.getelementptr %headerPointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %isMarked = arith.cmpi ne, %mark, %zero : i64
    cf.cond_br %isMarked, ^survive, ^free
  ^survive:
    cf.br ^visit(%nextSlot : !llvm.ptr)
  ^free:
    %following = llvm.load %nextSlot : !llvm.ptr -> i64
    llvm.store %following, %link : i64, !llvm.ptr
    %pagesSlot = llvm.getelementptr %headerPointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %pages = llvm.load %pagesSlot : !llvm.ptr -> i64
    cf.br ^release(%zero : i64)
  ^release(%index: i64):
    %more = arith.cmpi ult, %index, %pages : i64
    cf.cond_br %more, ^releaseOne, ^visit(%link : !llvm.ptr)
  ^releaseOne:
    %pageOffset = arith.muli %index, %pageSize : i64
    %pageAddress = arith.addi %current, %pageOffset : i64
    func.call @strict_gc_release_page(%pageAddress) : (i64) -> ()
    %nextIndex = arith.addi %index, %one : i64
    cf.br ^release(%nextIndex : i64)
  ^done:
    return
  }
  llvm.mlir.global internal @strict_gc_pinned(0 : i64) : i64
  llvm.mlir.global internal @strict_gc_pinned_count(0 : i64) : i64
  func.func private @strict_pin(%value: f64) {
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %countSlot = llvm.mlir.addressof @strict_gc_pinned_count : !llvm.ptr
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %next = arith.addi %count, %one : i64
    %bytes = arith.muli %next, %eight : i64
    %pinnedSlot = llvm.mlir.addressof @strict_gc_pinned : !llvm.ptr
    %pinned = llvm.load %pinnedSlot : !llvm.ptr -> i64
    %old = llvm.inttoptr %pinned : i64 to !llvm.ptr
    %grown = llvm.call @realloc(%old, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %grownBits = llvm.ptrtoint %grown : !llvm.ptr to i64
    llvm.store %grownBits, %pinnedSlot : i64, !llvm.ptr
    %slot = llvm.getelementptr %grown[%count] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %slot : f64, !llvm.ptr
    llvm.store %next, %countSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_list_pin(%value: f64) {
    func.call @strict_pin(%value) : (f64) -> ()
    %view = func.call @strict_ptr(%value) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %fixed = arith.constant 0.0 : f64
    llvm.store %fixed, %capacitySlot : f64, !llvm.ptr
    return
  }
  func.func private @strict_gc_collect() {
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    %allocated = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %thresholdSlot = llvm.mlir.addressof @strict_gc_threshold : !llvm.ptr
    %threshold = llvm.load %thresholdSlot : !llvm.ptr -> i64
    %full = arith.cmpi ugt, %allocated, %threshold : i64
    func.call @strict_gc_run(%full) : (i1) -> ()
    return
  }
  // marks stay set (sticky): a young collection marks from the roots and the marked cards and stops at marked (old)
  // objects, so it only frees objects allocated since the last collection, a full collection clears all marks first
  func.func private @strict_gc_run(%requested: i1) {
    %zero = arith.constant 0 : i64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %shared = arith.cmpi ne, %busy, %zero : i64
    cf.cond_br %shared, ^deferred, ^collect
  ^deferred:
    return
  ^collect:
    %measuredSlot = llvm.mlir.addressof @strict_measured_text : !llvm.ptr
    llvm.store %zero, %measuredSlot : i64, !llvm.ptr
    %one = arith.constant 1 : i64
    %classCount = arith.constant 64 : i64
    %minimum = arith.constant 16777216 : i64
    llvm.call @llvm.eh.unwind.init() : () -> ()
    %stackPointer = llvm.call_intrinsic "llvm.stacksave.p0"() : () -> !llvm.ptr
    %limitSlot = llvm.mlir.addressof @strict_gc_young_limit : !llvm.ptr
    %limit = llvm.load %limitSlot : !llvm.ptr -> i64
    %noYoung = arith.cmpi eq, %limit, %zero : i64
    %full = arith.ori %requested, %noYoung : i1
    cf.cond_br %full, ^unsweep(%zero : i64), ^young
  ^young:
    %scan = arith.constant true
    func.call @strict_gc_cards_done(%scan) : (i1) -> ()
    cf.br ^reset(%zero : i64)
  ^unsweep(%class: i64):
    %more = arith.cmpi ult, %class, %classCount : i64
    cf.cond_br %more, ^unsweepOne, ^unmarked
  ^unsweepOne:
    func.call @strict_gc_unsweep(%class) : (i64) -> ()
    %nextClass = arith.addi %class, %one : i64
    cf.br ^unsweep(%nextClass : i64)
  ^unmarked:
    func.call @strict_gc_unmark_large() : () -> ()
    %skip = arith.constant false
    func.call @strict_gc_cards_done(%skip) : (i1) -> ()
    cf.br ^reset(%zero : i64)
  ^reset(%resetClass: i64):
    %moreResets = arith.cmpi ult, %resetClass, %classCount : i64
    cf.cond_br %moreResets, ^resetOne, ^mark
  ^resetOne:
    %cursors = llvm.mlir.addressof @strict_gc_cursors : !llvm.ptr
    %cursorSlot = llvm.getelementptr %cursors[0, %resetClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %cursorSlot : i64, !llvm.ptr
    %ends = llvm.mlir.addressof @strict_gc_ends : !llvm.ptr
    %endSlot = llvm.getelementptr %ends[0, %resetClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %endSlot : i64, !llvm.ptr
    %spans = llvm.mlir.addressof @strict_gc_spans : !llvm.ptr
    %spanSlot = llvm.getelementptr %spans[0, %resetClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %spanSlot : i64, !llvm.ptr
    %nextReset = arith.addi %resetClass, %one : i64
    cf.br ^reset(%nextReset : i64)
  ^mark:
    %markedSlot = llvm.mlir.addressof @strict_gc_marked : !llvm.ptr
    llvm.store %zero, %markedSlot : i64, !llvm.ptr
    %from = llvm.ptrtoint %stackPointer : !llvm.ptr to i64
    %stackSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    %stackBase = llvm.load %stackSlot : !llvm.ptr -> i64
    %sixteen = arith.constant 16 : i64
    %to = arith.addi %stackBase, %sixteen : i64
    func.call @strict_gc_mark_range(%from, %to) : (i64, i64) -> ()
    %pinnedSlot = llvm.mlir.addressof @strict_gc_pinned : !llvm.ptr
    %pinnedStart = llvm.load %pinnedSlot : !llvm.ptr -> i64
    %pinnedCountSlot = llvm.mlir.addressof @strict_gc_pinned_count : !llvm.ptr
    %pinnedCount = llvm.load %pinnedCountSlot : !llvm.ptr -> i64
    %wordBytes = arith.constant 8 : i64
    %pinnedBytes = arith.muli %pinnedCount, %wordBytes : i64
    %pinnedEnd = arith.addi %pinnedStart, %pinnedBytes : i64
    func.call @strict_gc_mark_range(%pinnedStart, %pinnedEnd) : (i64, i64) -> ()
    func.call @strict_gc_drain() : () -> ()
    func.call @strict_gpu_sweep(%full) : (i1) -> ()
    cf.br ^sweeps(%zero : i64)
  ^sweeps(%sweepClass: i64):
    %moreSweeps = arith.cmpi ult, %sweepClass, %classCount : i64
    cf.cond_br %moreSweeps, ^moveOne, ^large
  ^moveOne:
    func.call @strict_gc_resweep(%sweepClass) : (i64) -> ()
    %nextSweep = arith.addi %sweepClass, %one : i64
    cf.br ^sweeps(%nextSweep : i64)
  ^large:
    func.call @strict_gc_sweep_large() : () -> ()
    %youngSlot = llvm.mlir.addressof @strict_gc_young : !llvm.ptr
    llvm.store %zero, %youngSlot : i64, !llvm.ptr
    cf.cond_br %full, ^threshold, ^done
  ^threshold:
    %marked = llvm.load %markedSlot : !llvm.ptr -> i64
    %twice = arith.addi %marked, %marked : i64
    %fourTimes = arith.addi %twice, %twice : i64
    %doubled = arith.addi %fourTimes, %fourTimes : i64
    %isSmall = arith.cmpi ult, %doubled, %minimum : i64
    %threshold = arith.select %isSmall, %minimum, %doubled : i64
    %thresholdSlot = llvm.mlir.addressof @strict_gc_threshold : !llvm.ptr
    llvm.store %threshold, %thresholdSlot : i64, !llvm.ptr
    %poolMinimum = arith.constant 1024 : i64
    %pageShift = arith.constant 16 : i64
    %thresholdPages = arith.shrui %threshold, %pageShift : i64
    %fewPages = arith.cmpi ult, %thresholdPages, %poolMinimum : i64
    %keep = arith.select %fewPages, %poolMinimum, %thresholdPages : i64
    func.call @strict_gc_trim_pool(%keep) : (i64) -> ()
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    llvm.store %zero, %allocatedSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_ptr(%bits: f64) -> !llvm.ptr {
    %address = arith.constant 281474976710655 : i64
    %i = arith.bitcast %bits : f64 to i64
    %untagged = arith.andi %i, %address : i64
    %p = llvm.inttoptr %untagged : i64 to !llvm.ptr
    return %p : !llvm.ptr
  }
  func.func private @strict_error_bits(%p: !llvm.ptr) -> f64 {
    %tag = arith.constant 9221683186994511872 : i64
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    %tagged = arith.ori %i, %tag : i64
    %f = arith.bitcast %tagged : i64 to f64
    return %f : f64
  }
  func.func private @strict_error_value_bits(%p: !llvm.ptr) -> f64 {
    %tag = arith.constant 9221964661971222528 : i64
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    %tagged = arith.ori %i, %tag : i64
    %f = arith.bitcast %tagged : i64 to f64
    return %f : f64
  }
  func.func private @strict_error_name(%value: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %shift = arith.constant 48 : i64
    %errorTag = arith.constant 32762 : i64
    %valueTag = arith.constant 32763 : i64
    %i = arith.bitcast %value : f64 to i64
    %tag = arith.shrui %i, %shift : i64
    %isError = arith.cmpi eq, %tag, %errorTag : i64
    %isValue = arith.cmpi eq, %tag, %valueTag : i64
    %isAny = arith.ori %isError, %isValue : i1
    cf.cond_br %isAny, ^error, ^none
  ^error:
    %p = func.call @strict_ptr(%value) : (f64) -> !llvm.ptr
    %first = llvm.load %p : !llvm.ptr -> f64
    cf.cond_br %isValue, ^wrapped, ^named
  ^wrapped:
    %inner = func.call @strict_error_name(%first) : (f64) -> f64
    return %inner : f64
  ^named:
    return %first : f64
  ^none:
    return %zero : f64
  }
  func.func private @strict_is_failure(%value: f64) -> i1 {
    %shift = arith.constant 48 : i64
    %errorTag = arith.constant 32762 : i64
    %valueTag = arith.constant 32763 : i64
    %bits = arith.bitcast %value : f64 to i64
    %tag = arith.shrui %bits, %shift : i64
    %isError = arith.cmpi eq, %tag, %errorTag : i64
    %isValue = arith.cmpi eq, %tag, %valueTag : i64
    %isFailure = arith.ori %isError, %isValue : i1
    return %isFailure : i1
  }
  func.func private @strict_failed_error(%value: f64) -> f64 {
    %shift = arith.constant 48 : i64
    %valueTag = arith.constant 32763 : i64
    %bits = arith.bitcast %value : f64 to i64
    %tag = arith.shrui %bits, %shift : i64
    %isValue = arith.cmpi eq, %tag, %valueTag : i64
    cf.cond_br %isValue, ^wrapped, ^plain
  ^wrapped:
    %outer = func.call @strict_ptr(%value) : (f64) -> !llvm.ptr
    %inner = llvm.load %outer : !llvm.ptr -> f64
    return %inner : f64
  ^plain:
    return %value : f64
  }
  func.func private @strict_trace(%value: f64, %where: !llvm.ptr, %line: i64) -> f64 {
    %failed = func.call @strict_is_failure(%value) : (f64) -> i1
    cf.cond_br %failed, ^trace, ^done(%value : f64)
  ^trace:
    %traced = func.call @strict_error_trace(%value, %where, %line) : (f64, !llvm.ptr, i64) -> f64
    cf.br ^done(%traced : f64)
  ^done(%result: f64):
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_trace_at("\09at \00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_trace_line(":line \00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_trace_in(" in \00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_error_prefix("Error: \00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_error_newline("\0A\00") : !llvm.array<2 x i8>
  func.func private @strict_error_trace(%value: f64, %location: !llvm.ptr, %line: i64) -> f64 attributes {passthrough = ["noinline"]} {
    %one = arith.constant 1 : i64
    %error = func.call @strict_failed_error(%value) : (f64) -> f64
    %method = func.call @strict_bits(%location) : (!llvm.ptr) -> f64
    %repeated = func.call @strict_trace_repeated(%error, %method, %line) : (f64, f64, i64) -> i1
    cf.cond_br %repeated, ^skip, ^append
  ^skip:
    return %value : f64
  ^append:
    %methodLength = llvm.call @strlen(%location) : (!llvm.ptr) -> i64
    %skip = arith.addi %methodLength, %one : i64
    %filePointer = llvm.getelementptr %location[%skip] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %file = func.call @strict_bits(%filePointer) : (!llvm.ptr) -> f64
    %lineNumber = arith.sitofp %line : i64 to f64
    %size = arith.constant 24 : i64
    %entryBlock = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    llvm.store %method, %entryBlock : f64, !llvm.ptr
    %fileSlot = llvm.getelementptr %entryBlock[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %file, %fileSlot : f64, !llvm.ptr
    %lineSlot = llvm.getelementptr %entryBlock[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %lineNumber, %lineSlot : f64, !llvm.ptr
    %entry = func.call @strict_bits(%entryBlock) : (!llvm.ptr) -> f64
    %block = func.call @strict_ptr(%error) : (f64) -> !llvm.ptr
    %tracesSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %traces = llvm.load %tracesSlot : !llvm.ptr -> f64
    %added = func.call @strict_list_add(%traces, %entry) : (f64, f64) -> f64
    llvm.store %added, %tracesSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%tracesSlot) : (!llvm.ptr) -> ()
    return %value : f64
  }
  func.func private @strict_trace_checked(%value: f64, %where: !llvm.ptr, %line: i64) -> f64 attributes {passthrough = ["noinline"]} {
    %failed = func.call @strict_is_failure(%value) : (f64) -> i1
    cf.cond_br %failed, ^check, ^done
  ^check:
    %error = func.call @strict_failed_error(%value) : (f64) -> f64
    %method = func.call @strict_bits(%where) : (!llvm.ptr) -> f64
    %here = func.call @strict_traced_in(%error, %method) : (f64, f64) -> i1
    cf.cond_br %here, ^done, ^trace
  ^trace:
    %traced = func.call @strict_error_trace(%value, %where, %line) : (f64, !llvm.ptr, i64) -> f64
    return %traced : f64
  ^done:
    return %value : f64
  }
  func.func private @strict_traced_in(%error: f64, %method: f64) -> i1 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %block = func.call @strict_ptr(%error) : (f64) -> !llvm.ptr
    %tracesSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %traces = llvm.load %tracesSlot : !llvm.ptr -> f64
    %count = func.call @strict_list_length(%traces) : (f64) -> f64
    %hasTraces = arith.cmpf ogt, %count, %zero : f64
    cf.cond_br %hasTraces, ^last, ^different
  ^last:
    %index = arith.subf %count, %one : f64
    %entry = func.call @strict_list_get(%traces, %index) : (f64, f64) -> f64
    %entryBlock = func.call @strict_ptr(%entry) : (f64) -> !llvm.ptr
    %entryMethod = llvm.load %entryBlock : !llvm.ptr -> f64
    %entryName = func.call @strict_ptr(%entryMethod) : (f64) -> !llvm.ptr
    %name = func.call @strict_ptr(%method) : (f64) -> !llvm.ptr
    %compared = llvm.call @strcmp(%entryName, %name) : (!llvm.ptr, !llvm.ptr) -> i32
    %equal = arith.constant 0 : i32
    %same = arith.cmpi eq, %compared, %equal : i32
    return %same : i1
  ^different:
    %false = arith.constant false
    return %false : i1
  }
  func.func private @strict_trace_repeated(%error: f64, %method: f64, %line: i64) -> i1 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %block = func.call @strict_ptr(%error) : (f64) -> !llvm.ptr
    %tracesSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %traces = llvm.load %tracesSlot : !llvm.ptr -> f64
    %count = func.call @strict_list_length(%traces) : (f64) -> f64
    %hasTraces = arith.cmpf ogt, %count, %zero : f64
    cf.cond_br %hasTraces, ^last, ^different
  ^last:
    %index = arith.subf %count, %one : f64
    %entry = func.call @strict_list_get(%traces, %index) : (f64, f64) -> f64
    %entryBlock = func.call @strict_ptr(%entry) : (f64) -> !llvm.ptr
    %entryMethod = llvm.load %entryBlock : !llvm.ptr -> f64
    %entryName = func.call @strict_ptr(%entryMethod) : (f64) -> !llvm.ptr
    %name = func.call @strict_ptr(%method) : (f64) -> !llvm.ptr
    %compared = llvm.call @strcmp(%entryName, %name) : (!llvm.ptr, !llvm.ptr) -> i32
    %equal = arith.constant 0 : i32
    %sameMethod = arith.cmpi eq, %compared, %equal : i32
    %lineSlot = llvm.getelementptr %entryBlock[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %entryLine = llvm.load %lineSlot : !llvm.ptr -> f64
    %lineNumber = arith.sitofp %line : i64 to f64
    %sameLine = arith.cmpf oeq, %entryLine, %lineNumber : f64
    %same = arith.andi %sameMethod, %sameLine : i1
    return %same : i1
  ^different:
    %false = arith.constant false
    return %false : i1
  }
  llvm.mlir.global internal constant @strict_unhandled_format("%s\0A\00") : !llvm.array<4 x i8>
  func.func private @strict_unhandled(%value: f64) -> i1 {
    %failed = func.call @strict_is_failure(%value) : (f64) -> i1
    cf.cond_br %failed, ^report, ^done
  ^report:
    %text = func.call @strict_error_report(%value) : (f64) -> f64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %format = llvm.mlir.addressof @strict_unhandled_format : !llvm.ptr
    %printed = llvm.call @printf(%format, %pointer) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @strict_crash_log(%pointer) : (!llvm.ptr) -> ()
    func.call @strict_flush() : () -> ()
    cf.br ^done
  ^done:
    return %failed : i1
  }
  func.func private @strict_trace_text(%entry: f64) -> f64 {
    %block = func.call @strict_ptr(%entry) : (f64) -> !llvm.ptr
    %method = llvm.load %block : !llvm.ptr -> f64
    %fileSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %file = llvm.load %fileSlot : !llvm.ptr -> f64
    %lineSlot = llvm.getelementptr %block[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %line = llvm.load %lineSlot : !llvm.ptr -> f64
    %at = llvm.mlir.addressof @strict_trace_at : !llvm.ptr
    %atBits = func.call @strict_bits(%at) : (!llvm.ptr) -> f64
    %in = llvm.mlir.addressof @strict_trace_in : !llvm.ptr
    %inBits = func.call @strict_bits(%in) : (!llvm.ptr) -> f64
    %lineLabel = llvm.mlir.addressof @strict_trace_line : !llvm.ptr
    %lineBits = func.call @strict_bits(%lineLabel) : (!llvm.ptr) -> f64
    %number = func.call @strict_number_text(%line) : (f64) -> f64
    %first = func.call @strict_text_concat(%atBits, %method) : (f64, f64) -> f64
    %second = func.call @strict_text_concat(%first, %inBits) : (f64, f64) -> f64
    %third = func.call @strict_text_concat(%second, %file) : (f64, f64) -> f64
    %fourth = func.call @strict_text_concat(%third, %lineBits) : (f64, f64) -> f64
    %result = func.call @strict_text_concat(%fourth, %number) : (f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_error_report(%value: f64) -> f64 attributes {passthrough = ["noinline"]} {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %error = func.call @strict_failed_error(%value) : (f64) -> f64
    %prefix = llvm.mlir.addressof @strict_error_prefix : !llvm.ptr
    %prefixBits = func.call @strict_bits(%prefix) : (!llvm.ptr) -> f64
    %name = func.call @strict_error_name(%error) : (f64) -> f64
    %named = func.call @strict_text_concat(%prefixBits, %name) : (f64, f64) -> f64
    %newline = llvm.mlir.addressof @strict_error_newline : !llvm.ptr
    %newlineBits = func.call @strict_bits(%newline) : (!llvm.ptr) -> f64
    %block = func.call @strict_ptr(%error) : (f64) -> !llvm.ptr
    %tracesSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %traces = llvm.load %tracesSlot : !llvm.ptr -> f64
    %count = func.call @strict_list_length(%traces) : (f64) -> f64
    cf.br ^loop(%zero, %named : f64, f64)
  ^loop(%index: f64, %text: f64):
    %more = arith.cmpf olt, %index, %count : f64
    cf.cond_br %more, ^add, ^done
  ^add:
    %entry = func.call @strict_list_get(%traces, %index) : (f64, f64) -> f64
    %trace = func.call @strict_trace_text(%entry) : (f64) -> f64
    %lined = func.call @strict_text_concat(%text, %newlineBits) : (f64, f64) -> f64
    %longer = func.call @strict_text_concat(%lined, %trace) : (f64, f64) -> f64
    %next = arith.addf %index, %one : f64
    cf.br ^loop(%next, %longer : f64, f64)
  ^done:
    return %text : f64
  }
  func.func private @strict_shown(%value: f64) -> f64 {
    %failed = func.call @strict_is_failure(%value) : (f64) -> i1
    cf.cond_br %failed, ^report, ^plain
  ^report:
    %report = func.call @strict_error_report(%value) : (f64) -> f64
    return %report : f64
  ^plain:
    return %value : f64
  }
  func.func private @strict_error_equal(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %leftName = func.call @strict_error_name(%left) : (f64) -> f64
    %rightName = func.call @strict_error_name(%right) : (f64) -> f64
    %leftSet = arith.cmpf une, %leftName, %zero : f64
    %rightSet = arith.cmpf une, %rightName, %zero : f64
    %bothSet = arith.andi %leftSet, %rightSet : i1
    cf.cond_br %bothSet, ^compare, ^different
  ^compare:
    %nothing = arith.constant 0 : i8
    %leftText = func.call @strict_ptr(%leftName) : (f64) -> !llvm.ptr
    %leftFirst = llvm.load %leftText : !llvm.ptr -> i8
    %rightText = func.call @strict_ptr(%rightName) : (f64) -> !llvm.ptr
    %rightFirst = llvm.load %rightText : !llvm.ptr -> i8
    %leftAny = arith.cmpi eq, %leftFirst, %nothing : i8
    %rightAny = arith.cmpi eq, %rightFirst, %nothing : i8
    %any = arith.ori %leftAny, %rightAny : i1
    cf.cond_br %any, ^same, ^named
  ^named:
    %result = func.call @strict_text_equal(%leftName, %rightName) : (f64, f64) -> f64
    return %result : f64
  ^same:
    %one = arith.constant 1.0 : f64
    return %one : f64
  ^different:
    return %zero : f64
  }
  func.func private @strict_bits(%p: !llvm.ptr) -> f64 {
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    %f = arith.bitcast %i : i64 to f64
    return %f : f64
  }
  func.func private @strict_list_new(%capacity: f64, %length: f64) -> f64 {
    %headers = arith.constant 56 : i64
    %eight = arith.constant 8.0 : f64
    %bytesf = arith.mulf %capacity, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %total = arith.addi %headers, %bytes : i64
    %view = func.call @strict_alloc(%total) : (i64) -> !llvm.ptr
    %holder = llvm.getelementptr %view[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %items = llvm.getelementptr %view[56] : (!llvm.ptr) -> !llvm.ptr, i8
    %itemsbits = func.call @strict_bits(%items) : (!llvm.ptr) -> f64
    llvm.store %itemsbits, %holder : f64, !llvm.ptr
    %used = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %length, %used : f64, !llvm.ptr
    %cap = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %capacity, %cap : f64, !llvm.ptr
    %unindexed = arith.constant 0.0 : f64
    %index = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %unindexed, %index : f64, !llvm.ptr
    %writes = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %unindexed, %writes : f64, !llvm.ptr
    llvm.store %length, %view : f64, !llvm.ptr
    %holderbits = func.call @strict_bits(%holder) : (!llvm.ptr) -> f64
    %viewholder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %viewholder : f64, !llvm.ptr
    %result = func.call @strict_bits(%view) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_open_paren("(\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_close_paren(")\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal @strict_empty_holder(dense<0.0> : tensor<5xf64>) : !llvm.array<5 x f64>
  llvm.mlir.global internal @strict_empty_view(dense<0.0> : tensor<2xf64>) : !llvm.array<2 x f64>
  func.func private @strict_number_and(%self: f64, %other: f64) -> f64 {
    %left = arith.fptosi %self : f64 to i64
    %right = arith.fptosi %other : f64 to i64
    %bits = arith.andi %left, %right : i64
    %result = arith.sitofp %bits : i64 to f64
    return %result : f64
  }
  func.func private @strict_number_or(%self: f64, %other: f64) -> f64 {
    %left = arith.fptosi %self : f64 to i64
    %right = arith.fptosi %other : f64 to i64
    %bits = arith.ori %left, %right : i64
    %result = arith.sitofp %bits : i64 to f64
    return %result : f64
  }
  func.func private @strict_number_xor(%self: f64, %other: f64) -> f64 {
    %left = arith.fptosi %self : f64 to i64
    %right = arith.fptosi %other : f64 to i64
    %bits = arith.xori %left, %right : i64
    %result = arith.sitofp %bits : i64 to f64
    return %result : f64
  }
  func.func private @strict_number_logarithm(%self: f64) -> f64 {
    %result = llvm.call @log(%self) : (f64) -> f64
    return %result : f64
  }
  llvm.mlir.global internal @strict_random_state(0 : i64) : i64
  func.func private @strict_number_random(%self: f64) -> f64 {
    %stateAddress = llvm.mlir.addressof @strict_random_state : !llvm.ptr
    %state = llvm.load %stateAddress : !llvm.ptr -> i64
    %zero = arith.constant 0 : i64
    %unseeded = arith.cmpi eq, %state, %zero : i64
    cf.cond_br %unseeded, ^seed, ^next(%state : i64)
  ^seed:
    %seeded = func.call @strict_random_seed() : () -> i64
    cf.br ^next(%seeded : i64)
  ^next(%current: i64):
    %golden = arith.constant -7046029254386353131 : i64
    %advanced = arith.addi %current, %golden : i64
    llvm.store %advanced, %stateAddress : i64, !llvm.ptr
    %shift30 = arith.constant 30 : i64
    %high30 = arith.shrui %advanced, %shift30 : i64
    %first = arith.xori %advanced, %high30 : i64
    %firstFactor = arith.constant -4658895280553007687 : i64
    %firstMixed = arith.muli %first, %firstFactor : i64
    %shift27 = arith.constant 27 : i64
    %high27 = arith.shrui %firstMixed, %shift27 : i64
    %second = arith.xori %firstMixed, %high27 : i64
    %secondFactor = arith.constant -7723592293110705685 : i64
    %secondMixed = arith.muli %second, %secondFactor : i64
    %shift31 = arith.constant 31 : i64
    %high31 = arith.shrui %secondMixed, %shift31 : i64
    %bits = arith.xori %secondMixed, %high31 : i64
    %shift11 = arith.constant 11 : i64
    %top = arith.shrui %bits, %shift11 : i64
    %fraction = arith.uitofp %top : i64 to f64
    %unitScale = arith.constant 1.1102230246251565e-16 : f64
    %unit = arith.mulf %fraction, %unitScale : f64
    %result = arith.mulf %unit, %self : f64
    return %result : f64
  }
  func.func private @strict_increment(%self: f64, %amount: f64) -> f64 {
    %result = arith.addf %self, %amount : f64
    return %result : f64
  }
  func.func private @strict_increment_one(%self: f64) -> f64 {
    %one = arith.constant 1.0 : f64
    %result = arith.addf %self, %one : f64
    return %result : f64
  }
  func.func private @strict_decrement(%self: f64, %amount: f64) -> f64 {
    %result = arith.subf %self, %amount : f64
    return %result : f64
  }
  func.func private @strict_decrement_one(%self: f64) -> f64 {
    %one = arith.constant 1.0 : f64
    %result = arith.subf %self, %one : f64
    return %result : f64
  }
  func.func private @strict_list_remove(%list: f64, %item: f64, %equal: (f64, f64) -> f64) -> f64 attributes {passthrough = ["alwaysinline"]} {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    cf.br ^scan(%start, %start : i64, i64)
  ^scan(%read: i64, %write: i64):
    %more = arith.cmpi slt, %read, %count : i64
    cf.cond_br %more, ^check, ^done
  ^check:
    %slot = llvm.getelementptr %items[%read] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %matched = func.call @strict_list_matches(%value, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> i1
    %next = arith.addi %read, %step : i64
    cf.cond_br %matched, ^scan(%next, %write : i64, i64), ^keep
  ^keep:
    %target = llvm.getelementptr %items[%write] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
    func.call @strict_gc_written(%target) : (!llvm.ptr) -> ()
    %written = arith.addi %write, %step : i64
    cf.br ^scan(%next, %written : i64, i64)
  ^done:
    %remaining = arith.sitofp %write : i64 to f64
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %remaining, %usedSlot : f64, !llvm.ptr
    %result = func.call @strict_alloc(%sixteen) : (i64) -> !llvm.ptr
    llvm.store %remaining, %result : f64, !llvm.ptr
    %resultHolder = llvm.getelementptr %result[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderBits, %resultHolder : f64, !llvm.ptr
    %bits = func.call @strict_bits(%result) : (!llvm.ptr) -> f64
    return %bits : f64
  }
  func.func private @strict_list_removed(%list: f64, %item: f64, %equal: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %copy = func.call @strict_list_copy(%list, %zero) : (f64, f64) -> f64
    %result = func.call @strict_list_remove(%copy, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_removed_text(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_removed(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_removed_number(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_removed(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_remove_text(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_remove(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_remove_number(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_remove(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_is_dot_name(%name: !llvm.ptr) -> i1 {
    %dot = arith.constant 46 : i8
    %end = arith.constant 0 : i8
    %first = llvm.load %name : !llvm.ptr -> i8
    %secondSlot = llvm.getelementptr %name[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %second = llvm.load %secondSlot : !llvm.ptr -> i8
    %startsWithDot = arith.cmpi eq, %first, %dot : i8
    %isSingle = arith.cmpi eq, %second, %end : i8
    %secondIsDot = arith.cmpi eq, %second, %dot : i8
    cf.cond_br %secondIsDot, ^third, ^decide(%isSingle : i1)
  ^third:
    %thirdSlot = llvm.getelementptr %name[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %third = llvm.load %thirdSlot : !llvm.ptr -> i8
    %isDouble = arith.cmpi eq, %third, %end : i8
    cf.br ^decide(%isDouble : i1)
  ^decide(%isDotted: i1):
    %result = arith.andi %startsWithDot, %isDotted : i1
    return %result : i1
  }
  func.func private @strict_list_empty() -> f64 {
    %view = llvm.mlir.addressof @strict_empty_view : !llvm.ptr
    %result = func.call @strict_bits(%view) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_list_items(%list: f64) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderbits = llvm.load %slot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderbits) : (f64) -> !llvm.ptr
    %itemsbits = llvm.load %holder : !llvm.ptr -> f64
    %stale = arith.cmpf oeq, %itemsbits, %zero : f64
    cf.cond_br %stale, ^restore, ^items
  ^restore:
    %restored = func.call @strict_list_restore(%list) : (f64) -> !llvm.ptr
    return %restored : !llvm.ptr
  ^items:
    %items = func.call @strict_ptr(%itemsbits) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  }
  llvm.mlir.global internal @strict_restore_mutex(0 : i64) : i64
  // in a parallel loop one thread unpacks or restores a list while the others wait and then find its items
  func.func private @strict_list_restore(%list: f64) -> !llvm.ptr {
    %none = arith.constant 0 : i64
    %unset = arith.constant 0.0 : f64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %shared = arith.cmpi ne, %busy, %none : i64
    cf.cond_br %shared, ^lock, ^alone
  ^alone:
    %items = func.call @strict_list_restoring(%list) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  ^lock:
    %mutex = llvm.mlir.addressof @strict_restore_mutex : !llvm.ptr
    llvm.call @strict_spin_lock(%mutex) : (!llvm.ptr) -> ()
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %itemsBits = llvm.load %holder : !llvm.ptr -> f64
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %isNewest = arith.cmpf oge, %used, %unset : f64
    %narrow = arith.cmpf olt, %capacity, %unset : f64
    %isPacked = arith.andi %isNewest, %narrow : i1
    %stale = arith.cmpf oeq, %itemsBits, %unset : f64
    %outdated = arith.ori %isPacked, %stale : i1
    cf.cond_br %outdated, ^restore, ^restored
  ^restore:
    %fresh = func.call @strict_list_restoring(%list) : (f64) -> !llvm.ptr
    cf.br ^done(%fresh : !llvm.ptr)
  ^restored:
    %present = func.call @strict_ptr(%itemsBits) : (f64) -> !llvm.ptr
    cf.br ^done(%present : !llvm.ptr)
  ^done(%result: !llvm.ptr):
    llvm.call @strict_spin_unlock(%mutex) : (!llvm.ptr) -> ()
    return %result : !llvm.ptr
  }
  func.func private @strict_list_restoring(%list: f64) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
    %eight = arith.constant 8.0 : f64
    %minusOne = arith.constant -1.0 : f64
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %slot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %slot : !llvm.ptr -> f64
    %noLength = arith.cmpf oeq, %length, %zero : f64
    cf.cond_br %noLength, ^empty, ^copy
  ^empty:
    %emptyHolder = llvm.mlir.addressof @strict_empty_holder : !llvm.ptr
    %emptyBits = func.call @strict_bits(%emptyHolder) : (!llvm.ptr) -> f64
    llvm.store %emptyBits, %slot : f64, !llvm.ptr
    return %emptyHolder : !llvm.ptr
  ^copy:
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %isNewest = arith.cmpf oge, %used, %zero : f64
    %narrow = arith.cmpf olt, %capacity, %zero : f64
    %isPacked = arith.andi %isNewest, %narrow : i1
    cf.cond_br %isPacked, ^unpack, ^history
  ^unpack:
    %unpacked = func.call @strict_list_unpacked(%list) : (f64) -> !llvm.ptr
    return %unpacked : !llvm.ptr
  ^history:
    %copy = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %copyView = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    %copyHolderSlot = llvm.getelementptr %copyView[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %copyHolderBits = llvm.load %copyHolderSlot : !llvm.ptr -> f64
    %copyHolder = func.call @strict_ptr(%copyHolderBits) : (f64) -> !llvm.ptr
    %copyItemsBits = llvm.load %copyHolder : !llvm.ptr -> f64
    %copyItems = func.call @strict_ptr(%copyItemsBits) : (f64) -> !llvm.ptr
    cf.br ^reverse(%holderBits, %zero : f64, f64)
  ^reverse(%current: f64, %previous: f64):
    %node = func.call @strict_ptr(%current) : (f64) -> !llvm.ptr
    %nodeItems = llvm.load %node : !llvm.ptr -> f64
    %isStale = arith.cmpf oeq, %nodeItems, %zero : f64
    cf.cond_br %isStale, ^flip, ^live
  ^flip:
    %link = llvm.getelementptr %node[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %following = llvm.load %link : !llvm.ptr -> f64
    llvm.store %previous, %link : f64, !llvm.ptr
    func.call @strict_gc_written(%link) : (!llvm.ptr) -> ()
    cf.br ^reverse(%following, %current : f64, f64)
  ^live:
    %liveItems = func.call @strict_ptr(%nodeItems) : (f64) -> !llvm.ptr
    %bytesf = arith.mulf %length, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %copied = llvm.call @memcpy(%copyItems, %liveItems, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    cf.br ^apply(%previous, %current : f64, f64)
  ^apply(%undo: f64, %after: f64):
    %done = arith.cmpf oeq, %undo, %zero : f64
    cf.cond_br %done, ^finish, ^undo
  ^undo:
    %diff = func.call @strict_ptr(%undo) : (f64) -> !llvm.ptr
    %encodedSlot = llvm.getelementptr %diff[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %encoded = llvm.load %encodedSlot : !llvm.ptr -> f64
    %position = arith.subf %minusOne, %encoded : f64
    %oldSlot = llvm.getelementptr %diff[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %old = llvm.load %oldSlot : !llvm.ptr -> f64
    %inside = arith.cmpf olt, %position, %length : f64
    cf.cond_br %inside, ^set, ^relink
  ^set:
    %at = arith.fptosi %position : f64 to i64
    %target = llvm.getelementptr %copyItems[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %old, %target : f64, !llvm.ptr
    cf.br ^relink
  ^relink:
    %backSlot = llvm.getelementptr %diff[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %back = llvm.load %backSlot : !llvm.ptr -> f64
    llvm.store %after, %backSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%backSlot) : (!llvm.ptr) -> ()
    cf.br ^apply(%back, %undo : f64, f64)
  ^finish:
    llvm.store %copyHolderBits, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    return %copyItems : !llvm.ptr
  }
  func.func private @strict_list_unpacked(%list: f64) -> !llvm.ptr {
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %halfCapacity = arith.constant -16.0 : f64
    %isHalf = arith.cmpf oeq, %capacity, %halfCapacity : f64
    %quantizedCapacity = arith.constant -8.0 : f64
    %isQuantized = arith.cmpf oeq, %capacity, %quantizedCapacity : f64
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    %copy = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %copyView = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    %copyHolderSlot = llvm.getelementptr %copyView[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %copyHolderBits = llvm.load %copyHolderSlot : !llvm.ptr -> f64
    %copyHolder = func.call @strict_ptr(%copyHolderBits) : (f64) -> !llvm.ptr
    %copyItemsBits = llvm.load %copyHolder : !llvm.ptr -> f64
    %copyItems = func.call @strict_ptr(%copyItemsBits) : (f64) -> !llvm.ptr
    %count = arith.fptosi %length : f64 to i64
    %widthNumber = arith.negf %capacity : f64
    %width = arith.fptosi %widthNumber : f64 to i64
    %scales = func.call @strict_packed_scales(%data, %count, %width) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    %sixCapacity = arith.constant -6.0 : f64
    %isSix = arith.cmpf oeq, %capacity, %sixCapacity : f64
    %fourCapacity = arith.constant -4.0 : f64
    %isFour = arith.cmpf oeq, %capacity, %fourCapacity : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %isBlocked = arith.ori %isSix, %isFour : i1
    %eight = arith.constant 8 : i64
    %blockSize = arith.constant 256 : i64
    %blockFloats = llvm.alloca %blockSize x f32 : (i64) -> !llvm.ptr
    %spare = arith.constant 255 : i64
    %roundedCount = arith.addi %count, %spare : i64
    %blockCount = arith.shrui %roundedCount, %eight : i64
    cf.cond_br %isBlocked, ^blocks(%start : i64), ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^element, ^done
  ^element:
    cf.cond_br %isHalf, ^half, ^wider
  ^half:
    %halfSlot = llvm.getelementptr %data[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %halfBits = llvm.load %halfSlot : !llvm.ptr -> i16
    %fromHalf = func.call @strict_half_number(%halfBits) : (i16) -> f64
    cf.br ^store(%fromHalf : f64)
  ^wider:
    cf.cond_br %isQuantized, ^quantized, ^narrower
  ^narrower:
    cf.br ^single
  ^blocks(%blockIndex: i64):
    %moreBlocks = arith.cmpi slt, %blockIndex, %blockCount : i64
    cf.cond_br %moreBlocks, ^decodeBlock, ^done
  ^decodeBlock:
    func.call @strict_packed_block(%data, %count, %width, %blockIndex, %blockFloats) : (!llvm.ptr, i64, i64, i64, !llvm.ptr) -> ()
    %blockStart = arith.shli %blockIndex, %eight : i64
    %blockEnd = arith.addi %blockStart, %blockSize : i64
    %lastEnd = arith.minui %blockEnd, %count : i64
    cf.br ^copied(%blockStart : i64)
  ^copied(%copyAt: i64):
    %moreCopies = arith.cmpi slt, %copyAt, %lastEnd : i64
    cf.cond_br %moreCopies, ^copyOne, ^nextBlock
  ^copyOne:
    %inBlock = arith.subi %copyAt, %blockStart : i64
    %floatSlot = llvm.getelementptr %blockFloats[%inBlock] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %blockFloat = llvm.load %floatSlot : !llvm.ptr -> f32
    %blockNumber = arith.extf %blockFloat : f32 to f64
    %copyTarget = llvm.getelementptr %copyItems[%copyAt] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %blockNumber, %copyTarget : f64, !llvm.ptr
    %nextAt = arith.addi %copyAt, %step : i64
    cf.br ^copied(%nextAt : i64)
  ^nextBlock:
    %followingBlock = arith.addi %blockIndex, %step : i64
    cf.br ^blocks(%followingBlock : i64)
  ^quantized:
    %byteSlot = llvm.getelementptr %data[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %byteSlot : !llvm.ptr -> i8
    %byteNumber = arith.sitofp %byte : i8 to f64
    %five = arith.constant 5 : i64
    %block = arith.shrui %index, %five : i64
    %scaleSlot = llvm.getelementptr %scales[%block] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %scale = llvm.load %scaleSlot : !llvm.ptr -> f32
    %wideScale = arith.extf %scale : f32 to f64
    %fromQuantized = arith.mulf %byteNumber, %wideScale : f64
    cf.br ^store(%fromQuantized : f64)
  ^single:
    %singleSlot = llvm.getelementptr %data[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %single = llvm.load %singleSlot : !llvm.ptr -> f32
    %fromSingle = arith.extf %single : f32 to f64
    cf.br ^store(%fromSingle : f64)
  ^store(%value: f64):
    %target = llvm.getelementptr %copyItems[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next : i64)
  ^done:
    llvm.store %copyHolderBits, %holderSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%holderSlot) : (!llvm.ptr) -> ()
    return %copyItems : !llvm.ptr
  }
  func.func private @strict_half_number(%half: i16) -> f64 {
    %bits = arith.extui %half : i16 to i64
    %signMask = arith.constant 32768 : i64
    %ten = arith.constant 10 : i64
    %exponentMask = arith.constant 31 : i64
    %mantissaMask = arith.constant 1023 : i64
    %signShift = arith.constant 48 : i64
    %exponentShift = arith.constant 52 : i64
    %mantissaShift = arith.constant 42 : i64
    %rebias = arith.constant 1008 : i64
    %infinite = arith.constant 9218868437227405312 : i64
    %zero = arith.constant 0 : i64
    %sign = arith.andi %bits, %signMask : i64
    %signBits = arith.shli %sign, %signShift : i64
    %shifted = arith.shrui %bits, %ten : i64
    %exponent = arith.andi %shifted, %exponentMask : i64
    %mantissa = arith.andi %bits, %mantissaMask : i64
    %mantissaBits = arith.shli %mantissa, %mantissaShift : i64
    %wideExponent = arith.addi %exponent, %rebias : i64
    %exponentBits = arith.shli %wideExponent, %exponentShift : i64
    %normalHigh = arith.ori %signBits, %exponentBits : i64
    %normal = arith.ori %normalHigh, %mantissaBits : i64
    %specialHigh = arith.ori %signBits, %infinite : i64
    %special = arith.ori %specialHigh, %mantissaBits : i64
    %isSpecial = arith.cmpi eq, %exponent, %exponentMask : i64
    %chosen = arith.select %isSpecial, %special, %normal : i64
    %wide = arith.bitcast %chosen : i64 to f64
    %tiny = arith.uitofp %mantissa : i64 to f64
    %scale = arith.constant 5.9604644775390625e-08 : f64
    %magnitude = arith.mulf %tiny, %scale : f64
    %negative = arith.negf %magnitude : f64
    %isNegative = arith.cmpi ne, %sign, %zero : i64
    %small = arith.select %isNegative, %negative, %magnitude : f64
    %isSmall = arith.cmpi eq, %exponent, %zero : i64
    %result = arith.select %isSmall, %small, %wide : f64
    return %result : f64
  }
  llvm.mlir.global internal @strict_versions(0 : i64) : i64
  func.func private @strict_list_version() -> i64 {
    %slot = llvm.mlir.addressof @strict_versions : !llvm.ptr
    %version = llvm.load %slot : !llvm.ptr -> i64
    return %version : i64
  }
  func.func private @strict_hoisted_item(%items: !llvm.ptr, %index: f64, %version: i64, %list: f64) -> f64 {
    %now = func.call @strict_list_version() : () -> i64
    %same = arith.cmpi eq, %now, %version : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %hasItems = llvm.icmp "ne" %items, %null : !llvm.ptr
    %unchanged = arith.andi %same, %hasItems : i1
    cf.cond_br %unchanged, ^hoisted, ^changed
  ^hoisted:
    %at = arith.fptosi %index : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    return %value : f64
  ^changed:
    %current = func.call @strict_list_get(%list, %index) : (f64, f64) -> f64
    return %current : f64
  }
  func.func private @strict_instance_with(%instance: f64, %slot: f64, %value: f64, %count: f64) -> f64 {
    %eight = arith.constant 8.0 : f64
    %bytesf = arith.mulf %count, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %copy = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    %source = func.call @strict_ptr(%instance) : (f64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%copy, %source, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %at = arith.fptosi %slot : f64 to i64
    %target = llvm.getelementptr %copy[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_list_write(%list: f64, %index: f64, %element: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %minusOne = arith.constant -1.0 : f64
    %blockSize = arith.constant 56 : i64
    %unset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %unset, ^copy, ^check
  ^check:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %newest = arith.cmpf oeq, %length, %used : f64
    %growable = arith.cmpf ogt, %capacity, %zero : f64
    %inPlace = arith.andi %newest, %growable : i1
    cf.cond_br %inPlace, ^check_separate, ^copy
  ^check_separate:
    %startBits = llvm.load %holder : !llvm.ptr -> f64
    %start = func.call @strict_ptr(%startBits) : (f64) -> !llvm.ptr
    %embedded = llvm.getelementptr %holder[40] : (!llvm.ptr) -> !llvm.ptr, i8
    %isEmbedded = llvm.icmp "eq" %start, %embedded : !llvm.ptr
    %writesSlot = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, f64
    %writes = llvm.load %writesSlot : !llvm.ptr -> f64
    %fewest = arith.constant 64.0 : f64
    %bound = arith.maximumf %capacity, %fewest : f64
    %exhausted = arith.cmpf oge, %writes, %bound : f64
    %fresh = arith.ori %isEmbedded, %exhausted : i1
    cf.cond_br %fresh, ^separate, ^write
  ^separate:
    %eight = arith.constant 8.0 : f64
    %separateBlock = func.call @strict_alloc(%blockSize) : (i64) -> !llvm.ptr
    %capacityBytesf = arith.mulf %capacity, %eight : f64
    %capacityBytes = arith.fptosi %capacityBytesf : f64 to i64
    %array = func.call @strict_alloc(%capacityBytes) : (i64) -> !llvm.ptr
    %usedBytesf = arith.mulf %used, %eight : f64
    %usedBytes = arith.fptosi %usedBytesf : f64 to i64
    %separated = llvm.call @memcpy(%array, %start, %usedBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %separateHolder = llvm.getelementptr %separateBlock[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %arrayBits = func.call @strict_bits(%array) : (!llvm.ptr) -> f64
    llvm.store %arrayBits, %separateHolder : f64, !llvm.ptr
    func.call @strict_gc_written(%separateHolder) : (!llvm.ptr) -> ()
    %separateUsed = llvm.getelementptr %separateHolder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %used, %separateUsed : f64, !llvm.ptr
    %separateCapacity = llvm.getelementptr %separateHolder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %capacity, %separateCapacity : f64, !llvm.ptr
    %separateIndex = llvm.getelementptr %separateHolder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %zero, %separateIndex : f64, !llvm.ptr
    %separateWrites = llvm.getelementptr %separateHolder[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %zero, %separateWrites : f64, !llvm.ptr
    llvm.store %length, %separateBlock : f64, !llvm.ptr
    %separateHolderBits = func.call @strict_bits(%separateHolder) : (!llvm.ptr) -> f64
    %separateView = llvm.getelementptr %separateBlock[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %separateHolderBits, %separateView : f64, !llvm.ptr
    %separateAt = arith.fptosi %index : f64 to i64
    %separateSlot = llvm.getelementptr %array[%separateAt] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %separateSlot : f64, !llvm.ptr
    %separateResult = func.call @strict_bits(%separateBlock) : (!llvm.ptr) -> f64
    return %separateResult : f64
  ^write:
    %block = func.call @strict_alloc(%blockSize) : (i64) -> !llvm.ptr
    %newer = llvm.getelementptr %block[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %itemsBits = llvm.load %holder : !llvm.ptr -> f64
    llvm.store %itemsBits, %newer : f64, !llvm.ptr
    %newerUsed = llvm.getelementptr %newer[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %used, %newerUsed : f64, !llvm.ptr
    %newerCapacity = llvm.getelementptr %newer[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %capacity, %newerCapacity : f64, !llvm.ptr
    %newerIndex = llvm.getelementptr %newer[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %zero, %newerIndex : f64, !llvm.ptr
    %one = arith.constant 1.0 : f64
    %counted = arith.addf %writes, %one : f64
    %newerWrites = llvm.getelementptr %newer[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %counted, %newerWrites : f64, !llvm.ptr
    %freshBits = func.call @strict_bits(%newer) : (!llvm.ptr) -> f64
    llvm.store %length, %block : f64, !llvm.ptr
    %blockHolder = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %freshBits, %blockHolder : f64, !llvm.ptr
    %items = func.call @strict_ptr(%itemsBits) : (f64) -> !llvm.ptr
    %at = arith.fptosi %index : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %old = llvm.load %slot : !llvm.ptr -> f64
    llvm.store %zero, %holder : f64, !llvm.ptr
    %encoded = arith.subf %minusOne, %index : f64
    llvm.store %encoded, %usedSlot : f64, !llvm.ptr
    llvm.store %old, %capacitySlot : f64, !llvm.ptr
    func.call @strict_gc_written(%capacitySlot) : (!llvm.ptr) -> ()
    %nextSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %freshBits, %nextSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%nextSlot) : (!llvm.ptr) -> ()
    llvm.store %element, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    %versionSlot = llvm.mlir.addressof @strict_versions : !llvm.ptr
    %version = llvm.load %versionSlot : !llvm.ptr -> i64
    %step = arith.constant 1 : i64
    %nextVersion = arith.addi %version, %step : i64
    llvm.store %nextVersion, %versionSlot : i64, !llvm.ptr
    %result = func.call @strict_bits(%block) : (!llvm.ptr) -> f64
    return %result : f64
  ^copy:
    %copied = func.call @strict_list_copy(%list, %zero) : (f64, f64) -> f64
    %set = func.call @strict_list_set(%copied, %index, %element) : (f64, f64, f64) -> f64
    return %copied : f64
  }
  func.func private @strict_list_length(%list: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %empty = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %empty, ^none, ^some
  ^none:
    return %zero : f64
  ^some:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    return %length : f64
  }
  func.func private @strict_list_get(%list: f64, %index: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %negative = arith.cmpf olt, %index, %zero : f64
    %wrapped = arith.addf %index, %length : f64
    %chosen = arith.select %negative, %wrapped, %index : f64
    %at = arith.fptosi %chosen : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    return %value : f64
  }
  func.func private @strict_error_text(%name: f64) -> f64 {
    %pointer = func.call @strict_ptr(%name) : (f64) -> !llvm.ptr
    %error = func.call @strict_error_named(%pointer) : (!llvm.ptr) -> f64
    return %error : f64
  }
  func.func private @strict_index_error(%name: f64) -> f64 attributes {passthrough = ["noinline"]} {
    %one = arith.constant 1 : i64
    %errorName = llvm.mlir.addressof @strict_index_out_of_range : !llvm.ptr
    %errorNameBits = func.call @strict_bits(%errorName) : (!llvm.ptr) -> f64
    %error = func.call @strict_error_text(%errorNameBits) : (f64) -> f64
    %pointer = func.call @strict_ptr(%name) : (f64) -> !llvm.ptr
    %nameLength = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %afterName = arith.addi %nameLength, %one : i64
    %location = llvm.getelementptr %pointer[%afterName] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %methodLength = llvm.call @strlen(%location) : (!llvm.ptr) -> i64
    %afterMethod = arith.addi %methodLength, %one : i64
    %file = llvm.getelementptr %location[%afterMethod] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %fileLength = llvm.call @strlen(%file) : (!llvm.ptr) -> i64
    %afterFile = arith.addi %fileLength, %one : i64
    %lineText = llvm.getelementptr %file[%afterFile] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lineBits = func.call @strict_bits(%lineText) : (!llvm.ptr) -> f64
    %lineNumber = func.call @strict_text_number(%lineBits) : (f64) -> f64
    %line = arith.fptosi %lineNumber : f64 to i64
    %traced = func.call @strict_error_trace(%error, %location, %line) : (f64, !llvm.ptr, i64) -> f64
    return %traced : f64
  }
  func.func private @strict_list_read(%list: f64, %index: f64, %name: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %negative = arith.cmpf olt, %index, %zero : f64
    %wrapped = arith.addf %index, %length : f64
    %chosen = arith.select %negative, %wrapped, %index : f64
    %below = arith.cmpf olt, %chosen, %zero : f64
    %above = arith.cmpf oge, %chosen, %length : f64
    %outside = arith.ori %below, %above : i1
    cf.cond_br %outside, ^error, ^inside
  ^error:
    %error = func.call @strict_index_error(%name) : (f64) -> f64
    return %error : f64
  ^inside:
    %value = func.call @strict_list_get(%list, %chosen) : (f64, f64) -> f64
    return %value : f64
  }
  func.func private @strict_hoisted_read(%items: !llvm.ptr, %length: f64, %index: f64, %name: f64, %version: i64, %list: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %negative = arith.cmpf olt, %index, %zero : f64
    %wrapped = arith.addf %index, %length : f64
    %chosen = arith.select %negative, %wrapped, %index : f64
    %below = arith.cmpf olt, %chosen, %zero : f64
    %above = arith.cmpf uge, %chosen, %length : f64
    %outside = arith.ori %below, %above : i1
    cf.cond_br %outside, ^error, ^inside
  ^error:
    %error = func.call @strict_index_error(%name) : (f64) -> f64
    return %error : f64
  ^inside:
    %value = func.call @strict_hoisted_item(%items, %chosen, %version, %list) : (!llvm.ptr, f64, i64, f64) -> f64
    return %value : f64
  }
  func.func private @strict_list_start(%list: f64) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
    %unset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %unset, ^none, ^some
  ^none:
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  ^some:
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  }
  func.func private @strict_list_hoisted(%list: f64) -> !llvm.ptr {
    %packed = func.call @strict_list_packed(%list) : (f64) -> i64
    %none = arith.constant 0 : i64
    %isPlain = arith.cmpi eq, %packed, %none : i64
    cf.cond_br %isPlain, ^plain, ^packed
  ^plain:
    %items = func.call @strict_list_start(%list) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  ^packed:
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }
  func.func private @strict_list_packed(%list: f64) -> i64 attributes {passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %unset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %unset, ^plain, ^check
  ^check:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %itemsBits = llvm.load %holder : !llvm.ptr -> f64
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %noItems = arith.cmpf oeq, %itemsBits, %zero : f64
    %isNewest = arith.cmpf oge, %used, %zero : f64
    %narrow = arith.cmpf olt, %capacity, %zero : f64
    %newestItems = arith.andi %noItems, %isNewest : i1
    %isPacked = arith.andi %newestItems, %narrow : i1
    %bitsNumber = arith.negf %capacity : f64
    %bits = arith.fptosi %bitsNumber : f64 to i64
    %result = arith.select %isPacked, %bits, %none : i64
    return %result : i64
  ^plain:
    return %none : i64
  }
  func.func private @strict_list_scales(%list: f64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %count = arith.fptosi %length : f64 to i64
    %bits = func.call @strict_list_packed(%list) : (f64) -> i64
    %scales = func.call @strict_packed_scales(%data, %count, %bits) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    return %scales : !llvm.ptr
  }
  // ponytail: one buffer per thread, a fast loop has at most one packed source; rows of 4 or 6 bits decode to floats
  llvm.mlir.global internal thread_local @strict_float_buffer(0 : i64) : i64
  llvm.mlir.global internal thread_local @strict_float_capacity(0 : i64) : i64
  func.func private @strict_list_floats(%list: f64) -> !llvm.ptr {
    %eight = arith.constant 8 : i64
    %spare = arith.constant 255 : i64
    %four = arith.constant 4 : i64
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %count = arith.fptosi %length : f64 to i64
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    %bits = func.call @strict_list_packed(%list) : (f64) -> i64
    %rounded = arith.addi %count, %spare : i64
    %blocks = arith.shrui %rounded, %eight : i64
    %floats = arith.shli %blocks, %eight : i64
    %bufferSlot = llvm.mlir.addressof @strict_float_buffer : !llvm.ptr
    %capacitySlot = llvm.mlir.addressof @strict_float_capacity : !llvm.ptr
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %enough = arith.cmpi sge, %capacity, %floats : i64
    cf.cond_br %enough, ^ready, ^grow
  ^grow:
    %oldBits = llvm.load %bufferSlot : !llvm.ptr -> i64
    %old = llvm.inttoptr %oldBits : i64 to !llvm.ptr
    llvm.call @free(%old) : (!llvm.ptr) -> ()
    %bytes = arith.muli %floats, %four : i64
    %fresh = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    %freshBits = llvm.ptrtoint %fresh : !llvm.ptr to i64
    llvm.store %freshBits, %bufferSlot : i64, !llvm.ptr
    llvm.store %floats, %capacitySlot : i64, !llvm.ptr
    cf.br ^ready
  ^ready:
    %bufferBits = llvm.load %bufferSlot : !llvm.ptr -> i64
    %buffer = llvm.inttoptr %bufferBits : i64 to !llvm.ptr
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    cf.br ^block(%none : i64)
  ^block(%block: i64):
    %more = arith.cmpi slt, %block, %blocks : i64
    cf.cond_br %more, ^decode, ^done
  ^decode:
    %first = arith.shli %block, %eight : i64
    %target = llvm.getelementptr %buffer[%first] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    func.call @strict_packed_block(%data, %count, %bits, %block, %target) : (!llvm.ptr, i64, i64, i64, !llvm.ptr) -> ()
    %next = arith.addi %block, %one : i64
    cf.br ^block(%next : i64)
  ^done:
    return %buffer : !llvm.ptr
  }
  // the 256 numbers of block number block of a list of 4 or 6 bits as floats
  func.func private @strict_packed_block(%data: !llvm.ptr, %count: i64, %bits: i64, %block: i64, %out: !llvm.ptr) {
    %four = arith.constant 4 : i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    cf.cond_br %isFour, ^four, ^six
  ^four:
    %fourBytes = arith.constant 144 : i64
    %fourOffset = arith.muli %block, %fourBytes : i64
    %fourBlock = llvm.getelementptr %data[%fourOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    func.call @strict_k4_block(%fourBlock, %out) : (!llvm.ptr, !llvm.ptr) -> ()
    return
  ^six:
    %sixBytes = arith.constant 208 : i64
    %sixOffset = arith.muli %block, %sixBytes : i64
    %sixBlock = llvm.getelementptr %data[%sixOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halves = func.call @strict_packed_scales(%data, %count, %bits) : (!llvm.ptr, i64, i64) -> !llvm.ptr
    %halfSlot = llvm.getelementptr %halves[%block] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %half = llvm.load %halfSlot : !llvm.ptr -> i16
    %wideScale = func.call @strict_half_number(%half) : (i16) -> f64
    %scale = arith.truncf %wideScale : f64 to f32
    func.call @strict_k6_block(%sixBlock, %scale, %out) : (!llvm.ptr, f32, !llvm.ptr) -> ()
    return
  }
  // the 6 bit scale and minimum of group number group (0 to 7) of a Q4_K block (get_scale_min_k4 of ggml)
  func.func private @strict_k4_scale(%scales: !llvm.ptr, %group: i64) -> (f32, f32) attributes {passthrough = ["alwaysinline"]} {
    %four = arith.constant 4 : i64
    %sixBits = arith.constant 63 : i32
    %fourBits = arith.constant 15 : i32
    %six = arith.constant 6 : i32
    %shift = arith.constant 4 : i32
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
    %scaleNumber = arith.uitofp %scale : i32 to f32
    %minimumNumber = arith.uitofp %minimum : i32 to f32
    return %scaleNumber, %minimumNumber : f32, f32
  }
  // a float in all 32 lanes of a vector
  func.func private @strict_splat(%value: f32) -> vector<32xf32> attributes {passthrough = ["alwaysinline"]} {
    %undefined = llvm.mlir.undef : vector<32xf32>
    %first = llvm.mlir.constant(0 : i32) : i32
    %inserted = llvm.insertelement %value, %undefined[%first : i32] : vector<32xf32>
    %splat = llvm.shufflevector %inserted, %undefined [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<32xf32>
    return %splat : vector<32xf32>
  }
  // the 256 numbers of a Q4_K block (half d and dmin, 12 bytes of 6 bit scales and minimums, 128 bytes of nibbles: the
  // low ones of 32 bytes are the first 32 numbers, their high ones the next 32), each the scale times the nibble minus
  // the minimum like dequantize_row_q4_K of ggml, 32 numbers at once
  func.func private @strict_k4_block(%block: !llvm.ptr, %out: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %partCount = arith.constant 4 : i64
    %fifteen = arith.constant dense<15> : vector<32xi32>
    %four = arith.constant dense<4> : vector<32xi32>
    %d = llvm.load %block : !llvm.ptr -> i16
    %wideScale = func.call @strict_half_number(%d) : (i16) -> f64
    %scale = arith.truncf %wideScale : f64 to f32
    %minimumSlot = llvm.getelementptr %block[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %dmin = llvm.load %minimumSlot : !llvm.ptr -> i16
    %wideMinimum = func.call @strict_half_number(%dmin) : (i16) -> f64
    %minimumScale = arith.truncf %wideMinimum : f64 to f32
    %scales = llvm.getelementptr %block[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %nibbles = llvm.getelementptr %block[16] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^part(%none : i64)
  ^part(%part: i64):
    %more = arith.cmpi slt, %part, %partCount : i64
    cf.cond_br %more, ^decode, ^done
  ^decode:
    %lowGroup = arith.shli %part, %one : i64
    %highGroup = arith.addi %lowGroup, %one : i64
    %low:2 = func.call @strict_k4_scale(%scales, %lowGroup) : (!llvm.ptr, i64) -> (f32, f32)
    %high:2 = func.call @strict_k4_scale(%scales, %highGroup) : (!llvm.ptr, i64) -> (f32, f32)
    %lowFactor = arith.mulf %scale, %low#0 : f32
    %lowOffset = arith.mulf %minimumScale, %low#1 : f32
    %highFactor = arith.mulf %scale, %high#0 : f32
    %highOffset = arith.mulf %minimumScale, %high#1 : f32
    %thirtyTwo = arith.constant 32 : i64
    %byteStart = arith.muli %part, %thirtyTwo : i64
    %byteSlot = llvm.getelementptr %nibbles[%byteStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %bytes = llvm.load %byteSlot {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
    %wide = arith.extui %bytes : vector<32xi8> to vector<32xi32>
    %lowBits = arith.andi %wide, %fifteen : vector<32xi32>
    %highBits = arith.shrui %wide, %four : vector<32xi32>
    %lowLevels = arith.uitofp %lowBits : vector<32xi32> to vector<32xf32>
    %highLevels = arith.uitofp %highBits : vector<32xi32> to vector<32xf32>
    %lowFactors = func.call @strict_splat(%lowFactor) : (f32) -> vector<32xf32>
    %lowOffsets = func.call @strict_splat(%lowOffset) : (f32) -> vector<32xf32>
    %highFactors = func.call @strict_splat(%highFactor) : (f32) -> vector<32xf32>
    %highOffsets = func.call @strict_splat(%highOffset) : (f32) -> vector<32xf32>
    %lowScaled = arith.mulf %lowFactors, %lowLevels : vector<32xf32>
    %lowValues = arith.subf %lowScaled, %lowOffsets : vector<32xf32>
    %highScaled = arith.mulf %highFactors, %highLevels : vector<32xf32>
    %highValues = arith.subf %highScaled, %highOffsets : vector<32xf32>
    %sixtyFour = arith.constant 64 : i64
    %first = arith.muli %part, %sixtyFour : i64
    %lowTarget = llvm.getelementptr %out[%first] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %lowValues, %lowTarget {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %highTarget = llvm.getelementptr %lowTarget[32] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %highValues, %highTarget {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %next = arith.addi %part, %one : i64
    cf.br ^part(%next : i64)
  ^done:
    return
  }
  // the 256 numbers of a Q6_K block without its scale d (128 bytes of low nibbles, 64 of high bit pairs, a signed byte
  // scale per 16), each half of 128 numbers is four quarters: the low nibbles of the first and second 32 bytes, then their
  // high nibbles, with bit pairs 0, 2, 4 and 6 of the same 32 bytes, like dequantize_row_q6_K of ggml, 32 numbers at once
  func.func private @strict_k6_block(%body: !llvm.ptr, %scale: f32, %out: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fifteen = arith.constant dense<15> : vector<32xi32>
    %four = arith.constant dense<4> : vector<32xi32>
    %three = arith.constant dense<3> : vector<32xi32>
    %middle = arith.constant dense<32> : vector<32xi32>
    cf.br ^half(%none : i64)
  ^half(%half: i64):
    %more = arith.cmpi slt, %half, %two : i64
    cf.cond_br %more, ^decode, ^done
  ^decode:
    %sixtyFour = arith.constant 64 : i64
    %thirtyTwo = arith.constant 32 : i64
    %eight = arith.constant 8 : i64
    %highStart = arith.constant 128 : i64
    %scaleStart = arith.constant 192 : i64
    %lowStart = arith.muli %half, %sixtyFour : i64
    %lowSlot = llvm.getelementptr %body[%lowStart] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %lowBytes = llvm.load %lowSlot {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
    %secondSlot = llvm.getelementptr %lowSlot[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondBytes = llvm.load %secondSlot {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
    %halfHigh = arith.muli %half, %thirtyTwo : i64
    %highAt = arith.addi %highStart, %halfHigh : i64
    %highSlot = llvm.getelementptr %body[%highAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %highBytes = llvm.load %highSlot {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
    %lowWide = arith.extui %lowBytes : vector<32xi8> to vector<32xi32>
    %secondWide = arith.extui %secondBytes : vector<32xi8> to vector<32xi32>
    %highWide = arith.extui %highBytes : vector<32xi8> to vector<32xi32>
    %halfScales = arith.muli %half, %eight : i64
    %groupAt = arith.addi %scaleStart, %halfScales : i64
    %groupScales = llvm.getelementptr %body[%groupAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %halfFirst = arith.constant 128 : i64
    %outStart = arith.muli %half, %halfFirst : i64
    %halfOut = llvm.getelementptr %out[%outStart] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %nibble0 = arith.andi %lowWide, %fifteen : vector<32xi32>
    %pairShift0 = arith.constant dense<0> : vector<32xi32>
    %pairs0 = arith.shrui %highWide, %pairShift0 : vector<32xi32>
    %pair0 = arith.andi %pairs0, %three : vector<32xi32>
    %placed0 = arith.shli %pair0, %four : vector<32xi32>
    %combined0 = arith.ori %nibble0, %placed0 : vector<32xi32>
    %centered0 = arith.subi %combined0, %middle : vector<32xi32>
    %level0 = arith.sitofp %centered0 : vector<32xi32> to vector<32xf32>
    %firstScaleSlot0 = llvm.getelementptr %groupScales[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %firstScaleByte0 = llvm.load %firstScaleSlot0 : !llvm.ptr -> i8
    %firstScale0 = arith.sitofp %firstScaleByte0 : i8 to f32
    %secondScaleSlot0 = llvm.getelementptr %groupScales[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondScaleByte0 = llvm.load %secondScaleSlot0 : !llvm.ptr -> i8
    %secondScale0 = arith.sitofp %secondScaleByte0 : i8 to f32
    %firstFactor0 = arith.mulf %scale, %firstScale0 : f32
    %secondFactor0 = arith.mulf %scale, %secondScale0 : f32
    %firstFactors0 = func.call @strict_splat(%firstFactor0) : (f32) -> vector<32xf32>
    %secondFactors0 = func.call @strict_splat(%secondFactor0) : (f32) -> vector<32xf32>
    %factors0 = llvm.shufflevector %firstFactors0, %secondFactors0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf32>
    %values0 = arith.mulf %factors0, %level0 : vector<32xf32>
    %target0 = llvm.getelementptr %halfOut[0] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %values0, %target0 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %nibble1 = arith.andi %secondWide, %fifteen : vector<32xi32>
    %pairShift1 = arith.constant dense<2> : vector<32xi32>
    %pairs1 = arith.shrui %highWide, %pairShift1 : vector<32xi32>
    %pair1 = arith.andi %pairs1, %three : vector<32xi32>
    %placed1 = arith.shli %pair1, %four : vector<32xi32>
    %combined1 = arith.ori %nibble1, %placed1 : vector<32xi32>
    %centered1 = arith.subi %combined1, %middle : vector<32xi32>
    %level1 = arith.sitofp %centered1 : vector<32xi32> to vector<32xf32>
    %firstScaleSlot1 = llvm.getelementptr %groupScales[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %firstScaleByte1 = llvm.load %firstScaleSlot1 : !llvm.ptr -> i8
    %firstScale1 = arith.sitofp %firstScaleByte1 : i8 to f32
    %secondScaleSlot1 = llvm.getelementptr %groupScales[3] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondScaleByte1 = llvm.load %secondScaleSlot1 : !llvm.ptr -> i8
    %secondScale1 = arith.sitofp %secondScaleByte1 : i8 to f32
    %firstFactor1 = arith.mulf %scale, %firstScale1 : f32
    %secondFactor1 = arith.mulf %scale, %secondScale1 : f32
    %firstFactors1 = func.call @strict_splat(%firstFactor1) : (f32) -> vector<32xf32>
    %secondFactors1 = func.call @strict_splat(%secondFactor1) : (f32) -> vector<32xf32>
    %factors1 = llvm.shufflevector %firstFactors1, %secondFactors1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf32>
    %values1 = arith.mulf %factors1, %level1 : vector<32xf32>
    %target1 = llvm.getelementptr %halfOut[32] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %values1, %target1 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %nibble2 = arith.shrui %lowWide, %four : vector<32xi32>
    %pairShift2 = arith.constant dense<4> : vector<32xi32>
    %pairs2 = arith.shrui %highWide, %pairShift2 : vector<32xi32>
    %pair2 = arith.andi %pairs2, %three : vector<32xi32>
    %placed2 = arith.shli %pair2, %four : vector<32xi32>
    %combined2 = arith.ori %nibble2, %placed2 : vector<32xi32>
    %centered2 = arith.subi %combined2, %middle : vector<32xi32>
    %level2 = arith.sitofp %centered2 : vector<32xi32> to vector<32xf32>
    %firstScaleSlot2 = llvm.getelementptr %groupScales[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %firstScaleByte2 = llvm.load %firstScaleSlot2 : !llvm.ptr -> i8
    %firstScale2 = arith.sitofp %firstScaleByte2 : i8 to f32
    %secondScaleSlot2 = llvm.getelementptr %groupScales[5] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondScaleByte2 = llvm.load %secondScaleSlot2 : !llvm.ptr -> i8
    %secondScale2 = arith.sitofp %secondScaleByte2 : i8 to f32
    %firstFactor2 = arith.mulf %scale, %firstScale2 : f32
    %secondFactor2 = arith.mulf %scale, %secondScale2 : f32
    %firstFactors2 = func.call @strict_splat(%firstFactor2) : (f32) -> vector<32xf32>
    %secondFactors2 = func.call @strict_splat(%secondFactor2) : (f32) -> vector<32xf32>
    %factors2 = llvm.shufflevector %firstFactors2, %secondFactors2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf32>
    %values2 = arith.mulf %factors2, %level2 : vector<32xf32>
    %target2 = llvm.getelementptr %halfOut[64] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %values2, %target2 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %nibble3 = arith.shrui %secondWide, %four : vector<32xi32>
    %pairShift3 = arith.constant dense<6> : vector<32xi32>
    %pairs3 = arith.shrui %highWide, %pairShift3 : vector<32xi32>
    %pair3 = arith.andi %pairs3, %three : vector<32xi32>
    %placed3 = arith.shli %pair3, %four : vector<32xi32>
    %combined3 = arith.ori %nibble3, %placed3 : vector<32xi32>
    %centered3 = arith.subi %combined3, %middle : vector<32xi32>
    %level3 = arith.sitofp %centered3 : vector<32xi32> to vector<32xf32>
    %firstScaleSlot3 = llvm.getelementptr %groupScales[6] : (!llvm.ptr) -> !llvm.ptr, i8
    %firstScaleByte3 = llvm.load %firstScaleSlot3 : !llvm.ptr -> i8
    %firstScale3 = arith.sitofp %firstScaleByte3 : i8 to f32
    %secondScaleSlot3 = llvm.getelementptr %groupScales[7] : (!llvm.ptr) -> !llvm.ptr, i8
    %secondScaleByte3 = llvm.load %secondScaleSlot3 : !llvm.ptr -> i8
    %secondScale3 = arith.sitofp %secondScaleByte3 : i8 to f32
    %firstFactor3 = arith.mulf %scale, %firstScale3 : f32
    %secondFactor3 = arith.mulf %scale, %secondScale3 : f32
    %firstFactors3 = func.call @strict_splat(%firstFactor3) : (f32) -> vector<32xf32>
    %secondFactors3 = func.call @strict_splat(%secondFactor3) : (f32) -> vector<32xf32>
    %factors3 = llvm.shufflevector %firstFactors3, %secondFactors3 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf32>
    %values3 = arith.mulf %factors3, %level3 : vector<32xf32>
    %target3 = llvm.getelementptr %halfOut[96] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %values3, %target3 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %next = arith.addi %half, %one : i64
    cf.br ^half(%next : i64)
  ^done:
    return
  }
  func.func private @strict_list_data(%list: f64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %dataBits = llvm.load %dataSlot : !llvm.ptr -> f64
    %zero = arith.constant 0.0 : f64
    %missing = arith.cmpf oeq, %dataBits, %zero : f64
    cf.cond_br %missing, ^download, ^present
  ^download:
    %downloaded = func.call @strict_gpu_download(%holder) : (!llvm.ptr) -> !llvm.ptr
    return %downloaded : !llvm.ptr
  ^present:
    %data = func.call @strict_ptr(%dataBits) : (f64) -> !llvm.ptr
    return %data : !llvm.ptr
  }
  func.func private @strict_list_copy(%list: f64, %extra: f64) -> f64 {
    %eight = arith.constant 8.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %capacity = arith.addf %length, %extra : f64
    %copy = func.call @strict_list_new(%capacity, %length) : (f64, f64) -> f64
    %from = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %to = func.call @strict_list_items(%copy) : (f64) -> !llvm.ptr
    %bytesf = arith.mulf %length, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %done = llvm.call @memcpy(%to, %from, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    func.call @strict_gc_written_range(%to, %bytes) : (!llvm.ptr, i64) -> ()
    return %copy : f64
  }
  func.func private @strict_list_set(%list: f64, %index: f64, %element: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %at = arith.fptosi %index : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    return %zero : f64
  }
  func.func private @strict_list_add(%list: f64, %element: f64) -> f64 {
    %nothing = arith.constant 0.0 : f64
    %unset = arith.cmpf oeq, %list, %nothing : f64
    cf.cond_br %unset, ^fresh, ^existing
  ^fresh:
    %capacity = arith.constant 4.0 : f64
    %created = func.call @strict_list_new(%capacity, %nothing) : (f64, f64) -> f64
    %first = func.call @strict_list_add(%created, %element) : (f64, f64) -> f64
    return %first : f64
  ^existing:
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %four = arith.constant 4.0 : f64
    %eight = arith.constant 8.0 : f64
    %c16 = arith.constant 16 : i64
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %viewholder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderbits = llvm.load %viewholder : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderbits) : (f64) -> !llvm.ptr
    %usedslot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedslot : !llvm.ptr -> f64
    %newest = arith.cmpf oeq, %length, %used : f64
    %capacityslot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %holdCapacity = llvm.load %capacityslot : !llvm.ptr -> f64
    %nothingYet = arith.constant 0.0 : f64
    %growable = arith.cmpf ogt, %holdCapacity, %nothingYet : f64
    %inPlace = arith.andi %newest, %growable : i1
    cf.cond_br %inPlace, ^grow, ^copy
  ^copy:
    %copied = func.call @strict_list_copy(%list, %four) : (f64, f64) -> f64
    %again = func.call @strict_list_add(%copied, %element) : (f64, f64) -> f64
    return %again : f64
  ^grow:
    %capslot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %room = llvm.load %capslot : !llvm.ptr -> f64
    %full = arith.cmpf oge, %used, %room : f64
    cf.cond_br %full, ^resize, ^store
  ^resize:
    %doubled = arith.mulf %room, %two : f64
    %larger = arith.addf %doubled, %four : f64
    %bytesf = arith.mulf %larger, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %oldbits = llvm.load %holder : !llvm.ptr -> f64
    %old = func.call @strict_ptr(%oldbits) : (f64) -> !llvm.ptr
    %moved = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    %usedBytesf = arith.mulf %used, %eight : f64
    %usedBytes = arith.fptosi %usedBytesf : f64 to i64
    %grown = llvm.call @memcpy(%moved, %old, %usedBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %movedbits = func.call @strict_bits(%moved) : (!llvm.ptr) -> f64
    llvm.store %movedbits, %holder : f64, !llvm.ptr
    func.call @strict_gc_written(%holder) : (!llvm.ptr) -> ()
    llvm.store %larger, %capslot : f64, !llvm.ptr
    cf.br ^store
  ^store:
    %itemsbits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsbits) : (f64) -> !llvm.ptr
    %at = arith.fptosi %used : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    %next = arith.addf %used, %one : f64
    llvm.store %next, %usedslot : f64, !llvm.ptr
    %result = func.call @strict_alloc(%c16) : (i64) -> !llvm.ptr
    llvm.store %next, %result : f64, !llvm.ptr
    %resultholder = llvm.getelementptr %result[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %resultholder : f64, !llvm.ptr
    %bits = func.call @strict_bits(%result) : (!llvm.ptr) -> f64
    return %bits : f64
  }
  // compiled loops write their results straight into the items (parallel ones from other threads), so all cards
  // of the filled items are marked here
  func.func private @strict_list_filled(%list: f64, %length: f64) -> f64 {
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    llvm.store %length, %view : f64, !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %length, %usedSlot : f64, !llvm.ptr
    %itemsBits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsBits) : (f64) -> !llvm.ptr
    %eight = arith.constant 8.0 : f64
    %bytesNumber = arith.mulf %length, %eight : f64
    %bytes = arith.fptosi %bytesNumber : f64 to i64
    func.call @strict_gc_written_range(%items, %bytes) : (!llvm.ptr, i64) -> ()
    return %list : f64
  }
  func.func private @strict_list_push(%list: f64, %element: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %unset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %unset, ^add, ^check
  ^check:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %length = llvm.load %view : !llvm.ptr -> f64
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %newest = arith.cmpf oeq, %length, %used : f64
    %room = arith.cmpf olt, %used, %capacity : f64
    %inPlace = arith.andi %newest, %room : i1
    cf.cond_br %inPlace, ^push, ^add
  ^push:
    %itemsBits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsBits) : (f64) -> !llvm.ptr
    %at = arith.fptosi %used : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    %next = arith.addf %used, %one : f64
    llvm.store %next, %usedSlot : f64, !llvm.ptr
    llvm.store %next, %view : f64, !llvm.ptr
    return %list : f64
  ^add:
    %added = func.call @strict_list_add(%list, %element) : (f64, f64) -> f64
    return %added : f64
  }
  // two lists computed on the GPU are joined there, an empty list before one is just that list
  func.func private @strict_list_concat(%left: f64, %right: f64) -> f64 {
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f64
    %rightDevice = func.call @strict_gpu_device(%right) : (f64) -> i64
    %rightResides = arith.cmpi ne, %rightDevice, %none : i64
    cf.cond_br %rightResides, ^device, ^host
  ^device:
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %isEmpty = arith.cmpf oeq, %leftLength, %zero : f64
    cf.cond_br %isEmpty, ^right, ^both
  ^right:
    return %right : f64
  ^both:
    %leftDevice = func.call @strict_gpu_device(%left) : (f64) -> i64
    %leftResides = arith.cmpi ne, %leftDevice, %none : i64
    cf.cond_br %leftResides, ^joined(%left : f64), ^move
  ^move:
    %moved = func.call @strict_gpu_moved(%left) : (f64) -> f64
    %isMoved = arith.cmpf une, %moved, %zero : f64
    cf.cond_br %isMoved, ^joined(%moved : f64), ^host
  ^joined(%resident: f64):
    %joined = func.call @strict_gpu_concat(%resident, %right) : (f64, f64) -> f64
    %isJoined = arith.cmpf une, %joined, %zero : f64
    cf.cond_br %isJoined, ^done(%joined : f64), ^host
  ^host:
    %copied = func.call @strict_list_concat_host(%left, %right) : (f64, f64) -> f64
    cf.br ^done(%copied : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_list_concat_host(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %two = arith.constant 2.0 : f64
    %four = arith.constant 4.0 : f64
    %eight = arith.constant 8.0 : f64
    %c16 = arith.constant 16 : i64
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %hasLeft = arith.cmpf ogt, %leftLength, %zero : f64
    cf.cond_br %hasLeft, ^check, ^copy
  ^check:
    %view = func.call @strict_ptr(%left) : (f64) -> !llvm.ptr
    %viewholder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderbits = llvm.load %viewholder : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderbits) : (f64) -> !llvm.ptr
    %usedslot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedslot : !llvm.ptr -> f64
    %capacityslot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacityslot : !llvm.ptr -> f64
    %newest = arith.cmpf oeq, %leftLength, %used : f64
    %growable = arith.cmpf ogt, %capacity, %zero : f64
    %inPlace = arith.andi %newest, %growable : i1
    cf.cond_br %inPlace, ^append, ^copy
  ^append:
    %needed = arith.addf %used, %rightLength : f64
    %full = arith.cmpf ogt, %needed, %capacity : f64
    cf.cond_br %full, ^grow, ^place
  ^grow:
    %doubled = arith.mulf %capacity, %two : f64
    %padded = arith.addf %doubled, %four : f64
    %larger = arith.maximumf %padded, %needed : f64
    %bytesf = arith.mulf %larger, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %oldbits = llvm.load %holder : !llvm.ptr -> f64
    %old = func.call @strict_ptr(%oldbits) : (f64) -> !llvm.ptr
    %moved = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    %usedBytesf = arith.mulf %used, %eight : f64
    %usedBytes = arith.fptosi %usedBytesf : f64 to i64
    %grown = llvm.call @memcpy(%moved, %old, %usedBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %movedbits = func.call @strict_bits(%moved) : (!llvm.ptr) -> f64
    llvm.store %movedbits, %holder : f64, !llvm.ptr
    func.call @strict_gc_written(%holder) : (!llvm.ptr) -> ()
    llvm.store %larger, %capacityslot : f64, !llvm.ptr
    cf.br ^place
  ^place:
    %hasRight = arith.cmpf ogt, %rightLength, %zero : f64
    cf.cond_br %hasRight, ^fill, ^placed
  ^fill:
    %itemsbits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsbits) : (f64) -> !llvm.ptr
    %at = arith.fptosi %used : f64 to i64
    %into = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %source = func.call @strict_list_items(%right) : (f64) -> !llvm.ptr
    %rightBytesf = arith.mulf %rightLength, %eight : f64
    %rightBytes = arith.fptosi %rightBytesf : f64 to i64
    %filled = llvm.call @memcpy(%into, %source, %rightBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    func.call @strict_gc_written_range(%into, %rightBytes) : (!llvm.ptr, i64) -> ()
    cf.br ^placed
  ^placed:
    llvm.store %needed, %usedslot : f64, !llvm.ptr
    %result = func.call @strict_alloc(%c16) : (i64) -> !llvm.ptr
    llvm.store %needed, %result : f64, !llvm.ptr
    %resultholder = llvm.getelementptr %result[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %resultholder : f64, !llvm.ptr
    %bits = func.call @strict_bits(%result) : (!llvm.ptr) -> f64
    return %bits : f64
  ^copy:
    %extra = arith.addf %rightLength, %four : f64
    %copy = func.call @strict_list_copy(%left, %extra) : (f64, f64) -> f64
    %hasCopied = arith.cmpf ogt, %rightLength, %zero : f64
    cf.cond_br %hasCopied, ^copyRight, ^copied
  ^copyRight:
    %to = func.call @strict_list_items(%copy) : (f64) -> !llvm.ptr
    %offset = arith.fptosi %leftLength : f64 to i64
    %target = llvm.getelementptr %to[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %from = func.call @strict_list_items(%right) : (f64) -> !llvm.ptr
    %copyBytesf = arith.mulf %rightLength, %eight : f64
    %copyBytes = arith.fptosi %copyBytesf : f64 to i64
    %done = llvm.call @memcpy(%target, %from, %copyBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    func.call @strict_gc_written_range(%target, %copyBytes) : (!llvm.ptr, i64) -> ()
    cf.br ^copied
  ^copied:
    %total = arith.addf %leftLength, %rightLength : f64
    %copyView = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    llvm.store %total, %copyView : f64, !llvm.ptr
    %copyHolderSlot = llvm.getelementptr %copyView[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %copyHolderBits = llvm.load %copyHolderSlot : !llvm.ptr -> f64
    %copyHolder = func.call @strict_ptr(%copyHolderBits) : (f64) -> !llvm.ptr
    %copyUsed = llvm.getelementptr %copyHolder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %total, %copyUsed : f64, !llvm.ptr
    return %copy : f64
  }
  func.func private @strict_list_prepend(%element: f64, %list: f64) -> f64 {
    %nothing = arith.constant 0.0 : f64
    %single = func.call @strict_list_add(%nothing, %element) : (f64, f64) -> f64
    %result = func.call @strict_list_concat(%single, %list) : (f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_downcast(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %count = func.call @strict_list_length(%right) : (f64) -> f64
    %empty = arith.cmpf oeq, %count, %zero : f64
    cf.cond_br %empty, ^unchanged, ^error
  ^unchanged:
    return %left : f64
  ^error:
    %first = func.call @strict_list_get(%right, %zero) : (f64, f64) -> f64
    %prefixPointer = llvm.mlir.addressof @strict_downcast_prefix : !llvm.ptr
    %prefix = func.call @strict_bits(%prefixPointer) : (!llvm.ptr) -> f64
    %quotePointer = llvm.mlir.addressof @strict_quote : !llvm.ptr
    %quote = func.call @strict_bits(%quotePointer) : (!llvm.ptr) -> f64
    %named = func.call @strict_text_concat(%prefix, %first) : (f64, f64) -> f64
    %message = func.call @strict_text_concat(%named, %quote) : (f64, f64) -> f64
    %error = func.call @strict_error_text(%message) : (f64) -> f64
    return %error : f64
  }
  func.func private @strict_list_without(%left: f64, %right: f64, %equal: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %copy = func.call @strict_list_copy(%left, %zero) : (f64, f64) -> f64
    %length = func.call @strict_list_length(%right) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    cf.br ^loop(%start, %copy : i64, f64)
  ^loop(%index: i64, %current: f64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %items = func.call @strict_list_items(%right) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %item = llvm.load %slot : !llvm.ptr -> f64
    %removed = func.call @strict_list_remove(%current, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    %next = arith.addi %index, %step : i64
    cf.br ^loop(%next, %removed : i64, f64)
  ^done:
    return %current : f64
  }
  func.func private @strict_list_without_number(%left: f64, %right: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_without(%left, %right, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_without_text(%left: f64, %right: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_without(%left, %right, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_error_named(%name: !llvm.ptr) -> f64 {
    %sixteen = arith.constant 16 : i64
    %block = func.call @strict_alloc(%sixteen) : (i64) -> !llvm.ptr
    %nameBits = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    llvm.store %nameBits, %block : f64, !llvm.ptr
    %traces = func.call @strict_list_empty() : () -> f64
    %tracesSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %traces, %tracesSlot : f64, !llvm.ptr
    %error = func.call @strict_error_bits(%block) : (!llvm.ptr) -> f64
    return %error : f64
  }
  func.func private @strict_list_zip(%left: f64, %right: f64, %divide: i1, %broadcast: i1) -> f64 {
    %one = arith.constant 1.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %same = arith.cmpf oeq, %leftLength, %rightLength : f64
    %leftSingle = arith.cmpf oeq, %leftLength, %one : f64
    %rightSingle = arith.cmpf oeq, %rightLength, %one : f64
    %single = arith.ori %leftSingle, %rightSingle : i1
    %spread = arith.andi %broadcast, %single : i1
    %fits = arith.ori %same, %spread : i1
    cf.cond_br %fits, ^combine, ^mismatch
  ^mismatch:
    %name = llvm.mlir.addressof @strict_dimensions : !llvm.ptr
    %error = func.call @strict_error_named(%name) : (!llvm.ptr) -> f64
    return %error : f64
  ^combine:
    %longer = arith.cmpf ogt, %leftLength, %rightLength : f64
    %length = arith.select %longer, %leftLength, %rightLength : f64
    %result = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    cf.br ^loop(%start : i64)
  ^loop(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %leftAt = arith.select %leftSingle, %start, %index : i64
    %rightAt = arith.select %rightSingle, %start, %index : i64
    %leftItems = func.call @strict_list_items(%left) : (f64) -> !llvm.ptr
    %leftSlot = llvm.getelementptr %leftItems[%leftAt] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %first = llvm.load %leftSlot : !llvm.ptr -> f64
    %rightItems = func.call @strict_list_items(%right) : (f64) -> !llvm.ptr
    %rightSlot = llvm.getelementptr %rightItems[%rightAt] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %second = llvm.load %rightSlot : !llvm.ptr -> f64
    %product = arith.mulf %first, %second : f64
    %quotient = arith.divf %first, %second : f64
    %value = arith.select %divide, %quotient, %product : f64
    %resultItems = func.call @strict_list_items(%result) : (f64) -> !llvm.ptr
    %target = llvm.getelementptr %resultItems[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
    %next = arith.addi %index, %step : i64
    cf.br ^loop(%next : i64)
  ^done:
    return %result : f64
  }
  func.func private @strict_list_times(%left: f64, %right: f64) -> f64 {
    %no = arith.constant false
    %result = func.call @strict_list_zip(%left, %right, %no, %no) : (f64, f64, i1, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_divided(%left: f64, %right: f64) -> f64 {
    %yes = arith.constant true
    %result = func.call @strict_list_zip(%left, %right, %yes, %yes) : (f64, f64, i1, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_scalar(%list: f64, %number: f64, %divide: i1) -> f64 {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %result = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    cf.br ^loop(%start : i64)
  ^loop(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %element = llvm.load %slot : !llvm.ptr -> f64
    %product = arith.mulf %element, %number : f64
    %quotient = arith.divf %element, %number : f64
    %value = arith.select %divide, %quotient, %product : f64
    %resultItems = func.call @strict_list_items(%result) : (f64) -> !llvm.ptr
    %target = llvm.getelementptr %resultItems[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
    %next = arith.addi %index, %step : i64
    cf.br ^loop(%next : i64)
  ^done:
    return %result : f64
  }
  func.func private @strict_list_times_number(%list: f64, %number: f64) -> f64 {
    %no = arith.constant false
    %result = func.call @strict_list_scalar(%list, %number, %no) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_divided_number(%list: f64, %number: f64) -> f64 {
    %yes = arith.constant true
    %result = func.call @strict_list_scalar(%list, %number, %yes) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_hash_text(%key: f64) -> i64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %unset = arith.cmpf oeq, %key, %zero : f64
    cf.cond_br %unset, ^done(%start : i64), ^text
  ^text:
    %pointer = func.call @strict_ptr(%key) : (f64) -> !llvm.ptr
    cf.br ^loop(%pointer, %start : !llvm.ptr, i64)
  ^loop(%at: !llvm.ptr, %hash: i64):
    %byte = llvm.load %at : !llvm.ptr -> i8
    %end = arith.constant 0 : i8
    %finished = arith.cmpi eq, %byte, %end : i8
    cf.cond_br %finished, ^done(%hash : i64), ^next
  ^next:
    %code = arith.extui %byte : i8 to i64
    %thirtyOne = arith.constant 31 : i64
    %scaled = arith.muli %hash, %thirtyOne : i64
    %added = arith.addi %scaled, %code : i64
    %following = llvm.getelementptr %at[1] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^loop(%following, %added : !llvm.ptr, i64)
  ^done(%value: i64):
    %result = func.call @strict_hash_mixed(%value) : (i64) -> i64
    return %result : i64
  }
  func.func private @strict_hash_number(%key: f64) -> i64 {
    %bits = arith.bitcast %key : f64 to i64
    %result = func.call @strict_hash_mixed(%bits) : (i64) -> i64
    return %result : i64
  }
  func.func private @strict_hash_mixed(%value: i64) -> i64 {
    %shift = arith.constant 33 : i64
    %first = arith.constant -49064778989728563 : i64
    %second = arith.constant -4265267296055464877 : i64
    %high = arith.shrui %value, %shift : i64
    %folded = arith.xori %value, %high : i64
    %product = arith.muli %folded, %first : i64
    %productHigh = arith.shrui %product, %shift : i64
    %mixed = arith.xori %product, %productHigh : i64
    %spread = arith.muli %mixed, %second : i64
    %spreadHigh = arith.shrui %spread, %shift : i64
    %result = arith.xori %spread, %spreadHigh : i64
    return %result : i64
  }
  func.func private @strict_dictionary_index(%dictionary: f64, %hash: (f64) -> i64) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
    %restored = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %eight = arith.constant 8 : i64
    %view = func.call @strict_ptr(%dictionary) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %usedCount = llvm.load %usedSlot : !llvm.ptr -> f64
    %used = arith.fptosi %usedCount : f64 to i64
    %indexSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %indexBits = llvm.load %indexSlot : !llvm.ptr -> f64
    %needed = arith.muli %used, %two : i64
    %missing = arith.cmpf oeq, %indexBits, %zero : f64
    cf.cond_br %missing, ^build, ^check
  ^check:
    %existing = func.call @strict_ptr(%indexBits) : (f64) -> !llvm.ptr
    %slots = llvm.load %existing : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %existing[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %shrunk = arith.cmpi sgt, %count, %used : i64
    %crowded = arith.cmpi sge, %needed, %slots : i64
    %stale = arith.ori %shrunk, %crowded : i1
    cf.cond_br %stale, ^build, ^extend(%existing, %count : !llvm.ptr, i64)
  ^build:
    %sixteen = arith.constant 16 : i64
    cf.br ^size(%sixteen : i64)
  ^size(%candidate: i64):
    %small = arith.cmpi sle, %candidate, %needed : i64
    %doubled = arith.muli %candidate, %two : i64
    cf.cond_br %small, ^size(%doubled : i64), ^allocate(%candidate : i64)
  ^allocate(%size: i64):
    %entries = arith.addi %size, %two : i64
    %bytes = arith.muli %entries, %eight : i64
    %fresh = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %clear = arith.constant 0 : i32
    %cleared = llvm.call @memset(%fresh, %clear, %bytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    llvm.store %size, %fresh : i64, !llvm.ptr
    %freshBits = func.call @strict_bits(%fresh) : (!llvm.ptr) -> f64
    llvm.store %freshBits, %indexSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%indexSlot) : (!llvm.ptr) -> ()
    cf.br ^extend(%fresh, %none : !llvm.ptr, i64)
  ^extend(%table: !llvm.ptr, %position: i64):
    %more = arith.cmpi slt, %position, %used : i64
    cf.cond_br %more, ^insert, ^done
  ^insert:
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %pairSlot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %pairSlot : !llvm.ptr -> f64
    %key = func.call @strict_list_get(%pair, %zero) : (f64, f64) -> f64
    %keyHash = func.call_indirect %hash(%key) : (f64) -> i64
    %tableSize = llvm.load %table : !llvm.ptr -> i64
    %mask = arith.subi %tableSize, %one : i64
    %first = arith.andi %keyHash, %mask : i64
    cf.br ^probe(%first : i64)
  ^probe(%slot: i64):
    %offset = arith.addi %slot, %two : i64
    %entrySlot = llvm.getelementptr %table[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %entry = llvm.load %entrySlot : !llvm.ptr -> i64
    %free = arith.cmpi eq, %entry, %none : i64
    %following = arith.addi %slot, %one : i64
    %wrapped = arith.andi %following, %mask : i64
    cf.cond_br %free, ^place, ^probe(%wrapped : i64)
  ^place:
    %stored = arith.addi %position, %one : i64
    llvm.store %stored, %entrySlot : i64, !llvm.ptr
    cf.br ^extend(%table, %stored : !llvm.ptr, i64)
  ^done:
    %countTarget = llvm.getelementptr %table[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %used, %countTarget : i64, !llvm.ptr
    return %table : !llvm.ptr
  }
  func.func private @strict_dictionary_find(%dictionary: f64, %key: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> i64 {
    %none = arith.constant -1 : i64
    %empty = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0.0 : f64
    %length = func.call @strict_list_length(%dictionary) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %isEmpty = arith.cmpi eq, %count, %empty : i64
    cf.cond_br %isEmpty, ^missing, ^search
  ^search:
    %table = func.call @strict_dictionary_index(%dictionary, %hash) : (f64, (f64) -> i64) -> !llvm.ptr
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %keyHash = func.call_indirect %hash(%key) : (f64) -> i64
    %size = llvm.load %table : !llvm.ptr -> i64
    %mask = arith.subi %size, %one : i64
    %first = arith.andi %keyHash, %mask : i64
    cf.br ^probe(%first : i64)
  ^probe(%slot: i64):
    %offset = arith.addi %slot, %two : i64
    %entrySlot = llvm.getelementptr %table[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %entry = llvm.load %entrySlot : !llvm.ptr -> i64
    %following = arith.addi %slot, %one : i64
    %wrapped = arith.andi %following, %mask : i64
    %free = arith.cmpi eq, %entry, %empty : i64
    cf.cond_br %free, ^missing, ^candidate
  ^candidate:
    %position = arith.subi %entry, %one : i64
    %visible = arith.cmpi slt, %position, %count : i64
    cf.cond_br %visible, ^compare, ^probe(%wrapped : i64)
  ^compare:
    %pairSlot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %pairSlot : !llvm.ptr -> f64
    %pairKey = func.call @strict_list_get(%pair, %zero) : (f64, f64) -> f64
    %same = func.call_indirect %equal(%pairKey, %key) : (f64, f64) -> f64
    %matched = arith.cmpf one, %same, %zero : f64
    cf.cond_br %matched, ^found, ^probe(%wrapped : i64)
  ^found:
    return %position : i64
  ^missing:
    return %none : i64
  }
  func.func private @strict_dictionary_has(%dictionary: f64, %key: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> f64 {
    %none = arith.constant -1 : i64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %hash, %equal) : (f64, f64, (f64) -> i64, (f64, f64) -> f64) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    %result = func.call @strict_boolean(%found) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_get(%dictionary: f64, %key: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> f64 {
    %none = arith.constant -1 : i64
    %one = arith.constant 1.0 : f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %hash, %equal) : (f64, f64, (f64) -> i64, (f64, f64) -> f64) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    cf.cond_br %found, ^value, ^missing
  ^value:
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %slot : !llvm.ptr -> f64
    %value = func.call @strict_list_get(%pair, %one) : (f64, f64) -> f64
    return %value : f64
  ^missing:
    %name = llvm.mlir.addressof @strict_key_not_found : !llvm.ptr
    %error = func.call @strict_error_named(%name) : (!llvm.ptr) -> f64
    return %error : f64
  }
  func.func private @strict_dictionary_add(%dictionary: f64, %key: f64, %value: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> f64 {
    %none = arith.constant -1 : i64
    %nothing = arith.constant 0.0 : f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %hash, %equal) : (f64, f64, (f64) -> i64, (f64, f64) -> f64) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    cf.cond_br %found, ^duplicate, ^append
  ^duplicate:
    %name = llvm.mlir.addressof @strict_duplicate_key : !llvm.ptr
    %error = func.call @strict_error_named(%name) : (!llvm.ptr) -> f64
    return %error : f64
  ^append:
    %single = func.call @strict_list_add(%nothing, %key) : (f64, f64) -> f64
    %pair = func.call @strict_list_add(%single, %value) : (f64, f64) -> f64
    %result = func.call @strict_list_add(%dictionary, %pair) : (f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_set(%dictionary: f64, %key: f64, %value: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> f64 {
    %none = arith.constant -1 : i64
    %nothing = arith.constant 0.0 : f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %hash, %equal) : (f64, f64, (f64) -> i64, (f64, f64) -> f64) -> i64
    %single = func.call @strict_list_add(%nothing, %key) : (f64, f64) -> f64
    %pair = func.call @strict_list_add(%single, %value) : (f64, f64) -> f64
    %found = arith.cmpi ne, %position, %none : i64
    cf.cond_br %found, ^replace, ^append
  ^replace:
    %view = func.call @strict_ptr(%dictionary) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %indexSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %indexBits = llvm.load %indexSlot : !llvm.ptr -> f64
    %at = arith.sitofp %position : i64 to f64
    %written = func.call @strict_list_write(%dictionary, %at, %pair) : (f64, f64, f64) -> f64
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %inPlace = arith.cmpf olt, %used, %nothing : f64
    cf.cond_br %inPlace, ^keepIndex, ^done(%written : f64)
  ^keepIndex:
    %newView = func.call @strict_ptr(%written) : (f64) -> !llvm.ptr
    %newHolderSlot = llvm.getelementptr %newView[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %newHolderBits = llvm.load %newHolderSlot : !llvm.ptr -> f64
    %newHolder = func.call @strict_ptr(%newHolderBits) : (f64) -> !llvm.ptr
    %newIndexSlot = llvm.getelementptr %newHolder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %indexBits, %newIndexSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%newIndexSlot) : (!llvm.ptr) -> ()
    cf.br ^done(%written : f64)
  ^append:
    %appended = func.call @strict_list_add(%dictionary, %pair) : (f64, f64) -> f64
    cf.br ^done(%appended : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_dictionary_remove(%dictionary: f64, %key: f64, %hash: (f64) -> i64, %equal: (f64, f64) -> f64) -> f64 {
    %none = arith.constant -1 : i64
    %zero = arith.constant 0.0 : f64
    %bits = func.constant @strict_equal_number : (f64, f64) -> f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %hash, %equal) : (f64, f64, (f64) -> i64, (f64, f64) -> f64) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    cf.cond_br %found, ^remove, ^unchanged
  ^remove:
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %slot : !llvm.ptr -> f64
    %copy = func.call @strict_list_copy(%dictionary, %zero) : (f64, f64) -> f64
    %result = func.call @strict_list_remove(%copy, %pair, %bits) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  ^unchanged:
    return %dictionary : f64
  }
  llvm.mlir.global internal @strict_measured_text(0 : i64) : i64
  llvm.mlir.global internal @strict_measured_length(0 : i64) : i64
  func.func private @strict_text_length(%text: f64) -> f64 {
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %address = llvm.ptrtoint %pointer : !llvm.ptr to i64
    %textSlot = llvm.mlir.addressof @strict_measured_text : !llvm.ptr
    %lengthSlot = llvm.mlir.addressof @strict_measured_length : !llvm.ptr
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %idle = arith.constant 0 : i64
    %shared = arith.cmpi ne, %busy, %idle : i64
    cf.cond_br %shared, ^direct, ^lookup
  ^direct:
    %directLength = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %directResult = arith.uitofp %directLength : i64 to f64
    return %directResult : f64
  ^lookup:
    %measured = llvm.load %textSlot : !llvm.ptr -> i64
    %known = arith.cmpi eq, %measured, %address : i64
    cf.cond_br %known, ^cached, ^measure
  ^cached:
    %cachedLength = llvm.load %lengthSlot : !llvm.ptr -> i64
    %cachedResult = arith.uitofp %cachedLength : i64 to f64
    return %cachedResult : f64
  ^measure:
    // ponytail: only one long text stays cached, alternating two long texts re-measures, upgrade: length in a text header
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %longText = arith.constant 256 : i64
    %isLong = arith.cmpi uge, %length, %longText : i64
    cf.cond_br %isLong, ^remember, ^measured
  ^remember:
    llvm.store %address, %textSlot : i64, !llvm.ptr
    llvm.store %length, %lengthSlot : i64, !llvm.ptr
    cf.br ^measured
  ^measured:
    %result = arith.uitofp %length : i64 to f64
    return %result : f64
  }
  func.func private @strict_text_at(%text: f64, %index: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %negative = arith.cmpf olt, %index, %zero : f64
    cf.cond_br %negative, ^wrap, ^read(%index : f64)
  ^wrap:
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %wrapped = arith.addf %index, %length : f64
    cf.br ^read(%wrapped : f64)
  ^read(%chosen: f64):
    %at = arith.fptosi %chosen : f64 to i64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %pointer[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %slot : !llvm.ptr -> i8
    %result = arith.uitofp %byte : i8 to f64
    return %result : f64
  }
  func.func private @strict_boolean(%condition: i1) -> f64 {
    %result = arith.uitofp %condition : i1 to f64
    return %result : f64
  }
  func.func private @strict_text_equal(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0 : i32
    %a = func.call @strict_ptr(%left) : (f64) -> !llvm.ptr
    %b = func.call @strict_ptr(%right) : (f64) -> !llvm.ptr
    %identical = llvm.icmp "eq" %a, %b : !llvm.ptr
    cf.cond_br %identical, ^equal, ^first
  ^equal:
    %truth = func.call @strict_boolean(%identical) : (i1) -> f64
    return %truth : f64
  ^first:
    %firstLeft = llvm.load %a : !llvm.ptr -> i8
    %firstRight = llvm.load %b : !llvm.ptr -> i8
    %sameFirst = arith.cmpi eq, %firstLeft, %firstRight : i8
    cf.cond_br %sameFirst, ^compare, ^different
  ^different:
    %falsity = func.call @strict_boolean(%sameFirst) : (i1) -> f64
    return %falsity : f64
  ^compare:
    %compared = llvm.call @strcmp(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    %same = arith.cmpi eq, %compared, %zero : i32
    %result = func.call @strict_boolean(%same) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_starts(%text: f64, %prefix: f64, %start: f64) -> f64 {
    %zero = arith.constant 0 : i32
    %offset = arith.fptosi %start : f64 to i64
    %whole = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %a = llvm.getelementptr %whole[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %b = func.call @strict_ptr(%prefix) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%b) : (!llvm.ptr) -> i64
    %compared = llvm.call @strncmp(%a, %b, %length) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %same = arith.cmpi eq, %compared, %zero : i32
    %result = func.call @strict_boolean(%same) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_index(%text: f64, %search: f64, %start: f64) -> f64 {
    %none = arith.constant -1.0 : f64
    %nothing = arith.constant 0 : i8
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %b = func.call @strict_ptr(%search) : (f64) -> !llvm.ptr
    %offset = arith.fptosi %start : f64 to i64
    %zeroStart = arith.constant 0.0 : f64
    %fromStart = arith.cmpf ole, %start, %zeroStart : f64
    cf.cond_br %fromStart, ^search, ^measure
  ^measure:
    %textLength = func.call @strict_text_length(%text) : (f64) -> f64
    %short = arith.cmpf ogt, %start, %textLength : f64
    cf.cond_br %short, ^missing, ^search
  ^search:
    %from = llvm.getelementptr %a[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %first = llvm.load %b : !llvm.ptr -> i8
    %isEmpty = arith.cmpi eq, %first, %nothing : i8
    cf.cond_br %isEmpty, ^general, ^second
  ^second:
    %secondSlot = llvm.getelementptr %b[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %following = llvm.load %secondSlot : !llvm.ptr -> i8
    %isSingle = arith.cmpi eq, %following, %nothing : i8
    cf.cond_br %isSingle, ^single, ^general
  ^single:
    %character = arith.extui %first : i8 to i32
    %character_found = llvm.call @strchr(%from, %character) : (!llvm.ptr, i32) -> !llvm.ptr
    cf.br ^found(%character_found : !llvm.ptr)
  ^general:
    %text_found = llvm.call @strstr(%from, %b) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    cf.br ^found(%text_found : !llvm.ptr)
  ^found(%found: !llvm.ptr):
    %null = llvm.mlir.zero : !llvm.ptr
    %isNull = llvm.icmp "eq" %found, %null : !llvm.ptr
    cf.cond_br %isNull, ^missing, ^hit
  ^hit:
    %foundAt = llvm.ptrtoint %found : !llvm.ptr to i64
    %base = llvm.ptrtoint %a : !llvm.ptr to i64
    %distance = arith.subi %foundAt, %base : i64
    %result = arith.sitofp %distance : i64 to f64
    return %result : f64
  ^missing:
    return %none : f64
  }
  func.func private @strict_text_last_index(%text: f64, %search: f64) -> f64 {
    %none = arith.constant -1.0 : f64
    %one = arith.constant 1.0 : f64
    %begin = arith.constant 0.0 : f64
    %first = func.call @strict_text_index(%text, %search, %begin) : (f64, f64, f64) -> f64
    cf.br ^check(%none, %first : f64, f64)
  ^check(%last: f64, %current: f64):
    %missing = arith.cmpf oeq, %current, %none : f64
    cf.cond_br %missing, ^exit, ^next
  ^next:
    %after = arith.addf %current, %one : f64
    %following = func.call @strict_text_index(%text, %search, %after) : (f64, f64, f64) -> f64
    cf.br ^check(%current, %following : f64, f64)
  ^exit:
    return %last : f64
  }
  llvm.mlir.global internal @strict_characters(dense<0> : tensor<512xi8>) : !llvm.array<512 x i8>
  func.func private @strict_characters_init() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %count = arith.constant 256 : i64
    %table = llvm.mlir.addressof @strict_characters : !llvm.ptr
    cf.br ^loop(%zero : i64)
  ^loop(%code: i64):
    %more = arith.cmpi ult, %code, %count : i64
    cf.cond_br %more, ^fill, ^done
  ^fill:
    %offset = arith.addi %code, %code : i64
    %slot = llvm.getelementptr %table[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = arith.trunci %code : i64 to i8
    llvm.store %byte, %slot : i8, !llvm.ptr
    %next = arith.addi %code, %one : i64
    cf.br ^loop(%next : i64)
  ^done:
    return
  }
  func.func private @strict_text_substring(%text: f64, %start: f64, %length: f64) -> f64 {
    %one = arith.constant 1 : i64
    %terminator = arith.constant 0 : i8
    %zero = arith.constant 0.0 : f64
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %negativeStart = arith.cmpf olt, %start, %zero : f64
    %startOrZero = arith.select %negativeStart, %zero, %start : f64
    %negativeCount = arith.cmpf olt, %length, %zero : f64
    %countOrZero = arith.select %negativeCount, %zero, %length : f64
    %textLength = func.call @strict_text_length(%text) : (f64) -> f64
    %beyond = arith.cmpf ogt, %startOrZero, %textLength : f64
    %clampedStart = arith.select %beyond, %textLength, %startOrZero : f64
    %rest = arith.subf %textLength, %clampedStart : f64
    %tooLong = arith.cmpf ogt, %countOrZero, %rest : f64
    %clampedCount = arith.select %tooLong, %rest, %countOrZero : f64
    %offset = arith.fptosi %clampedStart : f64 to i64
    %count = arith.fptosi %clampedCount : f64 to i64
    %from = llvm.getelementptr %a[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %isCharacter = arith.cmpi eq, %count, %one : i64
    cf.cond_br %isCharacter, ^character, ^copy
  ^character:
    %byte = llvm.load %from : !llvm.ptr -> i8
    %code = arith.extui %byte : i8 to i64
    %twice = arith.addi %code, %code : i64
    %table = llvm.mlir.addressof @strict_characters : !llvm.ptr
    %shared = llvm.getelementptr %table[%twice] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %sharedBits = func.call @strict_bits(%shared) : (!llvm.ptr) -> f64
    return %sharedBits : f64
  ^copy:
    %size = arith.addi %count, %one : i64
    %copy = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %done = llvm.call @memcpy(%copy, %from, %count) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %end = llvm.getelementptr %copy[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_text_copy(%text: f64) -> f64 {
    %one = arith.constant 1 : i64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %size = arith.addi %length, %one : i64
    %copy = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %done = llvm.call @memcpy(%copy, %pointer, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_text_cased(%text: f64, %upper: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1 : i64
    %start = arith.constant 0 : i64
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %copy = func.call @strict_text_copy(%text) : (f64) -> f64
    %pointer = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    %count = arith.fptosi %length : f64 to i64
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^exit
  ^body:
    %slot = llvm.getelementptr %pointer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %slot : !llvm.ptr -> i8
    %wide = arith.extui %byte : i8 to i32
    %raised = llvm.call @toupper(%wide) : (i32) -> i32
    %lowered = llvm.call @tolower(%wide) : (i32) -> i32
    %chosen = arith.select %upper, %raised, %lowered : i32
    %narrow = arith.trunci %chosen : i32 to i8
    llvm.store %narrow, %slot : i8, !llvm.ptr
    %next = arith.addi %index, %one : i64
    cf.br ^check(%next : i64)
  ^exit:
    return %copy : f64
  }
  func.func private @strict_character_text(%code: f64) -> f64 {
    %two = arith.constant 2 : i64
    %terminator = arith.constant 0 : i8
    %copy = func.call @strict_alloc_text(%two) : (i64) -> !llvm.ptr
    %byte = arith.fptoui %code : f64 to i8
    llvm.store %byte, %copy : i8, !llvm.ptr
    %end = llvm.getelementptr %copy[1] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_floor(%number: f64) -> f64 {
    %result = llvm.intr.floor(%number) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_snapped(%number: f64) -> f64 {
    %limit = arith.constant 1.0e-10 : f64
    %rounded = llvm.intr.round(%number) : (f64) -> f64
    %difference = arith.subf %number, %rounded : f64
    %distance = llvm.intr.fabs(%difference) : (f64) -> f64
    %isNear = arith.cmpf olt, %distance, %limit : f64
    %result = arith.select %isNear, %rounded, %number : f64
    return %result : f64
  }
  func.func private @strict_radians(%degrees: f64) -> f64 {
    %pi = arith.constant 3.141592653589793 : f64
    %half = arith.constant 180.0 : f64
    %scaled = arith.mulf %degrees, %pi : f64
    %result = arith.divf %scaled, %half : f64
    return %result : f64
  }
  func.func private @strict_degrees(%radians: f64) -> f64 {
    %pi = arith.constant 3.141592653589793 : f64
    %half = arith.constant 180.0 : f64
    %scaled = arith.mulf %radians, %half : f64
    %degrees = arith.divf %scaled, %pi : f64
    %result = func.call @strict_snapped(%degrees) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_degrees_sin(%degrees: f64) -> f64 {
    %radians = func.call @strict_radians(%degrees) : (f64) -> f64
    %sine = llvm.call @sin(%radians) : (f64) -> f64
    %result = func.call @strict_snapped(%sine) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_degrees_cos(%degrees: f64) -> f64 {
    %radians = func.call @strict_radians(%degrees) : (f64) -> f64
    %cosine = llvm.call @cos(%radians) : (f64) -> f64
    %result = func.call @strict_snapped(%cosine) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_degrees_tan(%degrees: f64) -> f64 {
    %radians = func.call @strict_radians(%degrees) : (f64) -> f64
    %tangent = llvm.call @tan(%radians) : (f64) -> f64
    %result = func.call @strict_snapped(%tangent) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_ratio_asin(%ratio: f64) -> f64 {
    %angle = llvm.call @asin(%ratio) : (f64) -> f64
    %result = func.call @strict_degrees(%angle) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_ratio_acos(%ratio: f64) -> f64 {
    %angle = llvm.call @acos(%ratio) : (f64) -> f64
    %result = func.call @strict_degrees(%angle) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_ratio_atan(%ratio: f64) -> f64 {
    %angle = llvm.call @atan(%ratio) : (f64) -> f64
    %result = func.call @strict_degrees(%angle) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_vector2_atan(%vector: f64) -> f64 {
    %first = arith.constant 0.0 : f64
    %second = arith.constant 1.0 : f64
    %x = func.call @strict_list_get(%vector, %first) : (f64, f64) -> f64
    %y = func.call @strict_list_get(%vector, %second) : (f64, f64) -> f64
    %angle = llvm.call @atan2(%x, %y) : (f64, f64) -> f64
    %result = func.call @strict_degrees(%angle) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_text_number(%text: f64) -> f64 {
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %result = llvm.call @strtod(%a, %null) : (!llvm.ptr, !llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_text_upper(%text: f64) -> f64 {
    %upper = arith.constant 1 : i1
    %result = func.call @strict_text_cased(%text, %upper) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_lower(%text: f64) -> f64 {
    %upper = arith.constant 0 : i1
    %result = func.call @strict_text_cased(%text, %upper) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_argument(%count: i32, %values: !llvm.ptr, %index: i32) -> f64 {
    %inside = arith.cmpi slt, %index, %count : i32
    cf.cond_br %inside, ^read, ^missing
  ^read:
    %at = arith.extsi %index : i32 to i64
    %slot = llvm.getelementptr %values[%at] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %text = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %bits = func.call @strict_bits(%text) : (!llvm.ptr) -> f64
    return %bits : f64
  ^missing:
    %empty = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %none = func.call @strict_bits(%empty) : (!llvm.ptr) -> f64
    return %none : f64
  }
  func.func private @strict_argument_list(%count: i32, %values: !llvm.ptr, %numbers: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %first = arith.constant 1 : i32
    %one = arith.constant 1 : i32
    %created = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    cf.br ^check(%first, %created : i32, f64)
  ^check(%index: i32, %elements: f64):
    %more = arith.cmpi slt, %index, %count : i32
    cf.cond_br %more, ^read, ^done
  ^read:
    %text = func.call @strict_argument(%count, %values, %index) : (i32, !llvm.ptr, i32) -> f64
    cf.cond_br %numbers, ^parse, ^add(%text : f64)
  ^parse:
    %number = func.call @strict_text_number(%text) : (f64) -> f64
    cf.br ^add(%number : f64)
  ^add(%element: f64):
    %added = func.call @strict_list_add(%elements, %element) : (f64, f64) -> f64
    %next = arith.addi %index, %one : i32
    cf.br ^check(%next, %added : i32, f64)
  ^done:
    return %elements : f64
  }
  func.func private @strict_argument_numbers(%count: i32, %values: !llvm.ptr) -> f64 {
    %numbers = arith.constant true
    %result = func.call @strict_argument_list(%count, %values, %numbers) : (i32, !llvm.ptr, i1) -> f64
    return %result : f64
  }
  func.func private @strict_argument_texts(%count: i32, %values: !llvm.ptr) -> f64 {
    %numbers = arith.constant false
    %result = func.call @strict_argument_list(%count, %values, %numbers) : (i32, !llvm.ptr, i1) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_precise("%.*g\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_scientific("%.16e\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @strict_exponent("e%d\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_nan("NaN\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_infinity("Infinity\00") : !llvm.array<9 x i8>
  llvm.mlir.global internal constant @strict_negative_infinity("-Infinity\00") : !llvm.array<10 x i8>
  func.func private @strict_number_text(%number: f64) -> f64 {
    %failed = func.call @strict_is_failure(%number) : (f64) -> i1
    cf.cond_br %failed, ^report, ^format
  ^report:
    %report = func.call @strict_error_report(%number) : (f64) -> f64
    return %report : f64
  ^format:
    %zero = arith.constant 0.0 : f64
    %magnitude = llvm.intr.fabs(%number) : (f64) -> f64
    %infinity = arith.constant 0x7FF0000000000000 : f64
    %isFinite = arith.cmpf olt, %magnitude, %infinity : f64
    cf.cond_br %isFinite, ^finite, ^special
  ^special:
    %isNumber = arith.cmpf ord, %number, %number : f64
    %isBelow = arith.cmpf olt, %number, %zero : f64
    %nanText = llvm.mlir.addressof @strict_nan : !llvm.ptr
    %positiveText = llvm.mlir.addressof @strict_infinity : !llvm.ptr
    %negativeText = llvm.mlir.addressof @strict_negative_infinity : !llvm.ptr
    %infinite = arith.select %isBelow, %negativeText, %positiveText : !llvm.ptr
    %special = arith.select %isNumber, %infinite, %nanText : !llvm.ptr
    %specialBits = func.call @strict_bits(%special) : (!llvm.ptr) -> f64
    return %specialBits : f64
  ^finite:
    %size = arith.constant 40 : i64
    %large = arith.constant 1.0e7 : f64
    %tiny = arith.constant 1.0e-9 : f64
    %buffer = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %absolute = llvm.intr.fabs(%number) : (f64) -> f64
    %isLarge = arith.cmpf oge, %absolute, %large : f64
    %isPositive = arith.cmpf ogt, %absolute, %zero : f64
    %isTiny = arith.cmpf ole, %absolute, %tiny : f64
    %isSmall = arith.andi %isPositive, %isTiny : i1
    %scientific = arith.ori %isLarge, %isSmall : i1
    %first = arith.constant 15 : i32
    cf.cond_br %scientific, ^exponent, ^plain(%first : i32)
  ^plain(%precision: i32):
    %format = llvm.mlir.addressof @strict_precise : !llvm.ptr
    %written = llvm.call @snprintf(%buffer, %size, %format, %precision, %number) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, i32, f64) -> i32
    %null = llvm.mlir.zero : !llvm.ptr
    %parsed = llvm.call @strtod(%buffer, %null) : (!llvm.ptr, !llvm.ptr) -> f64
    %exact = arith.cmpf oeq, %parsed, %number : f64
    %last = arith.constant 17 : i32
    %atLast = arith.cmpi sge, %precision, %last : i32
    %finished = arith.ori %exact, %atLast : i1
    %one = arith.constant 1 : i32
    %more = arith.addi %precision, %one : i32
    cf.cond_br %finished, ^done, ^plain(%more : i32)
  ^exponent:
    %scientificFormat = llvm.mlir.addressof @strict_scientific : !llvm.ptr
    %digits = llvm.call @snprintf(%buffer, %size, %scientificFormat, %number) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, f64) -> i32
    %letter = arith.constant 101 : i32
    %marker = llvm.call @strchr(%buffer, %letter) : (!llvm.ptr, i32) -> !llvm.ptr
    %null2 = llvm.mlir.zero : !llvm.ptr
    %afterMarker = llvm.getelementptr %marker[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %decimal = arith.constant 10 : i32
    %power = llvm.call @strtol(%afterMarker, %null2, %decimal) : (!llvm.ptr, !llvm.ptr, i32) -> i64
    %markerAt = llvm.ptrtoint %marker : !llvm.ptr to i64
    %bufferAt = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %markerIndex = arith.subi %markerAt, %bufferAt : i64
    %one64 = arith.constant 1 : i64
    %lastDigit = arith.subi %markerIndex, %one64 : i64
    cf.br ^trim(%lastDigit : i64)
  ^trim(%end: i64):
    %slot = llvm.getelementptr %buffer[%end] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %zeroDigit = arith.constant 48 : i8
    %dot = arith.constant 46 : i8
    %isZero = arith.cmpi eq, %character, %zeroDigit : i8
    %previous = arith.subi %end, %one64 : i64
    cf.cond_br %isZero, ^trim(%previous : i64), ^dotted(%end, %character : i64, i8)
  ^dotted(%kept: i64, %keptCharacter: i8):
    %dot2 = arith.constant 46 : i8
    %isDot = arith.cmpi eq, %keptCharacter, %dot2 : i8
    %withoutDot = arith.subi %kept, %one64 : i64
    %mantissaEnd = arith.select %isDot, %withoutDot, %kept : i64
    %next = arith.addi %mantissaEnd, %one64 : i64
    %tail = llvm.getelementptr %buffer[%next] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %remaining = arith.subi %size, %next : i64
    %exponentFormat = llvm.mlir.addressof @strict_exponent : !llvm.ptr
    %wide = arith.trunci %power : i64 to i32
    %appended = llvm.call @snprintf(%tail, %remaining, %exponentFormat, %wide) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, i32) -> i32
    cf.br ^done
  ^done:
    %result = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_equal_number(%left: f64, %right: f64) -> f64 {
    %same = arith.cmpf oeq, %left, %right : f64
    %result = func.call @strict_boolean(%same) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_equal(%left: f64, %right: f64, %element: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %sameLength = arith.cmpf oeq, %leftLength, %rightLength : f64
    %count = arith.fptosi %leftLength : f64 to i64
    cf.cond_br %sameLength, ^check(%start : i64), ^different
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^compare, ^equal
  ^compare:
    %at = arith.sitofp %index : i64 to f64
    %leftItem = func.call @strict_list_get(%left, %at) : (f64, f64) -> f64
    %rightItem = func.call @strict_list_get(%right, %at) : (f64, f64) -> f64
    %equalItems = func.call_indirect %element(%leftItem, %rightItem) : (f64, f64) -> f64
    %same = arith.cmpf one, %equalItems, %zero : f64
    %next = arith.addi %index, %step : i64
    cf.cond_br %same, ^check(%next : i64), ^different
  ^equal:
    return %one : f64
  ^different:
    return %zero : f64
  }
  func.func private @strict_list_matches(%value: f64, %item: f64, %equal: (f64, f64) -> f64) -> i1 attributes {passthrough = ["alwaysinline"]} {
    %zero = arith.constant 0.0 : f64
    %equalItems = func.call_indirect %equal(%value, %item) : (f64, f64) -> f64
    %same = arith.cmpf one, %equalItems, %zero : f64
    return %same : i1
  }
  func.func private @strict_list_find(%list: f64, %item: f64, %equal: (f64, f64) -> f64) -> f64 attributes {passthrough = ["alwaysinline"]} {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %none = arith.constant -1.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^compare, ^missing
  ^compare:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %same = func.call @strict_list_matches(%value, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> i1
    %next = arith.addi %index, %step : i64
    cf.cond_br %same, ^found(%index : i64), ^check(%next : i64)
  ^found(%at: i64):
    %result = arith.sitofp %at : i64 to f64
    return %result : f64
  ^missing:
    return %none : f64
  }
  func.func private @strict_list_index_number(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_find(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_index_text(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_find(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_count(%list: f64, %item: f64, %equal: (f64, f64) -> f64) -> f64 attributes {passthrough = ["alwaysinline"]} {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    cf.br ^check(%start, %zero : i64, f64)
  ^check(%index: i64, %found: f64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^compare, ^done
  ^compare:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %same = func.call @strict_list_matches(%value, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> i1
    %added = arith.addf %found, %one : f64
    %total = arith.select %same, %added, %found : f64
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next, %total : i64, f64)
  ^done:
    return %found : f64
  }
  func.func private @strict_list_count_number(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_count(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_count_text(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_count(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_has_number(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_has(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_has_text(%list: f64, %item: f64) -> f64 {
    %equal = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_has(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_has(%list: f64, %item: f64, %equal: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %index = func.call @strict_list_find(%list, %item, %equal) : (f64, f64, (f64, f64) -> f64) -> f64
    %found = arith.cmpf oge, %index, %zero : f64
    %result = func.call @strict_boolean(%found) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_not(%value: f64) -> f64 {
    %one = arith.constant 1.0 : f64
    %result = arith.subf %one, %value : f64
    return %result : f64
  }
  func.func private @strict_and(%left: f64, %right: f64) -> f64 {
    %result = arith.mulf %left, %right : f64
    return %result : f64
  }
  func.func private @strict_or(%left: f64, %right: f64) -> f64 {
    %result = arith.maximumf %left, %right : f64
    return %result : f64
  }
  func.func private @strict_xor(%left: f64, %right: f64) -> f64 {
    %different = arith.cmpf one, %left, %right : f64
    %result = func.call @strict_boolean(%different) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_concat(%left: f64, %right: f64) -> f64 {
    %one = arith.constant 1 : i64
    %a = func.call @strict_ptr(%left) : (f64) -> !llvm.ptr
    %b = func.call @strict_ptr(%right) : (f64) -> !llvm.ptr
    %leftLength = llvm.call @strlen(%a) : (!llvm.ptr) -> i64
    %rightLength = llvm.call @strlen(%b) : (!llvm.ptr) -> i64
    %total = arith.addi %leftLength, %rightLength : i64
    %size = arith.addi %total, %one : i64
    %buffer = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %first = llvm.call @memcpy(%buffer, %a, %leftLength) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %end = llvm.getelementptr %buffer[%leftLength] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rightSize = arith.addi %rightLength, %one : i64
    %second = llvm.call @memcpy(%end, %b, %rightSize) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %result = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_comma(", \00") : !llvm.array<3 x i8>
  func.func private @strict_list_text(%list: f64, %isText: i1) -> f64 {
    %commaText = llvm.mlir.addressof @strict_comma : !llvm.ptr
    %comma = func.call @strict_bits(%commaText) : (!llvm.ptr) -> f64
    %brackets = arith.constant true
    %result = func.call @strict_join(%list, %isText, %comma, %brackets) : (f64, i1, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_combine(%self: f64, %texts: f64, %separator: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %isUnset = arith.cmpf oeq, %separator, %zero : f64
    %commaText = llvm.mlir.addressof @strict_comma : !llvm.ptr
    %comma = func.call @strict_bits(%commaText) : (!llvm.ptr) -> f64
    %chosen = arith.select %isUnset, %comma, %separator : f64
    %isText = arith.constant true
    %brackets = arith.constant false
    %result = func.call @strict_join(%texts, %isText, %chosen, %brackets) : (f64, i1, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_join(%list: f64, %isText: i1, %separator: f64, %brackets: i1) -> f64 {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %eight = arith.constant 8 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %separatorPointer = func.call @strict_ptr(%separator) : (f64) -> !llvm.ptr
    %separatorSize = llvm.call @strlen(%separatorPointer) : (!llvm.ptr) -> i64
    %partsBytes = arith.muli %count, %eight : i64
    %parts = func.call @strict_alloc(%partsBytes) : (i64) -> !llvm.ptr
    %hasItems = arith.cmpi sgt, %count, %start : i64
    %enclosed = arith.andi %hasItems, %brackets : i1
    %bracketSize = arith.select %enclosed, %two, %start : i64
    cf.br ^measure(%start, %bracketSize : i64, i64)
  ^measure(%index: i64, %total: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^part, ^allocate
  ^part:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    cf.cond_br %isText, ^keep(%value : f64), ^format
  ^format:
    %number = func.call @strict_number_text(%value) : (f64) -> f64
    cf.br ^keep(%number : f64)
  ^keep(%element: f64):
    %partSlot = llvm.getelementptr %parts[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %partSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%partSlot) : (!llvm.ptr) -> ()
    %pointer = func.call @strict_ptr(%element) : (f64) -> !llvm.ptr
    %size = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %isFirst = arith.cmpi eq, %index, %start : i64
    %gap = arith.select %isFirst, %start, %separatorSize : i64
    %withGap = arith.addi %total, %gap : i64
    %measured = arith.addi %withGap, %size : i64
    %next = arith.addi %index, %step : i64
    cf.br ^measure(%next, %measured : i64, i64)
  ^allocate:
    %bytes = arith.addi %total, %step : i64
    %buffer = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    cf.cond_br %enclosed, ^opening, ^write(%start, %start : i64, i64)
  ^opening:
    %open = arith.constant 40 : i8
    llvm.store %open, %buffer : i8, !llvm.ptr
    cf.br ^write(%start, %step : i64, i64)
  ^write(%at: i64, %offset: i64):
    %remaining = arith.cmpi slt, %at, %count : i64
    cf.cond_br %remaining, ^separate, ^closing
  ^separate:
    %isFirstPart = arith.cmpi eq, %at, %start : i64
    cf.cond_br %isFirstPart, ^copy(%offset : i64), ^gapped
  ^gapped:
    %gapTarget = llvm.getelementptr %buffer[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %copiedGap = llvm.call @memcpy(%gapTarget, %separatorPointer, %separatorSize) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %afterGap = arith.addi %offset, %separatorSize : i64
    cf.br ^copy(%afterGap : i64)
  ^copy(%position: i64):
    %partAt = llvm.getelementptr %parts[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %part = llvm.load %partAt : !llvm.ptr -> f64
    %partPointer = func.call @strict_ptr(%part) : (f64) -> !llvm.ptr
    %partSize = llvm.call @strlen(%partPointer) : (!llvm.ptr) -> i64
    %target = llvm.getelementptr %buffer[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %copiedPart = llvm.call @memcpy(%target, %partPointer, %partSize) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %advanced = arith.addi %position, %partSize : i64
    %following = arith.addi %at, %step : i64
    cf.br ^write(%following, %advanced : i64, i64)
  ^closing:
    cf.cond_br %enclosed, ^bracket, ^terminate(%offset : i64)
  ^bracket:
    %close = arith.constant 41 : i8
    %closeSlot = llvm.getelementptr %buffer[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %close, %closeSlot : i8, !llvm.ptr
    %afterBracket = arith.addi %offset, %step : i64
    cf.br ^terminate(%afterBracket : i64)
  ^terminate(%finalAt: i64):
    %terminator = arith.constant 0 : i8
    %terminatorSlot = llvm.getelementptr %buffer[%finalAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %terminatorSlot : i8, !llvm.ptr
    %result = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_list_mapped(%list: f64, %toText: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %empty = func.call @strict_list_new(%length, %zero) : (f64, f64) -> f64
    cf.br ^loop(%zero, %empty : f64, f64)
  ^loop(%index: f64, %built: f64):
    %more = arith.cmpf olt, %index, %length : f64
    cf.cond_br %more, ^body, ^done
  ^body:
    %element = func.call @strict_list_get(%list, %index) : (f64, f64) -> f64
    cf.cond_br %toText, ^text, ^number
  ^text:
    %asText = func.call @strict_number_text(%element) : (f64) -> f64
    cf.br ^add(%asText : f64)
  ^number:
    %asNumber = func.call @strict_text_number(%element) : (f64) -> f64
    cf.br ^add(%asNumber : f64)
  ^add(%converted: f64):
    %added = func.call @strict_list_add(%built, %converted) : (f64, f64) -> f64
    %next = arith.addf %index, %one : f64
    cf.br ^loop(%next, %added : f64, f64)
  ^done:
    return %built : f64
  }
  func.func private @strict_list_equal_text(%left: f64, %right: f64) -> f64 {
    %element = func.constant @strict_text_equal : (f64, f64) -> f64
    %result = func.call @strict_list_equal(%left, %right, %element) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_equal_number(%left: f64, %right: f64) -> f64 {
    %element = func.constant @strict_equal_number : (f64, f64) -> f64
    %result = func.call @strict_list_equal(%left, %right, %element) : (f64, f64, (f64, f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_pair_equal(%left: f64, %right: f64, %keyEqual: (f64, f64) -> f64, %valueEqual: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %leftKey = func.call @strict_list_get(%left, %zero) : (f64, f64) -> f64
    %rightKey = func.call @strict_list_get(%right, %zero) : (f64, f64) -> f64
    %equalKeys = func.call_indirect %keyEqual(%leftKey, %rightKey) : (f64, f64) -> f64
    %sameKey = arith.cmpf one, %equalKeys, %zero : f64
    cf.cond_br %sameKey, ^value, ^different
  ^value:
    %leftValue = func.call @strict_list_get(%left, %one) : (f64, f64) -> f64
    %rightValue = func.call @strict_list_get(%right, %one) : (f64, f64) -> f64
    %equalValues = func.call_indirect %valueEqual(%leftValue, %rightValue) : (f64, f64) -> f64
    return %equalValues : f64
  ^different:
    return %zero : f64
  }
  func.func private @strict_pairs_equal(%left: f64, %right: f64, %keyEqual: (f64, f64) -> f64, %valueEqual: (f64, f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %sameLength = arith.cmpf oeq, %leftLength, %rightLength : f64
    cf.cond_br %sameLength, ^check(%zero : f64), ^different
  ^check(%index: f64):
    %more = arith.cmpf olt, %index, %leftLength : f64
    cf.cond_br %more, ^compare, ^equal
  ^compare:
    %leftPair = func.call @strict_list_get(%left, %index) : (f64, f64) -> f64
    %rightPair = func.call @strict_list_get(%right, %index) : (f64, f64) -> f64
    %equalPairs = func.call @strict_pair_equal(%leftPair, %rightPair, %keyEqual, %valueEqual) : (f64, f64, (f64, f64) -> f64, (f64, f64) -> f64) -> f64
    %same = arith.cmpf one, %equalPairs, %zero : f64
    %next = arith.addf %index, %one : f64
    cf.cond_br %same, ^check(%next : f64), ^different
  ^equal:
    return %one : f64
  ^different:
    return %zero : f64
  }
  func.func private @strict_list_text_mapped(%list: f64, %converter: (f64) -> f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %empty = func.call @strict_list_new(%length, %zero) : (f64, f64) -> f64
    cf.br ^loop(%zero, %empty : f64, f64)
  ^loop(%index: f64, %built: f64):
    %more = arith.cmpf olt, %index, %length : f64
    cf.cond_br %more, ^body, ^done
  ^body:
    %element = func.call @strict_list_get(%list, %index) : (f64, f64) -> f64
    %text = func.call_indirect %converter(%element) : (f64) -> f64
    %added = func.call @strict_list_add(%built, %text) : (f64, f64) -> f64
    %next = arith.addf %index, %one : f64
    cf.br ^loop(%next, %added : f64, f64)
  ^done:
    %joined = func.call @strict_text_combine(%zero, %built, %zero) : (f64, f64, f64) -> f64
    %isEmpty = arith.cmpf oeq, %length, %zero : f64
    cf.cond_br %isEmpty, ^empty, ^enclose
  ^empty:
    return %joined : f64
  ^enclose:
    %openText = llvm.mlir.addressof @strict_open_paren : !llvm.ptr
    %open = func.call @strict_bits(%openText) : (!llvm.ptr) -> f64
    %closeText = llvm.mlir.addressof @strict_close_paren : !llvm.ptr
    %close = func.call @strict_bits(%closeText) : (!llvm.ptr) -> f64
    %opened = func.call @strict_text_concat(%open, %joined) : (f64, f64) -> f64
    %result = func.call @strict_text_concat(%opened, %close) : (f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_lists_text(%list: f64) -> f64 {
    %converter = func.constant @strict_list_text_text : (f64) -> f64
    %result = func.call @strict_list_text_mapped(%list, %converter) : (f64, (f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_lists_number(%list: f64) -> f64 {
    %converter = func.constant @strict_list_text_number : (f64) -> f64
    %result = func.call @strict_list_text_mapped(%list, %converter) : (f64, (f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_texts(%list: f64) -> f64 {
    %toText = arith.constant 1 : i1
    %result = func.call @strict_list_mapped(%list, %toText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_numbers(%list: f64) -> f64 {
    %toText = arith.constant 0 : i1
    %result = func.call @strict_list_mapped(%list, %toText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_text(%list: f64) -> f64 {
    %isText = arith.constant 1 : i1
    %result = func.call @strict_list_text(%list, %isText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_number(%list: f64) -> f64 {
    %isText = arith.constant 0 : i1
    %result = func.call @strict_list_text(%list, %isText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_add_set(%list: f64, %text: f64, %value: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %unset = arith.cmpf oeq, %value, %zero : f64
    cf.cond_br %unset, ^skip, ^add
  ^skip:
    return %list : f64
  ^add:
    %added = func.call @strict_list_add(%list, %text) : (f64, f64) -> f64
    return %added : f64
  }
  func.func private @strict_converted(%value: f64, %converter: (f64) -> f64) -> f64 {
    %failed = func.call @strict_is_failure(%value) : (f64) -> i1
    cf.cond_br %failed, ^report, ^convert
  ^report:
    %report = func.call @strict_error_report(%value) : (f64) -> f64
    return %report : f64
  ^convert:
    %text = func.call_indirect %converter(%value) : (f64) -> f64
    return %text : f64
  }
  func.func private @strict_text_itself(%text: f64) -> f64 {
    return %text : f64
  }
  func.func private @strict_pair_text(%pair: f64, %keyText: (f64) -> f64, %valueText: (f64) -> f64) -> f64 {
    %failed = func.call @strict_is_failure(%pair) : (f64) -> i1
    cf.cond_br %failed, ^report, ^convert
  ^report:
    %report = func.call @strict_error_report(%pair) : (f64) -> f64
    return %report : f64
  ^convert:
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %key = func.call @strict_list_get(%pair, %zero) : (f64, f64) -> f64
    %value = func.call @strict_list_get(%pair, %one) : (f64, f64) -> f64
    %keyConverted = func.call_indirect %keyText(%key) : (f64) -> f64
    %valueConverted = func.call_indirect %valueText(%value) : (f64) -> f64
    %empty = func.call @strict_list_new(%two, %zero) : (f64, f64) -> f64
    %first = func.call @strict_list_add(%empty, %keyConverted) : (f64, f64) -> f64
    %texts = func.call @strict_list_add(%first, %valueConverted) : (f64, f64) -> f64
    %isText = arith.constant true
    %result = func.call @strict_list_text(%texts, %isText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_pairs_text(%list: f64, %keyText: (f64) -> f64, %valueText: (f64) -> f64) -> f64 {
    %failed = func.call @strict_is_failure(%list) : (f64) -> i1
    cf.cond_br %failed, ^report, ^convert
  ^report:
    %report = func.call @strict_error_report(%list) : (f64) -> f64
    return %report : f64
  ^convert:
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %empty = func.call @strict_list_new(%length, %zero) : (f64, f64) -> f64
    cf.br ^loop(%zero, %empty : f64, f64)
  ^loop(%index: f64, %built: f64):
    %more = arith.cmpf olt, %index, %length : f64
    cf.cond_br %more, ^body, ^done
  ^body:
    %pair = func.call @strict_list_get(%list, %index) : (f64, f64) -> f64
    %text = func.call @strict_pair_text(%pair, %keyText, %valueText) : (f64, (f64) -> f64, (f64) -> f64) -> f64
    %added = func.call @strict_list_add(%built, %text) : (f64, f64) -> f64
    %next = arith.addf %index, %one : f64
    cf.br ^loop(%next, %added : f64, f64)
  ^done:
    %isText = arith.constant true
    %result = func.call @strict_list_text(%built, %isText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_boolean_text(%value: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %isTrue = arith.cmpf one, %value, %zero : f64
    %yes = llvm.mlir.addressof @strict_true : !llvm.ptr
    %no = llvm.mlir.addressof @strict_false : !llvm.ptr
    %chosen = llvm.select %isTrue, %yes, %no : i1, !llvm.ptr
    %result = func.call @strict_bits(%chosen) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_is_space(%character: i8) -> i1 {
    %space = arith.constant 32 : i8
    %tab = arith.constant 9 : i8
    %line = arith.constant 10 : i8
    %return = arith.constant 13 : i8
    %isSpace = arith.cmpi eq, %character, %space : i8
    %isTab = arith.cmpi eq, %character, %tab : i8
    %isLine = arith.cmpi eq, %character, %line : i8
    %isReturn = arith.cmpi eq, %character, %return : i8
    %first = arith.ori %isSpace, %isTab : i1
    %second = arith.ori %isLine, %isReturn : i1
    %result = arith.ori %first, %second : i1
    return %result : i1
  }
  func.func private @strict_text_trim_start(%text: f64) -> f64 {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    cf.br ^check(%start : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %length : i64
    cf.cond_br %more, ^inspect, ^cut(%index : i64)
  ^inspect:
    %slot = llvm.getelementptr %pointer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %isSpace = func.call @strict_is_space(%character) : (i8) -> i1
    %next = arith.addi %index, %step : i64
    cf.cond_br %isSpace, ^check(%next : i64), ^cut(%index : i64)
  ^cut(%at: i64):
    %rest = arith.subi %length, %at : i64
    %from = arith.sitofp %at : i64 to f64
    %count = arith.sitofp %rest : i64 to f64
    %result = func.call @strict_text_substring(%text, %from, %count) : (f64, f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_text_trim_end(%text: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    cf.br ^check(%length : i64)
  ^check(%end: i64):
    %more = arith.cmpi sgt, %end, %none : i64
    cf.cond_br %more, ^inspect, ^cut(%end : i64)
  ^inspect:
    %last = arith.subi %end, %step : i64
    %slot = llvm.getelementptr %pointer[%last] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %isSpace = func.call @strict_is_space(%character) : (i8) -> i1
    cf.cond_br %isSpace, ^check(%last : i64), ^cut(%end : i64)
  ^cut(%at: i64):
    %count = arith.sitofp %at : i64 to f64
    %result = func.call @strict_text_substring(%text, %zero, %count) : (f64, f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_text_trim(%text: f64) -> f64 {
    %started = func.call @strict_text_trim_start(%text) : (f64) -> f64
    %result = func.call @strict_text_trim_end(%started) : (f64) -> f64
    return %result : f64
  }
  func.func private @strict_text_capitalize(%text: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %copy = func.call @strict_text_copy(%text) : (f64) -> f64
    %pointer = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    %first = llvm.load %pointer : !llvm.ptr -> i8
    %wide = arith.extui %first : i8 to i32
    %raised = llvm.call @toupper(%wide) : (i32) -> i32
    %narrow = arith.trunci %raised : i32 to i8
    llvm.store %narrow, %pointer : i8, !llvm.ptr
    return %copy : f64
  }
  llvm.mlir.global internal constant @strict_read_mode("rb\00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_write_mode("wb\00") : !llvm.array<3 x i8>
  func.func private @strict_file_exists(%path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %overriddenText = func.call @strict_overridden(%path) : (f64) -> f64
    %isOverridden = arith.cmpf one, %overriddenText, %zero : f64
    cf.cond_br %isOverridden, ^overridden, ^disk
  ^overridden:
    %present = arith.constant 1.0 : f64
    return %present : f64
  ^disk:
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %missing, ^absent, ^present
  ^present:
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    %directory = func.call @strict_directory_exists(%zero, %path) : (f64, f64) -> f64
    %isDirectory = arith.cmpf one, %directory, %zero : f64
    %one = arith.constant 1.0 : f64
    %result = arith.select %isDirectory, %zero, %one : f64
    return %result : f64
  ^absent:
    return %zero : f64
  }
  llvm.mlir.global internal @strict_read_paths(0 : i64) : i64
  llvm.mlir.global internal @strict_read_count(0 : i64) : i64
  func.func private @strict_file_noted(%path: f64) {
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%name) : (!llvm.ptr) -> i64
    %size = arith.addi %length, %one : i64
    %copy = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%copy, %name, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %countSlot = llvm.mlir.addressof @strict_read_count : !llvm.ptr
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %next = arith.addi %count, %one : i64
    %bytes = arith.muli %next, %eight : i64
    %pathsSlot = llvm.mlir.addressof @strict_read_paths : !llvm.ptr
    %paths = llvm.load %pathsSlot : !llvm.ptr -> i64
    %old = llvm.inttoptr %paths : i64 to !llvm.ptr
    %grown = llvm.call @realloc(%old, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %grownBits = llvm.ptrtoint %grown : !llvm.ptr to i64
    llvm.store %grownBits, %pathsSlot : i64, !llvm.ptr
    %slot = llvm.getelementptr %grown[%count] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    llvm.store %copy, %slot : !llvm.ptr, !llvm.ptr
    llvm.store %next, %countSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_process_filesread(%self: f64) -> f64 {
    %start = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %countSlot = llvm.mlir.addressof @strict_read_count : !llvm.ptr
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %pathsSlot = llvm.mlir.addressof @strict_read_paths : !llvm.ptr
    %pathsBits = llvm.load %pathsSlot : !llvm.ptr -> i64
    %paths = llvm.inttoptr %pathsBits : i64 to !llvm.ptr
    %empty = func.call @strict_list_empty() : () -> f64
    cf.br ^loop(%start, %empty : i64, f64)
  ^loop(%index: i64, %built: f64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^body, ^done
  ^body:
    %slot = llvm.getelementptr %paths[%index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %path = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %text = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    %added = func.call @strict_list_add(%built, %text) : (f64, f64) -> f64
    %next = arith.addi %index, %one : i64
    cf.br ^loop(%next, %added : i64, f64)
  ^done:
    return %built : f64
  }
  llvm.mlir.global internal @strict_library_error(0 : i64) : i64
  func.func private @strict_library_enter(%stack: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %tableSlot = llvm.mlir.addressof @strict_gc_table : !llvm.ptr
    %table = llvm.load %tableSlot : !llvm.ptr -> i64
    %isReady = arith.cmpi ne, %table, %none : i64
    cf.cond_br %isReady, ^ready, ^start
  ^start:
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    return
  ^ready:
    %stackValue = llvm.ptrtoint %stack : !llvm.ptr to i64
    %stackSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    llvm.store %stackValue, %stackSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_library_text(%value: f64) -> !llvm.ptr {
    %kept = func.call @strict_kept_text(%value) : (f64) -> f64
    %pointer = func.call @strict_ptr(%kept) : (f64) -> !llvm.ptr
    return %pointer : !llvm.ptr
  }
  func.func private @strict_library_failed(%value: f64) -> i32 {
    %report = func.call @strict_error_report(%value) : (f64) -> f64
    %text = func.call @strict_library_text(%report) : (f64) -> !llvm.ptr
    %textValue = llvm.ptrtoint %text : !llvm.ptr to i64
    %errorSlot = llvm.mlir.addressof @strict_library_error : !llvm.ptr
    llvm.store %textValue, %errorSlot : i64, !llvm.ptr
    %failed = arith.constant 1 : i32
    return %failed : i32
  }
  func.func private @strict_library_list(%block: !llvm.ptr) -> f64 {
    %address = llvm.ptrtoint %block : !llvm.ptr to i64
    %raw = arith.bitcast %address : i64 to f64
    %list = func.call @strict_foreign_numbers(%raw) : (f64) -> f64
    return %list : f64
  }
  func.func private @strict_library_numbers(%list: f64) -> !llvm.ptr {
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %slots = arith.addi %count, %one : i64
    %bytes = arith.muli %slots, %eight : i64
    %block = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    llvm.store %length, %block : f64, !llvm.ptr
    %target = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %itemBytes = arith.muli %count, %eight : i64
    %copied = llvm.call @memcpy(%target, %items, %itemBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    return %block : !llvm.ptr
  }
  func.func private @strict_library_records(%block: !llvm.ptr, %fields: i64) -> f64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %length = llvm.load %block : !llvm.ptr -> f64
    %count = arith.fptosi %length : f64 to i64
    %list = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %source = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %bytes = arith.muli %fields, %eight : i64
    cf.br ^loop(%zero : i64)
  ^loop(%at: i64):
    %more = arith.cmpi slt, %at, %count : i64
    cf.cond_br %more, ^record, ^done
  ^record:
    %element = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    %offset = arith.muli %at, %fields : i64
    %from = llvm.getelementptr %source[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %copied = llvm.call @memcpy(%element, %from, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %bits = func.call @strict_bits(%element) : (!llvm.ptr) -> f64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %bits, %slot : f64, !llvm.ptr
    func.call @strict_gc_written(%slot) : (!llvm.ptr) -> ()
    %next = arith.addi %at, %one : i64
    cf.br ^loop(%next : i64)
  ^done:
    return %list : f64
  }
  func.func private @strict_library_recorded(%list: f64, %fields: i64) -> !llvm.ptr {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %values = arith.muli %count, %fields : i64
    %slots = arith.addi %values, %one : i64
    %total = arith.muli %slots, %eight : i64
    %block = llvm.call @malloc(%total) : (i64) -> !llvm.ptr
    llvm.store %length, %block : f64, !llvm.ptr
    %target = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %bytes = arith.muli %fields, %eight : i64
    cf.br ^loop(%zero : i64)
  ^loop(%at: i64):
    %more = arith.cmpi slt, %at, %count : i64
    cf.cond_br %more, ^record, ^done
  ^record:
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %bits = llvm.load %slot : !llvm.ptr -> f64
    %element = func.call @strict_ptr(%bits) : (f64) -> !llvm.ptr
    %offset = arith.muli %at, %fields : i64
    %to = llvm.getelementptr %target[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %copied = llvm.call @memcpy(%to, %element, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %next = arith.addi %at, %one : i64
    cf.br ^loop(%next : i64)
  ^done:
    return %block : !llvm.ptr
  }
  func.func @strict_last_error() -> !llvm.ptr {
    %errorSlot = llvm.mlir.addressof @strict_library_error : !llvm.ptr
    %textValue = llvm.load %errorSlot : !llvm.ptr -> i64
    %text = llvm.inttoptr %textValue : i64 to !llvm.ptr
    return %text : !llvm.ptr
  }
  func.func @strict_release(%text: !llvm.ptr) {
    llvm.call @free(%text) : (!llvm.ptr) -> ()
    return
  }
  llvm.mlir.global internal @strict_override_path(0.0 : f64) : f64
  llvm.mlir.global internal @strict_override_text(0.0 : f64) : f64
  func.func private @strict_kept_text(%text: f64) -> f64 {
    %one = arith.constant 1 : i64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %size = arith.addi %length, %one : i64
    %copy = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%copy, %pointer, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_file_override(%path: f64, %text: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %keptPath = func.call @strict_kept_text(%path) : (f64) -> f64
    %keptText = func.call @strict_kept_text(%text) : (f64) -> f64
    %pathSlot = llvm.mlir.addressof @strict_override_path : !llvm.ptr
    llvm.store %keptPath, %pathSlot : f64, !llvm.ptr
    %textSlot = llvm.mlir.addressof @strict_override_text : !llvm.ptr
    llvm.store %keptText, %textSlot : f64, !llvm.ptr
    return %zero : f64
  }
  func.func private @strict_overridden(%path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %pathSlot = llvm.mlir.addressof @strict_override_path : !llvm.ptr
    %stored = llvm.load %pathSlot : !llvm.ptr -> f64
    %isSet = arith.cmpf one, %stored, %zero : f64
    cf.cond_br %isSet, ^compare, ^none
  ^compare:
    %same = func.call @strict_text_equal(%path, %stored) : (f64, f64) -> f64
    %isSame = arith.cmpf one, %same, %zero : f64
    cf.cond_br %isSame, ^found, ^none
  ^found:
    %textSlot = llvm.mlir.addressof @strict_override_text : !llvm.ptr
    %text = llvm.load %textSlot : !llvm.ptr -> f64
    return %text : f64
  ^none:
    return %zero : f64
  }
  func.func private @strict_stream_text(%file: !llvm.ptr) -> f64 {
    %chunk = arith.constant 65536 : i64
    %one = arith.constant 1 : i64
    %start = arith.constant 0 : i64
    %terminator = arith.constant 0 : i8
    %null = llvm.mlir.zero : !llvm.ptr
    cf.br ^read(%null, %start : !llvm.ptr, i64)
  ^read(%buffer: !llvm.ptr, %used: i64):
    %capacity = arith.addi %used, %chunk : i64
    %withEnd = arith.addi %capacity, %one : i64
    %grown = llvm.call @realloc(%buffer, %withEnd) : (!llvm.ptr, i64) -> !llvm.ptr
    %at = llvm.getelementptr %grown[%used] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %count = llvm.call @fread(%at, %one, %chunk, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %total = arith.addi %used, %count : i64
    %more = arith.cmpi eq, %count, %chunk : i64
    cf.cond_br %more, ^read(%grown, %total : !llvm.ptr, i64), ^done(%grown, %total : !llvm.ptr, i64)
  ^done(%text: !llvm.ptr, %length: i64):
    %end = llvm.getelementptr %text[%length] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%text) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_file_text(%path: f64) -> f64 {
    %overriddenText = func.call @strict_overridden(%path) : (f64) -> f64
    %unsetText = arith.constant 0.0 : f64
    %isOverridden = arith.cmpf one, %overriddenText, %unsetText : f64
    cf.cond_br %isOverridden, ^overridden, ^disk
  ^overridden:
    return %overriddenText : f64
  ^disk:
    %start = arith.constant 0 : i64
    %ending = arith.constant 2 : i32
    %beginning = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %terminator = arith.constant 0 : i8
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %missing, ^absent, ^read
  ^read:
    func.call @strict_file_noted(%path) : (f64) -> ()
    %toEnd = llvm.call @fseek(%file, %start, %ending) : (!llvm.ptr, i64, i32) -> i32
    %size = llvm.call @ftell(%file) : (!llvm.ptr) -> i64
    %toStart = llvm.call @fseek(%file, %start, %beginning) : (!llvm.ptr, i64, i32) -> i32
    %capacity = arith.addi %size, %one : i64
    %buffer = func.call @strict_alloc_text(%capacity) : (i64) -> !llvm.ptr
    %count = llvm.call @fread(%buffer, %one, %size, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %end = llvm.getelementptr %buffer[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    %skipped = func.call @strict_bom_length(%buffer, %count) : (!llvm.ptr, i64) -> i64
    %text = llvm.getelementptr %buffer[%skipped] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %result = func.call @strict_bits(%text) : (!llvm.ptr) -> f64
    return %result : f64
  ^absent:
    %empty = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %none = func.call @strict_bits(%empty) : (!llvm.ptr) -> f64
    return %none : f64
  }
  func.func private @strict_bom_length(%buffer: !llvm.ptr, %count: i64) -> i64 {
    %none = arith.constant 0 : i64
    %three = arith.constant 3 : i64
    %enough = arith.cmpi sge, %count, %three : i64
    cf.cond_br %enough, ^check, ^done(%none : i64)
  ^check:
    %firstMark = arith.constant -17 : i8
    %secondMark = arith.constant -69 : i8
    %thirdMark = arith.constant -65 : i8
    %first = llvm.load %buffer : !llvm.ptr -> i8
    %secondSlot = llvm.getelementptr %buffer[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %second = llvm.load %secondSlot : !llvm.ptr -> i8
    %thirdSlot = llvm.getelementptr %buffer[2] : (!llvm.ptr) -> !llvm.ptr, i8
    %third = llvm.load %thirdSlot : !llvm.ptr -> i8
    %isFirst = arith.cmpi eq, %first, %firstMark : i8
    %isSecond = arith.cmpi eq, %second, %secondMark : i8
    %isThird = arith.cmpi eq, %third, %thirdMark : i8
    %firstTwo = arith.andi %isFirst, %isSecond : i1
    %isMark = arith.andi %firstTwo, %isThird : i1
    %skip = arith.select %isMark, %three, %none : i64
    cf.br ^done(%skip : i64)
  ^done(%length: i64):
    return %length : i64
  }
  func.func private @strict_file_read_lines(%path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %newline = arith.constant 10 : i8
    %carriage = arith.constant 13 : i8
    %text = func.call @strict_file_text(%path) : (f64) -> f64
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %lines = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    cf.br ^scan(%start, %start, %lines : i64, i64, f64)
  ^scan(%index: i64, %lineStart: i64, %found: f64):
    %more = arith.cmpi slt, %index, %length : i64
    cf.cond_br %more, ^inspect, ^finish
  ^inspect:
    %slot = llvm.getelementptr %pointer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %isNewline = arith.cmpi eq, %character, %newline : i8
    %next = arith.addi %index, %step : i64
    cf.cond_br %isNewline, ^line, ^scan(%next, %lineStart, %found : i64, i64, f64)
  ^line:
    %added = func.call @strict_text_line(%text, %lineStart, %index, %found) : (f64, i64, i64, f64) -> f64
    cf.br ^scan(%next, %next, %added : i64, i64, f64)
  ^finish:
    %last = func.call @strict_text_line(%text, %lineStart, %length, %found) : (f64, i64, i64, f64) -> f64
    return %last : f64
  }
  func.func private @strict_text_line(%text: f64, %start: i64, %end: i64, %lines: f64) -> f64 {
    %step = arith.constant 1 : i64
    %carriage = arith.constant 13 : i8
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %hasCharacters = arith.cmpi sgt, %end, %start : i64
    %last = arith.subi %end, %step : i64
    %safeLast = arith.select %hasCharacters, %last, %start : i64
    %slot = llvm.getelementptr %pointer[%safeLast] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %isCarriage = arith.cmpi eq, %character, %carriage : i8
    %stripped = arith.andi %hasCharacters, %isCarriage : i1
    %trimmedEnd = arith.select %stripped, %last, %end : i64
    %count = arith.subi %trimmedEnd, %start : i64
    %from = arith.sitofp %start : i64 to f64
    %length = arith.sitofp %count : i64 to f64
    %line = func.call @strict_text_substring(%text, %from, %length) : (f64, f64, f64) -> f64
    %result = func.call @strict_list_add(%lines, %line) : (f64, f64) -> f64
    return %result : f64
  }
  func.func private @strict_file_create(%path: f64) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
    %step = arith.constant 1 : i64
    %slash = arith.constant 47 : i8
    %backslash = arith.constant 92 : i8
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_write_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %failed, ^folders, ^done(%file : !llvm.ptr)
  ^folders:
    %length = llvm.call @strlen(%name) : (!llvm.ptr) -> i64
    cf.br ^character(%step : i64)
  ^character(%index: i64):
    %more = arith.cmpi slt, %index, %length : i64
    cf.cond_br %more, ^check, ^retry
  ^check:
    %slot = llvm.getelementptr %name[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %character = llvm.load %slot : !llvm.ptr -> i8
    %isSlash = arith.cmpi eq, %character, %slash : i8
    %isBackslash = arith.cmpi eq, %character, %backslash : i8
    %isSeparator = arith.ori %isSlash, %isBackslash : i1
    cf.cond_br %isSeparator, ^create, ^next
  ^create:
    %count = arith.sitofp %index : i64 to f64
    %folder = func.call @strict_text_substring(%path, %zero, %count) : (f64, f64, f64) -> f64
    %created = func.call @strict_directory_create(%zero, %folder) : (f64, f64) -> f64
    cf.br ^next
  ^next:
    %following = arith.addi %index, %step : i64
    cf.br ^character(%following : i64)
  ^retry:
    %again = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    cf.br ^done(%again : !llvm.ptr)
  ^done(%result: !llvm.ptr):
    return %result : !llvm.ptr
  }
  func.func private @strict_file_writelines(%path: f64, %lines: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %newline = arith.constant 10 : i32
    %file = func.call @strict_file_create(%path) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %failed, ^done, ^write
  ^write:
    %lengthNumber = func.call @strict_list_length(%lines) : (f64) -> f64
    %count = arith.fptosi %lengthNumber : f64 to i64
    %hasLines = arith.cmpi sgt, %count, %start : i64
    cf.cond_br %hasLines, ^items, ^close
  ^items:
    %items = func.call @strict_list_items(%lines) : (f64) -> !llvm.ptr
    cf.br ^line(%start : i64)
  ^line(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^separate, ^close
  ^separate:
    %first = arith.cmpi eq, %index, %start : i64
    cf.cond_br %first, ^text, ^newline
  ^newline:
    %put = llvm.call @fputc(%newline, %file) : (i32, !llvm.ptr) -> i32
    cf.br ^text
  ^text:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %line = llvm.load %slot : !llvm.ptr -> f64
    %content = func.call @strict_ptr(%line) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%content) : (!llvm.ptr) -> i64
    %written = llvm.call @fwrite(%content, %step, %length, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %next = arith.addi %index, %step : i64
    cf.br ^line(%next : i64)
  ^close:
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return %zero : f64
  }
  func.func private @strict_file_readbytes(%path: f64) -> f64 {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %ending = arith.constant 2 : i32
    %beginning = arith.constant 0 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %missing, ^absent, ^read
  ^read:
    func.call @strict_file_noted(%path) : (f64) -> ()
    %toEnd = llvm.call @fseek(%file, %start, %ending) : (!llvm.ptr, i64, i32) -> i32
    %told = llvm.call @ftell(%file) : (!llvm.ptr) -> i64
    %toStart = llvm.call @fseek(%file, %start, %beginning) : (!llvm.ptr, i64, i32) -> i32
    %known = arith.cmpi sge, %told, %start : i64
    %size = arith.select %known, %told, %start : i64
    %length = arith.sitofp %size : i64 to f64
    %bytes = func.call @strict_list_new(%length, %length) : (f64, f64) -> f64
    %items = func.call @strict_list_items(%bytes) : (f64) -> !llvm.ptr
    cf.br ^copy(%start : i64)
  ^copy(%index: i64):
    %more = arith.cmpi slt, %index, %size : i64
    cf.cond_br %more, ^byte, ^done
  ^byte:
    %read = llvm.call @fgetc(%file) : (!llvm.ptr) -> i32
    %number = arith.sitofp %read : i32 to f64
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %number, %slot : f64, !llvm.ptr
    %next = arith.addi %index, %step : i64
    cf.br ^copy(%next : i64)
  ^done:
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    return %bytes : f64
  ^absent:
    %empty = func.call @strict_list_empty() : () -> f64
    return %empty : f64
  }
  func.func private @strict_file_writebytes(%path: f64, %bytes: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %file = func.call @strict_file_create(%path) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %failed, ^done, ^write
  ^write:
    %lengthNumber = func.call @strict_list_length(%bytes) : (f64) -> f64
    %count = arith.fptosi %lengthNumber : f64 to i64
    %hasItems = arith.cmpi sgt, %count, %start : i64
    cf.cond_br %hasItems, ^items, ^close
  ^items:
    %items = func.call @strict_list_items(%bytes) : (f64) -> !llvm.ptr
    cf.br ^copy(%start : i64)
  ^copy(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^byte, ^close
  ^byte:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %number = llvm.load %slot : !llvm.ptr -> f64
    %byte = arith.fptosi %number : f64 to i32
    %put = llvm.call @fputc(%byte, %file) : (i32, !llvm.ptr) -> i32
    %next = arith.addi %index, %step : i64
    cf.br ^copy(%next : i64)
  ^close:
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return %zero : f64
  }
  func.func private @strict_process_result(%code: f64, %output: f64, %error: f64) -> f64 {
    %size = arith.constant 24 : i64
    %block = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    llvm.store %code, %block : f64, !llvm.ptr
    %outputSlot = llvm.getelementptr %block[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %output, %outputSlot : f64, !llvm.ptr
    %errorSlot = llvm.getelementptr %block[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %error, %errorSlot : f64, !llvm.ptr
    %result = func.call @strict_bits(%block) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_process_find(%self: f64, %name: f64) -> f64 {
    %path = func.call @strict_tool_path(%name) : (f64) -> f64
    return %path : f64
  }
  llvm.mlir.global internal constant @strict_tool_missing("tool not found: \00") : !llvm.array<17 x i8>
  func.func private @strict_process_runtool(%self: f64, %name: f64, %arguments: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %path = func.call @strict_tool_path(%name) : (f64) -> f64
    %length = func.call @strict_text_length(%path) : (f64) -> f64
    %missing = arith.cmpf oeq, %length, %zero : f64
    cf.cond_br %missing, ^absent, ^present
  ^absent:
    %code = arith.constant 127.0 : f64
    %prefixText = llvm.mlir.addressof @strict_tool_missing : !llvm.ptr
    %prefix = func.call @strict_bits(%prefixText) : (!llvm.ptr) -> f64
    %error = func.call @strict_text_concat(%prefix, %name) : (f64, f64) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %failure = func.call @strict_process_result(%code, %empty, %error) : (f64, f64, f64) -> f64
    return %failure : f64
  ^present:
    %result = func.call @strict_process_execute(%path, %arguments) : (f64, f64) -> f64
    return %result : f64
  }
  llvm.mlir.global internal @strict_gpu_state(0 : i64) : i64
  // ponytail: a launch costs about 130 us of copies and synchronization, so loops reading fewer than 4M numbers
  // stay on the CPU (STRICT_GPU=all sends every loop), loops reading a list that is already on the GPU run there at any size
  llvm.mlir.global internal @strict_gpu_minimum(4194304 : i64) : i64
  llvm.mlir.global internal @strict_gpu_setting(0 : i64) : i64
  llvm.mlir.global internal @strict_gpu_calls(dense<0> : tensor<26xi64>) : !llvm.array<26 x i64>
  llvm.mlir.global internal @strict_gpu_handles(dense<0> : tensor<4xi64>) : !llvm.array<4 x i64>
  llvm.mlir.global internal constant @strict_gpu_names("cuInit\00cuDeviceGet\00cuDevicePrimaryCtxRetain\00cuCtxSetCurrent\00cuModuleLoadData\00cuModuleGetFunction\00cuMemAlloc_v2\00cuMemFree_v2\00cuMemcpyHtoD_v2\00cuMemcpyDtoH_v2\00cuLaunchKernel\00cuCtxSynchronize\00cuMemAllocHost_v2\00cuMemFreeHost\00cuMemcpyHtoDAsync_v2\00cuMemcpyDtoDAsync_v2\00cuStreamCreate\00cuStreamBeginCapture_v2\00cuStreamEndCapture\00cuGraphInstantiateWithFlags\00cuGraphExecUpdate_v2\00cuGraphLaunch\00cuGraphDestroy\00cuGraphExecDestroy\00cuMemGetInfo_v2\00cuStreamQuery\00") : !llvm.array<447 x i8>
  llvm.mlir.global internal constant @strict_gpu_switch("STRICT_GPU\00") : !llvm.array<11 x i8>
  func.func private @strict_gpu_ready() -> i1 {
    %stateSlot = llvm.mlir.addressof @strict_gpu_state : !llvm.ptr
    %state = llvm.load %stateSlot : !llvm.ptr -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %untried = arith.cmpi eq, %state, %zero : i64
    cf.cond_br %untried, ^open, ^known
  ^known:
    %ready = arith.cmpi eq, %state, %one : i64
    return %ready : i1
  ^open:
    %opened = func.call @strict_gpu_open() : () -> i1
    %two = arith.constant 2 : i64
    %result = arith.select %opened, %one, %two : i64
    llvm.store %result, %stateSlot : i64, !llvm.ptr
    return %opened : i1
  }
  func.func private @strict_gpu_threshold() -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %settingSlot = llvm.mlir.addressof @strict_gpu_setting : !llvm.ptr
    %minimumSlot = llvm.mlir.addressof @strict_gpu_minimum : !llvm.ptr
    %read = llvm.load %settingSlot : !llvm.ptr -> i64
    %isRead = arith.cmpi ne, %read, %zero : i64
    cf.cond_br %isRead, ^known, ^read
  ^read:
    llvm.store %one, %settingSlot : i64, !llvm.ptr
    %name = llvm.mlir.addressof @strict_gpu_switch : !llvm.ptr
    %setting = llvm.call @getenv(%name) : (!llvm.ptr) -> !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %isUnset = llvm.icmp "eq" %setting, %unset : !llvm.ptr
    cf.cond_br %isUnset, ^known, ^setting
  ^setting:
    %first = llvm.load %setting : !llvm.ptr -> i8
    %off = arith.constant 48 : i8
    %letterA = arith.constant 97 : i8
    %disabled = arith.cmpi eq, %first, %off : i8
    %everything = arith.cmpi eq, %first, %letterA : i8
    %never = arith.constant 9223372036854775807 : i64
    %minimum = llvm.load %minimumSlot : !llvm.ptr -> i64
    %wanted = arith.select %everything, %one, %minimum : i64
    %chosen = arith.select %disabled, %never, %wanted : i64
    llvm.store %chosen, %minimumSlot : i64, !llvm.ptr
    cf.br ^known
  ^known:
    %current = llvm.load %minimumSlot : !llvm.ptr -> i64
    return %current : i64
  }
  func.func private @strict_gpu_open() -> i1 {
    %false = arith.constant false
    %unset = llvm.mlir.zero : !llvm.ptr
    %library = func.call @strict_gpu_library() : () -> !llvm.ptr
    %missing = llvm.icmp "eq" %library, %unset : !llvm.ptr
    cf.cond_br %missing, ^failed, ^resolve
  ^resolve:
    %resolved = func.call @strict_gpu_resolve(%library) : (!llvm.ptr) -> i1
    cf.cond_br %resolved, ^start, ^failed
  ^start:
    %started = func.call @strict_gpu_start() : () -> i1
    return %started : i1
  ^failed:
    return %false : i1
  }
  func.func private @strict_gpu_resolve(%library: !llvm.ptr) -> i1 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %count = arith.constant 26 : i64
    %names = llvm.mlir.addressof @strict_gpu_names : !llvm.ptr
    %calls = llvm.mlir.addressof @strict_gpu_calls : !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %true = arith.constant true
    %false = arith.constant false
    cf.br ^next(%zero, %names : i64, !llvm.ptr)
  ^next(%index: i64, %name: !llvm.ptr):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^lookup, ^done
  ^lookup:
    %function = func.call @strict_gpu_lookup(%library, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %missing = llvm.icmp "eq" %function, %unset : !llvm.ptr
    cf.cond_br %missing, ^failed, ^store
  ^store:
    %bits = llvm.ptrtoint %function : !llvm.ptr to i64
    %slot = llvm.getelementptr %calls[0, %index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<26 x i64>
    llvm.store %bits, %slot : i64, !llvm.ptr
    %length = llvm.call @strlen(%name) : (!llvm.ptr) -> i64
    %skipped = arith.addi %length, %one : i64
    %following = llvm.getelementptr %name[%skipped] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %nextIndex = arith.addi %index, %one : i64
    cf.br ^next(%nextIndex, %following : i64, !llvm.ptr)
  ^done:
    return %true : i1
  ^failed:
    return %false : i1
  }
  // module loads, frees, synchronous copies and pinned memory are not allowed while launches are captured or wait for
  // them: those calls (bits 4, 5, 7, 8, 9, 11, 12, 13) replay the captured launches first
  func.func private @strict_gpu_call(%index: i64) -> !llvm.ptr {
    %endingCalls = arith.constant 15280 : i64
    %one = arith.constant 1 : i64
    %shifted = arith.shrui %endingCalls, %index : i64
    %bit = arith.andi %shifted, %one : i64
    %ending = arith.cmpi eq, %bit, %one : i64
    cf.cond_br %ending, ^replay, ^lookup
  ^replay:
    func.call @strict_gpu_replay() : () -> ()
    cf.br ^lookup
  ^lookup:
    %calls = llvm.mlir.addressof @strict_gpu_calls : !llvm.ptr
    %slot = llvm.getelementptr %calls[0, %index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<26 x i64>
    %bits = llvm.load %slot : !llvm.ptr -> i64
    %function = llvm.inttoptr %bits : i64 to !llvm.ptr
    return %function : !llvm.ptr
  }
  func.func private @strict_gpu_start() -> i1 {
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %initIndex = arith.constant 0 : i64
    %init = func.call @strict_gpu_call(%initIndex) : (i64) -> !llvm.ptr
    %initialized = llvm.call %init(%zero32) : !llvm.ptr, (i32) -> i32
    %initFailed = arith.cmpi ne, %initialized, %zero32 : i32
    cf.cond_br %initFailed, ^failed, ^device
  ^device:
    %deviceSlot = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %deviceGetIndex = arith.constant 1 : i64
    %deviceGet = func.call @strict_gpu_call(%deviceGetIndex) : (i64) -> !llvm.ptr
    %found = llvm.call %deviceGet(%deviceSlot, %zero32) : !llvm.ptr, (!llvm.ptr, i32) -> i32
    %notFound = arith.cmpi ne, %found, %zero32 : i32
    cf.cond_br %notFound, ^failed, ^context
  ^context:
    %device = llvm.load %deviceSlot : !llvm.ptr -> i32
    %contextSlot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    %retainIndex = arith.constant 2 : i64
    %retain = func.call @strict_gpu_call(%retainIndex) : (i64) -> !llvm.ptr
    %retained = llvm.call %retain(%contextSlot, %device) : !llvm.ptr, (!llvm.ptr, i32) -> i32
    %notRetained = arith.cmpi ne, %retained, %zero32 : i32
    cf.cond_br %notRetained, ^failed, ^current
  ^current:
    %context = llvm.load %contextSlot : !llvm.ptr -> !llvm.ptr
    %contextBits = llvm.ptrtoint %context : !llvm.ptr to i64
    %handles = llvm.mlir.addressof @strict_gpu_handles : !llvm.ptr
    llvm.store %contextBits, %handles : i64, !llvm.ptr
    %current = func.call @strict_gpu_current() : () -> i1
    %streamSlot = llvm.getelementptr %handles[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %createIndex = arith.constant 16 : i64
    %create = func.call @strict_gpu_call(%createIndex) : (i64) -> !llvm.ptr
    %created = llvm.call %create(%streamSlot, %zero32) : !llvm.ptr, (!llvm.ptr, i32) -> i32
    %hasStream = arith.cmpi eq, %created, %zero32 : i32
    %ready = arith.andi %current, %hasStream : i1
    return %ready : i1
  ^failed:
    %false = arith.constant false
    return %false : i1
  }
  // the context is made current once per thread, cuCtxSetCurrent costs about 30 us on Windows
  llvm.mlir.global internal thread_local @strict_gpu_made_current(0 : i64) : i64
  func.func private @strict_gpu_current() -> i1 {
    %zero32 = arith.constant 0 : i32
    %none = arith.constant 0 : i64
    %true = arith.constant true
    %madeSlot = llvm.mlir.addressof @strict_gpu_made_current : !llvm.ptr
    %made = llvm.load %madeSlot : !llvm.ptr -> i64
    %handles = llvm.mlir.addressof @strict_gpu_handles : !llvm.ptr
    %contextBits = llvm.load %handles : !llvm.ptr -> i64
    %isMade = arith.cmpi eq, %made, %contextBits : i64
    cf.cond_br %isMade, ^done, ^set
  ^done:
    return %true : i1
  ^set:
    %context = llvm.inttoptr %contextBits : i64 to !llvm.ptr
    %setIndex = arith.constant 3 : i64
    %set = func.call @strict_gpu_call(%setIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %set(%context) : !llvm.ptr, (!llvm.ptr) -> i32
    %isSet = arith.cmpi eq, %result, %zero32 : i32
    %remembered = arith.select %isSet, %contextBits, %none : i64
    llvm.store %remembered, %madeSlot : i64, !llvm.ptr
    return %isSet : i1
  }
  func.func private @strict_gpu_function(%ptx: !llvm.ptr, %name: !llvm.ptr, %slot: !llvm.ptr) -> !llvm.ptr {
    %zero = arith.constant 0 : i64
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %unset = llvm.mlir.zero : !llvm.ptr
    %cached = llvm.load %slot : !llvm.ptr -> i64
    %isCached = arith.cmpi ne, %cached, %zero : i64
    cf.cond_br %isCached, ^cached, ^module
  ^cached:
    %function = llvm.inttoptr %cached : i64 to !llvm.ptr
    return %function : !llvm.ptr
  ^module:
    %current = func.call @strict_gpu_current() : () -> i1
    %handles = llvm.mlir.addressof @strict_gpu_handles : !llvm.ptr
    %moduleSlot = llvm.getelementptr %handles[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %moduleBits = llvm.load %moduleSlot : !llvm.ptr -> i64
    %isLoaded = arith.cmpi ne, %moduleBits, %zero : i64
    cf.cond_br %isLoaded, ^lookup(%moduleBits : i64), ^load
  ^load:
    %loadIndex = arith.constant 4 : i64
    %loadData = func.call @strict_gpu_call(%loadIndex) : (i64) -> !llvm.ptr
    %loaded = llvm.call %loadData(%moduleSlot, %ptx) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> i32
    %loadFailed = arith.cmpi ne, %loaded, %zero32 : i32
    %newModule = llvm.load %moduleSlot : !llvm.ptr -> i64
    cf.cond_br %loadFailed, ^failed, ^lookup(%newModule : i64)
  ^lookup(%moduleValue: i64):
    %module = llvm.inttoptr %moduleValue : i64 to !llvm.ptr
    %getIndex = arith.constant 5 : i64
    %getFunction = func.call @strict_gpu_call(%getIndex) : (i64) -> !llvm.ptr
    %got = llvm.call %getFunction(%slot, %module, %name) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %notFound = arith.cmpi ne, %got, %zero32 : i32
    %found = llvm.load %slot : !llvm.ptr -> i64
    %foundFunction = llvm.inttoptr %found : i64 to !llvm.ptr
    cf.cond_br %notFound, ^failed, ^done
  ^done:
    return %foundFunction : !llvm.ptr
  ^failed:
    return %unset : !llvm.ptr
  }
  func.func private @strict_gpu_alloc(%bytes: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %zero, %slot : i64, !llvm.ptr
    %allocIndex = arith.constant 6 : i64
    %alloc = func.call @strict_gpu_call(%allocIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %alloc(%slot, %bytes) : !llvm.ptr, (!llvm.ptr, i64) -> i32
    %failed = arith.cmpi ne, %result, %zero32 : i32
    %device = llvm.load %slot : !llvm.ptr -> i64
    %chosen = arith.select %failed, %zero, %device : i64
    return %chosen : i64
  }
  func.func private @strict_gpu_free(%device: i64) {
    %freeIndex = arith.constant 7 : i64
    %free = func.call @strict_gpu_call(%freeIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %free(%device) : !llvm.ptr, (i64) -> i32
    return
  }
  func.func private @strict_gpu_put(%device: i64, %host: !llvm.ptr, %bytes: i64) -> i1 {
    %zero32 = arith.constant 0 : i32
    %putIndex = arith.constant 8 : i64
    %put = func.call @strict_gpu_call(%putIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %put(%device, %host, %bytes) : !llvm.ptr, (i64, !llvm.ptr, i64) -> i32
    %done = arith.cmpi eq, %result, %zero32 : i32
    return %done : i1
  }
  func.func private @strict_gpu_get(%host: !llvm.ptr, %device: i64, %bytes: i64) -> i1 {
    %zero32 = arith.constant 0 : i32
    %getIndex = arith.constant 9 : i64
    %get = func.call @strict_gpu_call(%getIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %get(%host, %device, %bytes) : !llvm.ptr, (!llvm.ptr, i64, i64) -> i32
    %done = arith.cmpi eq, %result, %zero32 : i32
    return %done : i1
  }
  func.func private @strict_gpu_launch(%function: !llvm.ptr, %count: i64, %threads: i64, %parameters: !llvm.ptr) -> i1 {
    %zero32 = arith.constant 0 : i32
    %one32 = arith.constant 1 : i32
    %one = arith.constant 1 : i64
    %spare = arith.subi %threads, %one : i64
    %rounded = arith.addi %count, %spare : i64
    %blocks = arith.divui %rounded, %threads : i64
    %blocks32 = arith.trunci %blocks : i64 to i32
    %threads32 = arith.trunci %threads : i64 to i32
    %none = llvm.mlir.zero : !llvm.ptr
    %current = func.call @strict_gpu_current() : () -> i1
    %stream = func.call @strict_gpu_stream() : () -> !llvm.ptr
    func.call @strict_gpu_capture() : () -> ()
    %launchIndex = arith.constant 10 : i64
    %launch = func.call @strict_gpu_call(%launchIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %launch(%function, %blocks32, %one32, %one32, %threads32, %one32, %one32, %zero32, %stream, %parameters, %none) : !llvm.ptr, (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %launched = arith.cmpi eq, %result, %zero32 : i32
    func.call @strict_gpu_captured(%function) : (!llvm.ptr) -> ()
    return %launched : i1
  }
  llvm.func @free(!llvm.ptr)
  // a value is a list when its bit in mask is set, a list of rows when its bit in rows is set too
  // dots marks lists only read by dot products, with a list of rows they become 8 bit numbers
  func.func private @strict_gpu_run(%ptx: !llvm.ptr, %name: !llvm.ptr, %slot: !llvm.ptr, %count: i64, %source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64, %rows: i64, %dots: i64, %lanes: i64, %group: i64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %unset = llvm.mlir.zero : !llvm.ptr
    %work = func.call @strict_gpu_work(%count, %source, %kind) : (i64, f64, i64) -> i64
    %minimum = func.call @strict_gpu_threshold() : () -> i64
    %enough = arith.cmpi sge, %work, %minimum : i64
    cf.cond_br %enough, ^ready, ^small
  ^small:
    %never = arith.constant 9223372036854775807 : i64
    %disabled = arith.cmpi eq, %minimum, %never : i64
    cf.cond_br %disabled, ^failed, ^inputs
  ^inputs:
    %onDevice = func.call @strict_gpu_residents(%source, %kind, %values, %valueCount, %mask) : (f64, i64, !llvm.ptr, i64, i64) -> i1
    %hasRows = arith.cmpi ne, %rows, %none : i64
    %present = func.call @strict_gpu_present() : () -> i1
    %rowsLive = arith.andi %hasRows, %present : i1
    %wanted = arith.ori %onDevice, %rowsLive : i1
    cf.cond_br %wanted, ^ready, ^failed
  ^ready:
    %isReady = func.call @strict_gpu_ready() : () -> i1
    %groupFits = func.call @strict_gpu_grouping(%count, %group) : (i64, i64) -> i1
    %groupReady = arith.andi %isReady, %groupFits : i1
    cf.cond_br %groupReady, ^prepare, ^failed
  ^prepare:
    func.call @strict_gpu_reset() : () -> ()
    %triple = arith.muli %valueCount, %three : i64
    %slotCount = arith.addi %triple, %six : i64
    %slots = llvm.alloca %slotCount x i64 : (i64) -> !llvm.ptr
    %parameters = llvm.alloca %slotCount x !llvm.ptr : (i64) -> !llvm.ptr
    %bufferCount = arith.addi %valueCount, %three : i64
    %buffers = llvm.alloca %bufferCount x i64 : (i64) -> !llvm.ptr
    %flag = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %flag : i64, !llvm.ptr
    %narrowSlot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %narrowSlot : i64, !llvm.ptr
    %packedSlot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %packedSlot : i64, !llvm.ptr
    cf.br ^clear(%none : i64)
  ^clear(%clearIndex: i64):
    %clearing = arith.cmpi slt, %clearIndex, %slotCount : i64
    cf.cond_br %clearing, ^clearOne, ^sources
  ^clearOne:
    %pointed = llvm.getelementptr %slots[%clearIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %pointer = llvm.getelementptr %parameters[%clearIndex] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    llvm.store %pointed, %pointer : !llvm.ptr, !llvm.ptr
    %inBuffers = arith.cmpi slt, %clearIndex, %bufferCount : i64
    %clearNext = arith.addi %clearIndex, %one : i64
    cf.cond_br %inBuffers, ^clearBuffer, ^clear(%clearNext : i64)
  ^clearBuffer:
    %bufferSlot = llvm.getelementptr %buffers[%clearIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %none, %bufferSlot : i64, !llvm.ptr
    cf.br ^clear(%clearNext : i64)
  ^sources:
    llvm.store %count, %slots : i64, !llvm.ptr
    %sourceSlot = llvm.getelementptr %slots[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %sourceBuffer = llvm.getelementptr %buffers[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %sourceReady = func.call @strict_gpu_source(%source, %kind, %sourceSlot, %sourceBuffer) : (f64, i64, !llvm.ptr, !llvm.ptr) -> i1
    func.call @strict_gpu_narrowed(%sourceSlot, %narrowSlot) : (!llvm.ptr, !llvm.ptr) -> ()
    %rowsKind = arith.constant 2 : i64
    %isRowsSource = arith.cmpi eq, %kind, %rowsKind : i64
    cf.cond_br %isRowsSource, ^packedSource, ^sourced
  ^packedSource:
    func.call @strict_gpu_packed(%sourceSlot, %packedSlot) : (!llvm.ptr, !llvm.ptr) -> ()
    cf.br ^sourced
  ^sourced:
    cf.cond_br %sourceReady, ^values(%none, %six : i64, i64), ^release
  ^values(%valueIndex: i64, %slotIndex: i64):
    %moreValues = arith.cmpi slt, %valueIndex, %valueCount : i64
    cf.cond_br %moreValues, ^value, ^choose
  ^value:
    %valueSlot = llvm.getelementptr %values[%valueIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %valueSlot : !llvm.ptr -> f64
    %shifted = arith.shrui %mask, %valueIndex : i64
    %lowest = arith.andi %shifted, %one : i64
    %isList = arith.cmpi ne, %lowest, %none : i64
    %target = llvm.getelementptr %slots[%slotIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %nextValue = arith.addi %valueIndex, %one : i64
    cf.cond_br %isList, ^list, ^number
  ^number:
    llvm.store %value, %target : f64, !llvm.ptr
    %afterNumber = arith.addi %slotIndex, %one : i64
    cf.br ^values(%nextValue, %afterNumber : i64, i64)
  ^list:
    %listBuffer = arith.addi %valueIndex, %three : i64
    %listBufferSlot = llvm.getelementptr %buffers[%listBuffer] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %rowsShifted = arith.shrui %rows, %valueIndex : i64
    %rowsLowest = arith.andi %rowsShifted, %one : i64
    %isRows = arith.cmpi ne, %rowsLowest, %none : i64
    cf.cond_br %isRows, ^rowsValue, ^listValue
  ^rowsValue:
    %rowsReady = func.call @strict_gpu_rows(%value, %target, %listBufferSlot) : (f64, !llvm.ptr, !llvm.ptr) -> i1
    func.call @strict_gpu_packed(%target, %packedSlot) : (!llvm.ptr, !llvm.ptr) -> ()
    cf.br ^listed(%rowsReady : i1)
  ^listValue:
    %listReady = func.call @strict_gpu_list(%value, %target, %listBufferSlot) : (f64, !llvm.ptr, !llvm.ptr) -> i1
    cf.br ^listed(%listReady : i1)
  ^listed(%ready: i1):
    func.call @strict_gpu_narrowed(%target, %narrowSlot) : (!llvm.ptr, !llvm.ptr) -> ()
    %afterList = arith.addi %slotIndex, %three : i64
    cf.cond_br %ready, ^values(%nextValue, %afterList : i64, i64), ^release
  ^choose:
    %packedRows = llvm.load %packedSlot : !llvm.ptr -> i64
    %hasPackedRows = arith.cmpi ne, %packedRows, %none : i64
    %hasDots = arith.cmpi ne, %dots, %none : i64
    %quantizing = arith.andi %hasPackedRows, %hasDots : i1
    cf.cond_br %quantizing, ^quantize(%none, %six : i64, i64), ^chooseKernel
  ^quantize(%quantizeIndex: i64, %quantizeSlot: i64):
    %moreQuantized = arith.cmpi slt, %quantizeIndex, %valueCount : i64
    cf.cond_br %moreQuantized, ^quantizeValue, ^chooseKernel
  ^quantizeValue:
    %listShifted = arith.shrui %mask, %quantizeIndex : i64
    %listLowest = arith.andi %listShifted, %one : i64
    %isListValue = arith.cmpi ne, %listLowest, %none : i64
    %dotsShifted = arith.shrui %dots, %quantizeIndex : i64
    %dotsLowest = arith.andi %dotsShifted, %one : i64
    %isDotted = arith.cmpi ne, %dotsLowest, %none : i64
    %quantizedView = llvm.getelementptr %slots[%quantizeSlot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %width = arith.select %isListValue, %three, %one : i64
    %followingSlot = arith.addi %quantizeSlot, %width : i64
    %followingValue = arith.addi %quantizeIndex, %one : i64
    cf.cond_br %isDotted, ^quantizeOne, ^quantize(%followingValue, %followingSlot : i64, i64)
  ^quantizeOne:
    func.call @strict_gpu_quantized(%ptx, %quantizedView) : (!llvm.ptr, !llvm.ptr) -> ()
    cf.br ^quantize(%followingValue, %followingSlot : i64, i64)
  ^chooseKernel:
    %narrowed = llvm.load %narrowSlot : !llvm.ptr -> i64
    %single = arith.cmpi ne, %narrowed, %none : i64
    %nameLength = llvm.call @strlen(%name) : (!llvm.ptr) -> i64
    %skipped = arith.addi %nameLength, %one : i64
    %singleName = llvm.getelementptr %name[%skipped] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %singleSlot = llvm.getelementptr %slot[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %chosenName = llvm.select %single, %singleName, %name : i1, !llvm.ptr
    %chosenSlot = llvm.select %single, %singleSlot, %slot : i1, !llvm.ptr
    %function = func.call @strict_gpu_function(%ptx, %chosenName, %chosenSlot) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %missing = llvm.icmp "eq" %function, %unset : !llvm.ptr
    %widest = arith.constant 256 : i64
    %grouped = arith.cmpi sgt, %group, %none : i64
    %groupLanes = arith.minsi %group, %widest : i64
    %laneTotal = arith.muli %count, %lanes : i64
    %threadTotal = arith.select %grouped, %count, %laneTotal : i64
    %threads = arith.select %grouped, %groupLanes, %widest : i64
    cf.cond_br %missing, ^release, ^chosen
  ^chosen:
    cf.cond_br %single, ^async, ^outputs
  ^async:
    %singleBytes = arith.muli %count, %four : i64
    %class = func.call @strict_gpu_class(%singleBytes) : (i64) -> i64
    %resident = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    %failure = func.call @strict_gpu_failure_word() : () -> i64
    %hasResident = arith.cmpi ne, %resident, %none : i64
    %hasFailure = arith.cmpi ne, %failure, %none : i64
    %isPrepared = arith.andi %hasResident, %hasFailure : i1
    cf.cond_br %isPrepared, ^asyncFlush, ^asyncFailed
  ^asyncFlush:
    %residentSlot = llvm.getelementptr %slots[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %resident, %residentSlot : i64, !llvm.ptr
    %failureSlot = llvm.getelementptr %slots[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %failure, %failureSlot : i64, !llvm.ptr
    %asyncFlushed = func.call @strict_gpu_flush() : () -> i1
    cf.cond_br %asyncFlushed, ^asyncStart, ^asyncFailed
  ^asyncStart:
    %asyncLaunched = func.call @strict_gpu_launch(%function, %threadTotal, %threads, %parameters) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %asyncLaunched, ^asyncDone, ^asyncFailed
  ^asyncDone:
    %residentList = func.call @strict_gpu_resident(%count, %resident, %class) : (i64, i64, i64) -> f64
    func.call @strict_gpu_settle(%buffers, %bufferCount) : (!llvm.ptr, i64) -> ()
    return %residentList : f64
  ^asyncFailed:
    func.call @strict_gpu_give(%resident, %class) : (i64, i64) -> ()
    cf.br ^release
  ^outputs:
    %bytes = arith.muli %count, %eight : i64
    %withFlag = arith.addi %bytes, %eight : i64
    %result = func.call @strict_gpu_temporary(%withFlag, %buffers) : (i64, !llvm.ptr) -> i64
    %failedBuffer = llvm.getelementptr %buffers[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %failedDevice = arith.addi %result, %bytes : i64
    %allocated = arith.cmpi ne, %result, %none : i64
    cf.cond_br %allocated, ^clearFlag, ^release
  ^clearFlag:
    %cleared = func.call @strict_gpu_stage(%failedDevice, %flag, %eight) : (i64, !llvm.ptr, i64) -> i1
    cf.cond_br %cleared, ^fixed, ^release
  ^fixed:
    %resultSlot = llvm.getelementptr %slots[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %result, %resultSlot : i64, !llvm.ptr
    %failedSlot = llvm.getelementptr %slots[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %failedDevice, %failedSlot : i64, !llvm.ptr
    %flushed = func.call @strict_gpu_flush() : () -> i1
    cf.cond_br %flushed, ^start, ^release
  ^start:
    %launched = func.call @strict_gpu_launch(%function, %threadTotal, %threads, %parameters) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %launched, ^fetch, ^release
  ^fetch:
    %resultOwner = llvm.load %buffers : !llvm.ptr -> i64
    %flagOwner = llvm.load %failedBuffer : !llvm.ptr -> i64
    %owners = arith.ori %resultOwner, %flagOwner : i64
    %isScratch = arith.cmpi eq, %owners, %none : i64
    cf.cond_br %isScratch, ^together, ^check
  ^together:
    %fetched = func.call @strict_gpu_fetch(%result, %failedDevice, %count, %eight) : (i64, i64, i64, i64) -> f64
    func.call @strict_gpu_settle(%buffers, %bufferCount) : (!llvm.ptr, i64) -> ()
    return %fetched : f64
  ^check:
    %checked = func.call @strict_gpu_get(%flag, %failedDevice, %eight) : (!llvm.ptr, i64, i64) -> i1
    %failedValue = llvm.load %flag : !llvm.ptr -> i64
    %passed = arith.cmpi eq, %failedValue, %none : i64
    %usable = arith.andi %checked, %passed : i1
    cf.cond_br %usable, ^download, ^release
  ^download:
    %list = func.call @strict_gpu_result(%count, %eight) : (i64, i64) -> f64
    %items = func.call @strict_gpu_items(%list, %eight) : (f64, i64) -> !llvm.ptr
    %got = func.call @strict_gpu_get(%items, %result, %bytes) : (!llvm.ptr, i64, i64) -> i1
    func.call @strict_gpu_settle(%buffers, %bufferCount) : (!llvm.ptr, i64) -> ()
    %chosen = arith.select %got, %list, %zero : f64
    return %chosen : f64
  ^release:
    func.call @strict_gpu_settle(%buffers, %bufferCount) : (!llvm.ptr, i64) -> ()
    cf.br ^failed
  ^failed:
    return %zero : f64
  }
  // device memory for lists computed on the GPU: blocks of 256 << class bytes from 64 MB chunks, a free block goes on
  // the stack of its class; when a class has none, a young collection frees unreachable lists once 256 MB were taken
  // since the last collection, a full one once more than half of the free device memory (at least 1 GB) was cut from
  // chunks since the last full collection, the free memory is measured with every new chunk
  // ponytail: chunks are never returned and the rest of a chunk too small for the next block is lost
  llvm.mlir.global internal @strict_gpu_blocks(dense<0> : tensor<144xi64>) : !llvm.array<144 x i64>
  llvm.mlir.global internal @strict_gpu_chunk(dense<0> : tensor<6xi64>) : !llvm.array<6 x i64>
  llvm.mlir.global internal @strict_gpu_residing(dense<0> : tensor<3xi64>) : !llvm.array<3 x i64>
  llvm.mlir.global internal @strict_gpu_failure(0 : i64) : i64
  llvm.mlir.global internal @strict_gpu_mutex(0 : i64) : i64
  llvm.mlir.global internal constant @strict_gpu_failed_text("Error: a loop on the GPU read outside of a list\0A\00") : !llvm.array<49 x i8>
  func.func private @strict_gpu_class(%bytes: i64) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %smallest = arith.constant 256 : i64
    cf.br ^check(%none, %smallest : i64, i64)
  ^check(%class: i64, %size: i64):
    %fits = arith.cmpi uge, %size, %bytes : i64
    %next = arith.addi %class, %one : i64
    %doubled = arith.shli %size, %one : i64
    cf.cond_br %fits, ^done, ^check(%next, %doubled : i64, i64)
  ^done:
    return %class : i64
  }
  func.func private @strict_gpu_stack(%class: i64) -> !llvm.ptr {
    %three = arith.constant 3 : i64
    %blocks = llvm.mlir.addressof @strict_gpu_blocks : !llvm.ptr
    %offset = arith.muli %class, %three : i64
    %stack = llvm.getelementptr %blocks[0, %offset] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<144 x i64>
    return %stack : !llvm.ptr
  }
  func.func private @strict_gpu_obtain(%class: i64) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %stack = func.call @strict_gpu_stack(%class) : (i64) -> !llvm.ptr
    %countSlot = llvm.getelementptr %stack[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %hasFree = arith.cmpi sgt, %count, %none : i64
    cf.cond_br %hasFree, ^reuse, ^fresh
  ^reuse:
    %itemsBits = llvm.load %stack : !llvm.ptr -> i64
    %items = llvm.inttoptr %itemsBits : i64 to !llvm.ptr
    %top = arith.subi %count, %one : i64
    %topSlot = llvm.getelementptr %items[%top] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %device = llvm.load %topSlot : !llvm.ptr -> i64
    llvm.store %top, %countSlot : i64, !llvm.ptr
    %smallestBlock = arith.constant 256 : i64
    %blockSize = arith.shli %smallestBlock, %class : i64
    func.call @strict_gpu_obtained(%blockSize) : (i64) -> ()
    return %device : i64
  ^fresh:
    %chunk = llvm.mlir.addressof @strict_gpu_chunk : !llvm.ptr
    %sinceSlot = llvm.getelementptr %chunk[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %since = llvm.load %sinceSlot : !llvm.ptr -> i64
    %limitSlot = llvm.getelementptr %chunk[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %measured = llvm.load %limitSlot : !llvm.ptr -> i64
    %gigabyte = arith.constant 1073741824 : i64
    %limit = arith.maxui %measured, %gigabyte : i64
    %crowded = arith.cmpi ugt, %since, %limit : i64
    %obtainedSlot = llvm.getelementptr %chunk[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %obtained = llvm.load %obtainedSlot : !llvm.ptr -> i64
    %youngBytes = arith.constant 268435456 : i64
    %taken = arith.cmpi ugt, %obtained, %youngBytes : i64
    %youngLimitSlot = llvm.mlir.addressof @strict_gc_young_limit : !llvm.ptr
    %youngLimit = llvm.load %youngLimitSlot : !llvm.ptr -> i64
    %hasYoung = arith.cmpi ne, %youngLimit, %none : i64
    %minor = arith.andi %taken, %hasYoung : i1
    %due = arith.ori %crowded, %minor : i1
    cf.cond_br %due, ^collect, ^carve
  ^collect:
    func.call @strict_gc_run(%crowded) : (i1) -> ()
    %afterCount = llvm.load %countSlot : !llvm.ptr -> i64
    %freed = arith.cmpi sgt, %afterCount, %none : i64
    cf.cond_br %freed, ^again, ^carve
  ^again:
    %again = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    return %again : i64
  ^carve:
    %smallest = arith.constant 256 : i64
    %size = arith.shli %smallest, %class : i64
    %base = llvm.load %chunk : !llvm.ptr -> i64
    %offsetSlot = llvm.getelementptr %chunk[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %offset = llvm.load %offsetSlot : !llvm.ptr -> i64
    %chunkSizeSlot = llvm.getelementptr %chunk[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %chunkSize = llvm.load %chunkSizeSlot : !llvm.ptr -> i64
    %end = arith.addi %offset, %size : i64
    %fits = arith.cmpi ule, %end, %chunkSize : i64
    %hasChunk = arith.cmpi ne, %base, %none : i64
    %usable = arith.andi %fits, %hasChunk : i1
    cf.cond_br %usable, ^cut(%base, %offset : i64, i64), ^chunk
  ^chunk:
    %sixtyFourMegabytes = arith.constant 67108864 : i64
    %wanted = arith.maxui %size, %sixtyFourMegabytes : i64
    %fresh = func.call @strict_gpu_alloc(%wanted) : (i64) -> i64
    %allocated = arith.cmpi ne, %fresh, %none : i64
    cf.cond_br %allocated, ^keep, ^none
  ^keep:
    llvm.store %fresh, %chunk : i64, !llvm.ptr
    llvm.store %wanted, %chunkSizeSlot : i64, !llvm.ptr
    %free = func.call @strict_gpu_free_memory() : () -> i64
    %half = arith.shrui %free, %one : i64
    llvm.store %half, %limitSlot : i64, !llvm.ptr
    cf.br ^cut(%fresh, %none : i64, i64)
  ^cut(%from: i64, %at: i64):
    %device2 = arith.addi %from, %at : i64
    %after = arith.addi %at, %size : i64
    llvm.store %after, %offsetSlot : i64, !llvm.ptr
    %sinceNow = llvm.load %sinceSlot : !llvm.ptr -> i64
    %grown = arith.addi %sinceNow, %size : i64
    llvm.store %grown, %sinceSlot : i64, !llvm.ptr
    func.call @strict_gpu_obtained(%size) : (i64) -> ()
    return %device2 : i64
  ^none:
    return %none : i64
  }
  func.func private @strict_gpu_obtained(%bytes: i64) {
    %chunk = llvm.mlir.addressof @strict_gpu_chunk : !llvm.ptr
    %obtainedSlot = llvm.getelementptr %chunk[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %obtained = llvm.load %obtainedSlot : !llvm.ptr -> i64
    %more = arith.addi %obtained, %bytes : i64
    llvm.store %more, %obtainedSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_free_memory() -> i64 {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %sizes = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %sizes : i64, !llvm.ptr
    %infoIndex = arith.constant 24 : i64
    %info = func.call @strict_gpu_call(%infoIndex) : (i64) -> !llvm.ptr
    %totalSlot = llvm.getelementptr %sizes[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %result = llvm.call %info(%sizes, %totalSlot) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> i32
    %free = llvm.load %sizes : !llvm.ptr -> i64
    return %free : i64
  }
  func.func private @strict_gpu_give(%device: i64, %class: i64) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %isSet = arith.cmpi ne, %device, %none : i64
    cf.cond_br %isSet, ^push, ^done
  ^push:
    %stack = func.call @strict_gpu_stack(%class) : (i64) -> !llvm.ptr
    %countSlot = llvm.getelementptr %stack[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %capacitySlot = llvm.getelementptr %stack[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %full = arith.cmpi sge, %count, %capacity : i64
    cf.cond_br %full, ^grow, ^store
  ^grow:
    %oldBits = llvm.load %stack : !llvm.ptr -> i64
    %old = llvm.inttoptr %oldBits : i64 to !llvm.ptr
    %doubled = arith.shli %capacity, %one : i64
    %sixteen = arith.constant 16 : i64
    %grown = arith.maxui %doubled, %sixteen : i64
    %bytes = arith.muli %grown, %eight : i64
    %fresh = llvm.call @realloc(%old, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %freshBits = llvm.ptrtoint %fresh : !llvm.ptr to i64
    llvm.store %freshBits, %stack : i64, !llvm.ptr
    llvm.store %grown, %capacitySlot : i64, !llvm.ptr
    cf.br ^store
  ^store:
    %itemsBits = llvm.load %stack : !llvm.ptr -> i64
    %items = llvm.inttoptr %itemsBits : i64 to !llvm.ptr
    %slot = llvm.getelementptr %items[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %device, %slot : i64, !llvm.ptr
    %more = arith.addi %count, %one : i64
    llvm.store %more, %countSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  // the pinned word a loop on the GPU sets when it reads outside of a list, the host reads it after downloading
  func.func private @strict_gpu_failure_word() -> i64 {
    %none = arith.constant 0 : i64
    %slot = llvm.mlir.addressof @strict_gpu_failure : !llvm.ptr
    %word = llvm.load %slot : !llvm.ptr -> i64
    %isSet = arith.cmpi ne, %word, %none : i64
    cf.cond_br %isSet, ^done(%word : i64), ^allocate
  ^allocate:
    %eight = arith.constant 8 : i64
    %pinned = func.call @strict_gpu_pinned(%eight) : (i64) -> i64
    %isPinned = arith.cmpi ne, %pinned, %none : i64
    cf.cond_br %isPinned, ^clear, ^done(%none : i64)
  ^clear:
    %pointer = llvm.inttoptr %pinned : i64 to !llvm.ptr
    llvm.store %none, %pointer : i64, !llvm.ptr
    llvm.store %pinned, %slot : i64, !llvm.ptr
    cf.br ^done(%pinned : i64)
  ^done(%result: i64):
    return %result : i64
  }
  func.func private @strict_gpu_resident(%count: i64, %device: i64, %class: i64) -> f64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %zero = arith.constant 0.0 : f64
    %thirtyTwo = arith.constant 32 : i64
    %tagged = arith.ori %device, %class : i64
    %deviceBits = arith.bitcast %tagged : i64 to f64
    %list = func.call @strict_packed_view(%count, %thirtyTwo, %zero, %deviceBits) : (i64, i64, f64, f64) -> f64
    %residing = llvm.mlir.addressof @strict_gpu_residing : !llvm.ptr
    %usedSlot = llvm.getelementptr %residing[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    %capacitySlot = llvm.getelementptr %residing[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %full = arith.cmpi sge, %used, %capacity : i64
    cf.cond_br %full, ^grow, ^store
  ^grow:
    %oldBits = llvm.load %residing : !llvm.ptr -> i64
    %old = llvm.inttoptr %oldBits : i64 to !llvm.ptr
    %doubled = arith.shli %capacity, %one : i64
    %smallest = arith.constant 1024 : i64
    %grown = arith.maxui %doubled, %smallest : i64
    %entryBytes = arith.muli %grown, %three : i64
    %bytes = arith.muli %entryBytes, %eight : i64
    %fresh = llvm.call @realloc(%old, %bytes) : (!llvm.ptr, i64) -> !llvm.ptr
    %freshBits = llvm.ptrtoint %fresh : !llvm.ptr to i64
    llvm.store %freshBits, %residing : i64, !llvm.ptr
    llvm.store %grown, %capacitySlot : i64, !llvm.ptr
    cf.br ^store
  ^store:
    %entriesBits = llvm.load %residing : !llvm.ptr -> i64
    %entries = llvm.inttoptr %entriesBits : i64 to !llvm.ptr
    %offset = arith.muli %used, %three : i64
    %entry = llvm.getelementptr %entries[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %listBits = arith.bitcast %list : f64 to i64
    llvm.store %listBits, %entry : i64, !llvm.ptr
    %deviceSlot = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %device, %deviceSlot : i64, !llvm.ptr
    %classSlot = llvm.getelementptr %entry[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %class, %classSlot : i64, !llvm.ptr
    %more = arith.addi %used, %one : i64
    llvm.store %more, %usedSlot : i64, !llvm.ptr
    return %list : f64
  }
  func.func private @strict_gpu_sweep_residents(%full: i1) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %residing = llvm.mlir.addressof @strict_gpu_residing : !llvm.ptr
    %usedSlot = llvm.getelementptr %residing[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %entriesBits = llvm.load %residing : !llvm.ptr -> i64
    %entries = llvm.inttoptr %entriesBits : i64 to !llvm.ptr
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    cf.br ^next(%none, %used : i64, i64)
  ^next(%index: i64, %count: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^check, ^done
  ^check:
    %offset = arith.muli %index, %three : i64
    %entry = llvm.getelementptr %entries[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %listBits = llvm.load %entry : !llvm.ptr -> i64
    %marked = func.call @strict_gc_is_marked(%listBits) : (i64) -> i1
    %following = arith.addi %index, %one : i64
    cf.cond_br %marked, ^next(%following, %count : i64, i64), ^free
  ^free:
    %deviceSlot = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %device = llvm.load %deviceSlot : !llvm.ptr -> i64
    %classSlot = llvm.getelementptr %entry[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %class = llvm.load %classSlot : !llvm.ptr -> i64
    func.call @strict_gpu_give(%device, %class) : (i64, i64) -> ()
    %last = arith.subi %count, %one : i64
    %lastOffset = arith.muli %last, %three : i64
    %lastEntry = llvm.getelementptr %entries[%lastOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %lastList = llvm.load %lastEntry : !llvm.ptr -> i64
    llvm.store %lastList, %entry : i64, !llvm.ptr
    %lastDeviceSlot = llvm.getelementptr %lastEntry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %lastDevice = llvm.load %lastDeviceSlot : !llvm.ptr -> i64
    llvm.store %lastDevice, %deviceSlot : i64, !llvm.ptr
    %lastClassSlot = llvm.getelementptr %lastEntry[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %lastClass = llvm.load %lastClassSlot : !llvm.ptr -> i64
    llvm.store %lastClass, %classSlot : i64, !llvm.ptr
    cf.br ^next(%index, %last : i64, i64)
  ^done:
    llvm.store %count, %usedSlot : i64, !llvm.ptr
    %chunk = llvm.mlir.addressof @strict_gpu_chunk : !llvm.ptr
    %obtainedSlot = llvm.getelementptr %chunk[5] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %obtainedSlot : i64, !llvm.ptr
    cf.cond_br %full, ^carved, ^kept
  ^carved:
    %sinceSlot = llvm.getelementptr %chunk[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %sinceSlot : i64, !llvm.ptr
    return
  ^kept:
    return
  }
  // a loop of at least 256 elements tries the GPU, a smaller one too once lists live there, it may read one of them
  func.func private @strict_gpu_wanted(%count: i64) -> i1 attributes {passthrough = ["alwaysinline"]} {
    %threshold = arith.constant 256 : i64
    %enough = arith.cmpi sge, %count, %threshold : i64
    %present = func.call @strict_gpu_present() : () -> i1
    %wanted = arith.ori %enough, %present : i1
    return %wanted : i1
  }
  func.func private @strict_gpu_present() -> i1 attributes {passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %residing = llvm.mlir.addressof @strict_gpu_residing : !llvm.ptr
    %usedSlot = llvm.getelementptr %residing[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    %present = arith.cmpi sgt, %used, %none : i64
    return %present : i1
  }
  // a list a kernel reads without the CPU: on the GPU already, or packed numbers (decoded there) once lists live there
  func.func private @strict_gpu_readable(%list: f64) -> i1 {
    %none = arith.constant 0 : i64
    %device = func.call @strict_gpu_device(%list) : (f64) -> i64
    %resides = arith.cmpi ne, %device, %none : i64
    %bits = func.call @strict_list_packed(%list) : (f64) -> i64
    %packed = arith.cmpi sgt, %bits, %none : i64
    %present = func.call @strict_gpu_present() : () -> i1
    %uploaded = arith.andi %packed, %present : i1
    %readable = arith.ori %resides, %uploaded : i1
    return %readable : i1
  }
  func.func private @strict_gpu_device(%list: f64) -> i64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %isUnset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %isUnset, ^plain, ^check
  ^check:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %capacitySlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> f64
    %isPacked = arith.cmpf olt, %capacity, %zero : f64
    cf.cond_br %isPacked, ^packed(%holder : !llvm.ptr), ^plain
  ^packed(%packedHolder: !llvm.ptr):
    %deviceSlot = llvm.getelementptr %packedHolder[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %tagged = llvm.load %deviceSlot : !llvm.ptr -> i64
    %addressMask = arith.constant -256 : i64
    %device = arith.andi %tagged, %addressMask : i64
    return %device : i64
  ^plain:
    return %none : i64
  }
  func.func private @strict_gpu_residents(%source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64) -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %true = arith.constant true
    %false = arith.constant false
    %isList = arith.cmpi eq, %kind, %one : i64
    cf.cond_br %isList, ^source, ^next(%none : i64)
  ^source:
    %sourceResides = func.call @strict_gpu_readable(%source) : (f64) -> i1
    cf.cond_br %sourceResides, ^found, ^next(%none : i64)
  ^next(%index: i64):
    %more = arith.cmpi slt, %index, %valueCount : i64
    cf.cond_br %more, ^value, ^missing
  ^value:
    %shifted = arith.shrui %mask, %index : i64
    %lowest = arith.andi %shifted, %one : i64
    %following = arith.addi %index, %one : i64
    %isListValue = arith.cmpi ne, %lowest, %none : i64
    cf.cond_br %isListValue, ^listed, ^next(%following : i64)
  ^listed:
    %slot = llvm.getelementptr %values[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %resides = func.call @strict_gpu_readable(%value) : (f64) -> i1
    cf.cond_br %resides, ^found, ^next(%following : i64)
  ^found:
    return %true : i1
  ^missing:
    return %false : i1
  }
  // the first read on the CPU waits for the GPU and copies the numbers into the host block of the list
  func.func private @strict_gpu_download(%holder: !llvm.ptr) -> !llvm.ptr {
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f64
    %eight = arith.constant 8 : i64
    %busySlot = llvm.mlir.addressof @strict_pool_busy : !llvm.ptr
    %busy = llvm.load %busySlot atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
    %shared = arith.cmpi ne, %busy, %none : i64
    cf.cond_br %shared, ^lock, ^check
  ^lock:
    %mutex = llvm.mlir.addressof @strict_gpu_mutex : !llvm.ptr
    llvm.call @strict_spin_lock(%mutex) : (!llvm.ptr) -> ()
    cf.br ^check
  ^check:
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %dataBits = llvm.load %dataSlot : !llvm.ptr -> f64
    %deviceSlot = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %tagged = llvm.load %deviceSlot : !llvm.ptr -> i64
    %addressMask = arith.constant -256 : i64
    %device = arith.andi %tagged, %addressMask : i64
    %stillMissing = arith.cmpf oeq, %dataBits, %zero : f64
    %hasDevice = arith.cmpi ne, %device, %none : i64
    %fetching = arith.andi %stillMissing, %hasDevice : i1
    cf.cond_br %fetching, ^fetch, ^present
  ^fetch:
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %count = arith.fptosi %used : f64 to i64
    %four = arith.constant 4 : i64
    %bytes = arith.muli %count, %four : i64
    %size = arith.maxui %bytes, %eight : i64
    %host = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %current = func.call @strict_gpu_current() : () -> i1
    %bounce = func.call @strict_gpu_bounce(%size) : (i64) -> !llvm.ptr
    %got = func.call @strict_gpu_get(%bounce, %device, %bytes) : (!llvm.ptr, i64, i64) -> i1
    %copied = llvm.call @memcpy(%host, %bounce, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %failureSlot = llvm.mlir.addressof @strict_gpu_failure : !llvm.ptr
    %word = llvm.load %failureSlot : !llvm.ptr -> i64
    %pointer = llvm.inttoptr %word : i64 to !llvm.ptr
    %failed = llvm.load %pointer : !llvm.ptr -> i64
    %isFailed = arith.cmpi ne, %failed, %none : i64
    %problem = arith.xori %got, %isFailed : i1
    %fine = arith.andi %got, %problem : i1
    cf.cond_br %fine, ^store, ^fail
  ^store:
    %false = arith.constant false
    func.call @strict_packed_canonical(%host, %count, %false) : (!llvm.ptr, i64, i1) -> ()
    %hostBits = func.call @strict_bits(%host) : (!llvm.ptr) -> f64
    llvm.store %hostBits, %dataSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%dataSlot) : (!llvm.ptr) -> ()
    cf.cond_br %shared, ^unlock(%host : !llvm.ptr), ^done(%host : !llvm.ptr)
  ^present:
    %data = func.call @strict_ptr(%dataBits) : (f64) -> !llvm.ptr
    cf.cond_br %shared, ^unlock(%data : !llvm.ptr), ^done(%data : !llvm.ptr)
  ^unlock(%unlocked: !llvm.ptr):
    %unlockedMutex = llvm.mlir.addressof @strict_gpu_mutex : !llvm.ptr
    llvm.call @strict_spin_unlock(%unlockedMutex) : (!llvm.ptr) -> ()
    cf.br ^done(%unlocked : !llvm.ptr)
  ^done(%result: !llvm.ptr):
    return %result : !llvm.ptr
  ^fail:
    %message = llvm.mlir.addressof @strict_gpu_failed_text : !llvm.ptr
    %printed = llvm.call @printf(%message) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %code = arith.constant 1 : i32
    llvm.call @exit(%code) : (i32) -> ()
    %nothing = llvm.mlir.zero : !llvm.ptr
    return %nothing : !llvm.ptr
  }
  // downloads go through pinned memory, copies into pageable memory ran at about 1 GB/s
  llvm.mlir.global internal @strict_gpu_bounced(dense<0> : tensor<2xi64>) : !llvm.array<2 x i64>
  func.func private @strict_gpu_bounce(%bytes: i64) -> !llvm.ptr {
    %bounced = llvm.mlir.addressof @strict_gpu_bounced : !llvm.ptr
    %capacitySlot = llvm.getelementptr %bounced[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %enough = arith.cmpi uge, %capacity, %bytes : i64
    cf.cond_br %enough, ^done, ^grow
  ^grow:
    %old = llvm.load %bounced : !llvm.ptr -> i64
    func.call @strict_gpu_unpinned(%old) : (i64) -> ()
    %one = arith.constant 1 : i64
    %doubled = arith.shli %bytes, %one : i64
    %fresh = func.call @strict_gpu_pinned(%doubled) : (i64) -> i64
    llvm.store %fresh, %bounced : i64, !llvm.ptr
    llvm.store %doubled, %capacitySlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    %bufferBits = llvm.load %bounced : !llvm.ptr -> i64
    %buffer = llvm.inttoptr %bufferBits : i64 to !llvm.ptr
    return %buffer : !llvm.ptr
  }
  // appends in place when the left list is the newest version of its block and the block has room, otherwise both go
  // into a block of twice the size, so a list growing one step at a time (a cache of keys) copies rarely
  // a host list of doubles appended with a list on the GPU moves there as 32 bit floats (like every result there), so the
  // append and the next loops stay on the GPU instead of downloading the new part and uploading everything again
  func.func private @strict_gpu_moved(%list: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %sixtyFour = arith.constant 64 : i64
    %bits = func.call @strict_gpu_bits(%list) : (f64) -> i64
    %isPlain = arith.cmpi eq, %bits, %sixtyFour : i64
    cf.cond_br %isPlain, ^plain, ^failed
  ^plain:
    %lengthNumber = func.call @strict_list_length(%list) : (f64) -> f64
    %length = arith.fptosi %lengthNumber : f64 to i64
    %bytes = arith.muli %length, %four : i64
    %doubled = arith.shli %bytes, %one : i64
    %class = func.call @strict_gpu_class(%doubled) : (i64) -> i64
    %block = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    %hasBlock = arith.cmpi ne, %block, %none : i64
    cf.cond_br %hasBlock, ^convert, ^failed
  ^convert:
    %floats = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    cf.br ^loop(%none : i64)
  ^loop(%index: i64):
    %more = arith.cmpi slt, %index, %length : i64
    cf.cond_br %more, ^item, ^upload
  ^item:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %single = arith.truncf %value : f64 to f32
    %target = llvm.getelementptr %floats[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %single, %target : f32, !llvm.ptr
    %next = arith.addi %index, %one : i64
    cf.br ^loop(%next : i64)
  ^upload:
    func.call @strict_gpu_replay() : () -> ()
    %put = func.call @strict_gpu_put_async(%block, %floats, %bytes) : (i64, !llvm.ptr, i64) -> i1
    llvm.call @free(%floats) : (!llvm.ptr) -> ()
    cf.cond_br %put, ^moved, ^returned
  ^moved:
    %moved = func.call @strict_gpu_resident(%length, %block, %class) : (i64, i64, i64) -> f64
    return %moved : f64
  ^returned:
    func.call @strict_gpu_give(%block, %class) : (i64, i64) -> ()
    cf.br ^failed
  ^failed:
    return %zero : f64
  }
  func.func private @strict_gpu_concat(%left: f64, %right: f64) -> f64 {
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %smallest = arith.constant 256 : i64
    %classMask = arith.constant 63 : i64
    %addressMask = arith.constant -256 : i64
    %leftNumber = func.call @strict_list_length(%left) : (f64) -> f64
    %rightNumber = func.call @strict_list_length(%right) : (f64) -> f64
    %leftCount = arith.fptosi %leftNumber : f64 to i64
    %rightCount = arith.fptosi %rightNumber : f64 to i64
    %total = arith.addi %leftCount, %rightCount : i64
    %bytes = arith.muli %total, %four : i64
    %leftBytes = arith.muli %leftCount, %four : i64
    %rightBytes = arith.muli %rightCount, %four : i64
    %rightDevice = func.call @strict_gpu_device(%right) : (f64) -> i64
    %view = func.call @strict_ptr(%left) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %taggedSlot = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %tagged = llvm.load %taggedSlot : !llvm.ptr -> i64
    %leftDevice = arith.andi %tagged, %addressMask : i64
    %class = arith.andi %tagged, %classMask : i64
    %capacity = arith.shli %smallest, %class : i64
    %newest = arith.cmpf oeq, %leftNumber, %used : f64
    %fits = arith.cmpi ule, %bytes, %capacity : i64
    %inPlace = arith.andi %newest, %fits : i1
    cf.cond_br %inPlace, ^append, ^copy
  ^append:
    %end = arith.addi %leftDevice, %leftBytes : i64
    %appended = func.call @strict_gpu_copy(%end, %rightDevice, %rightBytes) : (i64, i64, i64) -> i1
    cf.cond_br %appended, ^extend, ^failed
  ^extend:
    %totalNumber = arith.sitofp %total : i64 to f64
    llvm.store %totalNumber, %usedSlot : f64, !llvm.ptr
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %zero, %dataSlot : f64, !llvm.ptr
    %sixteen = arith.constant 16 : i64
    %longer = func.call @strict_alloc(%sixteen) : (i64) -> !llvm.ptr
    llvm.store %totalNumber, %longer : f64, !llvm.ptr
    %longerHolder = llvm.getelementptr %longer[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderBits, %longerHolder : f64, !llvm.ptr
    %longerList = func.call @strict_bits(%longer) : (!llvm.ptr) -> f64
    return %longerList : f64
  ^copy:
    %doubled = arith.shli %bytes, %one : i64
    %newClass = func.call @strict_gpu_class(%doubled) : (i64) -> i64
    %fresh = func.call @strict_gpu_obtain(%newClass) : (i64) -> i64
    %hasFresh = arith.cmpi ne, %fresh, %none : i64
    cf.cond_br %hasFresh, ^first, ^failed
  ^first:
    %firstCopied = func.call @strict_gpu_copy(%fresh, %leftDevice, %leftBytes) : (i64, i64, i64) -> i1
    %rightStart = arith.addi %fresh, %leftBytes : i64
    %secondCopied = func.call @strict_gpu_copy(%rightStart, %rightDevice, %rightBytes) : (i64, i64, i64) -> i1
    %bothCopied = arith.andi %firstCopied, %secondCopied : i1
    cf.cond_br %bothCopied, ^joined, ^returned
  ^joined:
    %joined = func.call @strict_gpu_resident(%total, %fresh, %newClass) : (i64, i64, i64) -> f64
    return %joined : f64
  ^returned:
    func.call @strict_gpu_give(%fresh, %newClass) : (i64, i64) -> ()
    cf.br ^failed
  ^failed:
    return %zero : f64
  }
  func.func private @strict_gpu_copy(%target: i64, %source: i64, %bytes: i64) -> i1 {
    %zero32 = arith.constant 0 : i32
    %none = arith.constant 0 : i64
    %true = arith.constant true
    %isEmpty = arith.cmpi eq, %bytes, %none : i64
    cf.cond_br %isEmpty, ^done, ^copy
  ^copy:
    %copyIndex = arith.constant 15 : i64
    %copy = func.call @strict_gpu_call(%copyIndex) : (i64) -> !llvm.ptr
    %stream = func.call @strict_gpu_stream() : () -> !llvm.ptr
    %result = llvm.call %copy(%target, %source, %bytes, %stream) : !llvm.ptr, (i64, i64, i64, !llvm.ptr) -> i32
    func.call @strict_gpu_captured(%copy) : (!llvm.ptr) -> ()
    %copied = arith.cmpi eq, %result, %zero32 : i32
    return %copied : i1
  ^done:
    return %true : i1
  }
  // launches are captured into graphs of 48 and replayed, a graph of the same kernels runs again with updated arguments
  // (cuGraphExecUpdate), so a decode step costs about 1.5 instead of 6 us of driver time per launch, STRICT_GRAPHS=0
  // launches directly
  llvm.mlir.global internal @strict_gpu_graph(dense<0> : tensor<4xi64>) : !llvm.array<4 x i64>
  llvm.mlir.global internal @strict_gpu_graphs(dense<0> : tensor<512xi64>) : !llvm.array<512 x i64>
  llvm.mlir.global internal constant @strict_gpu_graph_switch("STRICT_GRAPHS\00") : !llvm.array<14 x i8>
  llvm.mlir.global internal constant @strict_gpu_graph_failed("Error: the GPU launches of a graph failed\0A\00") : !llvm.array<43 x i8>
  func.func private @strict_gpu_stream() -> !llvm.ptr {
    %handles = llvm.mlir.addressof @strict_gpu_handles : !llvm.ptr
    %streamSlot = llvm.getelementptr %handles[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %streamBits = llvm.load %streamSlot : !llvm.ptr -> i64
    %stream = llvm.inttoptr %streamBits : i64 to !llvm.ptr
    return %stream : !llvm.ptr
  }
  func.func private @strict_gpu_graphing() -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %graph = llvm.mlir.addressof @strict_gpu_graph : !llvm.ptr
    %state = llvm.load %graph : !llvm.ptr -> i64
    %isKnown = arith.cmpi ne, %state, %none : i64
    cf.cond_br %isKnown, ^known(%state : i64), ^read
  ^read:
    %name = llvm.mlir.addressof @strict_gpu_graph_switch : !llvm.ptr
    %setting = llvm.call @getenv(%name) : (!llvm.ptr) -> !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %isUnset = llvm.icmp "eq" %setting, %unset : !llvm.ptr
    cf.cond_br %isUnset, ^decided(%one : i64), ^setting
  ^setting:
    %first = llvm.load %setting : !llvm.ptr -> i8
    %off = arith.constant 48 : i8
    %isOff = arith.cmpi eq, %first, %off : i8
    %chosen = arith.select %isOff, %two, %one : i64
    cf.br ^decided(%chosen : i64)
  ^decided(%decided: i64):
    llvm.store %decided, %graph : i64, !llvm.ptr
    cf.br ^known(%decided : i64)
  ^known(%known: i64):
    %enabled = arith.cmpi eq, %known, %one : i64
    return %enabled : i1
  }
  func.func private @strict_gpu_capture() {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %enabled = func.call @strict_gpu_graphing() : () -> i1
    cf.cond_br %enabled, ^check, ^done
  ^check:
    %graph = llvm.mlir.addressof @strict_gpu_graph : !llvm.ptr
    %capturingSlot = llvm.getelementptr %graph[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capturing = llvm.load %capturingSlot : !llvm.ptr -> i64
    %isCapturing = arith.cmpi ne, %capturing, %none : i64
    cf.cond_br %isCapturing, ^done, ^begin
  ^begin:
    %stream = func.call @strict_gpu_stream() : () -> !llvm.ptr
    %beginIndex = arith.constant 17 : i64
    %begin = func.call @strict_gpu_call(%beginIndex) : (i64) -> !llvm.ptr
    %relaxed = arith.constant 2 : i32
    %begun = llvm.call %begin(%stream, %relaxed) : !llvm.ptr, (!llvm.ptr, i32) -> i32
    llvm.store %one, %capturingSlot : i64, !llvm.ptr
    %launchesSlot = llvm.getelementptr %graph[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %launchesSlot : i64, !llvm.ptr
    %printSlot = llvm.getelementptr %graph[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %printSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_captured(%function: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %graph = llvm.mlir.addressof @strict_gpu_graph : !llvm.ptr
    %capturingSlot = llvm.getelementptr %graph[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capturing = llvm.load %capturingSlot : !llvm.ptr -> i64
    %isCapturing = arith.cmpi ne, %capturing, %none : i64
    cf.cond_br %isCapturing, ^count, ^done
  ^count:
    %launchesSlot = llvm.getelementptr %graph[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %launches = llvm.load %launchesSlot : !llvm.ptr -> i64
    %more = arith.addi %launches, %one : i64
    llvm.store %more, %launchesSlot : i64, !llvm.ptr
    %printSlot = llvm.getelementptr %graph[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %print = llvm.load %printSlot : !llvm.ptr -> i64
    %prime = arith.constant 1099511628211 : i64
    %mixed = arith.muli %print, %prime : i64
    %functionBits = llvm.ptrtoint %function : !llvm.ptr to i64
    %combined = arith.xori %mixed, %functionBits : i64
    llvm.store %combined, %printSlot : i64, !llvm.ptr
    %segment = arith.constant 96 : i64
    %full = arith.cmpi sge, %more, %segment : i64
    cf.cond_br %full, ^replay, ^done
  ^replay:
    func.call @strict_gpu_replay() : () -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_replay() {
    %none = arith.constant 0 : i64
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %graph = llvm.mlir.addressof @strict_gpu_graph : !llvm.ptr
    %capturingSlot = llvm.getelementptr %graph[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capturing = llvm.load %capturingSlot : !llvm.ptr -> i64
    %isCapturing = arith.cmpi ne, %capturing, %none : i64
    cf.cond_br %isCapturing, ^end, ^done
  ^end:
    llvm.store %none, %capturingSlot : i64, !llvm.ptr
    %stream = func.call @strict_gpu_stream() : () -> !llvm.ptr
    %graphSlot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %graphSlot : i64, !llvm.ptr
    %endIndex = arith.constant 18 : i64
    %endCapture = func.call @strict_gpu_call(%endIndex) : (i64) -> !llvm.ptr
    %ended = llvm.call %endCapture(%stream, %graphSlot) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> i32
    %endFailed = arith.cmpi ne, %ended, %zero32 : i32
    cf.cond_br %endFailed, ^fail, ^find
  ^find:
    %graphBits = llvm.load %graphSlot : !llvm.ptr -> i64
    %captured = llvm.inttoptr %graphBits : i64 to !llvm.ptr
    %printSlot = llvm.getelementptr %graph[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %print = llvm.load %printSlot : !llvm.ptr -> i64
    %entry = func.call @strict_gpu_graph_entry(%print) : (i64) -> !llvm.ptr
    %stored = llvm.load %entry : !llvm.ptr -> i64
    %execSlot = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %execBits = llvm.load %execSlot : !llvm.ptr -> i64
    %samePrint = arith.cmpi eq, %stored, %print : i64
    %hasExec = arith.cmpi ne, %execBits, %none : i64
    %reusable = arith.andi %samePrint, %hasExec : i1
    cf.cond_br %reusable, ^update, ^replace
  ^update:
    %exec = llvm.inttoptr %execBits : i64 to !llvm.ptr
    %three = arith.constant 3 : i64
    %info = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    %updateIndex = arith.constant 20 : i64
    %update = func.call @strict_gpu_call(%updateIndex) : (i64) -> !llvm.ptr
    %updated = llvm.call %update(%exec, %captured, %info) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %updateFailed = arith.cmpi ne, %updated, %zero32 : i32
    cf.cond_br %updateFailed, ^replace, ^launch(%exec : !llvm.ptr)
  ^replace:
    func.call @strict_gpu_forget(%execSlot) : (!llvm.ptr) -> ()
    %newSlot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %newSlot : i64, !llvm.ptr
    %instantiateIndex = arith.constant 19 : i64
    %instantiate = func.call @strict_gpu_call(%instantiateIndex) : (i64) -> !llvm.ptr
    %instantiated = llvm.call %instantiate(%newSlot, %captured, %none) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, i64) -> i32
    %instantiateFailed = arith.cmpi ne, %instantiated, %zero32 : i32
    cf.cond_br %instantiateFailed, ^fail, ^keep
  ^keep:
    %newBits = llvm.load %newSlot : !llvm.ptr -> i64
    llvm.store %print, %entry : i64, !llvm.ptr
    llvm.store %newBits, %execSlot : i64, !llvm.ptr
    %newExec = llvm.inttoptr %newBits : i64 to !llvm.ptr
    cf.br ^launch(%newExec : !llvm.ptr)
  ^launch(%runnable: !llvm.ptr):
    %launchIndex = arith.constant 21 : i64
    %launchGraph = func.call @strict_gpu_call(%launchIndex) : (i64) -> !llvm.ptr
    %launched = llvm.call %launchGraph(%runnable, %stream) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> i32
    // under WDDM the launch waits in the driver until something submits it, a query does (it is not ready yet)
    %queryIndex = arith.constant 25 : i64
    %query = func.call @strict_gpu_call(%queryIndex) : (i64) -> !llvm.ptr
    %queried = llvm.call %query(%stream) : !llvm.ptr, (!llvm.ptr) -> i32
    %destroyIndex = arith.constant 22 : i64
    %destroy = func.call @strict_gpu_call(%destroyIndex) : (i64) -> !llvm.ptr
    %destroyed = llvm.call %destroy(%captured) : !llvm.ptr, (!llvm.ptr) -> i32
    %launchFailed = arith.cmpi ne, %launched, %zero32 : i32
    cf.cond_br %launchFailed, ^fail, ^done
  ^fail:
    %message = llvm.mlir.addressof @strict_gpu_graph_failed : !llvm.ptr
    %printed = llvm.call @printf(%message) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %code = arith.constant 1 : i32
    llvm.call @exit(%code) : (i32) -> ()
    return
  ^done:
    return
  }
  // the cache entry (print and executable graph) of a print: 8 entries from the top byte of the print are probed for it or
  // an empty one, otherwise one of them is replaced
  func.func private @strict_gpu_graph_entry(%print: i64) -> !llvm.ptr {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %probes = arith.constant 8 : i64
    %mask = arith.constant 255 : i64
    %seven = arith.constant 7 : i64
    %highest = arith.constant 56 : i64
    %graphs = llvm.mlir.addressof @strict_gpu_graphs : !llvm.ptr
    %first = arith.shrui %print, %highest : i64
    cf.br ^probe(%none : i64)
  ^probe(%probe: i64):
    %more = arith.cmpi slt, %probe, %probes : i64
    cf.cond_br %more, ^check, ^victim
  ^check:
    %slotIndex = arith.addi %first, %probe : i64
    %index = arith.andi %slotIndex, %mask : i64
    %pair = arith.shli %index, %one : i64
    %entry = llvm.getelementptr %graphs[0, %pair] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i64>
    %stored = llvm.load %entry : !llvm.ptr -> i64
    %isSame = arith.cmpi eq, %stored, %print : i64
    %isEmpty = arith.cmpi eq, %stored, %none : i64
    %usable = arith.ori %isSame, %isEmpty : i1
    %next = arith.addi %probe, %one : i64
    cf.cond_br %usable, ^found(%entry : !llvm.ptr), ^probe(%next : i64)
  ^victim:
    %offset = arith.andi %print, %seven : i64
    %victimIndex = arith.addi %first, %offset : i64
    %wrapped = arith.andi %victimIndex, %mask : i64
    %victimPair = arith.shli %wrapped, %one : i64
    %victim = llvm.getelementptr %graphs[0, %victimPair] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i64>
    cf.br ^found(%victim : !llvm.ptr)
  ^found(%chosen: !llvm.ptr):
    return %chosen : !llvm.ptr
  }
  func.func private @strict_gpu_forget(%execSlot: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %execBits = llvm.load %execSlot : !llvm.ptr -> i64
    llvm.store %none, %execSlot : i64, !llvm.ptr
    %hasExec = arith.cmpi ne, %execBits, %none : i64
    cf.cond_br %hasExec, ^destroy, ^done
  ^destroy:
    %exec = llvm.inttoptr %execBits : i64 to !llvm.ptr
    %destroyIndex = arith.constant 23 : i64
    %destroy = func.call @strict_gpu_call(%destroyIndex) : (i64) -> !llvm.ptr
    %destroyed = llvm.call %destroy(%exec) : !llvm.ptr, (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_settle(%buffers: !llvm.ptr, %count: i64) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    cf.br ^next(%none, %none : i64, i64)
  ^next(%index: i64, %owned: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^check, ^settled
  ^check:
    %slot = llvm.getelementptr %buffers[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %device = llvm.load %slot : !llvm.ptr -> i64
    %combined = arith.ori %owned, %device : i64
    %following = arith.addi %index, %one : i64
    cf.br ^next(%following, %combined : i64, i64)
  ^settled:
    %hasOwned = arith.cmpi ne, %owned, %none : i64
    cf.cond_br %hasOwned, ^wait, ^done
  ^wait:
    func.call @strict_gpu_synchronize() : () -> ()
    func.call @strict_gpu_free_all(%buffers, %count) : (!llvm.ptr, i64) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_synchronize() {
    %synchronizeIndex = arith.constant 11 : i64
    %synchronize = func.call @strict_gpu_call(%synchronizeIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %synchronize() : !llvm.ptr, () -> i32
    return
  }
  func.func private @strict_gpu_put_async(%device: i64, %host: !llvm.ptr, %bytes: i64) -> i1 {
    %zero32 = arith.constant 0 : i32
    %putIndex = arith.constant 14 : i64
    %put = func.call @strict_gpu_call(%putIndex) : (i64) -> !llvm.ptr
    %stream = func.call @strict_gpu_stream() : () -> !llvm.ptr
    %result = llvm.call %put(%device, %host, %bytes, %stream) : !llvm.ptr, (i64, !llvm.ptr, i64, !llvm.ptr) -> i32
    func.call @strict_gpu_captured(%put) : (!llvm.ptr) -> ()
    %done = arith.cmpi eq, %result, %zero32 : i32
    return %done : i1
  }
  // a launch runs the 32 bit kernel when any list it reads is packed (bits below 64), the double one otherwise
  llvm.mlir.global internal constant @strict_gpu_quantize_name("strict_device_quantize\00") : !llvm.array<23 x i8>
  llvm.mlir.global internal @strict_gpu_quantize_slot(dense<0> : tensor<2xi64>) : !llvm.array<2 x i64>
  // device address, size and next offset of the 8 bit numbers made for dot products, one stream runs everything in order
  // so a quantization only overwrites bytes of dot products that ran already
  llvm.mlir.global internal @strict_gpu_scratch(dense<0> : tensor<3xi64>) : !llvm.array<3 x i64>
  llvm.mlir.global internal constant @strict_gpu_dp4a_switch("STRICT_DP4A\00") : !llvm.array<12 x i8>
  llvm.mlir.global internal @strict_gpu_dp4a_state(0 : i64) : i64
  // STRICT_DP4A=0 keeps the 32 bit numbers of dot products with rows
  func.func private @strict_gpu_dp4a() -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %stateSlot = llvm.mlir.addressof @strict_gpu_dp4a_state : !llvm.ptr
    %state = llvm.load %stateSlot : !llvm.ptr -> i64
    %isKnown = arith.cmpi ne, %state, %none : i64
    cf.cond_br %isKnown, ^known(%state : i64), ^read
  ^read:
    %name = llvm.mlir.addressof @strict_gpu_dp4a_switch : !llvm.ptr
    %setting = llvm.call @getenv(%name) : (!llvm.ptr) -> !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %isUnset = llvm.icmp "eq" %setting, %unset : !llvm.ptr
    cf.cond_br %isUnset, ^decided(%one : i64), ^setting
  ^setting:
    %first = llvm.load %setting : !llvm.ptr -> i8
    %off = arith.constant 48 : i8
    %isOff = arith.cmpi eq, %first, %off : i8
    %chosen = arith.select %isOff, %two, %one : i64
    cf.br ^decided(%chosen : i64)
  ^decided(%decided: i64):
    llvm.store %decided, %stateSlot : i64, !llvm.ptr
    cf.br ^known(%decided : i64)
  ^known(%known: i64):
    %enabled = arith.cmpi eq, %known, %one : i64
    return %enabled : i1
  }
  // 32 bit numbers on the GPU only read by dot products with rows become 8 bit numbers (a scale per 32) in the scratch
  // block, dp4a multiplies 4 of them with 4 packed numbers of a row at once
  func.func private @strict_gpu_quantized(%ptx: !llvm.ptr, %view: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %five = arith.constant 5 : i64
    %eight = arith.constant 8 : i64
    %thirtyOne = arith.constant 31 : i64
    %thirtyTwo = arith.constant 32 : i64
    %enabled = func.call @strict_gpu_dp4a() : () -> i1
    %device = llvm.load %view : !llvm.ptr -> i64
    %lengthSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %length = llvm.load %lengthSlot : !llvm.ptr -> i64
    %bitsSlot = llvm.getelementptr %view[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %bits = llvm.load %bitsSlot : !llvm.ptr -> i64
    %isSingle = arith.cmpi eq, %bits, %thirtyTwo : i64
    %rest = arith.andi %length, %thirtyOne : i64
    %whole = arith.cmpi eq, %rest, %none : i64
    %some = arith.cmpi sge, %length, %thirtyTwo : i64
    %wholeSingle = arith.andi %isSingle, %whole : i1
    %fits = arith.andi %wholeSingle, %some : i1
    %wanted = arith.andi %fits, %enabled : i1
    cf.cond_br %wanted, ^room, ^done
  ^room:
    %blocks = arith.shrui %length, %five : i64
    %scaleBytes = arith.shli %blocks, %two : i64
    %needed = arith.addi %length, %scaleBytes : i64
    %spare = arith.constant 255 : i64
    %roundedUp = arith.addi %needed, %spare : i64
    %alignMask = arith.constant -256 : i64
    %bytes = arith.andi %roundedUp, %alignMask : i64
    %scratch = llvm.mlir.addressof @strict_gpu_scratch : !llvm.ptr
    %capacitySlot = llvm.getelementptr %scratch[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %nextSlot = llvm.getelementptr %scratch[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %base = llvm.load %scratch : !llvm.ptr -> i64
    %hasBase = arith.cmpi ne, %base, %none : i64
    cf.cond_br %hasBase, ^place(%base : i64), ^obtain
  ^obtain:
    %size = arith.constant 16777216 : i64
    %class = func.call @strict_gpu_class(%size) : (i64) -> i64
    %block = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    %obtained = arith.cmpi ne, %block, %none : i64
    llvm.store %block, %scratch : i64, !llvm.ptr
    llvm.store %size, %capacitySlot : i64, !llvm.ptr
    cf.cond_br %obtained, ^place(%block : i64), ^done
  ^place(%start: i64):
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %quarter = arith.shrui %capacity, %two : i64
    %small = arith.cmpi ule, %bytes, %quarter : i64
    cf.cond_br %small, ^offset, ^done
  ^offset:
    %next = llvm.load %nextSlot : !llvm.ptr -> i64
    %end = arith.addi %next, %bytes : i64
    %wraps = arith.cmpi ugt, %end, %capacity : i64
    %at = arith.select %wraps, %none, %next : i64
    %after = arith.addi %at, %bytes : i64
    llvm.store %after, %nextSlot : i64, !llvm.ptr
    %target = arith.addi %start, %at : i64
    %name = llvm.mlir.addressof @strict_gpu_quantize_name : !llvm.ptr
    %slot = llvm.mlir.addressof @strict_gpu_quantize_slot : !llvm.ptr
    %function = func.call @strict_gpu_function(%ptx, %name, %slot) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %function, %unset : !llvm.ptr
    cf.cond_br %missing, ^done, ^launch
  ^launch:
    %three = arith.constant 3 : i64
    %arguments = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    %parameters = llvm.alloca %three x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %device, %arguments : i64, !llvm.ptr
    %lengthArgument = llvm.getelementptr %arguments[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %length, %lengthArgument : i64, !llvm.ptr
    %targetArgument = llvm.getelementptr %arguments[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %target, %targetArgument : i64, !llvm.ptr
    llvm.store %arguments, %parameters : !llvm.ptr, !llvm.ptr
    %lengthParameter = llvm.getelementptr %parameters[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %lengthArgument, %lengthParameter : !llvm.ptr, !llvm.ptr
    %targetParameter = llvm.getelementptr %parameters[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %targetArgument, %targetParameter : !llvm.ptr, !llvm.ptr
    %threads = arith.constant 256 : i64
    %launched = func.call @strict_gpu_launch(%function, %length, %threads, %parameters) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %launched, ^swap, ^done
  ^swap:
    llvm.store %target, %view : i64, !llvm.ptr
    llvm.store %eight, %bitsSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  // the blocks of a grouped launch (one element per lane, up to 256 lanes) compute the call of their group together, a
  // block must not reach into the next group: whole warps up to 256 lanes, beyond that whole blocks of 256
  func.func private @strict_gpu_grouping(%count: i64, %group: i64) -> i1 {
    %none = arith.constant 0 : i64
    %thirtyOne = arith.constant 31 : i64
    %widest = arith.constant 256 : i64
    %ungrouped = arith.cmpi sle, %group, %none : i64
    cf.cond_br %ungrouped, ^fits, ^check
  ^check:
    %rest = arith.remui %count, %group : i64
    %divides = arith.cmpi eq, %rest, %none : i64
    %lanes = arith.minsi %group, %widest : i64
    %blockRest = arith.remui %group, %lanes : i64
    %blocks = arith.cmpi eq, %blockRest, %none : i64
    %warpRest = arith.andi %lanes, %thirtyOne : i64
    %warps = arith.cmpi eq, %warpRest, %none : i64
    %whole = arith.andi %blocks, %warps : i1
    %fit = arith.andi %divides, %whole : i1
    return %fit : i1
  ^fits:
    %true = arith.constant true
    return %true : i1
  }
  // rows of 4, 6 or 8 bits are what dot products with 8 bit numbers are made for
  func.func private @strict_gpu_packed(%view: !llvm.ptr, %packedSlot: !llvm.ptr) {
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %eight = arith.constant 8 : i64
    %bitsSlot = llvm.getelementptr %view[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %bits = llvm.load %bitsSlot : !llvm.ptr -> i64
    %isFour = arith.cmpi eq, %bits, %four : i64
    %isSix = arith.cmpi eq, %bits, %six : i64
    %isEight = arith.cmpi eq, %bits, %eight : i64
    %isBlocked = arith.ori %isFour, %isSix : i1
    %isPacked = arith.ori %isBlocked, %isEight : i1
    %packed = arith.extui %isPacked : i1 to i64
    %before = llvm.load %packedSlot : !llvm.ptr -> i64
    %combined = arith.ori %before, %packed : i64
    llvm.store %combined, %packedSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_narrowed(%view: !llvm.ptr, %narrowSlot: !llvm.ptr) {
    %sixtyFour = arith.constant 64 : i64
    %bitsSlot = llvm.getelementptr %view[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %bits = llvm.load %bitsSlot : !llvm.ptr -> i64
    %isNarrow = arith.cmpi ult, %bits, %sixtyFour : i64
    %narrow = arith.extui %isNarrow : i1 to i64
    %before = llvm.load %narrowSlot : !llvm.ptr -> i64
    %combined = arith.ori %before, %narrow : i64
    llvm.store %combined, %narrowSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_result(%count: i64, %width: i64) -> f64 {
    %eight = arith.constant 8 : i64
    %thirtyTwo = arith.constant 32 : i64
    %isDouble = arith.cmpi eq, %width, %eight : i64
    cf.cond_br %isDouble, ^doubles, ^singles
  ^doubles:
    %countNumber = arith.sitofp %count : i64 to f64
    %list = func.call @strict_list_new(%countNumber, %countNumber) : (f64, f64) -> f64
    return %list : f64
  ^singles:
    %packed = func.call @strict_packed_new(%count, %thirtyTwo) : (i64, i64) -> f64
    return %packed : f64
  }
  func.func private @strict_gpu_items(%list: f64, %width: i64) -> !llvm.ptr {
    %eight = arith.constant 8 : i64
    %isDouble = arith.cmpi eq, %width, %eight : i64
    cf.cond_br %isDouble, ^doubles, ^singles
  ^doubles:
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  ^singles:
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    return %data : !llvm.ptr
  }
  func.func private @strict_gpu_canonical(%items: !llvm.ptr, %count: i64, %width: i64) {
    %four = arith.constant 4 : i64
    %isSingle = arith.cmpi eq, %width, %four : i64
    cf.cond_br %isSingle, ^singles, ^done
  ^singles:
    %false = arith.constant false
    func.call @strict_packed_canonical(%items, %count, %false) : (!llvm.ptr, i64, i1) -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_work(%count: i64, %source: f64, %kind: i64) -> i64 {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %isRows = arith.cmpi eq, %kind, %two : i64
    %hasRows = arith.cmpi sgt, %count, %none : i64
    %measured = arith.andi %isRows, %hasRows : i1
    cf.cond_br %measured, ^rows, ^elements
  ^rows:
    %items = func.call @strict_list_items(%source) : (f64) -> !llvm.ptr
    %first = llvm.load %items : !llvm.ptr -> f64
    %columnsNumber = func.call @strict_list_length(%first) : (f64) -> f64
    %columns = arith.fptosi %columnsNumber : f64 to i64
    %cells = arith.muli %count, %columns : i64
    return %cells : i64
  ^elements:
    return %count : i64
  }
  func.func private @strict_gpu_free_all(%buffers: !llvm.ptr, %count: i64) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    cf.br ^next(%none : i64)
  ^next(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^free, ^done
  ^free:
    %slot = llvm.getelementptr %buffers[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %device = llvm.load %slot : !llvm.ptr -> i64
    func.call @strict_gpu_release(%device) : (i64) -> ()
    %following = arith.addi %index, %one : i64
    cf.br ^next(%following : i64)
  ^done:
    return
  }
  func.func private @strict_gpu_source(%source: f64, %kind: i64, %out: !llvm.ptr, %buffer: !llvm.ptr) -> i1 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %isNumbers = arith.cmpi eq, %kind, %one : i64
    cf.cond_br %isNumbers, ^numbers, ^other
  ^numbers:
    %listed = func.call @strict_gpu_list(%source, %out, %buffer) : (f64, !llvm.ptr, !llvm.ptr) -> i1
    return %listed : i1
  ^other:
    %isRows = arith.cmpi eq, %kind, %two : i64
    cf.cond_br %isRows, ^rows, ^counted
  ^rows:
    %gathered = func.call @strict_gpu_rows(%source, %out, %buffer) : (f64, !llvm.ptr, !llvm.ptr) -> i1
    return %gathered : i1
  ^counted:
    %none = arith.constant 0 : i64
    %sixtyFour = arith.constant 64 : i64
    %start = llvm.bitcast %source : f64 to i64
    func.call @strict_gpu_view(%out, %none, %start, %sixtyFour) : (!llvm.ptr, i64, i64, i64) -> ()
    %true = arith.constant true
    return %true : i1
  }
  func.func private @strict_gpu_view(%out: !llvm.ptr, %device: i64, %length: i64, %bits: i64) {
    llvm.store %device, %out : i64, !llvm.ptr
    %lengthSlot = llvm.getelementptr %out[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %length, %lengthSlot : i64, !llvm.ptr
    %bitsSlot = llvm.getelementptr %out[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %bits, %bitsSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_host(%list: f64) -> !llvm.ptr {
    %sixtyFour = arith.constant 64 : i64
    %bits = func.call @strict_gpu_bits(%list) : (f64) -> i64
    %isPlain = arith.cmpi eq, %bits, %sixtyFour : i64
    cf.cond_br %isPlain, ^plain, ^narrow
  ^plain:
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
  ^narrow:
    %data = func.call @strict_list_data(%list) : (f64) -> !llvm.ptr
    return %data : !llvm.ptr
  }
  func.func private @strict_gpu_bits(%list: f64) -> i64 {
    %none = arith.constant 0 : i64
    %sixtyFour = arith.constant 64 : i64
    %packed = func.call @strict_list_packed(%list) : (f64) -> i64
    %isPlain = arith.cmpi eq, %packed, %none : i64
    %bits = arith.select %isPlain, %sixtyFour, %packed : i64
    return %bits : i64
  }
  func.func private @strict_gpu_list(%list: f64, %out: !llvm.ptr, %buffer: !llvm.ptr) -> i1 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %sixtyFour = arith.constant 64 : i64
    %cacheable = arith.constant 65536 : i64
    %true = arith.constant true
    %false = arith.constant false
    %isUnset = arith.cmpf oeq, %list, %zero : f64
    cf.cond_br %isUnset, ^empty, ^sized
  ^empty:
    func.call @strict_gpu_view(%out, %none, %none, %sixtyFour) : (!llvm.ptr, i64, i64, i64) -> ()
    return %true : i1
  ^sized:
    %lengthNumber = func.call @strict_list_length(%list) : (f64) -> f64
    %length = arith.fptosi %lengthNumber : f64 to i64
    %resident = func.call @strict_gpu_device(%list) : (f64) -> i64
    %isResident = arith.cmpi ne, %resident, %none : i64
    cf.cond_br %isResident, ^resident, ^host
  ^resident:
    %thirtyTwo = arith.constant 32 : i64
    func.call @strict_gpu_view(%out, %resident, %length, %thirtyTwo) : (!llvm.ptr, i64, i64, i64) -> ()
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %true : i1
  ^host:
    %bits = func.call @strict_gpu_bits(%list) : (f64) -> i64
    %bytes = func.call @strict_packed_bytes(%length, %bits) : (i64, i64) -> i64
    %size = arith.maxui %bytes, %eight : i64
    %big = arith.cmpi uge, %bytes, %cacheable : i64
    %hit = func.call @strict_gpu_cached(%list, %one, %length, %out) : (f64, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %hit, ^owned, ^fresh
  ^owned:
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %true : i1
  ^fresh:
    cf.cond_br %big, ^dedicated, ^small
  ^small:
    %seen = func.call @strict_gpu_seen(%list) : (f64) -> i1
    cf.cond_br %seen, ^promote, ^temporary
  ^promote:
    %class = func.call @strict_gpu_class(%size) : (i64) -> i64
    %block = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    %hasBlock = arith.cmpi ne, %block, %none : i64
    cf.cond_br %hasBlock, ^promoted, ^temporary
  ^promoted:
    func.call @strict_gpu_replay() : () -> ()
    %source = func.call @strict_gpu_host(%list) : (f64) -> !llvm.ptr
    %uploaded = func.call @strict_gpu_put_async(%block, %source, %bytes) : (i64, !llvm.ptr, i64) -> i1
    func.call @strict_gpu_view(%out, %block, %length, %bits) : (!llvm.ptr, i64, i64, i64) -> ()
    %pool = arith.addi %class, %one : i64
    func.call @strict_gpu_remember(%list, %one, %length, %out, %pool) : (f64, i64, i64, !llvm.ptr, i64) -> ()
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %uploaded : i1
  ^dedicated:
    %owned = func.call @strict_gpu_alloc(%size) : (i64) -> i64
    llvm.store %owned, %buffer : i64, !llvm.ptr
    cf.br ^allocated(%owned : i64)
  ^temporary:
    %scratch = func.call @strict_gpu_temporary(%size, %buffer) : (i64, !llvm.ptr) -> i64
    cf.br ^allocated(%scratch : i64)
  ^allocated(%device: i64):
    func.call @strict_gpu_view(%out, %device, %length, %bits) : (!llvm.ptr, i64, i64, i64) -> ()
    %hasDevice = arith.cmpi ne, %device, %none : i64
    %hasBytes = arith.cmpi ne, %bytes, %none : i64
    %copying = arith.andi %hasDevice, %hasBytes : i1
    cf.cond_br %copying, ^copy, ^done
  ^copy:
    %host = func.call @strict_gpu_host(%list) : (f64) -> !llvm.ptr
    %put = func.call @strict_gpu_stage(%device, %host, %bytes) : (i64, !llvm.ptr, i64) -> i1
    %keeping = arith.andi %put, %big : i1
    cf.cond_br %keeping, ^remember, ^copied
  ^remember:
    func.call @strict_gpu_remember(%list, %one, %length, %out, %none) : (f64, i64, i64, !llvm.ptr, i64) -> ()
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %true : i1
  ^copied:
    return %put : i1
  ^done:
    return %hasDevice : i1
  }
  // the rows of a matrix go up behind a header: the row stride (i64) and one byte per row with its width (64 for doubles),
  // padded to 16 bytes like the stride, so rows start aligned for vector loads, every row keeps its own format (Qwen3
  // has Q4_K query and key rows next to Q6_K value rows)
  func.func private @strict_gpu_rows(%rows: f64, %out: !llvm.ptr, %buffer: !llvm.ptr) -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %seven = arith.constant 7 : i64
    %eight = arith.constant 8 : i64
    %sixtyFour = arith.constant 64 : i64
    %cacheable = arith.constant 65536 : i64
    %alignMask = arith.constant -8 : i64
    %true = arith.constant true
    %false = arith.constant false
    %countNumber = func.call @strict_list_length(%rows) : (f64) -> f64
    %count = arith.fptosi %countNumber : f64 to i64
    %isEmpty = arith.cmpi eq, %count, %none : i64
    cf.cond_br %isEmpty, ^empty, ^first
  ^empty:
    func.call @strict_gpu_view(%out, %none, %none, %sixtyFour) : (!llvm.ptr, i64, i64, i64) -> ()
    return %true : i1
  ^first:
    %hit = func.call @strict_gpu_cached(%rows, %two, %count, %out) : (f64, i64, i64, !llvm.ptr) -> i1
    cf.cond_br %hit, ^owned, ^measure
  ^owned:
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %true : i1
  ^measure:
    %items = func.call @strict_list_items(%rows) : (f64) -> !llvm.ptr
    %firstRow = llvm.load %items : !llvm.ptr -> f64
    %columnsNumber = func.call @strict_list_length(%firstRow) : (f64) -> f64
    %columns = arith.fptosi %columnsNumber : f64 to i64
    cf.br ^widest(%none, %none, %sixtyFour : i64, i64, i64)
  ^widest(%index: i64, %stride: i64, %narrowest: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^width, ^sized
  ^width:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %row = llvm.load %slot : !llvm.ptr -> f64
    %lengthNumber = func.call @strict_list_length(%row) : (f64) -> f64
    %length = arith.fptosi %lengthNumber : f64 to i64
    %sameLength = arith.cmpi eq, %length, %columns : i64
    %bits = func.call @strict_gpu_bits(%row) : (f64) -> i64
    %bytes = func.call @strict_packed_bytes(%columns, %bits) : (i64, i64) -> i64
    %wider = arith.maxui %stride, %bytes : i64
    %narrower = arith.minui %narrowest, %bits : i64
    %following = arith.addi %index, %one : i64
    cf.cond_br %sameLength, ^widest(%following, %wider, %narrower : i64, i64, i64), ^mismatch
  ^mismatch:
    return %false : i1
  ^sized:
    %fifteen = arith.constant 15 : i64
    %sixteenMask = arith.constant -16 : i64
    %withTable = arith.addi %count, %eight : i64
    %paddedHeader = arith.addi %withTable, %fifteen : i64
    %header = arith.andi %paddedHeader, %sixteenMask : i64
    %paddedStride = arith.addi %stride, %fifteen : i64
    %rowStride = arith.andi %paddedStride, %sixteenMask : i64
    %rowsBytes = arith.muli %rowStride, %count : i64
    %total = arith.addi %header, %rowsBytes : i64
    %big = arith.cmpi uge, %total, %cacheable : i64
    %staging = llvm.call @malloc(%total) : (i64) -> !llvm.ptr
    %countShift = arith.constant 32 : i64
    %countHigh = arith.shli %count, %countShift : i64
    %strideAndCount = arith.ori %rowStride, %countHigh : i64
    llvm.store %strideAndCount, %staging : i64, !llvm.ptr
    cf.br ^row(%none : i64)
  ^row(%rowIndex: i64):
    %moreRows = arith.cmpi slt, %rowIndex, %count : i64
    cf.cond_br %moreRows, ^gather, ^upload
  ^gather:
    %rowSlot = llvm.getelementptr %items[%rowIndex] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %gathered = llvm.load %rowSlot : !llvm.ptr -> f64
    %rowBits = func.call @strict_gpu_bits(%gathered) : (f64) -> i64
    %widthAt = arith.addi %rowIndex, %eight : i64
    %widthSlot = llvm.getelementptr %staging[%widthAt] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %width = arith.trunci %rowBits : i64 to i8
    llvm.store %width, %widthSlot : i8, !llvm.ptr
    %rowBytes = func.call @strict_packed_bytes(%columns, %rowBits) : (i64, i64) -> i64
    %rowOffset = arith.muli %rowIndex, %rowStride : i64
    %offset = arith.addi %header, %rowOffset : i64
    %destination = llvm.getelementptr %staging[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rowData = func.call @strict_gpu_host(%gathered) : (f64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%destination, %rowData, %rowBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %nextRow = arith.addi %rowIndex, %one : i64
    cf.br ^row(%nextRow : i64)
  ^upload:
    cf.cond_br %big, ^dedicated, ^temporary
  ^dedicated:
    %dedicated = func.call @strict_gpu_alloc(%total) : (i64) -> i64
    llvm.store %dedicated, %buffer : i64, !llvm.ptr
    cf.br ^allocated(%dedicated : i64)
  ^temporary:
    %scratch = func.call @strict_gpu_temporary(%total, %buffer) : (i64, !llvm.ptr) -> i64
    cf.br ^allocated(%scratch : i64)
  ^allocated(%device: i64):
    func.call @strict_gpu_view(%out, %device, %columns, %narrowest) : (!llvm.ptr, i64, i64, i64) -> ()
    %hasDevice = arith.cmpi ne, %device, %none : i64
    cf.cond_br %hasDevice, ^copy, ^unavailable
  ^copy:
    %put = func.call @strict_gpu_stage(%device, %staging, %total) : (i64, !llvm.ptr, i64) -> i1
    llvm.call @free(%staging) : (!llvm.ptr) -> ()
    %keeping = arith.andi %put, %big : i1
    cf.cond_br %keeping, ^remember, ^copiedRows
  ^remember:
    func.call @strict_gpu_remember(%rows, %two, %count, %out, %none) : (f64, i64, i64, !llvm.ptr, i64) -> ()
    llvm.store %none, %buffer : i64, !llvm.ptr
    return %true : i1
  ^copiedRows:
    return %put : i1
  ^unavailable:
    llvm.call @free(%staging) : (!llvm.ptr) -> ()
    return %false : i1
  }
  llvm.mlir.global internal @strict_gpu_cache(dense<0> : tensor<524288xi64>) : !llvm.array<524288 x i64>
  llvm.mlir.global internal @strict_gpu_clock(0 : i64) : i64
  func.func private @strict_gpu_entry(%index: i64) -> !llvm.ptr {
    %eight = arith.constant 8 : i64
    %cache = llvm.mlir.addressof @strict_gpu_cache : !llvm.ptr
    %offset = arith.muli %index, %eight : i64
    %entry = llvm.getelementptr %cache[0, %offset] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<524288 x i64>
    return %entry : !llvm.ptr
  }
  func.func private @strict_gpu_tick() -> i64 {
    %one = arith.constant 1 : i64
    %clockSlot = llvm.mlir.addressof @strict_gpu_clock : !llvm.ptr
    %clock = llvm.load %clockSlot : !llvm.ptr -> i64
    %next = arith.addi %clock, %one : i64
    llvm.store %next, %clockSlot : i64, !llvm.ptr
    return %next : i64
  }
  // device copies of host lists are found by a hash of the list block, 16 of the 65536 entries are probed
  func.func private @strict_gpu_cached(%list: f64, %kind: i64, %count: i64, %out: !llvm.ptr) -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %probes = arith.constant 16 : i64
    %true = arith.constant true
    %false = arith.constant false
    %listBits = arith.bitcast %list : f64 to i64
    cf.br ^next(%none : i64)
  ^next(%probe: i64):
    %more = arith.cmpi slt, %probe, %probes : i64
    cf.cond_br %more, ^compare, ^missing
  ^compare:
    %entry = func.call @strict_gpu_slot(%listBits, %probe) : (i64, i64) -> !llvm.ptr
    %stored = llvm.load %entry : !llvm.ptr -> i64
    %kindSlot = llvm.getelementptr %entry[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %storedKind = llvm.load %kindSlot : !llvm.ptr -> i64
    %countSlot = llvm.getelementptr %entry[6] : (!llvm.ptr) -> !llvm.ptr, i64
    %storedCount = llvm.load %countSlot : !llvm.ptr -> i64
    %sameList = arith.cmpi eq, %stored, %listBits : i64
    %sameKind = arith.cmpi eq, %storedKind, %kind : i64
    %sameCount = arith.cmpi eq, %storedCount, %count : i64
    %sameShape = arith.andi %sameKind, %sameCount : i1
    %found = arith.andi %sameList, %sameShape : i1
    %following = arith.addi %probe, %one : i64
    cf.cond_br %found, ^hit, ^next(%following : i64)
  ^hit:
    %deviceSlot = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %device = llvm.load %deviceSlot : !llvm.ptr -> i64
    %lengthSlot = llvm.getelementptr %entry[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %length = llvm.load %lengthSlot : !llvm.ptr -> i64
    %bitsSlot = llvm.getelementptr %entry[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %bits = llvm.load %bitsSlot : !llvm.ptr -> i64
    func.call @strict_gpu_view(%out, %device, %length, %bits) : (!llvm.ptr, i64, i64, i64) -> ()
    %used = func.call @strict_gpu_tick() : () -> i64
    %usedSlot = llvm.getelementptr %entry[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %used, %usedSlot : i64, !llvm.ptr
    return %true : i1
  ^missing:
    return %false : i1
  }
  func.func private @strict_gpu_slot(%listBits: i64, %probe: i64) -> !llvm.ptr {
    %four = arith.constant 4 : i64
    %golden = arith.constant -7046029254386353131 : i64
    %topShift = arith.constant 48 : i64
    %mask = arith.constant 65535 : i64
    %shifted = arith.shrui %listBits, %four : i64
    %mixed = arith.muli %shifted, %golden : i64
    %top = arith.shrui %mixed, %topShift : i64
    %probed = arith.addi %top, %probe : i64
    %index = arith.andi %probed, %mask : i64
    %entry = func.call @strict_gpu_entry(%index) : (i64) -> !llvm.ptr
    return %entry : !llvm.ptr
  }
  // a small host list uploaded the second time (weights of a layer, a rotation used by every block) gets a device copy
  llvm.mlir.global internal @strict_gpu_seen_lists(dense<0> : tensor<4096xi64>) : !llvm.array<4096 x i64>
  // sets of 4 by a multiplicative hash, lists 64 KB apart would collide forever in the low bits
  func.func private @strict_gpu_seen(%list: f64) -> i1 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %golden = arith.constant -7046029254386353131 : i64
    %topShift = arith.constant 52 : i64
    %setMask = arith.constant 4092 : i64
    %true = arith.constant true
    %listBits = arith.bitcast %list : f64 to i64
    %shifted = arith.shrui %listBits, %four : i64
    %mixed = arith.muli %shifted, %golden : i64
    %top = arith.shrui %mixed, %topShift : i64
    %set = arith.andi %top, %setMask : i64
    %seen = llvm.mlir.addressof @strict_gpu_seen_lists : !llvm.ptr
    cf.br ^next(%none : i64)
  ^next(%way: i64):
    %more = arith.cmpi slt, %way, %four : i64
    cf.cond_br %more, ^compare, ^insert
  ^compare:
    %index = arith.addi %set, %way : i64
    %slot = llvm.getelementptr %seen[0, %index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i64>
    %stored = llvm.load %slot : !llvm.ptr -> i64
    %same = arith.cmpi eq, %stored, %listBits : i64
    %following = arith.addi %way, %one : i64
    cf.cond_br %same, ^found, ^next(%following : i64)
  ^found:
    return %true : i1
  ^insert:
    %tick = func.call @strict_gpu_tick() : () -> i64
    %chosen = arith.andi %tick, %three : i64
    %target = arith.addi %set, %chosen : i64
    %targetSlot = llvm.getelementptr %seen[0, %target] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i64>
    llvm.store %listBits, %targetSlot : i64, !llvm.ptr
    %false = arith.constant false
    return %false : i1
  }
  // pool is 0 for a device copy of its own allocation, else the class of its pool block plus 1
  func.func private @strict_gpu_evict(%entry: !llvm.ptr) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %deviceSlot = llvm.getelementptr %entry[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %device = llvm.load %deviceSlot : !llvm.ptr -> i64
    %poolSlot = llvm.getelementptr %entry[7] : (!llvm.ptr) -> !llvm.ptr, i64
    %pool = llvm.load %poolSlot : !llvm.ptr -> i64
    llvm.store %none, %entry : i64, !llvm.ptr
    %usedSlot = llvm.getelementptr %entry[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %usedSlot : i64, !llvm.ptr
    %isPooled = arith.cmpi ne, %pool, %none : i64
    cf.cond_br %isPooled, ^give, ^release
  ^give:
    %class = arith.subi %pool, %one : i64
    func.call @strict_gpu_give(%device, %class) : (i64, i64) -> ()
    return
  ^release:
    func.call @strict_gpu_release(%device) : (i64) -> ()
    return
  }
  func.func private @strict_gpu_remember(%list: f64, %kind: i64, %count: i64, %out: !llvm.ptr, %pool: i64) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %probes = arith.constant 16 : i64
    %latest = arith.constant 9223372036854775807 : i64
    %listBits = arith.bitcast %list : f64 to i64
    %first = func.call @strict_gpu_slot(%listBits, %none) : (i64, i64) -> !llvm.ptr
    cf.br ^scan(%none, %first, %latest : i64, !llvm.ptr, i64)
  ^scan(%probe: i64, %oldest: !llvm.ptr, %oldestUse: i64):
    %more = arith.cmpi slt, %probe, %probes : i64
    cf.cond_br %more, ^compare, ^replace(%oldest : !llvm.ptr)
  ^compare:
    %entry = func.call @strict_gpu_slot(%listBits, %probe) : (i64, i64) -> !llvm.ptr
    %stored = llvm.load %entry : !llvm.ptr -> i64
    %isFree = arith.cmpi eq, %stored, %none : i64
    %usedSlot = llvm.getelementptr %entry[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %used = llvm.load %usedSlot : !llvm.ptr -> i64
    %entryPoolSlot = llvm.getelementptr %entry[7] : (!llvm.ptr) -> !llvm.ptr, i64
    %entryPool = llvm.load %entryPoolSlot : !llvm.ptr -> i64
    %isDedicated = arith.cmpi eq, %entryPool, %none : i64
    %dedicatedAge = arith.constant 4611686018427387904 : i64
    %penalty = arith.select %isDedicated, %dedicatedAge, %none : i64
    %weighted = arith.addi %used, %penalty : i64
    %older = arith.cmpi slt, %weighted, %oldestUse : i64
    %chosen = llvm.select %older, %entry, %oldest : i1, !llvm.ptr
    %chosenUse = arith.select %older, %weighted, %oldestUse : i64
    %following = arith.addi %probe, %one : i64
    cf.cond_br %isFree, ^replace(%entry : !llvm.ptr), ^scan(%following, %chosen, %chosenUse : i64, !llvm.ptr, i64)
  ^replace(%slot: !llvm.ptr):
    %previous = llvm.load %slot : !llvm.ptr -> i64
    %isUsed = arith.cmpi ne, %previous, %none : i64
    cf.cond_br %isUsed, ^evict, ^store
  ^evict:
    func.call @strict_gpu_evict(%slot) : (!llvm.ptr) -> ()
    cf.br ^store
  ^store:
    llvm.store %listBits, %slot : i64, !llvm.ptr
    %device = llvm.load %out : !llvm.ptr -> i64
    %deviceSlot = llvm.getelementptr %slot[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %device, %deviceSlot : i64, !llvm.ptr
    %lengthSource = llvm.getelementptr %out[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %length = llvm.load %lengthSource : !llvm.ptr -> i64
    %lengthSlot = llvm.getelementptr %slot[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %length, %lengthSlot : i64, !llvm.ptr
    %bitsSource = llvm.getelementptr %out[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %bits = llvm.load %bitsSource : !llvm.ptr -> i64
    %bitsSlot = llvm.getelementptr %slot[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %bits, %bitsSlot : i64, !llvm.ptr
    %stamp = func.call @strict_gpu_tick() : () -> i64
    %stampSlot = llvm.getelementptr %slot[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %stamp, %stampSlot : i64, !llvm.ptr
    %kindSlot = llvm.getelementptr %slot[5] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %kind, %kindSlot : i64, !llvm.ptr
    %countSlot = llvm.getelementptr %slot[6] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %count, %countSlot : i64, !llvm.ptr
    %poolSlot = llvm.getelementptr %slot[7] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %pool, %poolSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_sweep(%full: i1) {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %entries = arith.constant 65536 : i64
    cf.br ^next(%none : i64)
  ^next(%index: i64):
    %more = arith.cmpi slt, %index, %entries : i64
    cf.cond_br %more, ^check, ^done
  ^check:
    %entry = func.call @strict_gpu_entry(%index) : (i64) -> !llvm.ptr
    %listBits = llvm.load %entry : !llvm.ptr -> i64
    %following = arith.addi %index, %one : i64
    %isUsed = arith.cmpi ne, %listBits, %none : i64
    cf.cond_br %isUsed, ^alive, ^next(%following : i64)
  ^alive:
    %marked = func.call @strict_gc_is_marked(%listBits) : (i64) -> i1
    cf.cond_br %marked, ^next(%following : i64), ^release
  ^release:
    func.call @strict_gpu_evict(%entry) : (!llvm.ptr) -> ()
    cf.br ^next(%following : i64)
  ^done:
    func.call @strict_gpu_sweep_residents(%full) : (i1) -> ()
    return
  }
  llvm.mlir.global internal @strict_gpu_arena(dense<0> : tensor<6xi64>) : !llvm.array<6 x i64>
  // the arena is a ring: small inputs of a launch are staged in pinned memory and copied asynchronously, an input that
  // does not fit behind the previous ones waits for the GPU and starts again at the front, the arena grows at the start
  // of a launch when the previous one needed more than half of it
  func.func private @strict_gpu_reset() {
    %none = arith.constant 0 : i64
    %arena = llvm.mlir.addressof @strict_gpu_arena : !llvm.ptr
    %capacitySlot = llvm.getelementptr %arena[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %wantedSlot = llvm.getelementptr %arena[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %wanted = llvm.load %wantedSlot : !llvm.ptr -> i64
    %offsetSlot = llvm.getelementptr %arena[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %none, %wantedSlot : i64, !llvm.ptr
    %lowSlot = llvm.getelementptr %arena[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %nothingStaged = arith.constant -1 : i64
    llvm.store %nothingStaged, %lowSlot : i64, !llvm.ptr
    %one = arith.constant 1 : i64
    %doubled = arith.shli %wanted, %one : i64
    %tooSmall = arith.cmpi ugt, %doubled, %capacity : i64
    cf.cond_br %tooSmall, ^grow, ^done
  ^grow:
    func.call @strict_gpu_synchronize() : () -> ()
    %base = llvm.load %arena : !llvm.ptr -> i64
    func.call @strict_gpu_release(%base) : (i64) -> ()
    %stagingSlot = llvm.getelementptr %arena[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %oldStaging = llvm.load %stagingSlot : !llvm.ptr -> i64
    func.call @strict_gpu_unpinned(%oldStaging) : (i64) -> ()
    %two = arith.constant 2 : i64
    %quadrupled = arith.shli %wanted, %two : i64
    %smallest = arith.constant 1048576 : i64
    %grown = arith.maxui %quadrupled, %smallest : i64
    %fresh = func.call @strict_gpu_alloc(%grown) : (i64) -> i64
    llvm.store %fresh, %arena : i64, !llvm.ptr
    %staging = func.call @strict_gpu_pinned(%grown) : (i64) -> i64
    llvm.store %staging, %stagingSlot : i64, !llvm.ptr
    %isAllocated = arith.cmpi ne, %fresh, %none : i64
    %isStaged = arith.cmpi ne, %staging, %none : i64
    %isUsable = arith.andi %isAllocated, %isStaged : i1
    %kept = arith.select %isUsable, %grown, %none : i64
    llvm.store %kept, %capacitySlot : i64, !llvm.ptr
    llvm.store %none, %offsetSlot : i64, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_pinned(%bytes: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %zero, %slot : i64, !llvm.ptr
    %allocIndex = arith.constant 12 : i64
    %alloc = func.call @strict_gpu_call(%allocIndex) : (i64) -> !llvm.ptr
    %result = llvm.call %alloc(%slot, %bytes) : !llvm.ptr, (!llvm.ptr, i64) -> i32
    %failed = arith.cmpi ne, %result, %zero32 : i32
    %host = llvm.load %slot : !llvm.ptr -> i64
    %chosen = arith.select %failed, %zero, %host : i64
    return %chosen : i64
  }
  func.func private @strict_gpu_unpinned(%host: i64) {
    %zero = arith.constant 0 : i64
    %isSet = arith.cmpi ne, %host, %zero : i64
    cf.cond_br %isSet, ^free, ^done
  ^free:
    %freeIndex = arith.constant 13 : i64
    %free = func.call @strict_gpu_call(%freeIndex) : (i64) -> !llvm.ptr
    %pointer = llvm.inttoptr %host : i64 to !llvm.ptr
    %result = llvm.call %free(%pointer) : !llvm.ptr, (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_gpu_stage(%device: i64, %host: !llvm.ptr, %bytes: i64) -> i1 {
    %arena = llvm.mlir.addressof @strict_gpu_arena : !llvm.ptr
    %base = llvm.load %arena : !llvm.ptr -> i64
    %capacitySlot = llvm.getelementptr %arena[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %offset = arith.subi %device, %base : i64
    %isScratch = arith.cmpi ult, %offset, %capacity : i64
    cf.cond_br %isScratch, ^stage, ^put
  ^stage:
    %stagingSlot = llvm.getelementptr %arena[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %stagingBits = llvm.load %stagingSlot : !llvm.ptr -> i64
    %staging = llvm.inttoptr %stagingBits : i64 to !llvm.ptr
    %target = llvm.getelementptr %staging[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %copied = llvm.call @memcpy(%target, %host, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %lowSlot = llvm.getelementptr %arena[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %low = llvm.load %lowSlot : !llvm.ptr -> i64
    %lowest = arith.minui %low, %offset : i64
    llvm.store %lowest, %lowSlot : i64, !llvm.ptr
    %true = arith.constant true
    return %true : i1
  ^put:
    %put = func.call @strict_gpu_put(%device, %host, %bytes) : (i64, !llvm.ptr, i64) -> i1
    return %put : i1
  }
  func.func private @strict_gpu_flush() -> i1 {
    %arena = llvm.mlir.addressof @strict_gpu_arena : !llvm.ptr
    %lowSlot = llvm.getelementptr %arena[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %low = llvm.load %lowSlot : !llvm.ptr -> i64
    %offsetSlot = llvm.getelementptr %arena[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %offset = llvm.load %offsetSlot : !llvm.ptr -> i64
    %isPending = arith.cmpi ult, %low, %offset : i64
    cf.cond_br %isPending, ^upload, ^done
  ^upload:
    %base = llvm.load %arena : !llvm.ptr -> i64
    %stagingSlot = llvm.getelementptr %arena[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %stagingBits = llvm.load %stagingSlot : !llvm.ptr -> i64
    %staging = llvm.inttoptr %stagingBits : i64 to !llvm.ptr
    %device = arith.addi %base, %low : i64
    %host = llvm.getelementptr %staging[%low] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %bytes = arith.subi %offset, %low : i64
    %nothingStaged = arith.constant -1 : i64
    llvm.store %nothingStaged, %lowSlot : i64, !llvm.ptr
    %put = func.call @strict_gpu_put_async(%device, %host, %bytes) : (i64, !llvm.ptr, i64) -> i1
    return %put : i1
  ^done:
    %true = arith.constant true
    return %true : i1
  }
  func.func private @strict_gpu_fetch(%result: i64, %flagDevice: i64, %count: i64, %width: i64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %arena = llvm.mlir.addressof @strict_gpu_arena : !llvm.ptr
    %base = llvm.load %arena : !llvm.ptr -> i64
    %stagingSlot = llvm.getelementptr %arena[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %stagingBits = llvm.load %stagingSlot : !llvm.ptr -> i64
    %staging = llvm.inttoptr %stagingBits : i64 to !llvm.ptr
    %resultOffset = arith.subi %result, %base : i64
    %flagEnd = arith.addi %flagDevice, %eight : i64
    %span = arith.subi %flagEnd, %result : i64
    %target = llvm.getelementptr %staging[%resultOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %got = func.call @strict_gpu_get(%target, %result, %span) : (!llvm.ptr, i64, i64) -> i1
    %flagOffset = arith.subi %flagDevice, %base : i64
    %flagHost = llvm.getelementptr %staging[%flagOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %failed = llvm.load %flagHost : !llvm.ptr -> i64
    %passed = arith.cmpi eq, %failed, %none : i64
    %usable = arith.andi %got, %passed : i1
    cf.cond_br %usable, ^list, ^failed
  ^list:
    %list = func.call @strict_gpu_result(%count, %width) : (i64, i64) -> f64
    %items = func.call @strict_gpu_items(%list, %width) : (f64, i64) -> !llvm.ptr
    %bytes = arith.muli %count, %width : i64
    %copied = llvm.call @memcpy(%items, %target, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    func.call @strict_gpu_canonical(%items, %count, %width) : (!llvm.ptr, i64, i64) -> ()
    return %list : f64
  ^failed:
    return %zero : f64
  }
  func.func private @strict_gpu_temporary(%bytes: i64, %buffer: !llvm.ptr) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %spare = arith.constant 255 : i64
    %alignment = arith.constant -256 : i64
    %arena = llvm.mlir.addressof @strict_gpu_arena : !llvm.ptr
    %padded = arith.addi %bytes, %spare : i64
    %aligned = arith.andi %padded, %alignment : i64
    %wantedSlot = llvm.getelementptr %arena[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %wanted = llvm.load %wantedSlot : !llvm.ptr -> i64
    %moreWanted = arith.addi %wanted, %aligned : i64
    llvm.store %moreWanted, %wantedSlot : i64, !llvm.ptr
    %capacitySlot = llvm.getelementptr %arena[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %capacity = llvm.load %capacitySlot : !llvm.ptr -> i64
    %offsetSlot = llvm.getelementptr %arena[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %offset = llvm.load %offsetSlot : !llvm.ptr -> i64
    %end = arith.addi %offset, %aligned : i64
    %fits = arith.cmpi ule, %end, %capacity : i64
    cf.cond_br %fits, ^scratch(%offset : i64), ^full
  ^full:
    %doubled = arith.shli %aligned, %one : i64
    %wraps = arith.cmpi ule, %doubled, %capacity : i64
    cf.cond_br %wraps, ^wrap, ^owned
  ^wrap:
    %flushed = func.call @strict_gpu_flush() : () -> i1
    func.call @strict_gpu_synchronize() : () -> ()
    cf.br ^scratch(%none : i64)
  ^scratch(%start: i64):
    %finish = arith.addi %start, %aligned : i64
    llvm.store %finish, %offsetSlot : i64, !llvm.ptr
    llvm.store %none, %buffer : i64, !llvm.ptr
    %base = llvm.load %arena : !llvm.ptr -> i64
    %address = arith.addi %base, %start : i64
    return %address : i64
  ^owned:
    %device = func.call @strict_gpu_alloc(%bytes) : (i64) -> i64
    llvm.store %device, %buffer : i64, !llvm.ptr
    return %device : i64
  }
  func.func private @strict_gpu_release(%device: i64) {
    %none = arith.constant 0 : i64
    %isSet = arith.cmpi ne, %device, %none : i64
    cf.cond_br %isSet, ^free, ^done
  ^free:
    func.call @strict_gpu_free(%device) : (i64) -> ()
    cf.br ^done
  ^done:
    return
  }
