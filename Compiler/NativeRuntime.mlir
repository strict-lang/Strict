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
  llvm.func @strftime(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i64
  llvm.func @remove(!llvm.ptr) -> i32
  llvm.func @fgetc(!llvm.ptr) -> i32
  llvm.func @fputc(i32, !llvm.ptr) -> i32
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
  llvm.mlir.global internal @strict_gc_threshold(4194304 : i64) : i64
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
  llvm.mlir.global internal constant @strict_gc_sizes(dense<[16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 12288, 16384]> : tensor<32xi64>) : !llvm.array<32 x i64>
  llvm.mlir.global internal constant @strict_gc_classes(dense<[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19]> : tensor<65xi8>) : !llvm.array<65 x i8>
  llvm.mlir.global internal constant @strict_gc_out_of_memory("Strict: out of memory\0A\00") : !llvm.array<23 x i8>
  llvm.mlir.global internal constant @strict_dimensions("listsHaveDifferentDimensions\00") : !llvm.array<29 x i8>
  llvm.mlir.global internal constant @strict_downcast_prefix("Cannot downcast Text to Number for list: Text \22\00") : !llvm.array<48 x i8>
  llvm.mlir.global internal constant @strict_quote("\22\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_key_not_found("KeyNotFound\00") : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @strict_duplicate_key("DuplicateKey\00") : !llvm.array<13 x i8>
  func.func private @strict_gc_init(%stack: !llvm.ptr) {
    %tableBytes = arith.constant 8388608 : i64
    %heapBytes = arith.constant 68719476736 : i64
    %pageSize = arith.constant 65536 : i64
    %pageRest = arith.constant 65535 : i64
    %pageMask = arith.constant -65536 : i64
    %slack = arith.addi %tableBytes, %pageSize : i64
    %total = arith.addi %heapBytes, %slack : i64
    %reserved = func.call @strict_reserve(%total) : (i64) -> !llvm.ptr
    %start = llvm.ptrtoint %reserved : !llvm.ptr to i64
    %padded = arith.addi %start, %pageRest : i64
    %table = arith.andi %padded, %pageMask : i64
    %tablePointer = llvm.inttoptr %table : i64 to !llvm.ptr
    func.call @strict_commit(%tablePointer, %tableBytes) : (!llvm.ptr, i64) -> ()
    %base = arith.addi %table, %tableBytes : i64
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
    %stackBits = llvm.ptrtoint %stack : !llvm.ptr to i64
    %stackSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    llvm.store %stackBits, %stackSlot : i64, !llvm.ptr
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
  func.func private @strict_gc_allocate(%size: i64, %list: i64) -> !llvm.ptr {
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
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    %allocated = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %counted = arith.addi %allocated, %bytes : i64
    llvm.store %counted, %allocatedSlot : i64, !llvm.ptr
    %thresholdSlot = llvm.mlir.addressof @strict_gc_threshold : !llvm.ptr
    %threshold = llvm.load %thresholdSlot : !llvm.ptr -> i64
    %due = arith.cmpi ugt, %counted, %threshold : i64
    cf.cond_br %due, ^collect, ^done
  ^collect:
    func.call @strict_gc_collect() : () -> ()
    cf.br ^done
  ^done:
    return
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
    %allocatedSlot = llvm.mlir.addressof @strict_gc_allocated : !llvm.ptr
    %allocated = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %thresholdSlot = llvm.mlir.addressof @strict_gc_threshold : !llvm.ptr
    %threshold = llvm.load %thresholdSlot : !llvm.ptr -> i64
    %due = arith.cmpi ugt, %allocated, %threshold : i64
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
    %counted = llvm.load %allocatedSlot : !llvm.ptr -> i64
    %more = arith.addi %counted, %bytes : i64
    llvm.store %more, %allocatedSlot : i64, !llvm.ptr
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
    llvm.store %unmarked, %markSlot : i8, !llvm.ptr
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
  func.func private @strict_gc_mark(%tagged: i64) {
    %address = arith.constant 281474976710655 : i64
    %word = arith.andi %tagged, %address : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %shift = arith.constant 16 : i64
    %eight = arith.constant 8 : i64
    %kindMask = arith.constant 255 : i64
    %unmarked = arith.constant 0 : i8
    %marked = arith.constant 1 : i8
    %baseSlot = llvm.mlir.addressof @strict_gc_base : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %topSlot = llvm.mlir.addressof @strict_gc_top : !llvm.ptr
    %top = llvm.load %topSlot : !llvm.ptr -> i64
    %aboveBase = arith.cmpi uge, %word, %base : i64
    %belowTop = arith.cmpi ult, %word, %top : i64
    %inHeap = arith.andi %aboveBase, %belowTop : i1
    cf.cond_br %inHeap, ^page, ^done
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
    cf.cond_br %pastHeader, ^object, ^done
  ^object:
    %relative = arith.subi %within, %first : i64
    %reciprocalSlot = llvm.getelementptr %pagePointer[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %reciprocal = llvm.load %reciprocalSlot : !llvm.ptr -> i64
    %product = arith.muli %relative, %reciprocal : i64
    %thirtyTwo = arith.constant 32 : i64
    %objectIndex = arith.shrui %product, %thirtyTwo : i64
    %valid = arith.cmpi ult, %objectIndex, %count : i64
    cf.cond_br %valid, ^check, ^done
  ^check:
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %markSlot = llvm.getelementptr %marks[%objectIndex] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %mark = llvm.load %markSlot : !llvm.ptr -> i8
    %isUnmarked = arith.cmpi eq, %mark, %unmarked : i8
    cf.cond_br %isUnmarked, ^set, ^done
  ^set:
    llvm.store %marked, %markSlot : i8, !llvm.ptr
    %objectOffset = arith.muli %objectIndex, %objectSize : i64
    %objectStart = arith.addi %first, %objectOffset : i64
    %objectAddress = arith.addi %page, %objectStart : i64
    func.call @strict_gc_add_marked(%objectSize) : (i64) -> ()
    %pageListSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %pageList = llvm.load %pageListSlot : !llvm.ptr -> i64
    %textLists = arith.constant 32 : i64
    %isScanned = arith.cmpi ult, %pageList, %textLists : i64
    cf.cond_br %isScanned, ^scan(%objectAddress, %objectSize : i64, i64), ^done
  ^scan(%scannedAddress: i64, %scannedSize: i64):
    func.call @strict_gc_push(%scannedAddress, %scannedSize) : (i64, i64) -> ()
    cf.br ^done
  ^other:
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %isFirst = arith.cmpi eq, %kind, %two : i64
    %isContinued = arith.cmpi eq, %kind, %three : i64
    %isLarge = arith.ori %isFirst, %isContinued : i1
    %continuedFrom = arith.shrui %entry, %eight : i64
    %firstIndex = arith.select %isContinued, %continuedFrom, %index : i64
    cf.cond_br %isLarge, ^large(%firstIndex : i64), ^done
  ^large(%startIndex: i64):
    %largeStart = arith.shli %startIndex, %shift : i64
    %header = arith.addi %base, %largeStart : i64
    %headerPointer = llvm.inttoptr %header : i64 to !llvm.ptr
    %payloadSize = llvm.load %headerPointer : !llvm.ptr -> i64
    %largeMarkSlot = llvm.getelementptr %headerPointer[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %largeMark = llvm.load %largeMarkSlot : !llvm.ptr -> i64
    %isLargeUnmarked = arith.cmpi eq, %largeMark, %zero : i64
    cf.cond_br %isLargeUnmarked, ^setLarge, ^done
  ^setLarge:
    llvm.store %one, %largeMarkSlot : i64, !llvm.ptr
    %payloadOffset = arith.constant 64 : i64
    %payload = arith.addi %header, %payloadOffset : i64
    func.call @strict_gc_add_marked(%payloadSize) : (i64) -> ()
    %largeListSlot = llvm.getelementptr %headerPointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %largeList = llvm.load %largeListSlot : !llvm.ptr -> i64
    %largeTexts = arith.constant 32 : i64
    %isLargeScanned = arith.cmpi ult, %largeList, %largeTexts : i64
    cf.cond_br %isLargeScanned, ^scan(%payload, %payloadSize : i64, i64), ^done
  ^done:
    return
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
    %clear = arith.constant 0 : i32
    %sweeps = llvm.mlir.addressof @strict_gc_sweeps : !llvm.ptr
    %sweepSlot = llvm.getelementptr %sweeps[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    cf.br ^next
  ^next:
    %page = llvm.load %sweepSlot : !llvm.ptr -> i64
    %done = arith.cmpi eq, %page, %zero : i64
    cf.cond_br %done, ^finished, ^clear
  ^clear:
    %pagePointer = llvm.inttoptr %page : i64 to !llvm.ptr
    %nextSlot = llvm.getelementptr %pagePointer[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %following = llvm.load %nextSlot : !llvm.ptr -> i64
    llvm.store %following, %sweepSlot : i64, !llvm.ptr
    %countSlot = llvm.getelementptr %pagePointer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %marks = llvm.getelementptr %pagePointer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %cleared = llvm.call @memset(%marks, %clear, %count) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %classSlot = llvm.getelementptr %pagePointer[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %pageClass = llvm.load %classSlot : !llvm.ptr -> i64
    func.call @strict_gc_keep_page(%page, %pageClass) : (i64, i64) -> ()
    cf.br ^next
  ^finished:
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
    llvm.store %zero, %markSlot : i64, !llvm.ptr
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
  func.func private @strict_gc_collect() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %classCount = arith.constant 64 : i64
    %minimum = arith.constant 4194304 : i64
    llvm.call @llvm.eh.unwind.init() : () -> ()
    %stackPointer = llvm.call_intrinsic "llvm.stacksave.p0"() : () -> !llvm.ptr
    cf.br ^unsweep(%zero : i64)
  ^unsweep(%class: i64):
    %more = arith.cmpi ult, %class, %classCount : i64
    cf.cond_br %more, ^unsweepOne, ^mark
  ^unsweepOne:
    func.call @strict_gc_unsweep(%class) : (i64) -> ()
    %cursors = llvm.mlir.addressof @strict_gc_cursors : !llvm.ptr
    %cursorSlot = llvm.getelementptr %cursors[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %cursorSlot : i64, !llvm.ptr
    %ends = llvm.mlir.addressof @strict_gc_ends : !llvm.ptr
    %endSlot = llvm.getelementptr %ends[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %endSlot : i64, !llvm.ptr
    %spans = llvm.mlir.addressof @strict_gc_spans : !llvm.ptr
    %spanSlot = llvm.getelementptr %spans[0, %class] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    llvm.store %zero, %spanSlot : i64, !llvm.ptr
    %nextClass = arith.addi %class, %one : i64
    cf.br ^unsweep(%nextClass : i64)
  ^mark:
    %markedSlot = llvm.mlir.addressof @strict_gc_marked : !llvm.ptr
    llvm.store %zero, %markedSlot : i64, !llvm.ptr
    %from = llvm.ptrtoint %stackPointer : !llvm.ptr to i64
    %stackSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    %stackBase = llvm.load %stackSlot : !llvm.ptr -> i64
    %sixteen = arith.constant 16 : i64
    %to = arith.addi %stackBase, %sixteen : i64
    func.call @strict_gc_mark_range(%from, %to) : (i64, i64) -> ()
    func.call @strict_gc_drain() : () -> ()
    cf.br ^sweeps(%zero : i64)
  ^sweeps(%sweepClass: i64):
    %moreSweeps = arith.cmpi ult, %sweepClass, %classCount : i64
    cf.cond_br %moreSweeps, ^moveOne, ^large
  ^moveOne:
    %pages = llvm.mlir.addressof @strict_gc_pages : !llvm.ptr
    %pagesSlot = llvm.getelementptr %pages[0, %sweepClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %sweeps = llvm.mlir.addressof @strict_gc_sweeps : !llvm.ptr
    %sweepSlot = llvm.getelementptr %sweeps[0, %sweepClass] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<64 x i64>
    %classPages = llvm.load %pagesSlot : !llvm.ptr -> i64
    llvm.store %classPages, %sweepSlot : i64, !llvm.ptr
    llvm.store %zero, %pagesSlot : i64, !llvm.ptr
    %nextSweep = arith.addi %sweepClass, %one : i64
    cf.br ^sweeps(%nextSweep : i64)
  ^large:
    func.call @strict_gc_sweep_large() : () -> ()
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
  func.func private @strict_error_equal(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %leftName = func.call @strict_error_name(%left) : (f64) -> f64
    %rightName = func.call @strict_error_name(%right) : (f64) -> f64
    %leftSet = arith.cmpf une, %leftName, %zero : f64
    %rightSet = arith.cmpf une, %rightName, %zero : f64
    %bothSet = arith.andi %leftSet, %rightSet : i1
    cf.cond_br %bothSet, ^compare, ^different
  ^compare:
    %result = func.call @strict_text_equal(%leftName, %rightName) : (f64, f64) -> f64
    return %result : f64
  ^different:
    return %zero : f64
  }
  func.func private @strict_bits(%p: !llvm.ptr) -> f64 {
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    %f = arith.bitcast %i : i64 to f64
    return %f : f64
  }
  func.func private @strict_list_new(%capacity: f64, %length: f64) -> f64 {
    %headers = arith.constant 48 : i64
    %eight = arith.constant 8.0 : f64
    %bytesf = arith.mulf %capacity, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %total = arith.addi %headers, %bytes : i64
    %view = func.call @strict_alloc(%total) : (i64) -> !llvm.ptr
    %holder = llvm.getelementptr %view[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %items = llvm.getelementptr %view[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %itemsbits = func.call @strict_bits(%items) : (!llvm.ptr) -> f64
    llvm.store %itemsbits, %holder : f64, !llvm.ptr
    %used = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %length, %used : f64, !llvm.ptr
    %cap = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %capacity, %cap : f64, !llvm.ptr
    %unindexed = arith.constant 0.0 : f64
    %index = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %unindexed, %index : f64, !llvm.ptr
    llvm.store %length, %view : f64, !llvm.ptr
    %holderbits = func.call @strict_bits(%holder) : (!llvm.ptr) -> f64
    %viewholder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %viewholder : f64, !llvm.ptr
    %result = func.call @strict_bits(%view) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_open_paren("(\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_close_paren(")\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal @strict_empty_holder(dense<0.0> : tensor<4xf64>) : !llvm.array<4 x f64>
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
  func.func private @strict_list_remove(%list: f64, %item: f64, %isText: i1) -> f64 {
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
    %matched = func.call @strict_list_matches(%value, %item, %isText) : (f64, f64, i1) -> i1
    %next = arith.addi %read, %step : i64
    cf.cond_br %matched, ^scan(%next, %write : i64, i64), ^keep
  ^keep:
    %target = llvm.getelementptr %items[%write] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %value, %target : f64, !llvm.ptr
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
  func.func private @strict_list_remove_text(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_list_remove(%list, %item, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_remove_number(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_list_remove(%list, %item, %isText) : (f64, f64, i1) -> f64
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
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderbits = llvm.load %slot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderbits) : (f64) -> !llvm.ptr
    %itemsbits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsbits) : (f64) -> !llvm.ptr
    return %items : !llvm.ptr
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
  func.func private @strict_list_at(%list: f64, %index: f64, %name: f64) -> f64 {
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
    %error = func.call @strict_error_text(%name) : (f64) -> f64
    return %error : f64
  ^inside:
    %value = func.call @strict_list_get(%list, %chosen) : (f64, f64) -> f64
    return %value : f64
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
    return %copy : f64
  }
  func.func private @strict_list_set(%list: f64, %index: f64, %element: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %at = arith.fptosi %index : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %slot : f64, !llvm.ptr
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
    llvm.store %larger, %capslot : f64, !llvm.ptr
    cf.br ^store
  ^store:
    %itemsbits = llvm.load %holder : !llvm.ptr -> f64
    %items = func.call @strict_ptr(%itemsbits) : (f64) -> !llvm.ptr
    %at = arith.fptosi %used : f64 to i64
    %slot = llvm.getelementptr %items[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %element, %slot : f64, !llvm.ptr
    %next = arith.addf %used, %one : f64
    llvm.store %next, %usedslot : f64, !llvm.ptr
    %result = func.call @strict_alloc(%c16) : (i64) -> !llvm.ptr
    llvm.store %next, %result : f64, !llvm.ptr
    %resultholder = llvm.getelementptr %result[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %resultholder : f64, !llvm.ptr
    %bits = func.call @strict_bits(%result) : (!llvm.ptr) -> f64
    return %bits : f64
  }
  func.func private @strict_list_concat(%left: f64, %right: f64) -> f64 {
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
  func.func private @strict_list_without(%left: f64, %right: f64, %isText: i1) -> f64 {
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
    %removed = func.call @strict_list_remove(%current, %item, %isText) : (f64, f64, i1) -> f64
    %next = arith.addi %index, %step : i64
    cf.br ^loop(%next, %removed : i64, f64)
  ^done:
    return %current : f64
  }
  func.func private @strict_list_without_number(%left: f64, %right: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_list_without(%left, %right, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_without_text(%left: f64, %right: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_list_without(%left, %right, %isText) : (f64, f64, i1) -> f64
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
  func.func private @strict_hash_key(%key: f64, %isText: i1) -> i64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %unset = arith.cmpf oeq, %key, %zero : f64
    cf.cond_br %unset, ^mixed(%start : i64), ^kind
  ^kind:
    cf.cond_br %isText, ^text, ^number
  ^text:
    %pointer = func.call @strict_ptr(%key) : (f64) -> !llvm.ptr
    cf.br ^loop(%pointer, %start : !llvm.ptr, i64)
  ^loop(%at: !llvm.ptr, %hash: i64):
    %byte = llvm.load %at : !llvm.ptr -> i8
    %end = arith.constant 0 : i8
    %done = arith.cmpi eq, %byte, %end : i8
    cf.cond_br %done, ^mixed(%hash : i64), ^next
  ^next:
    %code = arith.extui %byte : i8 to i64
    %thirtyOne = arith.constant 31 : i64
    %scaled = arith.muli %hash, %thirtyOne : i64
    %added = arith.addi %scaled, %code : i64
    %following = llvm.getelementptr %at[1] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^loop(%following, %added : !llvm.ptr, i64)
  ^number:
    %bits = arith.bitcast %key : f64 to i64
    cf.br ^mixed(%bits : i64)
  ^mixed(%value: i64):
    %golden = arith.constant -7046029254386353131 : i64
    %product = arith.muli %value, %golden : i64
    %shift = arith.constant 29 : i64
    %high = arith.shrui %product, %shift : i64
    %result = arith.xori %product, %high : i64
    return %result : i64
  }
  func.func private @strict_dictionary_index(%dictionary: f64, %isText: i1) -> !llvm.ptr {
    %zero = arith.constant 0.0 : f64
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
    cf.br ^extend(%fresh, %none : !llvm.ptr, i64)
  ^extend(%table: !llvm.ptr, %position: i64):
    %more = arith.cmpi slt, %position, %used : i64
    cf.cond_br %more, ^insert, ^done
  ^insert:
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %pairSlot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %pairSlot : !llvm.ptr -> f64
    %key = func.call @strict_list_get(%pair, %zero) : (f64, f64) -> f64
    %hash = func.call @strict_hash_key(%key, %isText) : (f64, i1) -> i64
    %tableSize = llvm.load %table : !llvm.ptr -> i64
    %mask = arith.subi %tableSize, %one : i64
    %first = arith.andi %hash, %mask : i64
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
  func.func private @strict_dictionary_find(%dictionary: f64, %key: f64, %isText: i1) -> i64 {
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
    %table = func.call @strict_dictionary_index(%dictionary, %isText) : (f64, i1) -> !llvm.ptr
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %hash = func.call @strict_hash_key(%key, %isText) : (f64, i1) -> i64
    %size = llvm.load %table : !llvm.ptr -> i64
    %mask = arith.subi %size, %one : i64
    %first = arith.andi %hash, %mask : i64
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
    %matched = func.call @strict_list_matches(%pairKey, %key, %isText) : (f64, f64, i1) -> i1
    cf.cond_br %matched, ^found, ^probe(%wrapped : i64)
  ^found:
    return %position : i64
  ^missing:
    return %none : i64
  }
  func.func private @strict_dictionary_has(%dictionary: f64, %key: f64, %isText: i1) -> f64 {
    %none = arith.constant -1 : i64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %isText) : (f64, f64, i1) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    %result = func.call @strict_boolean(%found) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_get(%dictionary: f64, %key: f64, %isText: i1) -> f64 {
    %none = arith.constant -1 : i64
    %one = arith.constant 1.0 : f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %isText) : (f64, f64, i1) -> i64
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
  func.func private @strict_dictionary_add(%dictionary: f64, %key: f64, %value: f64, %isText: i1) -> f64 {
    %none = arith.constant -1 : i64
    %nothing = arith.constant 0.0 : f64
    %position = func.call @strict_dictionary_find(%dictionary, %key, %isText) : (f64, f64, i1) -> i64
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
  func.func private @strict_dictionary_remove(%dictionary: f64, %key: f64, %isText: i1) -> f64 {
    %none = arith.constant -1 : i64
    %zero = arith.constant 0.0 : f64
    %bits = arith.constant false
    %position = func.call @strict_dictionary_find(%dictionary, %key, %isText) : (f64, f64, i1) -> i64
    %found = arith.cmpi ne, %position, %none : i64
    cf.cond_br %found, ^remove, ^unchanged
  ^remove:
    %items = func.call @strict_list_items(%dictionary) : (f64) -> !llvm.ptr
    %slot = llvm.getelementptr %items[%position] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %pair = llvm.load %slot : !llvm.ptr -> f64
    %copy = func.call @strict_list_copy(%dictionary, %zero) : (f64, f64) -> f64
    %result = func.call @strict_list_remove(%copy, %pair, %bits) : (f64, f64, i1) -> f64
    return %result : f64
  ^unchanged:
    return %dictionary : f64
  }
  func.func private @strict_dictionary_has_number(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_dictionary_has(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_has_text(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_dictionary_has(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_get_number(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_dictionary_get(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_get_text(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_dictionary_get(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_add_number(%dictionary: f64, %key: f64, %value: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_dictionary_add(%dictionary, %key, %value, %isText) : (f64, f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_add_text(%dictionary: f64, %key: f64, %value: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_dictionary_add(%dictionary, %key, %value, %isText) : (f64, f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_remove_number(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_dictionary_remove(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_dictionary_remove_text(%dictionary: f64, %key: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_dictionary_remove(%dictionary, %key, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_text_length(%text: f64) -> f64 {
    %pointer = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%pointer) : (!llvm.ptr) -> i64
    %result = arith.uitofp %length : i64 to f64
    return %result : f64
  }
  func.func private @strict_text_at(%text: f64, %index: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %negative = arith.cmpf olt, %index, %zero : f64
    %wrapped = arith.addf %index, %length : f64
    %chosen = arith.select %negative, %wrapped, %index : f64
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
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %b = func.call @strict_ptr(%search) : (f64) -> !llvm.ptr
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %beyond = arith.cmpf ogt, %start, %length : f64
    cf.cond_br %beyond, ^missing, ^search
  ^search:
    %offset = arith.fptosi %start : f64 to i64
    %from = llvm.getelementptr %a[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %found = llvm.call @strstr(%from, %b) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
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
  func.func private @strict_text_substring(%text: f64, %start: f64, %length: f64) -> f64 {
    %one = arith.constant 1 : i64
    %terminator = arith.constant 0 : i8
    %zero = arith.constant 0.0 : f64
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %textLength = func.call @strict_text_length(%text) : (f64) -> f64
    %negativeStart = arith.cmpf olt, %start, %zero : f64
    %startOrZero = arith.select %negativeStart, %zero, %start : f64
    %beyond = arith.cmpf ogt, %startOrZero, %textLength : f64
    %clampedStart = arith.select %beyond, %textLength, %startOrZero : f64
    %rest = arith.subf %textLength, %clampedStart : f64
    %negativeCount = arith.cmpf olt, %length, %zero : f64
    %countOrZero = arith.select %negativeCount, %zero, %length : f64
    %tooLong = arith.cmpf ogt, %countOrZero, %rest : f64
    %clampedCount = arith.select %tooLong, %rest, %countOrZero : f64
    %offset = arith.fptosi %clampedStart : f64 to i64
    %count = arith.fptosi %clampedCount : f64 to i64
    %size = arith.addi %count, %one : i64
    %copy = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %from = llvm.getelementptr %a[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %done = llvm.call @memcpy(%copy, %from, %count) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %end = llvm.getelementptr %copy[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%copy) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_text_cased(%text: f64, %upper: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1 : i64
    %start = arith.constant 0 : i64
    %length = func.call @strict_text_length(%text) : (f64) -> f64
    %copy = func.call @strict_text_substring(%text, %zero, %length) : (f64, f64, f64) -> f64
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
  func.func private @strict_number_text(%number: f64) -> f64 {
    %size = arith.constant 40 : i64
    %zero = arith.constant 0.0 : f64
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
  func.func private @strict_list_matches(%value: f64, %item: f64, %isText: i1) -> i1 {
    %zero = arith.constant 0.0 : f64
    cf.cond_br %isText, ^text, ^number
  ^text:
    %equal = func.call @strict_text_equal(%value, %item) : (f64, f64) -> f64
    %same = arith.cmpf one, %equal, %zero : f64
    return %same : i1
  ^number:
    %identical = arith.cmpf oeq, %value, %item : f64
    return %identical : i1
  }
  func.func private @strict_list_find(%list: f64, %item: f64, %isText: i1) -> f64 {
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
    %same = func.call @strict_list_matches(%value, %item, %isText) : (f64, f64, i1) -> i1
    %next = arith.addi %index, %step : i64
    cf.cond_br %same, ^found(%index : i64), ^check(%next : i64)
  ^found(%at: i64):
    %result = arith.sitofp %at : i64 to f64
    return %result : f64
  ^missing:
    return %none : f64
  }
  func.func private @strict_list_index_number(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_list_find(%list, %item, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_index_text(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_list_find(%list, %item, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_count(%list: f64, %item: f64, %isText: i1) -> f64 {
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
    %same = func.call @strict_list_matches(%value, %item, %isText) : (f64, f64, i1) -> i1
    %added = arith.addf %found, %one : f64
    %total = arith.select %same, %added, %found : f64
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next, %total : i64, f64)
  ^done:
    return %found : f64
  }
  func.func private @strict_list_count_number(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant false
    %result = func.call @strict_list_count(%list, %item, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_count_text(%list: f64, %item: f64) -> f64 {
    %isText = arith.constant true
    %result = func.call @strict_list_count(%list, %item, %isText) : (f64, f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_has_number(%list: f64, %item: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %index = func.call @strict_list_index_number(%list, %item) : (f64, f64) -> f64
    %found = arith.cmpf oge, %index, %zero : f64
    %result = func.call @strict_boolean(%found) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_has_text(%list: f64, %item: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %index = func.call @strict_list_index_text(%list, %item) : (f64, f64) -> f64
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
    %isMultiple = arith.cmpi sgt, %count, %step : i64
    %enclosed = arith.andi %isMultiple, %brackets : i1
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
  func.func private @strict_list_equal_pair(%left: f64, %right: f64, %keyIsText: i1, %valueIsText: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %leftKey = func.call @strict_list_get(%left, %zero) : (f64, f64) -> f64
    %rightKey = func.call @strict_list_get(%right, %zero) : (f64, f64) -> f64
    %leftValue = func.call @strict_list_get(%left, %one) : (f64, f64) -> f64
    %rightValue = func.call @strict_list_get(%right, %one) : (f64, f64) -> f64
    %sameKey = func.call @strict_list_matches(%leftKey, %rightKey, %keyIsText) : (f64, f64, i1) -> i1
    %sameValue = func.call @strict_list_matches(%leftValue, %rightValue, %valueIsText) : (f64, f64, i1) -> i1
    %same = arith.andi %sameKey, %sameValue : i1
    %result = arith.select %same, %one, %zero : f64
    return %result : f64
  }
  func.func private @strict_list_equal_pair_text_number(%left: f64, %right: f64) -> f64 {
    %keyIsText = arith.constant true
    %valueIsText = arith.constant false
    %result = func.call @strict_list_equal_pair(%left, %right, %keyIsText, %valueIsText) : (f64, f64, i1, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_equal_pair_number_text(%left: f64, %right: f64) -> f64 {
    %keyIsText = arith.constant false
    %valueIsText = arith.constant true
    %result = func.call @strict_list_equal_pair(%left, %right, %keyIsText, %valueIsText) : (f64, f64, i1, i1) -> f64
    return %result : f64
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
  func.func private @strict_textual(%value: f64, %isText: i1) -> f64 {
    cf.cond_br %isText, ^text, ^number
  ^text:
    return %value : f64
  ^number:
    %text = func.call @strict_number_text(%value) : (f64) -> f64
    return %text : f64
  }
  func.func private @strict_list_text_pair(%pair: f64, %keyIsText: i1, %valueIsText: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %key = func.call @strict_list_get(%pair, %zero) : (f64, f64) -> f64
    %value = func.call @strict_list_get(%pair, %one) : (f64, f64) -> f64
    %keyText = func.call @strict_textual(%key, %keyIsText) : (f64, i1) -> f64
    %valueText = func.call @strict_textual(%value, %valueIsText) : (f64, i1) -> f64
    %empty = func.call @strict_list_new(%two, %zero) : (f64, f64) -> f64
    %first = func.call @strict_list_add(%empty, %keyText) : (f64, f64) -> f64
    %texts = func.call @strict_list_add(%first, %valueText) : (f64, f64) -> f64
    %isText = arith.constant true
    %result = func.call @strict_list_text(%texts, %isText) : (f64, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_pair_text_number(%pair: f64) -> f64 {
    %keyIsText = arith.constant true
    %valueIsText = arith.constant false
    %result = func.call @strict_list_text_pair(%pair, %keyIsText, %valueIsText) : (f64, i1, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_pair_number_text(%pair: f64) -> f64 {
    %keyIsText = arith.constant false
    %valueIsText = arith.constant true
    %result = func.call @strict_list_text_pair(%pair, %keyIsText, %valueIsText) : (f64, i1, i1) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_lists_pair_text_number(%list: f64) -> f64 {
    %converter = func.constant @strict_list_text_pair_text_number : (f64) -> f64
    %result = func.call @strict_list_text_mapped(%list, %converter) : (f64, (f64) -> f64) -> f64
    return %result : f64
  }
  func.func private @strict_list_text_lists_pair_number_text(%list: f64) -> f64 {
    %converter = func.constant @strict_list_text_pair_number_text : (f64) -> f64
    %result = func.call @strict_list_text_mapped(%list, %converter) : (f64, (f64) -> f64) -> f64
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
    %copy = func.call @strict_text_substring(%text, %zero, %length) : (f64, f64, f64) -> f64
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
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %missing = llvm.icmp "eq" %file, %null : !llvm.ptr
    cf.cond_br %missing, ^absent, ^present
  ^present:
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    %one = arith.constant 1.0 : f64
    return %one : f64
  ^absent:
    return %zero : f64
  }
  func.func private @strict_file_text(%path: f64) -> f64 {
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
  func.func private @strict_file_writelines(%path: f64, %lines: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %newline = arith.constant 10 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_write_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
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
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_write_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
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
