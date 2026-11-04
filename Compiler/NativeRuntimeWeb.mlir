  // the heap the collector reserves (its page table and cards are an 8192th and a 512th of it), 3.5 GB of the 4 GB of
  // wasm32, the browser commits what it uses (strict_commit); a pause while spinning
  // (no threads in the browser), fresh memory of the browser is zero
  llvm.mlir.global internal constant @strict_heap_bytes(3758096384 : i64) : i64
  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    llvm.return
  }
  // pointers are 4 bytes, locals of 4 and 8 bytes follow each other: the collector reads every 4 bytes as a possible
  // reference (the address of a boxed value is its lower half), locals live in wasm, Asyncify unwinds them into memory
  llvm.mlir.global internal constant @strict_word_bytes(4 : i64) : i64
  // a reclaim unwinds the whole stack to find the values in locals, the collector mostly frees instead
  llvm.mlir.global internal constant @strict_reclaim_bytes(268435456 : i64) : i64
  llvm.func @strict_word(%at: i64) -> i64 attributes {passthrough = ["alwaysinline"]} {
    %pointer = llvm.inttoptr %at : i64 to !llvm.ptr
    %word = llvm.load %pointer : !llvm.ptr -> i32
    %wide = llvm.zext %word : i32 to i64
    llvm.return %wide : i64
  }
  llvm.mlir.global internal @strict_registers(dense<0> : tensor<131072xi64>) : !llvm.array<131072 x i64>
  llvm.mlir.global internal @strict_registers_end(0 : i64) : i64
  llvm.func @strict_scan_registers(!llvm.ptr)
  llvm.func @emscripten_stack_get_base() -> !llvm.ptr
  llvm.func @strict_registers_copy(%begin: !llvm.ptr, %end: !llvm.ptr) {
    %start = llvm.ptrtoint %begin : !llvm.ptr to i64
    %finish = llvm.ptrtoint %end : !llvm.ptr to i64
    %length = llvm.sub %finish, %start : i64
    %most = llvm.mlir.constant(1048568 : i64) : i64
    %isLarger = llvm.icmp "ugt" %length, %most : i64
    %bytes = llvm.select %isLarger, %most, %length : i1, i64
    %buffer = llvm.mlir.addressof @strict_registers : !llvm.ptr
    %copiedData = llvm.call @strict_c_memcpy(%buffer, %begin, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %bufferStart = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %copiedEnd = llvm.add %bufferStart, %bytes : i64
    %endSlot = llvm.mlir.addressof @strict_registers_end : !llvm.ptr
    llvm.store %copiedEnd, %endSlot : i64, !llvm.ptr
    llvm.return
  }
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %copy = llvm.mlir.addressof @strict_registers_copy : !llvm.ptr
    llvm.call @strict_scan_registers(%copy) : (!llvm.ptr) -> ()
    %buffer = llvm.mlir.addressof @strict_registers : !llvm.ptr
    %start = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %endSlot = llvm.mlir.addressof @strict_registers_end : !llvm.ptr
    %end = llvm.load %endSlot : !llvm.ptr -> i64
    return %start, %end : i64, i64
  }
  // wasm32 has 32 bit sizes and longs: the shared runtime and the program call these instead (WebBuild.Sized)
  llvm.func @malloc(i32) -> !llvm.ptr
  llvm.func @calloc(i32, i32) -> !llvm.ptr
  llvm.func @realloc(!llvm.ptr, i32) -> !llvm.ptr
  llvm.func @memcpy(!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
  llvm.func @memset(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @strlen(!llvm.ptr) -> i32
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @strftime(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @fread(!llvm.ptr, i32, i32, !llvm.ptr) -> i32
  llvm.func @fwrite(!llvm.ptr, i32, i32, !llvm.ptr) -> i32
  llvm.func @fseek(!llvm.ptr, i32, i32) -> i32
  llvm.func @ftell(!llvm.ptr) -> i32
  llvm.func @strtol(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @snprintf(!llvm.ptr, i32, !llvm.ptr, ...) -> i32
  llvm.func @strict_c_malloc(%size: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @malloc(%narrow) : (i32) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_c_calloc(%count: i64, %size: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %narrowCount = llvm.trunc %count : i64 to i32
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @calloc(%narrowCount, %narrow) : (i32, i32) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_c_realloc(%old: !llvm.ptr, %size: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @realloc(%old, %narrow) : (!llvm.ptr, i32) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_c_memcpy(%target: !llvm.ptr, %source: !llvm.ptr, %size: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @memcpy(%target, %source, %narrow) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_c_memset(%target: !llvm.ptr, %byte: i32, %size: i64) -> !llvm.ptr attributes {passthrough = ["alwaysinline"]} {
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @memset(%target, %byte, %narrow) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_c_strlen(%text: !llvm.ptr) -> i64 attributes {passthrough = ["alwaysinline"]} {
    %length = llvm.call @strlen(%text) : (!llvm.ptr) -> i32
    %wide = llvm.zext %length : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_c_strncmp(%left: !llvm.ptr, %right: !llvm.ptr, %size: i64) -> i32 attributes {passthrough = ["alwaysinline"]} {
    %narrow = llvm.trunc %size : i64 to i32
    %result = llvm.call @strncmp(%left, %right, %narrow) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_c_strftime(%text: !llvm.ptr, %size: i64, %format: !llvm.ptr, %time: !llvm.ptr) -> i64 {
    %narrow = llvm.trunc %size : i64 to i32
    %written = llvm.call @strftime(%text, %narrow, %format, %time) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
    %wide = llvm.zext %written : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_c_fread(%data: !llvm.ptr, %size: i64, %count: i64, %stream: !llvm.ptr) -> i64 {
    %narrowSize = llvm.trunc %size : i64 to i32
    %narrowCount = llvm.trunc %count : i64 to i32
    %read = llvm.call @fread(%data, %narrowSize, %narrowCount, %stream) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
    %wide = llvm.zext %read : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_c_fwrite(%data: !llvm.ptr, %size: i64, %count: i64, %stream: !llvm.ptr) -> i64 {
    %narrowSize = llvm.trunc %size : i64 to i32
    %narrowCount = llvm.trunc %count : i64 to i32
    %written = llvm.call @fwrite(%data, %narrowSize, %narrowCount, %stream) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
    %wide = llvm.zext %written : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_c_fseek(%stream: !llvm.ptr, %offset: i64, %origin: i32) -> i32 {
    %narrow = llvm.trunc %offset : i64 to i32
    %result = llvm.call @fseek(%stream, %narrow, %origin) : (!llvm.ptr, i32, i32) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_c_ftell(%stream: !llvm.ptr) -> i64 {
    %position = llvm.call @ftell(%stream) : (!llvm.ptr) -> i32
    %wide = llvm.sext %position : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_c_strtol(%text: !llvm.ptr, %end: !llvm.ptr, %base: i32) -> i64 {
    %number = llvm.call @strtol(%text, %end, %base) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %wide = llvm.sext %number : i32 to i64
    llvm.return %wide : i64
  }
  llvm.func @strict_print_precise(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %precision: i32, %number: f64) -> i32 {
    %narrow = llvm.trunc %size : i64 to i32
    %written = llvm.call @snprintf(%buffer, %narrow, %format, %precision, %number) vararg(!llvm.func<i32 (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, i32, f64) -> i32
    llvm.return %written : i32
  }
  llvm.func @strict_print_number(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %number: f64) -> i32 {
    %narrow = llvm.trunc %size : i64 to i32
    %written = llvm.call @snprintf(%buffer, %narrow, %format, %number) vararg(!llvm.func<i32 (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, f64) -> i32
    llvm.return %written : i32
  }
  llvm.func @strict_print_integer(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %number: i32) -> i32 {
    %narrow = llvm.trunc %size : i64 to i32
    %written = llvm.call @snprintf(%buffer, %narrow, %format, %number) vararg(!llvm.func<i32 (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, i32) -> i32
    llvm.return %written : i32
  }
  func.func private @strict_stack_top(%stack: !llvm.ptr) -> i64 {
    %base = llvm.call @emscripten_stack_get_base() : () -> !llvm.ptr
    %top = llvm.ptrtoint %base : !llvm.ptr to i64
    return %top : i64
  }
  llvm.func @strict_crash_log(%text: !llvm.ptr) {
    llvm.return
  }
  // the collector takes the address space 256 MB above the start of malloc (strict_reserve) and the browser's memory
  // grows as it commits pages (strict_commit), malloc stays below it: its sbrk replaces emscripten's (sbrk.c, these four
  // functions), which grows into memory the collector already grew past
  llvm.mlir.global internal @strict_malloc_end(0 : i32) : i32
  llvm.mlir.global internal @strict_sbrk(0 : i32) : i32
  llvm.mlir.global external @__heap_base() : i8
  llvm.func @emscripten_resize_heap(i32) -> i32
  llvm.func @emscripten_get_heap_size() -> i32
  llvm.func @emscripten_get_sbrk_ptr() -> !llvm.ptr {
    %slot = llvm.mlir.addressof @strict_sbrk : !llvm.ptr
    %value = llvm.load %slot : !llvm.ptr -> i32
    %unset = llvm.mlir.constant(0 : i32) : i32
    %isUnset = llvm.icmp "eq" %value, %unset : i32
    llvm.cond_br %isUnset, ^set, ^done
  ^set:
    %base = llvm.mlir.addressof @__heap_base : !llvm.ptr
    %baseBits = llvm.ptrtoint %base : !llvm.ptr to i32
    llvm.store %baseBits, %slot : i32, !llvm.ptr
    llvm.br ^done
  ^done:
    llvm.return %slot : !llvm.ptr
  }
  llvm.func @_sbrk64(%increment: i64) -> !llvm.ptr {
    %slot = llvm.call @emscripten_get_sbrk_ptr() : () -> !llvm.ptr
    %old = llvm.load %slot : !llvm.ptr -> i32
    %oldWide = llvm.zext %old : i32 to i64
    %rest = llvm.mlir.constant(15 : i64) : i64
    %mask = llvm.mlir.constant(-16 : i64) : i64
    %padded = llvm.add %increment, %rest : i64
    %aligned = llvm.and %padded, %mask : i64
    %new = llvm.add %oldWide, %aligned : i64
    %endSlot = llvm.mlir.addressof @strict_malloc_end : !llvm.ptr
    %end = llvm.load %endSlot : !llvm.ptr -> i32
    %endWide = llvm.zext %end : i32 to i64
    %none = llvm.mlir.constant(0 : i64) : i64
    %isLimited = llvm.icmp "ne" %endWide, %none : i64
    %isPast = llvm.icmp "ugt" %new, %endWide : i64
    %isOver = llvm.and %isLimited, %isPast : i1
    %largest = llvm.mlir.constant(4294901760 : i64) : i64
    %isOutside = llvm.icmp "ugt" %new, %largest : i64
    %isRefused = llvm.or %isOver, %isOutside : i1
    llvm.cond_br %isRefused, ^fail, ^sized
  ^sized:
    %heap = llvm.call @emscripten_get_heap_size() : () -> i32
    %heapWide = llvm.zext %heap : i32 to i64
    %isGrowing = llvm.icmp "ugt" %new, %heapWide : i64
    %narrow = llvm.trunc %new : i64 to i32
    llvm.cond_br %isGrowing, ^grow, ^taken
  ^grow:
    %grown = llvm.call @emscripten_resize_heap(%narrow) : (i32) -> i32
    %failed = llvm.mlir.constant(0 : i32) : i32
    %isGrown = llvm.icmp "ne" %grown, %failed : i32
    llvm.cond_br %isGrown, ^taken, ^fail
  ^taken:
    llvm.store %narrow, %slot : i32, !llvm.ptr
    %previous = llvm.inttoptr %oldWide : i64 to !llvm.ptr
    llvm.return %previous : !llvm.ptr
  ^fail:
    %minusOne = llvm.mlir.constant(-1 : i32) : i32
    %failure = llvm.inttoptr %minusOne : i32 to !llvm.ptr
    llvm.return %failure : !llvm.ptr
  }
  llvm.func @sbrk(%increment: i32) -> !llvm.ptr {
    %wide = llvm.zext %increment : i32 to i64
    %result = llvm.call @_sbrk64(%wide) : (i64) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @brk(%target: !llvm.ptr) -> i32 {
    %nothing = llvm.mlir.constant(0 : i32) : i32
    %last = llvm.call @sbrk(%nothing) : (i32) -> !llvm.ptr
    %lastBits = llvm.ptrtoint %last : !llvm.ptr to i32
    %targetBits = llvm.ptrtoint %target : !llvm.ptr to i32
    %increment = llvm.sub %targetBits, %lastBits : i32
    %moved = llvm.call @sbrk(%increment) : (i32) -> !llvm.ptr
    %minusOne = llvm.mlir.constant(-1 : i32) : i32
    %failure = llvm.inttoptr %minusOne : i32 to !llvm.ptr
    %isFailed = llvm.icmp "eq" %moved, %failure : !llvm.ptr
    %result = llvm.select %isFailed, %minusOne, %nothing : i1, i32
    llvm.return %result : i32
  }
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %slot = llvm.call @emscripten_get_sbrk_ptr() : () -> !llvm.ptr
    %brk = llvm.load %slot : !llvm.ptr -> i32
    %brkWide = arith.extui %brk : i32 to i64
    %mallocBytes = arith.constant 268435456 : i64
    %pageRest = arith.constant 65535 : i64
    %pageMask = arith.constant -65536 : i64
    %spaced = arith.addi %brkWide, %mallocBytes : i64
    %padded = arith.addi %spaced, %pageRest : i64
    %start = arith.andi %padded, %pageMask : i64
    %end = arith.addi %start, %size : i64
    %largest = arith.constant 4294901760 : i64
    %fits = arith.cmpi ule, %end, %largest : i64
    cf.cond_br %fits, ^done, ^fail
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    %startNarrow = arith.trunci %start : i64 to i32
    %endSlot = llvm.mlir.addressof @strict_malloc_end : !llvm.ptr
    llvm.store %startNarrow, %endSlot : i32, !llvm.ptr
    %reserved = llvm.inttoptr %start : i64 to !llvm.ptr
    return %reserved : !llvm.ptr
  }
  llvm.func @emscripten_stack_get_end() -> !llvm.ptr
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %end = llvm.call @emscripten_stack_get_end() : () -> !llvm.ptr
    %lowest = llvm.ptrtoint %end : !llvm.ptr to i64
    return %lowest : i64
  }
  // fresh memory of the browser is zero, reused pages are cleared where they are taken (strict_gc_take_pages): the
  // memory grows up to the end of what the collector commits
  func.func private @strict_committed(%at: !llvm.ptr, %size: i64) -> i1 {
    %start = llvm.ptrtoint %at : !llvm.ptr to i64
    %end = arith.addi %start, %size : i64
    %heap = llvm.call @emscripten_get_heap_size() : () -> i32
    %heapWide = arith.extui %heap : i32 to i64
    %isGrowing = arith.cmpi ugt, %end, %heapWide : i64
    %fits = arith.constant true
    cf.cond_br %isGrowing, ^grow, ^done(%fits : i1)
  ^grow:
    %narrow = arith.trunci %end : i64 to i32
    %grown = llvm.call @emscripten_resize_heap(%narrow) : (i32) -> i32
    %failed = arith.constant 0 : i32
    %isGrown = arith.cmpi ne, %grown, %failed : i32
    cf.br ^done(%isGrown : i1)
  ^done(%committed: i1):
    return %committed : i1
  }
  func.func private @strict_decommit(%at: !llvm.ptr, %size: i64) {
    return
  }
  llvm.func @opendir(!llvm.ptr) -> !llvm.ptr
  llvm.func @readdir(!llvm.ptr) -> !llvm.ptr
  llvm.func @closedir(!llvm.ptr) -> i32
  llvm.func @mkdir(!llvm.ptr, i32) -> i32
  llvm.func @fnmatch(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @qsort(!llvm.ptr, i32, i32, !llvm.ptr)
  llvm.func @strcasecmp(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @getenv(!llvm.ptr) -> !llvm.ptr
  llvm.func @stat(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @localtime(!llvm.ptr) -> !llvm.ptr
  llvm.mlir.global internal constant @strict_separator_path("/\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_any_file("*\00") : !llvm.array<2 x i8>
  llvm.func @access(!llvm.ptr, i32) -> i32
  // whether a file (no directory) is at name, without opening it
  func.func private @strict_file_present(%name: !llvm.ptr) -> i1 {
    %false = arith.constant false
    %existing = arith.constant 0 : i32
    %checked = llvm.call @access(%name, %existing) : (!llvm.ptr, i32) -> i32
    %found = arith.cmpi eq, %checked, %existing : i32
    cf.cond_br %found, ^kind, ^done(%false : i1)
  ^kind:
    %directory = llvm.call @opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %isDirectory = llvm.icmp "ne" %directory, %null : !llvm.ptr
    cf.cond_br %isDirectory, ^close, ^done(%found : i1)
  ^close:
    %closed = llvm.call @closedir(%directory) : (!llvm.ptr) -> i32
    cf.br ^done(%false : i1)
  ^done(%present: i1):
    return %present : i1
  }
  func.func private @strict_directory_exists(%self: f64, %path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %directory = llvm.call @opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %exists = llvm.icmp "ne" %directory, %null : !llvm.ptr
    cf.cond_br %exists, ^close, ^done
  ^close:
    %closed = llvm.call @closedir(%directory) : (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    %result = func.call @strict_boolean(%exists) : (i1) -> f64
    return %result : f64
  }
  llvm.func @getcwd(!llvm.ptr, i32) -> !llvm.ptr
  llvm.func @fdopen(i32, !llvm.ptr) -> !llvm.ptr
  func.func private @strict_standard_input() -> !llvm.ptr {
    %standardInput = arith.constant 0 : i32
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %stream = llvm.call @fdopen(%standardInput, %mode) : (i32, !llvm.ptr) -> !llvm.ptr
    return %stream : !llvm.ptr
  }
  func.func private @strict_directory_current(%self: f64) -> f64 {
    %size = arith.constant 4096 : i64
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %path : i8, !llvm.ptr
    %narrow = arith.trunci %size : i64 to i32
    %found = llvm.call @getcwd(%path, %narrow) : (!llvm.ptr, i32) -> !llvm.ptr
    %result = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_directory_create(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %mode = arith.constant 511 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %created = llvm.call @mkdir(%name, %mode) : (!llvm.ptr, i32) -> i32
    %success = arith.constant 0 : i32
    %made = arith.cmpi eq, %created, %success : i32
    %result = func.call @strict_directory_created(%made, %path) : (i1, f64) -> f64
    return %result : f64
  }
  llvm.func @chdir(!llvm.ptr) -> i32
  func.func private @strict_directory_change(%self: f64, %path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %changed = llvm.call @chdir(%name) : (!llvm.ptr) -> i32
    %zero = arith.constant 0 : i32
    %isChanged = arith.cmpi eq, %changed, %zero : i32
    %result = func.call @strict_boolean(%isChanged) : (i1) -> f64
    return %result : f64
  }
  func.func private @strict_path_join(%path: f64, %name: f64) -> f64 {
    %separatorText = llvm.mlir.addressof @strict_separator_path : !llvm.ptr
    %separator = func.call @strict_bits(%separatorText) : (!llvm.ptr) -> f64
    %prefix = func.call @strict_text_concat(%path, %separator) : (f64, f64) -> f64
    %result = func.call @strict_text_concat(%prefix, %name) : (f64, f64) -> f64
    return %result : f64
  }
  llvm.func @strict_compare_names(%left: !llvm.ptr, %right: !llvm.ptr) -> i32 {
    %leftName = llvm.load %left : !llvm.ptr -> !llvm.ptr
    %rightName = llvm.load %right : !llvm.ptr -> !llvm.ptr
    %order = llvm.call @strcasecmp(%leftName, %rightName) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %order : i32
  }
  func.func private @strict_directory_files(%self: f64, %path: f64, %pattern: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %none = arith.constant 0 : i32
    %slotSize = arith.constant 8 : i32
    %directoryType = arith.constant 4 : i8
    %anyText = llvm.mlir.addressof @strict_any_file : !llvm.ptr
    %any = func.call @strict_bits(%anyText) : (!llvm.ptr) -> f64
    %isUnset = arith.cmpf oeq, %pattern, %zero : f64
    %safePattern = arith.select %isUnset, %any, %pattern : f64
    %patternLength = func.call @strict_text_length(%safePattern) : (f64) -> f64
    %isEmpty = arith.cmpf oeq, %patternLength, %zero : f64
    %chosen = arith.select %isEmpty, %any, %safePattern : f64
    %patternText = func.call @strict_ptr(%chosen) : (f64) -> !llvm.ptr
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %created = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    %directory = llvm.call @opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %directory, %null : !llvm.ptr
    cf.cond_br %failed, ^done(%created : f64), ^next(%created : f64)
  ^next(%current: f64):
    %entry = llvm.call @readdir(%directory) : (!llvm.ptr) -> !llvm.ptr
    %finished = llvm.icmp "eq" %entry, %null : !llvm.ptr
    cf.cond_br %finished, ^close(%current : f64), ^entry(%current : f64)
  ^entry(%found: f64):
    %typeSlot = llvm.getelementptr %entry[18] : (!llvm.ptr) -> !llvm.ptr, i8
    %type = llvm.load %typeSlot : !llvm.ptr -> i8
    %fileName = llvm.getelementptr %entry[19] : (!llvm.ptr) -> !llvm.ptr, i8
    %isFile = arith.cmpi ne, %type, %directoryType : i8
    %matched = llvm.call @fnmatch(%patternText, %fileName, %none) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %isMatch = arith.cmpi eq, %matched, %none : i32
    %take = arith.andi %isFile, %isMatch : i1
    cf.cond_br %take, ^add(%found : f64), ^next(%found : f64)
  ^add(%before: f64):
    %nameBits = func.call @strict_bits(%fileName) : (!llvm.ptr) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %copied = func.call @strict_text_concat(%nameBits, %empty) : (f64, f64) -> f64
    %full = func.call @strict_path_join(%path, %copied) : (f64, f64) -> f64
    %added = func.call @strict_list_add(%before, %full) : (f64, f64) -> f64
    cf.br ^next(%added : f64)
  ^close(%all: f64):
    %closed = llvm.call @closedir(%directory) : (!llvm.ptr) -> i32
    %length = func.call @strict_list_length(%all) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i32
    %items = func.call @strict_list_items(%all) : (f64) -> !llvm.ptr
    %compare = llvm.mlir.addressof @strict_compare_names : !llvm.ptr
    llvm.call @qsort(%items, %count, %slotSize, %compare) : (!llvm.ptr, i32, i32, !llvm.ptr) -> ()
    cf.br ^done(%all : f64)
  ^done(%result: f64):
    return %result : f64
  }
  // only Windows needs more than the driver to know what other programs hold on the GPU
  // no CUDA here, the GPU lock (strict_gpu_lock_take) is always free
  func.func private @strict_gpu_lock_try(%buffer: !llvm.ptr) -> i64 {
    %free = arith.constant -1 : i64
    return %free : i64
  }
  func.func private @strict_gpu_lock_holder(%buffer: !llvm.ptr) -> !llvm.ptr {
    return %buffer : !llvm.ptr
  }
  func.func private @strict_gpu_lock_give(%handle: i64) {
    return
  }
  func.func private @strict_gpu_others_free() -> i64 {
    %unknown = arith.constant -1 : i64
    return %unknown : i64
  }
  func.func private @strict_directory_modified(%self: f64, %path: f64) -> f64 {
    %newest = func.call @strict_directory_newest(%path) : (f64) -> f64
    return %newest : f64
  }
  func.func private @strict_directory_directories(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %slotSize = arith.constant 8 : i32
    %directoryType = arith.constant 4 : i8
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %created = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    %directory = llvm.call @opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %directory, %null : !llvm.ptr
    cf.cond_br %failed, ^done(%created : f64), ^next(%created : f64)
  ^next(%current: f64):
    %entry = llvm.call @readdir(%directory) : (!llvm.ptr) -> !llvm.ptr
    %finished = llvm.icmp "eq" %entry, %null : !llvm.ptr
    cf.cond_br %finished, ^close(%current : f64), ^entry(%current : f64)
  ^entry(%found: f64):
    %typeSlot = llvm.getelementptr %entry[18] : (!llvm.ptr) -> !llvm.ptr, i8
    %type = llvm.load %typeSlot : !llvm.ptr -> i8
    %fileName = llvm.getelementptr %entry[19] : (!llvm.ptr) -> !llvm.ptr, i8
    %isDirectory = arith.cmpi eq, %type, %directoryType : i8
    %isDotName = func.call @strict_is_dot_name(%fileName) : (!llvm.ptr) -> i1
    %notDot = arith.xori %isDotName, %isDirectory : i1
    %take = arith.andi %isDirectory, %notDot : i1
    cf.cond_br %take, ^add(%found : f64), ^next(%found : f64)
  ^add(%before: f64):
    %nameBits = func.call @strict_bits(%fileName) : (!llvm.ptr) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %copied = func.call @strict_text_concat(%nameBits, %empty) : (f64, f64) -> f64
    %full = func.call @strict_path_join(%path, %copied) : (f64, f64) -> f64
    %added = func.call @strict_list_add(%before, %full) : (f64, f64) -> f64
    cf.br ^next(%added : f64)
  ^close(%all: f64):
    %closed = llvm.call @closedir(%directory) : (!llvm.ptr) -> i32
    %length = func.call @strict_list_length(%all) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i32
    %items = func.call @strict_list_items(%all) : (f64) -> !llvm.ptr
    %compare = llvm.mlir.addressof @strict_compare_names : !llvm.ptr
    llvm.call @qsort(%items, %count, %slotSize, %compare) : (!llvm.ptr, i32, i32, !llvm.ptr) -> ()
    cf.br ^done(%all : f64)
  ^done(%result: f64):
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_web("Web\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_start_failed("no processes in the browser, could not start \00") : !llvm.array<46 x i8>
  func.func private @strict_process_operatingsystem(%self: f64) -> f64 {
    %name = llvm.mlir.addressof @strict_web : !llvm.ptr
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_process_current(%self: f64) -> f64 {
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    return %empty : f64
  }
  llvm.func @time(!llvm.ptr) -> i64
  llvm.func @clock_gettime(i32, !llvm.ptr) -> i32
  func.func private @strict_clock() -> f64 {
    %monotonic = arith.constant 1 : i32
    %seconds = func.call @strict_clock_read(%monotonic) : (i32) -> f64
    return %seconds : f64
  }
  func.func private @strict_time_now(%self: f64) -> f64 {
    %realtime = arith.constant 0 : i32
    %seconds = func.call @strict_clock_read(%realtime) : (i32) -> f64
    return %seconds : f64
  }
  // no thread clock in the browser: the wall clock
  func.func private @strict_thread_seconds() -> f64 {
    %seconds = func.call @strict_clock() : () -> f64
    return %seconds : f64
  }
  func.func private @strict_clock_read(%clock: i32) -> f64 {
    %two = arith.constant 2 : i64
    %timespec = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    %read = llvm.call @clock_gettime(%clock, %timespec) : (i32, !llvm.ptr) -> i32
    %secondsSlot = llvm.getelementptr %timespec[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %nanosecondsSlot = llvm.getelementptr %timespec[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %whole = llvm.load %secondsSlot : !llvm.ptr -> i64
    %nanosecondsLong = llvm.load %nanosecondsSlot : !llvm.ptr -> i32
    %nanoseconds = arith.extsi %nanosecondsLong : i32 to i64
    %wholeFloat = arith.sitofp %whole : i64 to f64
    %nanosecondsFloat = arith.sitofp %nanoseconds : i64 to f64
    %billion = arith.constant 1.0e9 : f64
    %fraction = arith.divf %nanosecondsFloat, %billion : f64
    %seconds = arith.addf %wholeFloat, %fraction : f64
    return %seconds : f64
  }
  func.func private @strict_random_seed() -> i64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %now = llvm.call @time(%null) : (!llvm.ptr) -> i64
    %one = arith.constant 1 : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %address = llvm.ptrtoint %slot : !llvm.ptr to i64
    %mixed = arith.xori %now, %address : i64
    return %mixed : i64
  }
  func.func private @strict_file_locked(%path: f64) -> f64 {
    %held = arith.constant 1.0 : f64
    return %held : f64
  }
  func.func private @strict_file_rename(%path: f64, %newPath: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %newName = func.call @strict_ptr(%newPath) : (f64) -> !llvm.ptr
    %renamed = llvm.call @rename(%name, %newName) : (!llvm.ptr, !llvm.ptr) -> i32
    return %zero : f64
  }
  func.func private @strict_guard() {
    return
  }
  func.func private @strict_file_size(%name: !llvm.ptr) -> i64 {
    %size = arith.constant 256 : i64
    %buffer = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
    %status = llvm.call @stat(%name, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %zero = arith.constant 0 : i64
    cf.cond_br %failed, ^done(%zero : i64), ^size
  ^size:
    %sizeSlot = llvm.getelementptr %buffer[24] : (!llvm.ptr) -> !llvm.ptr, i8
    %bytes = llvm.load %sizeSlot : !llvm.ptr -> i64
    cf.br ^done(%bytes : i64)
  ^done(%result: i64):
    return %result : i64
  }
  func.func private @strict_file_modified(%path: f64) -> f64 {
    %overriddenText = func.call @strict_overridden(%path) : (f64) -> f64
    %unsetText = arith.constant 0.0 : f64
    %isOverridden = arith.cmpf one, %overriddenText, %unsetText : f64
    cf.cond_br %isOverridden, ^overridden, ^disk
  ^overridden:
    return %unsetText : f64
  ^disk:
    %size = arith.constant 256 : i64
    %buffer = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %status = llvm.call @stat(%name, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %zero = arith.constant 0.0 : f64
    cf.cond_br %failed, ^done(%zero : f64), ^time
  ^time:
    %secondsSlot = llvm.getelementptr %buffer[56] : (!llvm.ptr) -> !llvm.ptr, i8
    %seconds = llvm.load %secondsSlot : !llvm.ptr -> i64
    %nanosecondsSlot = llvm.getelementptr %buffer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %nanosecondsLong = llvm.load %nanosecondsSlot : !llvm.ptr -> i32
    %nanoseconds = arith.extsi %nanosecondsLong : i32 to i64
    %whole = arith.sitofp %seconds : i64 to f64
    %part = arith.sitofp %nanoseconds : i64 to f64
    %billion = arith.constant 1.0e9 : f64
    %fraction = arith.divf %part, %billion : f64
    %time = arith.addf %whole, %fraction : f64
    cf.br ^done(%time : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_process_built(%self: f64) -> f64 {
    %size = arith.constant 256 : i64
    %buffer = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
    %path = func.call @strict_ptr(%self) : (f64) -> !llvm.ptr
    %status = llvm.call @stat(%path, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %null = llvm.mlir.zero : !llvm.ptr
    cf.cond_br %failed, ^done(%null : !llvm.ptr), ^time
  ^time:
    %modified = llvm.getelementptr %buffer[56] : (!llvm.ptr) -> !llvm.ptr, i8
    %local = llvm.call @localtime(%modified) : (!llvm.ptr) -> !llvm.ptr
    cf.br ^done(%local : !llvm.ptr)
  ^done(%time: !llvm.ptr):
    %result = func.call @strict_time_text(%time) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_tool_path(%name: f64) -> f64 {
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    return %empty : f64
  }
  func.func private @strict_text_of(%global: !llvm.ptr) -> f64 {
    %bits = func.call @strict_bits(%global) : (!llvm.ptr) -> f64
    return %bits : f64
  }
  func.func private @strict_process_started(%path: f64, %arguments: f64) -> f64 {
    %missing = arith.constant 127.0 : f64
    return %missing : f64
  }
  func.func private @strict_process_execute(%path: f64, %arguments: f64) -> f64 {
    %startCode = arith.constant 1.0 : f64
    %startText = llvm.mlir.addressof @strict_start_failed : !llvm.ptr
    %startPrefix = func.call @strict_bits(%startText) : (!llvm.ptr) -> f64
    %startError = func.call @strict_text_concat(%startPrefix, %path) : (f64, f64) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %startFailure = func.call @strict_process_result(%startCode, %empty, %startError) : (f64, f64, f64) -> f64
    return %startFailure : f64
  }
  func.func private @strict_library_call(%path: f64, %symbol: f64, %arguments: f64) -> f64 {
    %libraryMessage = llvm.mlir.addressof @strict_library_missing : !llvm.ptr
    %libraryError = func.call @strict_error_named(%libraryMessage) : (!llvm.ptr) -> f64
    return %libraryError : f64
  }
  // bytes of physical memory, 0 when unknown
  func.func private @strict_host_memory() -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  // Process.MemoryUsed: bytes of the WebAssembly memory (the heap only grows, it is never given back)
  func.func private @strict_process_memoryused(%self: f64) -> f64 {
    %heap = llvm.call @emscripten_get_heap_size() : () -> i32
    %bytes = arith.uitofp %heap : i32 to f64
    return %bytes : f64
  }
  func.func private @strict_pool_spawn() -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  llvm.func @strict_pool_sleep(%seen: i64) {
    llvm.return
  }
  llvm.func @strict_pool_wake(%helpers: i64) {
    llvm.return
  }
  llvm.func @strict_pool_await(%count: i64) {
    llvm.return
  }
  llvm.func @strict_pool_finished() {
    llvm.return
  }
  func.func private @strict_gpu_spawn() -> i1 {
    %none = arith.constant false
    return %none : i1
  }
  // a loop with a WGSL shader (WgslShader) runs on WebGPU through strict_gpu_kernel like kind 2 (generated loops by the
  // second name of their table, the 32 bit one, the first is empty when a 64 bit kernel failed) when its result format
  // is 17 (bfloat16) or 32 bits, or when one of the lists of a loop of numbers is narrow (strict_gpu_shaded_format),
  // else it runs on the CPU (0); a runtime kernel of NativeRuntimeWgsl.js: each value is (number, -1, 0) or (data
  // address, length, bits of its numbers)
  llvm.func @strict_webgpu_ready(!llvm.ptr) -> i32
  func.func private @strict_gpu_shaded(%name: !llvm.ptr, %count: i64, %source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64, %format: i64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %nameLength = llvm.call @strict_c_strlen(%name) : (!llvm.ptr) -> i64
    %afterName = arith.addi %nameLength, %one : i64
    %singleName = llvm.getelementptr %name[%afterName] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %isGenerated = arith.cmpi ule, %kind, %one : i64
    %shaderName = arith.select %isGenerated, %singleName, %name : !llvm.ptr
    %ready = llvm.call @strict_webgpu_ready(%shaderName) : (!llvm.ptr) -> i32
    %unready = arith.constant 0 : i32
    %isReady = arith.cmpi ne, %ready, %unready : i32
    %isPlain = arith.cmpi ule, %kind, %one : i64
    %usable = arith.andi %isReady, %isPlain : i1
    %isKernel = arith.cmpi eq, %kind, %two : i64
    %kernelUsable = arith.andi %isReady, %isKernel : i1
    cf.cond_br %usable, ^plain, ^other
  ^plain:
    %brainFormat = arith.constant 17 : i64
    %singleFormat = arith.constant 32 : i64
    %isBrain = arith.cmpi eq, %format, %brainFormat : i64
    %isSingle = arith.cmpi eq, %format, %singleFormat : i64
    %isNarrow = arith.ori %isBrain, %isSingle : i1
    %widened = func.call @strict_gpu_shaded_format(%source, %kind, %values, %valueCount, %mask) : (f64, i64, !llvm.ptr, i64, i64) -> i64
    %chosen = arith.select %isNarrow, %format, %widened : i64
    %isChosen = arith.cmpi ne, %chosen, %none : i64
    cf.cond_br %isChosen, ^narrow, ^skip
  ^narrow:
    %slotBytes = arith.constant 8 : i64
    %narrowSlots = arith.addi %valueCount, %one : i64
    %sourced = llvm.alloca %narrowSlots x f64 : (i64) -> !llvm.ptr
    %valueBytes = arith.muli %valueCount, %slotBytes : i64
    %copied = llvm.call @strict_c_memcpy(%sourced, %values, %valueBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %sourceSlot = llvm.getelementptr %sourced[%valueCount] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %source, %sourceSlot : f64, !llvm.ptr
    %isNarrowList = arith.cmpi eq, %kind, %one : i64
    %sourceBit = arith.shli %one, %valueCount : i64
    %withSource = arith.ori %mask, %sourceBit : i64
    %narrowMask = arith.select %isNarrowList, %withSource, %mask : i64
    %narrowResult = func.call @strict_gpu_kernel(%shaderName, %count, %sourced, %narrowSlots, %narrowMask, %chosen) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> f64
    return %narrowResult : f64
  ^other:
    cf.cond_br %kernelUsable, ^kernel, ^skip
  ^kernel:
    %kernelResult = func.call @strict_gpu_kernel(%name, %count, %values, %valueCount, %mask, %format) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> f64
    return %kernelResult : f64
  ^skip:
    return %zero : f64
  }
  // the format a loop of numbers runs its 32 bit shader in, like the native 32 bit kernel: 32 when one of its lists
  // holds narrow numbers (also in a WebGPU buffer), 17 when all of them hold bfloat16, else 0 (it runs on the CPU)
  func.func private @strict_gpu_shaded_format(%source: f64, %kind: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %brainBits = arith.constant 17 : i64
    %singleBits = arith.constant 32 : i64
    %sixtyFour = arith.constant 64 : i64
    %true = arith.constant true
    %false = arith.constant false
    %isListSource = arith.cmpi eq, %kind, %one : i64
    cf.cond_br %isListSource, ^sourced, ^next(%none, %false, %true : i64, i1, i1)
  ^sourced:
    %sourceBits = func.call @strict_gpu_bits(%source) : (f64) -> i64
    %sourceNarrow = arith.cmpi ult, %sourceBits, %sixtyFour : i64
    %sourceBrain = arith.cmpi eq, %sourceBits, %brainBits : i64
    cf.br ^next(%none, %sourceNarrow, %sourceBrain : i64, i1, i1)
  ^next(%at: i64, %narrow: i1, %brains: i1):
    %more = arith.cmpi slt, %at, %valueCount : i64
    cf.cond_br %more, ^value, ^chosen
  ^value:
    %shifted = arith.shrui %mask, %at : i64
    %isList = arith.trunci %shifted : i64 to i1
    %after = arith.addi %at, %one : i64
    cf.cond_br %isList, ^list, ^next(%after, %narrow, %brains : i64, i1, i1)
  ^list:
    %valueSlot = llvm.getelementptr %values[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %valueSlot : !llvm.ptr -> f64
    %bits = func.call @strict_gpu_bits(%value) : (f64) -> i64
    %isNarrowList = arith.cmpi ult, %bits, %sixtyFour : i64
    %isBrainList = arith.cmpi eq, %bits, %brainBits : i64
    %anyNarrow = arith.ori %narrow, %isNarrowList : i1
    %allBrains = arith.andi %brains, %isBrainList : i1
    cf.br ^next(%after, %anyNarrow, %allBrains : i64, i1, i1)
  ^chosen:
    %narrowFormat = arith.select %brains, %brainBits, %singleBits : i64
    %format = arith.select %narrow, %narrowFormat, %none : i64
    return %format : i64
  }
  // the count numbers (of bits 17 or 32) of the runtime kernel name of NativeRuntimeWgsl.js from values (lists where
  // mask has their bit) as a list that stays in a WebGPU buffer (class 255, strict_gpu_web_get reads it on the first
  // CPU read), 0.0 when WebGPU fails; a list value already in a WebGPU buffer is copied there on the GPU; after 1 GB of
  // results a full collection gives back the buffers of dead lists first
  llvm.func @strict_webgpu_kernel(!llvm.ptr, i32, !llvm.ptr, i32, i32) -> i32
  llvm.func @strict_webgpu_crowded() -> i32
  func.func private @strict_gpu_kernel(%name: !llvm.ptr, %count: i64, %values: !llvm.ptr, %valueCount: i64, %mask: i64, %bits: i64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %unused = arith.constant 0 : i32
    %crowded = llvm.call @strict_webgpu_crowded() : () -> i32
    %isCrowded = arith.cmpi ne, %crowded, %unused : i32
    cf.cond_br %isCrowded, ^collect, ^table
  ^collect:
    %full = arith.constant true
    func.call @strict_gc_run(%full) : (i1) -> ()
    cf.br ^table
  ^table:
    %entries = arith.muli %valueCount, %three : i64
    %table = llvm.alloca %entries x f64 : (i64) -> !llvm.ptr
    cf.br ^slot(%none : i64)
  ^slot(%at: i64):
    %more = arith.cmpi slt, %at, %valueCount : i64
    cf.cond_br %more, ^value, ^filled
  ^value:
    %valueSlot = llvm.getelementptr %values[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %valueSlot : !llvm.ptr -> f64
    %bit = arith.shli %one, %at : i64
    %masked = arith.andi %mask, %bit : i64
    %isList = arith.cmpi ne, %masked, %none : i64
    %entry = arith.muli %at, %three : i64
    %number = llvm.getelementptr %table[%entry] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %length = llvm.getelementptr %number[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %format = llvm.getelementptr %number[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %next = arith.addi %at, %one : i64
    cf.cond_br %isList, ^list, ^plain
  ^plain:
    %unlisted = arith.constant -1.0 : f64
    llvm.store %value, %number : f64, !llvm.ptr
    llvm.store %unlisted, %length : f64, !llvm.ptr
    llvm.store %zero, %format : f64, !llvm.ptr
    cf.br ^slot(%next : i64)
  ^list:
    %listLength = func.call @strict_list_length(%value) : (f64) -> f64
    llvm.store %listLength, %length : f64, !llvm.ptr
    %handle = func.call @strict_gpu_web_handle(%value) : (f64) -> i64
    %isResident = arith.cmpi ne, %handle, %none : i64
    cf.cond_br %isResident, ^resident(%handle : i64), ^hosted
  ^resident(%residentHandle: i64):
    %handleNumber = arith.sitofp %residentHandle : i64 to f64
    %bufferFormat = arith.constant 255.0 : f64
    llvm.store %handleNumber, %number : f64, !llvm.ptr
    llvm.store %bufferFormat, %format : f64, !llvm.ptr
    cf.br ^slot(%next : i64)
  ^hosted:
    %valueBits = func.call @strict_gpu_bits(%value) : (f64) -> i64
    %listCount = arith.fptosi %listLength : f64 to i64
    %kept = func.call @strict_gpu_web_kept(%value, %listCount, %valueBits) : (f64, i64, i64) -> i64
    %isKept = arith.cmpi ne, %kept, %none : i64
    cf.cond_br %isKept, ^resident(%kept : i64), ^copied
  ^copied:
    %host = func.call @strict_gpu_host(%value) : (f64) -> !llvm.ptr
    %address = llvm.ptrtoint %host : !llvm.ptr to i64
    %addressNumber = arith.uitofp %address : i64 to f64
    %bitsNumber = arith.sitofp %valueBits : i64 to f64
    llvm.store %addressNumber, %number : f64, !llvm.ptr
    llvm.store %bitsNumber, %format : f64, !llvm.ptr
    cf.br ^slot(%next : i64)
  ^filled:
    %count32 = arith.trunci %count : i64 to i32
    %valueCount32 = arith.trunci %valueCount : i64 to i32
    %bits32 = arith.trunci %bits : i64 to i32
    %made = llvm.call @strict_webgpu_kernel(%name, %count32, %table, %valueCount32, %bits32) : (!llvm.ptr, i32, !llvm.ptr, i32, i32) -> i32
    %isMade = arith.cmpi ne, %made, %unused : i32
    cf.cond_br %isMade, ^resides, ^failed
  ^resides:
    %buffer = arith.extui %made : i32 to i64
    %device = arith.shli %buffer, %eight : i64
    %webClass = arith.constant 255 : i64
    %result = func.call @strict_gpu_resident(%count, %device, %webClass, %bits) : (i64, i64, i64, i64) -> f64
    return %result : f64
  ^failed:
    return %zero : f64
  }
  // the WebGPU buffer of a list computed on the GPU, also after the CPU read it (like native device copies: the lists
  // are not changed, FluxAttention's Products read the downloaded mixed values 13 times again from the host), 0 for any
  // other list
  func.func private @strict_gpu_web_handle(%list: f64) -> i64 {
    %none = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %webClass = arith.constant 255 : i64
    %sixtyFour = arith.constant 64 : i64
    %bits = func.call @strict_gpu_bits(%list) : (f64) -> i64
    %isPacked = arith.cmpi ne, %bits, %sixtyFour : i64
    cf.cond_br %isPacked, ^packed, ^other
  ^packed:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %deviceSlot = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %tagged = llvm.load %deviceSlot : !llvm.ptr -> i64
    %class = arith.andi %tagged, %webClass : i64
    %isWeb = arith.cmpi eq, %class, %webClass : i64
    %handle = arith.shrui %tagged, %eight : i64
    %result = arith.select %isWeb, %handle, %none : i64
    return %result : i64
  ^other:
    return %none : i64
  }
  llvm.mlir.global internal constant @strict_gpu_web_unread("Error: a result of the GPU could not be read back (WebGPU)\0A\00") : !llvm.array<60 x i8>
  llvm.func @strict_webgpu_download(i32, !llvm.ptr, i32, i32) -> i32
  llvm.func @strict_webgpu_release(i32)
  // the numbers of the WebGPU buffer of device in a new host block of the list holder, stops the program when its
  // loop read outside of a list or the buffer can not be read (lost device)
  func.func private @strict_gpu_web_get(%holder: !llvm.ptr, %device: i64) -> !llvm.ptr {
    %none = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %usedSlot = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %used = llvm.load %usedSlot : !llvm.ptr -> f64
    %count = arith.fptosi %used : f64 to i64
    %formatSlot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %negativeFormat = llvm.load %formatSlot : !llvm.ptr -> f64
    %formatNumber = arith.negf %negativeFormat : f64
    %format = arith.fptosi %formatNumber : f64 to i64
    %bytes = func.call @strict_packed_bytes(%count, %format) : (i64, i64) -> i64
    %size = arith.maxui %bytes, %eight : i64
    %host = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %handle = arith.shrui %device, %eight : i64
    %handle32 = arith.trunci %handle : i64 to i32
    %count32 = arith.trunci %count : i64 to i32
    %format32 = arith.trunci %format : i64 to i32
    %got = llvm.call @strict_webgpu_download(%handle32, %host, %count32, %format32) : (i32, !llvm.ptr, i32, i32) -> i32
    %failed = arith.constant 0 : i32
    %read = arith.constant 1 : i32
    %isGot = arith.cmpi eq, %got, %read : i32
    %isFailed = arith.cmpi eq, %got, %failed : i32
    cf.cond_br %isGot, ^store, ^fail(%isFailed : i1)
  ^store:
    func.call @strict_gc_written_range(%host, %bytes) : (!llvm.ptr, i64) -> ()
    %hostBits = func.call @strict_bits(%host) : (!llvm.ptr) -> f64
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %hostBits, %dataSlot : f64, !llvm.ptr
    func.call @strict_gc_written(%dataSlot) : (!llvm.ptr) -> ()
    return %host : !llvm.ptr
  ^fail(%isUnread: i1):
    %unread = llvm.mlir.addressof @strict_gpu_web_unread : !llvm.ptr
    %outside = llvm.mlir.addressof @strict_gpu_failed_text : !llvm.ptr
    %message = arith.select %isUnread, %unread, %outside : !llvm.ptr
    %printed = llvm.call @printf(%message) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %code = arith.constant 1 : i32
    llvm.call @exit(%code) : (i32) -> ()
    %nothing = llvm.mlir.zero : !llvm.ptr
    return %nothing : !llvm.ptr
  }
  // the handle of a new WebGPU buffer with the words at data as they are (halves bfloat16 numbers in them when above 0,
  // 32 bit floats kept as bfloat16 halves when all are exact bfloat16 numbers when -1), 0 before the first kernel made the
  // device
  // kernels run as WebGPU shaders here, native platforms leave the WebGPU paths of NativeRuntime.mlir at once
  func.func private @strict_gpu_webgpu() -> i1 {
    %yes = arith.constant true
    return %yes : i1
  }
  llvm.func @strict_webgpu_uploaded(!llvm.ptr, i32, i32) -> i32
  func.func private @strict_gpu_web_uploaded(%data: !llvm.ptr, %words: i64, %halves: i64) -> i64 {
    %words32 = arith.trunci %words : i64 to i32
    %halves32 = arith.trunci %halves : i64 to i32
    %made = llvm.call @strict_webgpu_uploaded(%data, %words32, %halves32) : (!llvm.ptr, i32, i32) -> i32
    %handle = arith.extui %made : i32 to i64
    return %handle : i64
  }
  // a packed host list of bfloat16 or 32 bit numbers (bits) with at least 4096 of them uploaded once as a kernel value
  // and kept in its holder like a downloaded result (the rotations and modulation all blocks of a Moss step read), its
  // handle, 0 for other lists or when WebGPU can not
  func.func private @strict_gpu_web_kept(%list: f64, %count: i64, %bits: i64) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %eight = arith.constant 8 : i64
    %webClass = arith.constant 255 : i64
    %halvesBits = arith.constant 17 : i64
    %floatBits = arith.constant 32 : i64
    %isHalves = arith.cmpi eq, %bits, %halvesBits : i64
    %isFloats = arith.cmpi eq, %bits, %floatBits : i64
    %isNarrow = arith.ori %isHalves, %isFloats : i1
    %least = arith.constant 4096 : i64
    %isLong = arith.cmpi sge, %count, %least : i64
    %isKept = arith.andi %isNarrow, %isLong : i1
    cf.cond_br %isKept, ^holder, ^none
  ^holder:
    %view = func.call @strict_ptr(%list) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %deviceSlot = llvm.getelementptr %holder[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %tagged = llvm.load %deviceSlot : !llvm.ptr -> i64
    %isFree = arith.cmpi eq, %tagged, %none : i64
    cf.cond_br %isFree, ^upload, ^none
  ^upload:
    %host = func.call @strict_gpu_host(%list) : (f64) -> !llvm.ptr
    %pairs = arith.addi %count, %one : i64
    %halfWords = arith.divui %pairs, %two : i64
    %words = arith.select %isHalves, %halfWords, %count : i64
    %halves = arith.select %isHalves, %count, %none : i64
    %handle = func.call @strict_gpu_web_uploaded(%host, %words, %halves) : (!llvm.ptr, i64, i64) -> i64
    %isUploaded = arith.cmpi ne, %handle, %none : i64
    cf.cond_br %isUploaded, ^keep, ^none
  ^keep:
    %shifted = arith.shli %handle, %eight : i64
    %device = arith.ori %shifted, %webClass : i64
    llvm.store %device, %deviceSlot : i64, !llvm.ptr
    return %handle : i64
  ^none:
    return %none : i64
  }
  // count 8 bit rows of elements numbers (bits, plus STRICT_KEPT_TOKENS) whose block (the codes of each row as in the
  // file, 4 bit halves of bytes when nibbles is 1, then a 32 bit scale per row) is only in the WebGPU buffer handle
  // (TensorFile reading a URL): views without data in one block, strict_gpu_tokens_web finds the buffer by their
  // addresses, it is given back once no view is left (its views can outlive this list in lists joined from them, Moss's
  // query, key and value rows); reading the rows on the CPU stops the program
  llvm.func @strict_webgpu_rows_views(!llvm.ptr, i32, i32, i32, i32)
  func.func @strict_rows_web(%count: i64, %elements: i64, %bits: f64, %handle: i64, %nibbles: i64) -> f64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %zero = arith.constant 0.0 : f64
    %viewBytes = arith.constant 64 : i64
    %markedBits = arith.fptosi %bits : f64 to i64
    %formatMask = arith.constant 255 : i64
    %wholeBits = arith.andi %markedBits, %formatMask : i64
    %kept = arith.shrui %markedBits, %eight : i64
    %two = arith.constant 2 : i64
    %halfBytes = arith.divui %elements, %two : i64
    %isNibbles = arith.cmpi ne, %nibbles, %none : i64
    %rowStride = arith.select %isNibbles, %halfBytes, %elements : i64
    %views = arith.muli %count, %viewBytes : i64
    %block = func.call @strict_alloc_text(%views) : (i64) -> !llvm.ptr
    %countNumber = arith.sitofp %count : i64 to f64
    %list = func.call @strict_list_new(%countNumber, %countNumber) : (f64, f64) -> f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    cf.br ^place(%none : i64)
  ^place(%index: i64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^row, ^placed
  ^row:
    %viewOffset = arith.muli %index, %viewBytes : i64
    %view = llvm.getelementptr %block[%viewOffset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %row = func.call @strict_packed_placed(%view, %elements, %wholeBits, %zero, %zero, %kept) : (!llvm.ptr, i64, i64, f64, f64, i64) -> f64
    %target = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %row, %target : f64, !llvm.ptr
    func.call @strict_gc_written(%target) : (!llvm.ptr) -> ()
    %next = arith.addi %index, %one : i64
    cf.br ^place(%next : i64)
  ^placed:
    %count32 = arith.trunci %count : i64 to i32
    %handle32 = arith.trunci %handle : i64 to i32
    %stride32 = arith.trunci %rowStride : i64 to i32
    %nibbles32 = arith.trunci %nibbles : i64 to i32
    llvm.call @strict_webgpu_rows_views(%block, %count32, %handle32, %stride32, %nibbles32) : (!llvm.ptr, i32, i32, i32, i32) -> ()
    %blockAddress = llvm.ptrtoint %block : !llvm.ptr to i64
    %device = arith.shli %handle, %eight : i64
    %webClass = arith.constant 255 : i64
    func.call @strict_gpu_reside(%blockAddress, %device, %webClass) : (i64, i64, i64) -> ()
    %pendingSlot = llvm.mlir.addressof @strict_foreign_pending : !llvm.ptr
    %listBits = arith.bitcast %list : f64 to i64
    llvm.store %listBits, %pendingSlot : i64, !llvm.ptr
    return %list : f64
  }
  // the count rows of items (views of strict_rows_web, rows joined from several matrices like Moss's query, key and
  // value rows) copied into a new WebGPU buffer on the GPU, its handle (0 when a row is in no buffer or the rows differ),
  // the stride in words and 1 for 4 bit codes at shape
  llvm.func @strict_webgpu_gathered(!llvm.ptr, i32, !llvm.ptr) -> i32
  func.func private @strict_gpu_web_gathered(%items: !llvm.ptr, %count: i64, %shape: !llvm.ptr) -> i64 {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %views = llvm.alloca %count x i32 : (i64) -> !llvm.ptr
    cf.br ^next(%none : i64)
  ^next(%row: i64):
    %more = arith.cmpi slt, %row, %count : i64
    cf.cond_br %more, ^view, ^gather
  ^view:
    %slot = llvm.getelementptr %items[%row] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %view = func.call @strict_ptr(%value) : (f64) -> !llvm.ptr
    %address = llvm.ptrtoint %view : !llvm.ptr to i64
    %address32 = arith.trunci %address : i64 to i32
    %target = llvm.getelementptr %views[%row] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %address32, %target : i32, !llvm.ptr
    %following = arith.addi %row, %one : i64
    cf.br ^next(%following : i64)
  ^gather:
    %count32 = arith.trunci %count : i64 to i32
    %made = llvm.call @strict_webgpu_gathered(%views, %count32, %shape) : (!llvm.ptr, i32, !llvm.ptr) -> i32
    %handle = arith.extui %made : i32 to i64
    return %handle : i64
  }
  // a list in a WebGPU buffer is gone: so is the buffer
  func.func private @strict_gpu_web_give(%device: i64) {
    %eight = arith.constant 8 : i64
    %handle = arith.shrui %device, %eight : i64
    %handle32 = arith.trunci %handle : i64 to i32
    llvm.call @strict_webgpu_release(%handle32) : (i32) -> ()
    return
  }
  // GpuMemory.Released and Evicted (strict_gpu_trim_unused) also destroy the WebGPU buffers kept for reuse
  llvm.func @strict_webgpu_trimmed()
  func.func private @strict_gpu_web_trim() {
    llvm.call @strict_webgpu_trimmed() : () -> ()
    return
  }
  // GpuMemory.Evicted also gives back the WebGPU buffers of rows read from files, they are uploaded again when used
  llvm.func @strict_webgpu_evicted()
  func.func private @strict_gpu_web_evict() {
    llvm.call @strict_webgpu_evicted() : () -> ()
    return
  }
  // a list in a WebGPU buffer after another one (or after an empty list) joined in a new buffer on the GPU, lists of
  // other formats by strict_gpu_web_mixed, 0.0 for any other pair (strict_list_concat joins those on the host):
  // FluxAttention appends 13 chunks of results this way
  llvm.func @strict_webgpu_joined(i32, i32) -> i32
  // the next line the page hands in with Module.strictInput(text) (WebRuntime.js), waits for it: its length in bytes,
  // then it is copied into a new text
  llvm.func @strict_webinput_length() -> i32
  llvm.func @strict_webinput_copy(!llvm.ptr, i32)
  // true when the first of rows has no numbers on the host: it is only in a WebGPU buffer (strict_rows_web)
  func.func private @strict_web_rows_only(%rows: f64) -> i1 {
    %zero = arith.constant 0.0 : f64
    %no = arith.constant false
    %length = func.call @strict_list_length(%rows) : (f64) -> f64
    %some = arith.cmpf ogt, %length, %zero : f64
    cf.cond_br %some, ^first, ^none
  ^first:
    %items = func.call @strict_list_items(%rows) : (f64) -> !llvm.ptr
    %row = llvm.load %items : !llvm.ptr -> f64
    %view = func.call @strict_ptr(%row) : (f64) -> !llvm.ptr
    %holderSlot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderBits = llvm.load %holderSlot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderBits) : (f64) -> !llvm.ptr
    %dataSlot = llvm.getelementptr %holder[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %data = llvm.load %dataSlot : !llvm.ptr -> f64
    %unhosted = arith.cmpf oeq, %data, %zero : f64
    return %unhosted : i1
  ^none:
    return %no : i1
  }
  // true when the page maps name to a URL in Module.strictFiles (WebRuntime.js): it exists without being fetched
  llvm.func @strict_webfile_mapped(!llvm.ptr) -> i32
  func.func private @strict_web_file_mapped(%name: !llvm.ptr) -> i1 {
    %mapped = llvm.call @strict_webfile_mapped(%name) : (!llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %isMapped = arith.cmpi ne, %mapped, %none : i32
    return %isMapped : i1
  }
  func.func private @strict_web_input_line() -> f64 {
    %length32 = llvm.call @strict_webinput_length() : () -> i32
    %length = arith.extui %length32 : i32 to i64
    %one = arith.constant 1 : i64
    %size = arith.addi %length, %one : i64
    %buffer = func.call @strict_malloc(%size) : (i64) -> !llvm.ptr
    %size32 = arith.trunci %size : i64 to i32
    llvm.call @strict_webinput_copy(%buffer, %size32) : (!llvm.ptr, i32) -> ()
    %text = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %text : f64
  }
  func.func private @strict_gpu_web_joined(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %rightHandle = func.call @strict_gpu_web_handle(%right) : (f64) -> i64
    %isRightKept = arith.cmpi ne, %rightHandle, %none : i64
    cf.cond_br %isRightKept, ^left, ^other
  ^left:
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %isEmpty = arith.cmpf oeq, %leftLength, %zero : f64
    cf.cond_br %isEmpty, ^right, ^both
  ^right:
    return %right : f64
  ^both:
    %leftHandle = func.call @strict_gpu_web_handle(%left) : (f64) -> i64
    %leftBits = func.call @strict_gpu_bits(%left) : (f64) -> i64
    %rightBits = func.call @strict_gpu_bits(%right) : (f64) -> i64
    %isLeftKept = arith.cmpi ne, %leftHandle, %none : i64
    %isSameFormat = arith.cmpi eq, %leftBits, %rightBits : i64
    %isJoinable = arith.andi %isLeftKept, %isSameFormat : i1
    cf.cond_br %isJoinable, ^join, ^mixed
  ^join:
    %leftHandle32 = arith.trunci %leftHandle : i64 to i32
    %rightHandle32 = arith.trunci %rightHandle : i64 to i32
    %joined = llvm.call @strict_webgpu_joined(%leftHandle32, %rightHandle32) : (i32, i32) -> i32
    %failed = arith.constant 0 : i32
    %isJoined = arith.cmpi ne, %joined, %failed : i32
    cf.cond_br %isJoined, ^resides, ^other
  ^resides:
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %total = arith.addf %leftLength, %rightLength : f64
    %count = arith.fptosi %total : f64 to i64
    %buffer = arith.extui %joined : i32 to i64
    %device = arith.shli %buffer, %eight : i64
    %webClass = arith.constant 255 : i64
    %result = func.call @strict_gpu_resident(%count, %device, %webClass, %leftBits) : (i64, i64, i64, i64) -> f64
    return %result : f64
  ^mixed:
    %mixed = func.call @strict_gpu_web_mixed(%left, %right) : (f64, f64) -> f64
    return %mixed : f64
  ^other:
    return %zero : f64
  }
  // two lists of 32 bit or bfloat16 numbers, one of them in a WebGPU buffer, joined into 32 bit floats on the GPU
  // (strict_joined: Moss's 32 bit image tokens and bfloat16 text tokens), 0.0 for any other pair
  llvm.mlir.global internal constant @strict_gpu_joined_web("strict_joined\00") : !llvm.array<14 x i8>
  func.func private @strict_gpu_web_mixed(%left: f64, %right: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %none = arith.constant 0 : i64
    %brainBits = arith.constant 17 : i64
    %singleBits = arith.constant 32 : i64
    %leftHandle = func.call @strict_gpu_web_handle(%left) : (f64) -> i64
    %rightHandle = func.call @strict_gpu_web_handle(%right) : (f64) -> i64
    %handles = arith.ori %leftHandle, %rightHandle : i64
    %anyKept = arith.cmpi ne, %handles, %none : i64
    %leftBits = func.call @strict_gpu_bits(%left) : (f64) -> i64
    %rightBits = func.call @strict_gpu_bits(%right) : (f64) -> i64
    %leftBrain = arith.cmpi eq, %leftBits, %brainBits : i64
    %leftSingle = arith.cmpi eq, %leftBits, %singleBits : i64
    %rightBrain = arith.cmpi eq, %rightBits, %brainBits : i64
    %rightSingle = arith.cmpi eq, %rightBits, %singleBits : i64
    %leftNarrow = arith.ori %leftBrain, %leftSingle : i1
    %rightNarrow = arith.ori %rightBrain, %rightSingle : i1
    %narrow = arith.andi %leftNarrow, %rightNarrow : i1
    %usable = arith.andi %anyKept, %narrow : i1
    cf.cond_br %usable, ^join, ^other
  ^join:
    %two = arith.constant 2 : i64
    %values = llvm.alloca %two x f64 : (i64) -> !llvm.ptr
    llvm.store %left, %values : f64, !llvm.ptr
    %rightSlot = llvm.getelementptr %values[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %right, %rightSlot : f64, !llvm.ptr
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %total = arith.addf %leftLength, %rightLength : f64
    %count = arith.fptosi %total : f64 to i64
    %name = llvm.mlir.addressof @strict_gpu_joined_web : !llvm.ptr
    %kernelKind = arith.constant 2 : i64
    %bothLists = arith.constant 3 : i64
    %joined = func.call @strict_gpu_shaded(%name, %count, %zero, %kernelKind, %values, %two, %bothLists, %singleBits) : (!llvm.ptr, i64, f64, i64, !llvm.ptr, i64, i64, i64) -> f64
    return %joined : f64
  ^other:
    return %zero : f64
  }
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }
  llvm.mlir.global internal constant @strict_gpu_ptx_ending(".ptx\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_gpu_older_ending(".older.ptx\00") : !llvm.array<11 x i8>

  func.func private @strict_process_wait(%self: f64, %seconds: f64) -> f64 {
    %none = arith.constant 0.0 : f64
    return %none : f64
  }
  func.func private @strict_test_pause() {
    return
  }
  func.func private @strict_test_spawn() -> i1 {
    %unavailable = arith.constant false
    return %unavailable : i1
  }

  func.func private @strict_test_quit() {
    %failure = arith.constant 1 : i32
    llvm.call @exit(%failure) : (i32) -> ()
    return
  }
