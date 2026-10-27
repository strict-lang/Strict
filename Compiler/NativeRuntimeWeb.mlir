  // the heap the collector reserves (its page table and cards are an 8192th and a 512th of it), a pause while spinning
  // (no threads in the browser), fresh memory of the browser is zero
  llvm.mlir.global internal constant @strict_heap_bytes(1073741824 : i64) : i64
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
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    %reserved = llvm.call @strict_c_malloc(%size) : (i64) -> !llvm.ptr
    %failed = llvm.icmp "eq" %reserved, %null : !llvm.ptr
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return %reserved : !llvm.ptr
  }
  llvm.func @emscripten_stack_get_end() -> !llvm.ptr
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %end = llvm.call @emscripten_stack_get_end() : () -> !llvm.ptr
    %lowest = llvm.ptrtoint %end : !llvm.ptr to i64
    return %lowest : i64
  }
  // committed memory is zero like on the other platforms, pages of the grown heap are not cleared again
  func.func private @strict_commit(%at: !llvm.ptr, %size: i64) {
    %zero = arith.constant 0 : i32
    %cleared = llvm.call @strict_c_memset(%at, %zero, %size) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    return
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
