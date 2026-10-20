  // the heap the collector reserves (its page table and cards are an 8192th and a 512th of it), a pause while spinning
  // (no threads in the browser), fresh memory of the browser is zero
  llvm.mlir.global internal constant @strict_heap_bytes(1073741824 : i64) : i64
  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    llvm.return
  }
  // locals live in wasm, not on the stack in memory: Asyncify unwinds them into memory, they are copied twice (the second
  // copy 4 bytes later, a local of 8 bytes may follow one of 4) so the collector reads every value 8 bytes at a time
  llvm.mlir.global internal @strict_registers(dense<0> : tensor<262144xi64>) : !llvm.array<262144 x i64>
  llvm.mlir.global internal @strict_registers_end(0 : i64) : i64
  llvm.func @emscripten_scan_registers(!llvm.ptr)
  llvm.func @emscripten_stack_get_base() -> i64
  llvm.func @strict_registers_copy(%begin: !llvm.ptr, %end: !llvm.ptr) {
    %start = llvm.ptrtoint %begin : !llvm.ptr to i64
    %finish = llvm.ptrtoint %end : !llvm.ptr to i64
    %length = llvm.sub %finish, %start : i64
    %most = llvm.mlir.constant(1048568 : i64) : i64
    %isLarger = llvm.icmp "ugt" %length, %most : i64
    %bytes = llvm.select %isLarger, %most, %length : i1, i64
    %buffer = llvm.mlir.addressof @strict_registers : !llvm.ptr
    %first = llvm.call @memcpy(%buffer, %begin, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %seven = llvm.mlir.constant(7 : i64) : i64
    %alignMask = llvm.mlir.constant(-8 : i64) : i64
    %rounded = llvm.add %bytes, %seven : i64
    %aligned = llvm.and %rounded, %alignMask : i64
    %four = llvm.mlir.constant(4 : i64) : i64
    %shift = llvm.add %aligned, %four : i64
    %shifted = llvm.getelementptr %buffer[%shift] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %second = llvm.call @memcpy(%shifted, %begin, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %bufferStart = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %copied = llvm.add %shift, %bytes : i64
    %copiedEnd = llvm.add %bufferStart, %copied : i64
    %endSlot = llvm.mlir.addressof @strict_registers_end : !llvm.ptr
    llvm.store %copiedEnd, %endSlot : i64, !llvm.ptr
    llvm.return
  }
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %copy = llvm.mlir.addressof @strict_registers_copy : !llvm.ptr
    llvm.call @emscripten_scan_registers(%copy) : (!llvm.ptr) -> ()
    %buffer = llvm.mlir.addressof @strict_registers : !llvm.ptr
    %start = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %endSlot = llvm.mlir.addressof @strict_registers_end : !llvm.ptr
    %end = llvm.load %endSlot : !llvm.ptr -> i64
    return %start, %end : i64, i64
  }
  func.func private @strict_stack_top(%stack: !llvm.ptr) -> i64 {
    %top = llvm.call @emscripten_stack_get_base() : () -> i64
    return %top : i64
  }
  llvm.func @strict_crash_log(%text: !llvm.ptr) {
    llvm.return
  }
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    %reserved = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    %failed = llvm.icmp "eq" %reserved, %null : !llvm.ptr
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return %reserved : !llvm.ptr
  }
  llvm.func @emscripten_stack_get_end() -> i64
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %lowest = llvm.call @emscripten_stack_get_end() : () -> i64
    return %lowest : i64
  }
  func.func private @strict_commit(%at: !llvm.ptr, %size: i64) {
    return
  }
  func.func private @strict_decommit(%at: !llvm.ptr, %size: i64) {
    %zero = arith.constant 0 : i32
    %cleared = llvm.call @memset(%at, %zero, %size) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    return
  }
  llvm.func @opendir(!llvm.ptr) -> !llvm.ptr
  llvm.func @readdir(!llvm.ptr) -> !llvm.ptr
  llvm.func @closedir(!llvm.ptr) -> i32
  llvm.func @mkdir(!llvm.ptr, i32) -> i32
  llvm.func @fnmatch(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @qsort(!llvm.ptr, i64, i64, !llvm.ptr)
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
  llvm.func @getcwd(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @fdopen(i32, !llvm.ptr) -> !llvm.ptr
  func.func private @strict_process_input(%self: f64) -> f64 {
    %standardInput = arith.constant 0 : i32
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %stream = llvm.call @fdopen(%standardInput, %mode) : (i32, !llvm.ptr) -> !llvm.ptr
    %result = func.call @strict_stream_text(%stream) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_directory_current(%self: f64) -> f64 {
    %size = arith.constant 4096 : i64
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %path : i8, !llvm.ptr
    %found = llvm.call @getcwd(%path, %size) : (!llvm.ptr, i64) -> !llvm.ptr
    %result = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_directory_create(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %mode = arith.constant 511 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %created = llvm.call @mkdir(%name, %mode) : (!llvm.ptr, i32) -> i32
    return %zero : f64
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
    %slotSize = arith.constant 8 : i64
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
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%all) : (f64) -> !llvm.ptr
    %compare = llvm.mlir.addressof @strict_compare_names : !llvm.ptr
    llvm.call @qsort(%items, %count, %slotSize, %compare) : (!llvm.ptr, i64, i64, !llvm.ptr) -> ()
    cf.br ^done(%all : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_directory_directories(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %slotSize = arith.constant 8 : i64
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
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%all) : (f64) -> !llvm.ptr
    %compare = llvm.mlir.addressof @strict_compare_names : !llvm.ptr
    llvm.call @qsort(%items, %count, %slotSize, %compare) : (!llvm.ptr, i64, i64, !llvm.ptr) -> ()
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
    %nanoseconds = llvm.load %nanosecondsSlot : !llvm.ptr -> i64
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
    %secondsSlot = llvm.getelementptr %buffer[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %seconds = llvm.load %secondsSlot : !llvm.ptr -> i64
    %nanosecondsSlot = llvm.getelementptr %buffer[72] : (!llvm.ptr) -> !llvm.ptr, i8
    %nanoseconds = llvm.load %nanosecondsSlot : !llvm.ptr -> i64
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
    %modified = llvm.getelementptr %buffer[64] : (!llvm.ptr) -> !llvm.ptr, i8
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
  func.func private @strict_pool_spawn() -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  llvm.func @strict_pool_sleep(%seen: i64) {
    llvm.return
  }
  llvm.func @strict_pool_wake() {
    llvm.return
  }
  func.func private @strict_gpu_library() -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }