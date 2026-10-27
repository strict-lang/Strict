  // the heap the collector reserves (its page table and cards are an 8192th and a 512th of it), the pause while spinning
  // and the registers the collector reads are in NativeRuntimeMacOSX64.mlir and NativeRuntimeArm64.mlir
  // all values the collector has to see are on the stack (locals in wasm are not, see NativeRuntimeWeb.mlir)
  // the collector reads possible references 8 bytes at a time (wasm32 keeps pointers in 4 bytes)
  llvm.mlir.global internal constant @strict_word_bytes(8 : i64) : i64
  // a reclaim of objects without references after every MB allocated
  llvm.mlir.global internal constant @strict_reclaim_bytes(1048576 : i64) : i64
  llvm.func @strict_word(%at: i64) -> i64 attributes {passthrough = ["alwaysinline"]} {
    %pointer = llvm.inttoptr %at : i64 to !llvm.ptr
    %word = llvm.load %pointer : !llvm.ptr -> i64
    llvm.return %word : i64
  }
  func.func private @strict_stack_top(%stack: !llvm.ptr) -> i64 {
    %top = llvm.ptrtoint %stack : !llvm.ptr to i64
    return %top : i64
  }
  // snprintf with one or two numbers, its size is 32 bits in wasm32
  llvm.func @strict_print_precise(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %precision: i32, %number: f64) -> i32 {
    %written = llvm.call @snprintf(%buffer, %size, %format, %precision, %number) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, i32, f64) -> i32
    llvm.return %written : i32
  }
  llvm.func @strict_print_number(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %number: f64) -> i32 {
    %written = llvm.call @snprintf(%buffer, %size, %format, %number) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, f64) -> i32
    llvm.return %written : i32
  }
  llvm.func @strict_print_integer(%buffer: !llvm.ptr, %size: i64, %format: !llvm.ptr, %number: i32) -> i32 {
    %written = llvm.call @snprintf(%buffer, %size, %format, %number) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, i32) -> i32
    llvm.return %written : i32
  }
  llvm.func @mmap(!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr
  llvm.func @strict_crash_log(%text: !llvm.ptr) {
    llvm.return
  }
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    %readWrite = arith.constant 3 : i32
    %privateAnonymous = arith.constant 4098 : i32
    %noFile = arith.constant -1 : i32
    %noOffset = arith.constant 0 : i64
    %failure = arith.constant -1 : i64
    %reserved = llvm.call @mmap(%null, %size, %readWrite, %privateAnonymous, %noFile, %noOffset) : (!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr
    %address = llvm.ptrtoint %reserved : !llvm.ptr to i64
    %failed = arith.cmpi eq, %address, %failure : i64
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return %reserved : !llvm.ptr
  }
  llvm.func @pthread_self() -> !llvm.ptr
  llvm.func @pthread_get_stacksize_np(!llvm.ptr) -> i64
  // the main stack has the size it was linked with (-stack_size, RLIMIT_STACK does not show it), at most 64 MB counted
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %most = arith.constant 67108864 : i64
    %thread = llvm.call @pthread_self() : () -> !llvm.ptr
    %current = llvm.call @pthread_get_stacksize_np(%thread) : (!llvm.ptr) -> i64
    %size = arith.minui %current, %most : i64
    %top = llvm.ptrtoint %stack : !llvm.ptr to i64
    %lowest = arith.subi %top, %size : i64
    return %lowest : i64
  }
  func.func private @strict_commit(%at: !llvm.ptr, %size: i64) {
    return
  }
  // madvise does not zero pages on Darwin, mapping them again does
  func.func private @strict_decommit(%at: !llvm.ptr, %size: i64) {
    %readWrite = arith.constant 3 : i32
    %fixedPrivateAnonymous = arith.constant 4114 : i32
    %noFile = arith.constant -1 : i32
    %noOffset = arith.constant 0 : i64
    %mapped = llvm.call @mmap(%at, %size, %readWrite, %fixedPrivateAnonymous, %noFile, %noOffset) : (!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr
    return
  }
  llvm.func @closedir(!llvm.ptr) -> i32
  llvm.func @mkdir(!llvm.ptr, i32) -> i32
  llvm.func @fnmatch(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @qsort(!llvm.ptr, i64, i64, !llvm.ptr)
  llvm.func @strcasecmp(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @getenv(!llvm.ptr) -> !llvm.ptr
  llvm.func @access(!llvm.ptr, i32) -> i32
  llvm.func @mkstemp(!llvm.ptr) -> i32
  llvm.func @close(i32) -> i32
  llvm.func @unlink(!llvm.ptr) -> i32
  llvm.func @fork() -> i32
  llvm.func @execl(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.func @_exit(i32)
  llvm.func @waitpid(i32, !llvm.ptr, i32) -> i32
  llvm.func @kill(i32, i32) -> i32
  llvm.func @usleep(i32) -> i32
  // stat64 has the 64 bit inode layout on Intel and arm64, opendir and readdir differ
  llvm.func @stat64(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @localtime(!llvm.ptr) -> !llvm.ptr
  llvm.mlir.global internal constant @strict_separator_path("/\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_any_file("*\00") : !llvm.array<2 x i8>
  func.func private @strict_directory_exists(%self: f64, %path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %directory = llvm.call @strict_opendir(%name) : (!llvm.ptr) -> !llvm.ptr
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
    %directory = llvm.call @strict_opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %directory, %null : !llvm.ptr
    cf.cond_br %failed, ^done(%created : f64), ^next(%created : f64)
  ^next(%current: f64):
    %entry = llvm.call @strict_readdir(%directory) : (!llvm.ptr) -> !llvm.ptr
    %finished = llvm.icmp "eq" %entry, %null : !llvm.ptr
    cf.cond_br %finished, ^close(%current : f64), ^entry(%current : f64)
  ^entry(%found: f64):
    %typeSlot = llvm.getelementptr %entry[20] : (!llvm.ptr) -> !llvm.ptr, i8
    %type = llvm.load %typeSlot : !llvm.ptr -> i8
    %fileName = llvm.getelementptr %entry[21] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %directory = llvm.call @strict_opendir(%name) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %directory, %null : !llvm.ptr
    cf.cond_br %failed, ^done(%created : f64), ^next(%created : f64)
  ^next(%current: f64):
    %entry = llvm.call @strict_readdir(%directory) : (!llvm.ptr) -> !llvm.ptr
    %finished = llvm.icmp "eq" %entry, %null : !llvm.ptr
    cf.cond_br %finished, ^close(%current : f64), ^entry(%current : f64)
  ^entry(%found: f64):
    %typeSlot = llvm.getelementptr %entry[20] : (!llvm.ptr) -> !llvm.ptr, i8
    %type = llvm.load %typeSlot : !llvm.ptr -> i8
    %fileName = llvm.getelementptr %entry[21] : (!llvm.ptr) -> !llvm.ptr, i8
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
  llvm.mlir.global internal constant @strict_path_variable("PATH\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_shell("/bin/sh\00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_shell_name("sh\00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_shell_command("-c\00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_exec_quote("exec \22\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_quote_space("\22 \00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_output_redirect(" > \22\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_error_redirect("\22 2> \22\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_start_failed("could not start \00") : !llvm.array<17 x i8>
  // tools and downloads stop after 10 minutes (30 s stopped clang in parallel builds and model downloads)
  llvm.mlir.global internal constant @strict_timed_out("timed out after 10 minutes\00") : !llvm.array<27 x i8>
  func.func private @strict_process_operatingsystem(%self: f64) -> f64 {
    %name = llvm.mlir.addressof @strict_macos : !llvm.ptr
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.func @_NSGetExecutablePath(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @realpath(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  func.func private @strict_process_current(%self: f64) -> f64 {
    %one = arith.constant 1 : i64
    %size = arith.constant 4096 : i64
    %limit = arith.constant 4095 : i32
    %terminator = arith.constant 0 : i8
    %launched = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    llvm.store %terminator, %launched : i8, !llvm.ptr
    %sizeSlot = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %limit, %sizeSlot : i32, !llvm.ptr
    %got = llvm.call @_NSGetExecutablePath(%launched, %sizeSlot) : (!llvm.ptr, !llvm.ptr) -> i32
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    llvm.store %terminator, %path : i8, !llvm.ptr
    %resolved = llvm.call @realpath(%launched, %path) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %resolved, %null : !llvm.ptr
    %chosen = arith.select %failed, %launched, %path : !llvm.ptr
    %result = func.call @strict_bits(%chosen) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.func @time(!llvm.ptr) -> i64
  llvm.func @clock_gettime(i32, !llvm.ptr) -> i32
  func.func private @strict_clock() -> f64 {
    %monotonic = arith.constant 6 : i32
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
  llvm.func @sigaction(i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @sigaltstack(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @write(i32, !llvm.ptr, i64) -> i64
  llvm.mlir.global internal constant @strict_overflow_text("Error: StackOverflow, methods called each other without end until the stack was used up\0A\00") : !llvm.array<89 x i8>
  llvm.func @strict_overflow_handler(%signal: i32, %info: !llvm.ptr, %context: !llvm.ptr) {
    %addressSlot = llvm.getelementptr %info[24] : (!llvm.ptr) -> !llvm.ptr, i8
    %address = llvm.load %addressSlot : !llvm.ptr -> i64
    %baseSlot = llvm.mlir.addressof @strict_gc_stack : !llvm.ptr
    %base = llvm.load %baseSlot : !llvm.ptr -> i64
    %below = llvm.sub %base, %address : i64
    %window = llvm.mlir.constant(1073741824 : i64) : i64
    %isStack = llvm.icmp "ult" %below, %window : i64
    llvm.cond_br %isStack, ^report, ^crash
  ^report:
    %null = llvm.mlir.zero : !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    %text = llvm.mlir.addressof @strict_overflow_text : !llvm.ptr
    %length = llvm.mlir.constant(88 : i64) : i64
    %output = llvm.mlir.constant(1 : i32) : i32
    %written = llvm.call @write(%output, %text, %length) : (i32, !llvm.ptr, i64) -> i64
    %code = llvm.mlir.constant(139 : i32) : i32
    llvm.call @_exit(%code) : (i32) -> ()
    llvm.return
  ^crash:
    llvm.return
  }
  func.func private @strict_guard() {
    %one = arith.constant 1 : i64
    %stackBytes = arith.constant 65536 : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %stack = llvm.call @malloc(%stackBytes) : (i64) -> !llvm.ptr
    %alternate = llvm.alloca %one x !llvm.array<24 x i8> : (i64) -> !llvm.ptr
    llvm.store %stack, %alternate : !llvm.ptr, !llvm.ptr
    %stackFlags = llvm.getelementptr %alternate[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %noFlags = arith.constant 0 : i32
    llvm.store %noFlags, %stackFlags : i32, !llvm.ptr
    %stackSize = llvm.getelementptr %alternate[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %stackBytes, %stackSize : i64, !llvm.ptr
    %alternated = llvm.call @sigaltstack(%alternate, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    %action = llvm.alloca %one x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    %actionBytes = arith.constant 16 : i64
    %cleared = llvm.call @memset(%action, %noFlags, %actionBytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    llvm.store %handler, %action : !llvm.ptr, !llvm.ptr
    %actionFlags = llvm.getelementptr %action[12] : (!llvm.ptr) -> !llvm.ptr, i8
    %resetOnStackWithInfo = arith.constant 69 : i32
    llvm.store %resetOnStackWithInfo, %actionFlags : i32, !llvm.ptr
    %segmentation = arith.constant 11 : i32
    %installed = llvm.call @sigaction(%segmentation, %action, %null) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    %bus = arith.constant 10 : i32
    %busInstalled = llvm.call @sigaction(%bus, %action, %null) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    return
  }
  func.func private @strict_environment_put(%setting: !llvm.ptr) {
    %put = llvm.call @putenv(%setting) : (!llvm.ptr) -> i32
    return
  }
  llvm.func @putenv(!llvm.ptr) -> i32
  func.func private @strict_file_size(%name: !llvm.ptr) -> i64 {
    %size = arith.constant 256 : i64
    %buffer = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
    %status = llvm.call @stat64(%name, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %zero = arith.constant 0 : i64
    cf.cond_br %failed, ^done(%zero : i64), ^size
  ^size:
    %sizeSlot = llvm.getelementptr %buffer[96] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %status = llvm.call @stat64(%name, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %zero = arith.constant 0.0 : f64
    cf.cond_br %failed, ^done(%zero : f64), ^time
  ^time:
    %secondsSlot = llvm.getelementptr %buffer[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %seconds = llvm.load %secondsSlot : !llvm.ptr -> i64
    %nanosecondsSlot = llvm.getelementptr %buffer[56] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %status = llvm.call @stat64(%path, %buffer) : (!llvm.ptr, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %failed = arith.cmpi ne, %status, %none : i32
    %null = llvm.mlir.zero : !llvm.ptr
    cf.cond_br %failed, ^done(%null : !llvm.ptr), ^time
  ^time:
    %modified = llvm.getelementptr %buffer[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %local = llvm.call @localtime(%modified) : (!llvm.ptr) -> !llvm.ptr
    cf.br ^done(%local : !llvm.ptr)
  ^done(%time: !llvm.ptr):
    %result = func.call @strict_time_text(%time) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_tool_path(%name: f64) -> f64 {
    %executable = arith.constant 1 : i32
    %ok = arith.constant 0 : i32
    %slash = arith.constant 47 : i32
    %colon = arith.constant 58 : i32
    %noBytes = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %slashCharacter = arith.constant 47 : i8
    %terminator = arith.constant 0 : i8
    %file = func.call @strict_ptr(%name) : (f64) -> !llvm.ptr
    %fileSize = llvm.call @strlen(%file) : (!llvm.ptr) -> i64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %null = llvm.mlir.zero : !llvm.ptr
    %slashAt = llvm.call @strchr(%file, %slash) : (!llvm.ptr, i32) -> !llvm.ptr
    %isPath = llvm.icmp "ne" %slashAt, %null : !llvm.ptr
    cf.cond_br %isPath, ^direct, ^search
  ^direct:
    %access = llvm.call @access(%file, %executable) : (!llvm.ptr, i32) -> i32
    %usable = arith.cmpi eq, %access, %ok : i32
    %direct = arith.select %usable, %name, %empty : f64
    return %direct : f64
  ^search:
    %variable = llvm.mlir.addressof @strict_path_variable : !llvm.ptr
    %paths = llvm.call @getenv(%variable) : (!llvm.ptr) -> !llvm.ptr
    %noPath = llvm.icmp "eq" %paths, %null : !llvm.ptr
    cf.cond_br %noPath, ^missing, ^segment(%paths : !llvm.ptr)
  ^segment(%start: !llvm.ptr):
    %end = llvm.call @strchr(%start, %colon) : (!llvm.ptr, i32) -> !llvm.ptr
    %isLast = llvm.icmp "eq" %end, %null : !llvm.ptr
    %restSize = llvm.call @strlen(%start) : (!llvm.ptr) -> i64
    %startAddress = llvm.ptrtoint %start : !llvm.ptr to i64
    %endAddress = llvm.ptrtoint %end : !llvm.ptr to i64
    %segmentSize = arith.subi %endAddress, %startAddress : i64
    %size = arith.select %isLast, %restSize, %segmentSize : i64
    %withName = arith.addi %size, %fileSize : i64
    %bytes = arith.addi %withName, %two : i64
    %candidate = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %copiedDirectory = llvm.call @memcpy(%candidate, %start, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %slashSlot = llvm.getelementptr %candidate[%size] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %slashCharacter, %slashSlot : i8, !llvm.ptr
    %nameSlot = llvm.getelementptr %slashSlot[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %copiedName = llvm.call @memcpy(%nameSlot, %file, %fileSize) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %terminatorSlot = llvm.getelementptr %nameSlot[%fileSize] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %terminatorSlot : i8, !llvm.ptr
    %found = llvm.call @access(%candidate, %executable) : (!llvm.ptr, i32) -> i32
    %isFound = arith.cmpi eq, %found, %ok : i32
    %notEmpty = arith.cmpi ne, %size, %noBytes : i64
    %usableCandidate = arith.andi %isFound, %notEmpty : i1
    cf.cond_br %usableCandidate, ^located(%candidate : !llvm.ptr), ^advance
  ^advance:
    %following = llvm.getelementptr %end[1] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.cond_br %isLast, ^missing, ^segment(%following : !llvm.ptr)
  ^located(%path: !llvm.ptr):
    %located = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %located : f64
  ^missing:
    return %empty : f64
  }
  func.func private @strict_removed_file(%path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %text = func.call @strict_file_text(%path) : (f64) -> f64
    %deleted = llvm.call @unlink(%name) : (!llvm.ptr) -> i32
    return %text : f64
  }
  func.func private @strict_text_of(%global: !llvm.ptr) -> f64 {
    %bits = func.call @strict_bits(%global) : (!llvm.ptr) -> f64
    return %bits : f64
  }
  llvm.func @system(!llvm.ptr) -> i32
  // runs a program with the console of this one (its output appears right away) and returns its exit code
  func.func private @strict_process_started(%path: f64, %arguments: f64) -> f64 {
    %eight = arith.constant 8 : i32
    %byte = arith.constant 255 : i32
    %signals = arith.constant 127 : i32
    %none = arith.constant 0 : i32
    %signaled = arith.constant 128 : i32
    %execQuote = llvm.mlir.addressof @strict_exec_quote : !llvm.ptr
    %quoteSpace = llvm.mlir.addressof @strict_quote_space : !llvm.ptr
    %execText = func.call @strict_text_of(%execQuote) : (!llvm.ptr) -> f64
    %quoteSpaceText = func.call @strict_text_of(%quoteSpace) : (!llvm.ptr) -> f64
    %quoted = func.call @strict_text_concat(%execText, %path) : (f64, f64) -> f64
    %program = func.call @strict_text_concat(%quoted, %quoteSpaceText) : (f64, f64) -> f64
    %command = func.call @strict_text_concat(%program, %arguments) : (f64, f64) -> f64
    %commandText = func.call @strict_ptr(%command) : (f64) -> !llvm.ptr
    func.call @strict_flush() : () -> ()
    %status = llvm.call @system(%commandText) : (!llvm.ptr) -> i32
    %signal = arith.andi %status, %signals : i32
    %shifted = arith.shrui %status, %eight : i32
    %exited = arith.andi %shifted, %byte : i32
    %isSignaled = arith.cmpi ne, %signal, %none : i32
    %killed = arith.addi %signal, %signaled : i32
    %code = arith.select %isSignaled, %killed, %exited : i32
    %number = arith.sitofp %code : i32 to f64
    return %number : f64
  }
  func.func private @strict_process_execute(%path: f64, %arguments: f64) -> f64 {
    %none = arith.constant 0 : i32
    %noHang = arith.constant 1 : i32
    %killSignal = arith.constant 9 : i32
    %failure = arith.constant 127 : i32
    %millisecond = arith.constant 1000 : i32
    %limit = arith.constant 600000 : i32
    %step = arith.constant 1 : i32
    %statusSize = arith.constant 8 : i64
    %execQuote = llvm.mlir.addressof @strict_exec_quote : !llvm.ptr
    %quoteSpace = llvm.mlir.addressof @strict_quote_space : !llvm.ptr
    %outputRedirect = llvm.mlir.addressof @strict_output_redirect : !llvm.ptr
    %errorRedirect = llvm.mlir.addressof @strict_error_redirect : !llvm.ptr
    %quote = llvm.mlir.addressof @strict_quote : !llvm.ptr
    %execText = func.call @strict_text_of(%execQuote) : (!llvm.ptr) -> f64
    %quoteSpaceText = func.call @strict_text_of(%quoteSpace) : (!llvm.ptr) -> f64
    %outputText = func.call @strict_text_of(%outputRedirect) : (!llvm.ptr) -> f64
    %errorText = func.call @strict_text_of(%errorRedirect) : (!llvm.ptr) -> f64
    %quoteText = func.call @strict_text_of(%quote) : (!llvm.ptr) -> f64
    %outputPath = func.call @strict_temp_file() : () -> f64
    %errorPath = func.call @strict_temp_file() : () -> f64
    %quoted = func.call @strict_text_concat(%execText, %path) : (f64, f64) -> f64
    %program = func.call @strict_text_concat(%quoted, %quoteSpaceText) : (f64, f64) -> f64
    %called = func.call @strict_text_concat(%program, %arguments) : (f64, f64) -> f64
    %toOutput = func.call @strict_text_concat(%called, %outputText) : (f64, f64) -> f64
    %withOutput = func.call @strict_text_concat(%toOutput, %outputPath) : (f64, f64) -> f64
    %toError = func.call @strict_text_concat(%withOutput, %errorText) : (f64, f64) -> f64
    %withError = func.call @strict_text_concat(%toError, %errorPath) : (f64, f64) -> f64
    %command = func.call @strict_text_concat(%withError, %quoteText) : (f64, f64) -> f64
    %commandText = func.call @strict_ptr(%command) : (f64) -> !llvm.ptr
    %shell = llvm.mlir.addressof @strict_shell : !llvm.ptr
    %shellName = llvm.mlir.addressof @strict_shell_name : !llvm.ptr
    %shellCommand = llvm.mlir.addressof @strict_shell_command : !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %status = func.call @strict_alloc_text(%statusSize) : (i64) -> !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    %child = llvm.call @fork() : () -> i32
    %isChild = arith.cmpi eq, %child, %none : i32
    cf.cond_br %isChild, ^child, ^parent
  ^child:
    %executed = llvm.call @execl(%shell, %shellName, %shellCommand, %commandText, %null) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @_exit(%failure) : (i32) -> ()
    llvm.unreachable
  ^parent:
    %notStarted = arith.cmpi slt, %child, %none : i32
    cf.cond_br %notStarted, ^startFailed, ^wait(%none : i32)
  ^startFailed:
    %unusedOutput = func.call @strict_removed_file(%outputPath) : (f64) -> f64
    %unusedError = func.call @strict_removed_file(%errorPath) : (f64) -> f64
    %startCode = arith.constant 1.0 : f64
    %startText = llvm.mlir.addressof @strict_start_failed : !llvm.ptr
    %startPrefix = func.call @strict_bits(%startText) : (!llvm.ptr) -> f64
    %startError = func.call @strict_text_concat(%startPrefix, %path) : (f64, f64) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %startFailure = func.call @strict_process_result(%startCode, %empty, %startError) : (f64, f64, f64) -> f64
    return %startFailure : f64
  ^wait(%waited: i32):
    %ended = llvm.call @waitpid(%child, %status, %noHang) : (i32, !llvm.ptr, i32) -> i32
    %hasEnded = arith.cmpi eq, %ended, %child : i32
    cf.cond_br %hasEnded, ^exited, ^check(%waited : i32)
  ^check(%elapsed: i32):
    %timedOut = arith.cmpi sge, %elapsed, %limit : i32
    cf.cond_br %timedOut, ^kill, ^sleep(%elapsed : i32)
  ^sleep(%sleeping: i32):
    %slept = llvm.call @usleep(%millisecond) : (i32) -> i32
    %later = arith.addi %sleeping, %step : i32
    cf.br ^wait(%later : i32)
  ^kill:
    %killed = llvm.call @kill(%child, %killSignal) : (i32, i32) -> i32
    %reaped = llvm.call @waitpid(%child, %status, %none) : (i32, !llvm.ptr, i32) -> i32
    %timeoutCode = arith.constant 124.0 : f64
    %timedOutText = llvm.mlir.addressof @strict_timed_out : !llvm.ptr
    %timedOutMessage = func.call @strict_bits(%timedOutText) : (!llvm.ptr) -> f64
    cf.br ^collect(%timeoutCode, %timedOutMessage, %step : f64, f64, i32)
  ^exited:
    %rawStatus = llvm.load %status : !llvm.ptr -> i32
    %eight = arith.constant 8 : i32
    %byteMask = arith.constant 255 : i32
    %signalMask = arith.constant 127 : i32
    %signalBase = arith.constant 128 : i32
    %shifted = arith.shrui %rawStatus, %eight : i32
    %exitCode = arith.andi %shifted, %byteMask : i32
    %signal = arith.andi %rawStatus, %signalMask : i32
    %wasSignaled = arith.cmpi ne, %signal, %none : i32
    %signalCode = arith.addi %signalBase, %signal : i32
    %code = arith.select %wasSignaled, %signalCode, %exitCode : i32
    %exitNumber = arith.sitofp %code : i32 to f64
    %noMessage = arith.constant 0.0 : f64
    cf.br ^collect(%exitNumber, %noMessage, %none : f64, f64, i32)
  ^collect(%result: f64, %override: f64, %hasOverride: i32):
    %written = func.call @strict_removed_file(%outputPath) : (f64) -> f64
    %reported = func.call @strict_removed_file(%errorPath) : (f64) -> f64
    %useOverride = arith.cmpi ne, %hasOverride, %none : i32
    %message = arith.select %useOverride, %override, %reported : f64
    %processResult = func.call @strict_process_result(%result, %written, %message) : (f64, f64, f64) -> f64
    return %processResult : f64
  }
  llvm.func @dlopen(!llvm.ptr, i32) -> !llvm.ptr
  llvm.func @dlsym(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  func.func private @strict_library_call(%path: f64, %symbol: f64, %arguments: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %now = arith.constant 2 : i32
    %library = llvm.call @dlopen(%name, %now) : (!llvm.ptr, i32) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %isMissing = llvm.icmp "eq" %library, %null : !llvm.ptr
    cf.cond_br %isMissing, ^noLibrary, ^loaded
  ^noLibrary:
    %libraryMessage = llvm.mlir.addressof @strict_library_missing : !llvm.ptr
    %libraryError = func.call @strict_error_named(%libraryMessage) : (!llvm.ptr) -> f64
    return %libraryError : f64
  ^loaded:
    %symbolName = func.call @strict_ptr(%symbol) : (f64) -> !llvm.ptr
    %function = llvm.call @dlsym(%library, %symbolName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %isAbsent = llvm.icmp "eq" %function, %null : !llvm.ptr
    cf.cond_br %isAbsent, ^noSymbol, ^found
  ^noSymbol:
    %symbolMessage = llvm.mlir.addressof @strict_library_symbol_missing : !llvm.ptr
    %symbolError = func.call @strict_error_named(%symbolMessage) : (!llvm.ptr) -> f64
    return %symbolError : f64
  ^found:
    %result = func.call @strict_library_invoke(%function, %arguments) : (!llvm.ptr, f64) -> f64
    %number = func.call @strict_canonical(%result) : (f64) -> f64
    return %number : f64
  }
  llvm.func @sysconf(i32) -> i64
  llvm.func @pthread_attr_init(!llvm.ptr) -> i32
  llvm.func @pthread_attr_setstacksize(!llvm.ptr, i64) -> i32
  llvm.func @pthread_create(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_mutex_lock(!llvm.ptr) -> i32
  llvm.func @pthread_mutex_unlock(!llvm.ptr) -> i32
  llvm.func @pthread_cond_wait(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_cond_broadcast(!llvm.ptr) -> i32
  // PTHREAD_MUTEX_INITIALIZER and PTHREAD_COND_INITIALIZER start with their signature on Darwin
  llvm.mlir.global internal @strict_pool_lock(dense<[850045863, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>) : !llvm.array<8 x i64>
  llvm.mlir.global internal @strict_pool_signal(dense<[1018212795, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>) : !llvm.array<8 x i64>
  llvm.func @strict_pool_thread(%argument: !llvm.ptr) -> !llvm.ptr {
    llvm.call @strict_pool_serve() : () -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  }
  llvm.func @strict_gpu_thread(%argument: !llvm.ptr) -> !llvm.ptr {
    func.call @strict_gpu_warming() : () -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  }
  func.func private @strict_gpu_spawn() -> i1 {
    %zero32 = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_gpu_thread : !llvm.ptr
    %handle = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %created = llvm.call @pthread_create(%handle, %null, %procedure, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %spawned = arith.cmpi eq, %created, %zero32 : i32
    return %spawned : i1
  }
  // bytes of physical memory, 0 when unknown
  func.func private @strict_host_memory() -> i64 {
    %pagesName = arith.constant 200 : i32
    %sizeName = arith.constant 29 : i32
    %pages = llvm.call @sysconf(%pagesName) : (i32) -> i64
    %size = llvm.call @sysconf(%sizeName) : (i32) -> i64
    %zero = arith.constant 0 : i64
    %product = arith.muli %pages, %size : i64
    %bytes = arith.maxsi %product, %zero : i64
    return %bytes : i64
  }
  func.func private @strict_pool_spawn() -> i64 {
    %online = arith.constant 58 : i32
    %processors = llvm.call @sysconf(%online) : (i32) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    // a quarter of the logical processors was fastest for memory bound GPT-2 decoding (13900KF), compute bound loops could use all
    %four = arith.constant 4 : i64
    %two = arith.constant 2 : i64
    %quarter = arith.divsi %processors, %four : i64
    %standard = arith.maxsi %quarter, %two : i64
    %threads = func.call @strict_pool_size(%standard) : (i64) -> i64
    %workers = arith.subi %threads, %one : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_pool_thread : !llvm.ptr
    %handle = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %attributes = llvm.alloca %one x !llvm.array<8 x i64> : (i64) -> !llvm.ptr
    %initialized = llvm.call @pthread_attr_init(%attributes) : (!llvm.ptr) -> i32
    // 64 MB per worker like the main thread and Windows (the default is 512 KB)
    %stackBytes = arith.constant 67108864 : i64
    %sized = llvm.call @pthread_attr_setstacksize(%attributes, %stackBytes) : (!llvm.ptr, i64) -> i32
    cf.br ^check(%zero : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %workers : i64
    cf.cond_br %more, ^spawn, ^done
  ^spawn:
    %created = llvm.call @pthread_create(%handle, %attributes, %procedure, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %next = arith.addi %index, %one : i64
    cf.br ^check(%next : i64)
  ^done:
    return %workers : i64
  }
  llvm.func @strict_pool_sleep(%seen: i64) {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_signal : !llvm.ptr
    %jobSlot = llvm.mlir.addressof @strict_pool_job : !llvm.ptr
    %sleepersSlot = llvm.mlir.addressof @strict_pool_sleepers : !llvm.ptr
    %one = llvm.mlir.constant(1 : i64) : i64
    %locked = llvm.call @pthread_mutex_lock(%lock) : (!llvm.ptr) -> i32
    %sleeping = llvm.atomicrmw add %sleepersSlot, %one seq_cst : !llvm.ptr, i64
    llvm.br ^check
  ^check:
    %current = llvm.load %jobSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %same = llvm.icmp "eq" %current, %seen : i64
    llvm.cond_br %same, ^sleep, ^leave
  ^sleep:
    %waited = llvm.call @pthread_cond_wait(%signal, %lock) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.br ^check
  ^leave:
    %awake = llvm.atomicrmw sub %sleepersSlot, %one seq_cst : !llvm.ptr, i64
    %unlocked = llvm.call @pthread_mutex_unlock(%lock) : (!llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @strict_pool_wake() {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_signal : !llvm.ptr
    %sleepersSlot = llvm.mlir.addressof @strict_pool_sleepers : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %sleepers = llvm.load %sleepersSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %anyone = llvm.icmp "ne" %sleepers, %zero : i64
    llvm.cond_br %anyone, ^wake, ^leave
  ^wake:
    %locked = llvm.call @pthread_mutex_lock(%lock) : (!llvm.ptr) -> i32
    %unlocked = llvm.call @pthread_mutex_unlock(%lock) : (!llvm.ptr) -> i32
    %woken = llvm.call @pthread_cond_broadcast(%signal) : (!llvm.ptr) -> i32
    llvm.return
  ^leave:
    llvm.return
  }
  llvm.mlir.global internal constant @strict_gpu_ptx_ending(".metal\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_gpu_older_ending(".older.metal\00") : !llvm.array<13 x i8>
  llvm.mlir.global internal constant @strict_metal_s_objectAtIndex("objectAtIndex:\00") : !llvm.array<15 x i8>
  llvm.mlir.global internal constant @strict_metal_s_newCommandQueue("newCommandQueue\00") : !llvm.array<16 x i8>
  llvm.mlir.global internal constant @strict_metal_s_recommendedMaxWorkingSetSize("recommendedMaxWorkingSetSize\00") : !llvm.array<29 x i8>
  llvm.mlir.global internal constant @strict_metal_s_maxBufferLength("maxBufferLength\00") : !llvm.array<16 x i8>
  llvm.mlir.global internal constant @strict_metal_s_newBufferWithLength_options("newBufferWithLength:options:\00") : !llvm.array<29 x i8>
  llvm.mlir.global internal constant @strict_metal_s_contents("contents\00") : !llvm.array<9 x i8>
  llvm.mlir.global internal constant @strict_metal_s_gpuAddress("gpuAddress\00") : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @strict_metal_s_stringWithUTF8String("stringWithUTF8String:\00") : !llvm.array<22 x i8>
  llvm.mlir.global internal constant @strict_metal_s_new("new\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_metal_s_setFastMathEnabled("setFastMathEnabled:\00") : !llvm.array<20 x i8>
  llvm.mlir.global internal constant @strict_metal_s_newLibraryWithSource_options_error("newLibraryWithSource:options:error:\00") : !llvm.array<36 x i8>
  llvm.mlir.global internal constant @strict_metal_s_newFunctionWithName("newFunctionWithName:\00") : !llvm.array<21 x i8>
  llvm.mlir.global internal constant @strict_metal_s_newComputePipelineStateWithFunction_error("newComputePipelineStateWithFunction:error:\00") : !llvm.array<43 x i8>
  llvm.mlir.global internal constant @strict_metal_s_commandBuffer("commandBuffer\00") : !llvm.array<14 x i8>
  llvm.mlir.global internal constant @strict_metal_s_computeCommandEncoder("computeCommandEncoder\00") : !llvm.array<22 x i8>
  llvm.mlir.global internal constant @strict_metal_s_setComputePipelineState("setComputePipelineState:\00") : !llvm.array<25 x i8>
  llvm.mlir.global internal constant @strict_metal_s_setBytes_length_atIndex("setBytes:length:atIndex:\00") : !llvm.array<25 x i8>
  llvm.mlir.global internal constant @strict_metal_s_useResource_usage("useResource:usage:\00") : !llvm.array<19 x i8>
  llvm.mlir.global internal constant @strict_metal_s_dispatchThreadgroups_threadsPerThreadgroup("dispatchThreadgroups:threadsPerThreadgroup:\00") : !llvm.array<44 x i8>
  llvm.mlir.global internal constant @strict_metal_s_endEncoding("endEncoding\00") : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @strict_metal_s_commit("commit\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_metal_s_count("count\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @strict_metal_s_respondsToSelector("respondsToSelector:\00") : !llvm.array<20 x i8>
  llvm.mlir.global internal constant @strict_metal_s_retain("retain\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_metal_s_release("release\00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_metal_s_waitUntilCompleted("waitUntilCompleted\00") : !llvm.array<19 x i8>
  llvm.mlir.global internal constant @strict_metal_s_status("status\00") : !llvm.array<7 x i8>
  llvm.mlir.global internal constant @strict_metal_s_localizedDescription("localizedDescription\00") : !llvm.array<21 x i8>
  llvm.mlir.global internal constant @strict_metal_s_UTF8String("UTF8String\00") : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @strict_metal_n_library("/System/Library/Frameworks/Metal.framework/Metal\00") : !llvm.array<49 x i8>
  llvm.mlir.global internal constant @strict_metal_n_send("objc_msgSend\00") : !llvm.array<13 x i8>
  llvm.mlir.global internal constant @strict_metal_n_register("sel_registerName\00") : !llvm.array<17 x i8>
  llvm.mlir.global internal constant @strict_metal_n_class("objc_getClass\00") : !llvm.array<14 x i8>
  llvm.mlir.global internal constant @strict_metal_n_push("objc_autoreleasePoolPush\00") : !llvm.array<25 x i8>
  llvm.mlir.global internal constant @strict_metal_n_pop("objc_autoreleasePoolPop\00") : !llvm.array<24 x i8>
  llvm.mlir.global internal constant @strict_metal_n_devices("MTLCopyAllDevices\00") : !llvm.array<18 x i8>
  llvm.mlir.global internal constant @strict_metal_n_string("NSString\00") : !llvm.array<9 x i8>
  llvm.mlir.global internal constant @strict_metal_n_options("MTLCompileOptions\00") : !llvm.array<18 x i8>
  llvm.mlir.global internal constant @strict_metal_marker("// parameters %s \00") : !llvm.array<18 x i8>
  llvm.mlir.global internal constant @strict_metal_failed("Metal: \00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_metal_newline("\0A\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_metal_command_failed("Metal: a command buffer failed\0A\00") : !llvm.array<32 x i8>
  // the CUDA driver functions NativeRuntime.mlir calls, done with Metal (macOS has no CUDA): kernels are the Metal
  // source MetalShader.strict writes next to the executable, device memory is one shared buffer (addresses are its
  // gpuAddress plus an offset, freed all at once when nothing is allocated anymore), launches run right away in one
  // queue (captured graphs are empty), copies wait for the last command buffer
  // slots: objc_msgSend, sel_registerName, objc_getClass, autorelease push and pop, device, queue, buffer, contents,
  // gpu address, size, top, live allocations, last command buffer, Metal library
  llvm.mlir.global internal @strict_metal(dense<0> : tensor<16xi64>) : !llvm.array<16 x i64>
  llvm.func @strict_metal_get(%index: i64) -> i64 {
    %state = llvm.mlir.addressof @strict_metal : !llvm.ptr
    %slot = llvm.getelementptr %state[0, %index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i64>
    %value = llvm.load %slot : !llvm.ptr -> i64
    llvm.return %value : i64
  }
  llvm.func @strict_metal_put(%index: i64, %value: i64) {
    %state = llvm.mlir.addressof @strict_metal : !llvm.ptr
    %slot = llvm.getelementptr %state[0, %index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i64>
    llvm.store %value, %slot : i64, !llvm.ptr
    llvm.return
  }
  llvm.func @strict_metal_pointer(%index: i64) -> !llvm.ptr {
    %value = llvm.call @strict_metal_get(%index) : (i64) -> i64
    %pointer = llvm.inttoptr %value : i64 to !llvm.ptr
    llvm.return %pointer : !llvm.ptr
  }
  llvm.func @strict_metal_selector(%name: !llvm.ptr) -> !llvm.ptr {
    %one = llvm.mlir.constant(1 : i64) : i64
    %register = llvm.call @strict_metal_pointer(%one) : (i64) -> !llvm.ptr
    %selector = llvm.call %register(%name) : !llvm.ptr, (!llvm.ptr) -> !llvm.ptr
    llvm.return %selector : !llvm.ptr
  }
  llvm.func @strict_metal_class(%name: !llvm.ptr) -> !llvm.ptr {
    %two = llvm.mlir.constant(2 : i64) : i64
    %lookup = llvm.call @strict_metal_pointer(%two) : (i64) -> !llvm.ptr
    %class = llvm.call %lookup(%name) : !llvm.ptr, (!llvm.ptr) -> !llvm.ptr
    llvm.return %class : !llvm.ptr
  }
  // a message without arguments that returns an object, a pointer or an integer (all come back in the same register)
  llvm.func @strict_metal_message(%object: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %selector = llvm.call @strict_metal_selector(%name) : (!llvm.ptr) -> !llvm.ptr
    %result = llvm.call %send(%object, %selector) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_metal_number(%object: !llvm.ptr, %name: !llvm.ptr) -> i64 {
    %result = llvm.call @strict_metal_message(%object, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %number = llvm.ptrtoint %result : !llvm.ptr to i64
    llvm.return %number : i64
  }
  llvm.func @strict_metal_with(%object: !llvm.ptr, %name: !llvm.ptr, %argument: !llvm.ptr) -> !llvm.ptr {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %selector = llvm.call @strict_metal_selector(%name) : (!llvm.ptr) -> !llvm.ptr
    %result = llvm.call %send(%object, %selector, %argument) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.return %result : !llvm.ptr
  }
  llvm.func @strict_metal_text(%text: !llvm.ptr) -> !llvm.ptr {
    %className = llvm.mlir.addressof @strict_metal_n_string : !llvm.ptr
    %class = llvm.call @strict_metal_class(%className) : (!llvm.ptr) -> !llvm.ptr
    %name = llvm.mlir.addressof @strict_metal_s_stringWithUTF8String : !llvm.ptr
    %string = llvm.call @strict_metal_with(%class, %name, %text) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.return %string : !llvm.ptr
  }
  llvm.func @strict_metal_push() -> !llvm.ptr {
    %three = llvm.mlir.constant(3 : i64) : i64
    %push = llvm.call @strict_metal_pointer(%three) : (i64) -> !llvm.ptr
    %pool = llvm.call %push() : !llvm.ptr, () -> !llvm.ptr
    llvm.return %pool : !llvm.ptr
  }
  llvm.func @strict_metal_pop(%pool: !llvm.ptr) {
    %four = llvm.mlir.constant(4 : i64) : i64
    %pop = llvm.call @strict_metal_pointer(%four) : (i64) -> !llvm.ptr
    llvm.call %pop(%pool) : !llvm.ptr, (!llvm.ptr) -> ()
    llvm.return
  }
  // writes "Metal: " and the description of an NSError to the error output
  llvm.func @strict_metal_report(%error: !llvm.ptr) {
    %output = llvm.mlir.constant(2 : i32) : i32
    %prefix = llvm.mlir.addressof @strict_metal_failed : !llvm.ptr
    %prefixLength = llvm.call @strlen(%prefix) : (!llvm.ptr) -> i64
    %wrotePrefix = llvm.call @write(%output, %prefix, %prefixLength) : (i32, !llvm.ptr, i64) -> i64
    %null = llvm.mlir.zero : !llvm.ptr
    %hasError = llvm.icmp "ne" %error, %null : !llvm.ptr
    llvm.cond_br %hasError, ^described, ^done
  ^described:
    %describe = llvm.mlir.addressof @strict_metal_s_localizedDescription : !llvm.ptr
    %description = llvm.call @strict_metal_message(%error, %describe) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %utf8 = llvm.mlir.addressof @strict_metal_s_UTF8String : !llvm.ptr
    %text = llvm.call @strict_metal_message(%description, %utf8) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %length = llvm.call @strlen(%text) : (!llvm.ptr) -> i64
    %wrote = llvm.call @write(%output, %text, %length) : (i32, !llvm.ptr, i64) -> i64
    llvm.br ^done
  ^done:
    %newline = llvm.mlir.addressof @strict_metal_newline : !llvm.ptr
    %one = llvm.mlir.constant(1 : i64) : i64
    %ended = llvm.call @write(%output, %newline, %one) : (i32, !llvm.ptr, i64) -> i64
    llvm.return
  }
  llvm.func @strict_metal_host(%device: i64) -> !llvm.ptr {
    %eight = llvm.mlir.constant(8 : i64) : i64
    %nine = llvm.mlir.constant(9 : i64) : i64
    %contents = llvm.call @strict_metal_pointer(%eight) : (i64) -> !llvm.ptr
    %base = llvm.call @strict_metal_get(%nine) : (i64) -> i64
    %offset = llvm.sub %device, %base : i64
    %host = llvm.getelementptr %contents[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.return %host : !llvm.ptr
  }
  // waits for the last command buffer, the queue runs them in order
  llvm.func @strict_metal_wait() -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %failure = llvm.mlir.constant(1 : i32) : i32
    %lastIndex = llvm.mlir.constant(13 : i64) : i64
    %last = llvm.call @strict_metal_pointer(%lastIndex) : (i64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %none = llvm.icmp "eq" %last, %null : !llvm.ptr
    llvm.cond_br %none, ^done(%success : i32), ^wait
  ^wait:
    %waitName = llvm.mlir.addressof @strict_metal_s_waitUntilCompleted : !llvm.ptr
    %waited = llvm.call @strict_metal_message(%last, %waitName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %statusName = llvm.mlir.addressof @strict_metal_s_status : !llvm.ptr
    %status = llvm.call @strict_metal_number(%last, %statusName) : (!llvm.ptr, !llvm.ptr) -> i64
    %errorStatus = llvm.mlir.constant(5 : i64) : i64
    %failed = llvm.icmp "eq" %status, %errorStatus : i64
    llvm.cond_br %failed, ^report, ^done(%success : i32)
  ^report:
    %output = llvm.mlir.constant(2 : i32) : i32
    %message = llvm.mlir.addressof @strict_metal_command_failed : !llvm.ptr
    %length = llvm.call @strlen(%message) : (!llvm.ptr) -> i64
    %wrote = llvm.call @write(%output, %message, %length) : (i32, !llvm.ptr, i64) -> i64
    llvm.br ^done(%failure : i32)
  ^done(%result: i32):
    llvm.return %result : i32
  }
  llvm.func @strict_metal_succeeded() -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    llvm.return %success : i32
  }
  llvm.func @strict_metal_cuInit(%flags: i32) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuDeviceGet(%device: !llvm.ptr, %ordinal: i32) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.store %zero, %device : i32, !llvm.ptr
    llvm.return %zero : i32
  }
  // the first Metal device (MTLCreateSystemDefaultDevice has none without a window server connection), its queue and
  // a shared buffer of a quarter of its recommended working set, Metal 3 GPUs only (gpuAddress)
  llvm.func @strict_metal_cuDevicePrimaryCtxRetain(%context: !llvm.ptr, %ordinal: i32) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %noDevice = llvm.mlir.constant(100 : i32) : i32
    %zero = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %libraryIndex = llvm.mlir.constant(14 : i64) : i64
    %library = llvm.call @strict_metal_pointer(%libraryIndex) : (i64) -> !llvm.ptr
    %copyName = llvm.mlir.addressof @strict_metal_n_devices : !llvm.ptr
    %copy = llvm.call @dlsym(%library, %copyName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %noCopy = llvm.icmp "eq" %copy, %null : !llvm.ptr
    llvm.cond_br %noCopy, ^failed, ^copy
  ^copy:
    %devices = llvm.call %copy() : !llvm.ptr, () -> !llvm.ptr
    %countName = llvm.mlir.addressof @strict_metal_s_count : !llvm.ptr
    %count = llvm.call @strict_metal_number(%devices, %countName) : (!llvm.ptr, !llvm.ptr) -> i64
    %empty = llvm.icmp "eq" %count, %zero : i64
    llvm.cond_br %empty, ^failed, ^first
  ^first:
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %atName = llvm.mlir.addressof @strict_metal_s_objectAtIndex : !llvm.ptr
    %at = llvm.call @strict_metal_selector(%atName) : (!llvm.ptr) -> !llvm.ptr
    %device = llvm.call %send(%devices, %at, %zero) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %respondsName = llvm.mlir.addressof @strict_metal_s_respondsToSelector : !llvm.ptr
    %responds = llvm.call @strict_metal_selector(%respondsName) : (!llvm.ptr) -> !llvm.ptr
    %addressName = llvm.mlir.addressof @strict_metal_s_gpuAddress : !llvm.ptr
    %addressSelector = llvm.call @strict_metal_selector(%addressName) : (!llvm.ptr) -> !llvm.ptr
    %queueName = llvm.mlir.addressof @strict_metal_s_newCommandQueue : !llvm.ptr
    %queue = llvm.call @strict_metal_message(%device, %queueName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %workingName = llvm.mlir.addressof @strict_metal_s_recommendedMaxWorkingSetSize : !llvm.ptr
    %working = llvm.call @strict_metal_number(%device, %workingName) : (!llvm.ptr, !llvm.ptr) -> i64
    %longestName = llvm.mlir.addressof @strict_metal_s_maxBufferLength : !llvm.ptr
    %longest = llvm.call @strict_metal_number(%device, %longestName) : (!llvm.ptr, !llvm.ptr) -> i64
    %two = llvm.mlir.constant(2 : i64) : i64
    %quarter = llvm.lshr %working, %two : i64
    %size = llvm.intr.umin(%quarter, %longest) : (i64, i64) -> i64
    %bufferName = llvm.mlir.addressof @strict_metal_s_newBufferWithLength_options : !llvm.ptr
    %newBuffer = llvm.call @strict_metal_selector(%bufferName) : (!llvm.ptr) -> !llvm.ptr
    %buffer = llvm.call %send(%device, %newBuffer, %size, %zero) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, i64, i64) -> !llvm.ptr
    %noBuffer = llvm.icmp "eq" %buffer, %null : !llvm.ptr
    llvm.cond_br %noBuffer, ^failed, ^addressed
  ^addressed:
    %canAddress = llvm.call %send(%buffer, %responds, %addressSelector) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i8
    %noAddress = llvm.mlir.constant(0 : i8) : i8
    %isOld = llvm.icmp "eq" %canAddress, %noAddress : i8
    llvm.cond_br %isOld, ^failed, ^ready
  ^ready:
    %contentsName = llvm.mlir.addressof @strict_metal_s_contents : !llvm.ptr
    %contents = llvm.call @strict_metal_number(%buffer, %contentsName) : (!llvm.ptr, !llvm.ptr) -> i64
    %base = llvm.call @strict_metal_number(%buffer, %addressName) : (!llvm.ptr, !llvm.ptr) -> i64
    %deviceBits = llvm.ptrtoint %device : !llvm.ptr to i64
    %queueBits = llvm.ptrtoint %queue : !llvm.ptr to i64
    %bufferBits = llvm.ptrtoint %buffer : !llvm.ptr to i64
    %five = llvm.mlir.constant(5 : i64) : i64
    %six = llvm.mlir.constant(6 : i64) : i64
    %seven = llvm.mlir.constant(7 : i64) : i64
    %eight = llvm.mlir.constant(8 : i64) : i64
    %nine = llvm.mlir.constant(9 : i64) : i64
    %ten = llvm.mlir.constant(10 : i64) : i64
    llvm.call @strict_metal_put(%five, %deviceBits) : (i64, i64) -> ()
    llvm.call @strict_metal_put(%six, %queueBits) : (i64, i64) -> ()
    llvm.call @strict_metal_put(%seven, %bufferBits) : (i64, i64) -> ()
    llvm.call @strict_metal_put(%eight, %contents) : (i64, i64) -> ()
    llvm.call @strict_metal_put(%nine, %base) : (i64, i64) -> ()
    llvm.call @strict_metal_put(%ten, %size) : (i64, i64) -> ()
    llvm.store %device, %context : !llvm.ptr, !llvm.ptr
    llvm.return %success : i32
  ^failed:
    llvm.return %noDevice : i32
  }
  llvm.func @strict_metal_cuCtxSetCurrent(%context: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  // compiles the Metal source (without fast math, Strict keeps NaN and infinity), keeps a copy for the parameter counts
  llvm.func @strict_metal_cuModuleLoadData(%module: !llvm.ptr, %image: !llvm.ptr) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %invalid = llvm.mlir.constant(200 : i32) : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %zero = llvm.mlir.constant(0 : i64) : i64
    %five = llvm.mlir.constant(5 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %first = llvm.load %image : !llvm.ptr -> i8
    %nothing = llvm.mlir.constant(0 : i8) : i8
    %isEmpty = llvm.icmp "eq" %first, %nothing : i8
    llvm.cond_br %isEmpty, ^empty, ^compile
  ^empty:
    llvm.return %invalid : i32
  ^compile:
    %pool = llvm.call @strict_metal_push() : () -> !llvm.ptr
    %source = llvm.call @strict_metal_text(%image) : (!llvm.ptr) -> !llvm.ptr
    %optionsClassName = llvm.mlir.addressof @strict_metal_n_options : !llvm.ptr
    %optionsClass = llvm.call @strict_metal_class(%optionsClassName) : (!llvm.ptr) -> !llvm.ptr
    %newName = llvm.mlir.addressof @strict_metal_s_new : !llvm.ptr
    %options = llvm.call @strict_metal_message(%optionsClass, %newName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %fastName = llvm.mlir.addressof @strict_metal_s_setFastMathEnabled : !llvm.ptr
    %fast = llvm.call @strict_metal_selector(%fastName) : (!llvm.ptr) -> !llvm.ptr
    llvm.call %send(%options, %fast, %nothing) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, i8) -> ()
    %errorSlot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %null, %errorSlot : !llvm.ptr, !llvm.ptr
    %device = llvm.call @strict_metal_pointer(%five) : (i64) -> !llvm.ptr
    %compileName = llvm.mlir.addressof @strict_metal_s_newLibraryWithSource_options_error : !llvm.ptr
    %compile = llvm.call @strict_metal_selector(%compileName) : (!llvm.ptr) -> !llvm.ptr
    %library = llvm.call %send(%device, %compile, %source, %options, %errorSlot) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %failed = llvm.icmp "eq" %library, %null : !llvm.ptr
    llvm.cond_br %failed, ^report, ^keep
  ^report:
    %error = llvm.load %errorSlot : !llvm.ptr -> !llvm.ptr
    llvm.call @strict_metal_report(%error) : (!llvm.ptr) -> ()
    llvm.call @strict_metal_pop(%pool) : (!llvm.ptr) -> ()
    llvm.return %invalid : i32
  ^keep:
    %length = llvm.call @strlen(%image) : (!llvm.ptr) -> i64
    %size = llvm.add %length, %one : i64
    %copy = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%copy, %image, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %sixteen = llvm.mlir.constant(16 : i64) : i64
    %record = llvm.call @malloc(%sixteen) : (i64) -> !llvm.ptr
    llvm.store %library, %record : !llvm.ptr, !llvm.ptr
    %copySlot = llvm.getelementptr %record[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %copy, %copySlot : !llvm.ptr, !llvm.ptr
    llvm.store %record, %module : !llvm.ptr, !llvm.ptr
    llvm.call @strict_metal_pop(%pool) : (!llvm.ptr) -> ()
    llvm.return %success : i32
  }
  // a kernel the source names in "// parameters Name count": its pipeline and the number of its 8 byte parameters
  llvm.func @strict_metal_cuModuleGetFunction(%function: !llvm.ptr, %module: !llvm.ptr, %name: !llvm.ptr) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %notFound = llvm.mlir.constant(500 : i32) : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %five = llvm.mlir.constant(5 : i64) : i64
    %ten = llvm.mlir.constant(10 : i32) : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %library = llvm.load %module : !llvm.ptr -> !llvm.ptr
    %sourceSlot = llvm.getelementptr %module[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %source = llvm.load %sourceSlot : !llvm.ptr -> !llvm.ptr
    %patternSize = llvm.mlir.constant(512 : i64) : i64
    %pattern = llvm.alloca %patternSize x i8 : (i64) -> !llvm.ptr
    %format = llvm.mlir.addressof @strict_metal_marker : !llvm.ptr
    %written = llvm.call @snprintf(%pattern, %patternSize, %format, %name) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
    %found = llvm.call @strstr(%source, %pattern) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %missing = llvm.icmp "eq" %found, %null : !llvm.ptr
    llvm.cond_br %missing, ^missing, ^named
  ^missing:
    llvm.return %notFound : i32
  ^named:
    %patternLength = llvm.call @strlen(%pattern) : (!llvm.ptr) -> i64
    %digits = llvm.getelementptr %found[%patternLength] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %count = llvm.call @strtol(%digits, %null, %ten) : (!llvm.ptr, !llvm.ptr, i32) -> i64
    %pool = llvm.call @strict_metal_push() : () -> !llvm.ptr
    %nameText = llvm.call @strict_metal_text(%name) : (!llvm.ptr) -> !llvm.ptr
    %functionName = llvm.mlir.addressof @strict_metal_s_newFunctionWithName : !llvm.ptr
    %metalFunction = llvm.call @strict_metal_with(%library, %functionName, %nameText) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %noFunction = llvm.icmp "eq" %metalFunction, %null : !llvm.ptr
    llvm.cond_br %noFunction, ^failed(%null : !llvm.ptr), ^pipeline
  ^pipeline:
    %errorSlot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %null, %errorSlot : !llvm.ptr, !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %device = llvm.call @strict_metal_pointer(%five) : (i64) -> !llvm.ptr
    %pipelineName = llvm.mlir.addressof @strict_metal_s_newComputePipelineStateWithFunction_error : !llvm.ptr
    %newPipeline = llvm.call @strict_metal_selector(%pipelineName) : (!llvm.ptr) -> !llvm.ptr
    %pipeline = llvm.call %send(%device, %newPipeline, %metalFunction, %errorSlot) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %noPipeline = llvm.icmp "eq" %pipeline, %null : !llvm.ptr
    %error = llvm.load %errorSlot : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %noPipeline, ^failed(%error : !llvm.ptr), ^made
  ^failed(%reported: !llvm.ptr):
    llvm.call @strict_metal_report(%reported) : (!llvm.ptr) -> ()
    llvm.call @strict_metal_pop(%pool) : (!llvm.ptr) -> ()
    llvm.return %notFound : i32
  ^made:
    %sixteen = llvm.mlir.constant(16 : i64) : i64
    %record = llvm.call @malloc(%sixteen) : (i64) -> !llvm.ptr
    llvm.store %pipeline, %record : !llvm.ptr, !llvm.ptr
    %countSlot = llvm.getelementptr %record[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %count, %countSlot : i64, !llvm.ptr
    llvm.store %record, %function : !llvm.ptr, !llvm.ptr
    llvm.call @strict_metal_pop(%pool) : (!llvm.ptr) -> ()
    llvm.return %success : i32
  }
  llvm.func @strict_metal_cuMemAlloc_v2(%pointer: !llvm.ptr, %bytes: i64) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %outOfMemory = llvm.mlir.constant(2 : i32) : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %nine = llvm.mlir.constant(9 : i64) : i64
    %ten = llvm.mlir.constant(10 : i64) : i64
    %eleven = llvm.mlir.constant(11 : i64) : i64
    %twelve = llvm.mlir.constant(12 : i64) : i64
    %rest = llvm.mlir.constant(255 : i64) : i64
    %mask = llvm.mlir.constant(-256 : i64) : i64
    %top = llvm.call @strict_metal_get(%eleven) : (i64) -> i64
    %padded = llvm.add %top, %rest : i64
    %aligned = llvm.and %padded, %mask : i64
    %end = llvm.add %aligned, %bytes : i64
    %size = llvm.call @strict_metal_get(%ten) : (i64) -> i64
    %tooBig = llvm.icmp "ugt" %end, %size : i64
    llvm.cond_br %tooBig, ^full, ^taken
  ^full:
    llvm.return %outOfMemory : i32
  ^taken:
    llvm.call @strict_metal_put(%eleven, %end) : (i64, i64) -> ()
    %live = llvm.call @strict_metal_get(%twelve) : (i64) -> i64
    %more = llvm.add %live, %one : i64
    llvm.call @strict_metal_put(%twelve, %more) : (i64, i64) -> ()
    %base = llvm.call @strict_metal_get(%nine) : (i64) -> i64
    %address = llvm.add %base, %aligned : i64
    llvm.store %address, %pointer : i64, !llvm.ptr
    llvm.return %success : i32
  }
  llvm.func @strict_metal_cuMemFree_v2(%device: i64) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %eleven = llvm.mlir.constant(11 : i64) : i64
    %twelve = llvm.mlir.constant(12 : i64) : i64
    %live = llvm.call @strict_metal_get(%twelve) : (i64) -> i64
    %fewer = llvm.sub %live, %one : i64
    llvm.call @strict_metal_put(%twelve, %fewer) : (i64, i64) -> ()
    %none = llvm.icmp "eq" %fewer, %zero : i64
    llvm.cond_br %none, ^reset, ^done
  ^reset:
    llvm.call @strict_metal_put(%eleven, %zero) : (i64, i64) -> ()
    llvm.br ^done
  ^done:
    llvm.return %success : i32
  }
  llvm.func @strict_metal_cuMemcpyHtoD_v2(%device: i64, %host: !llvm.ptr, %bytes: i64) -> i32 {
    %waited = llvm.call @strict_metal_wait() : () -> i32
    %target = llvm.call @strict_metal_host(%device) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%target, %host, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.return %waited : i32
  }
  llvm.func @strict_metal_cuMemcpyDtoH_v2(%host: !llvm.ptr, %device: i64, %bytes: i64) -> i32 {
    %waited = llvm.call @strict_metal_wait() : () -> i32
    %source = llvm.call @strict_metal_host(%device) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%host, %source, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.return %waited : i32
  }
  // one command buffer per launch, its parameters copied into the encoder; dynamic shared memory is not supported
  llvm.func @strict_metal_cuLaunchKernel(%function: !llvm.ptr, %gridX: i32, %gridY: i32, %gridZ: i32, %groupX: i32, %groupY: i32, %groupZ: i32, %shared: i32, %stream: !llvm.ptr, %parameters: !llvm.ptr, %extra: !llvm.ptr) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %unsupported = llvm.mlir.constant(801 : i32) : i32
    %noShared = llvm.mlir.constant(0 : i32) : i32
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %hasShared = llvm.icmp "ne" %shared, %noShared : i32
    llvm.cond_br %hasShared, ^refused, ^start
  ^refused:
    llvm.return %unsupported : i32
  ^start:
    %pipeline = llvm.load %function : !llvm.ptr -> !llvm.ptr
    %countSlot = llvm.getelementptr %function[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %slotCount = llvm.mlir.constant(64 : i64) : i64
    %slots = llvm.alloca %slotCount x i64 : (i64) -> !llvm.ptr
    llvm.br ^copy(%zero : i64)
  ^copy(%index: i64):
    %more = llvm.icmp "slt" %index, %count : i64
    llvm.cond_br %more, ^parameter, ^encode
  ^parameter:
    %pointerSlot = llvm.getelementptr %parameters[%index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %pointer = llvm.load %pointerSlot : !llvm.ptr -> !llvm.ptr
    %value = llvm.load %pointer : !llvm.ptr -> i64
    %slot = llvm.getelementptr %slots[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %value, %slot : i64, !llvm.ptr
    %next = llvm.add %index, %one : i64
    llvm.br ^copy(%next : i64)
  ^encode:
    %pool = llvm.call @strict_metal_push() : () -> !llvm.ptr
    %send = llvm.call @strict_metal_pointer(%zero) : (i64) -> !llvm.ptr
    %six = llvm.mlir.constant(6 : i64) : i64
    %queue = llvm.call @strict_metal_pointer(%six) : (i64) -> !llvm.ptr
    %bufferName = llvm.mlir.addressof @strict_metal_s_commandBuffer : !llvm.ptr
    %commands = llvm.call @strict_metal_message(%queue, %bufferName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %encoderName = llvm.mlir.addressof @strict_metal_s_computeCommandEncoder : !llvm.ptr
    %encoder = llvm.call @strict_metal_message(%commands, %encoderName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %stateName = llvm.mlir.addressof @strict_metal_s_setComputePipelineState : !llvm.ptr
    %stated = llvm.call @strict_metal_with(%encoder, %stateName, %pipeline) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %eight = llvm.mlir.constant(8 : i64) : i64
    %used = llvm.intr.smax(%count, %one) : (i64, i64) -> i64
    %bytes = llvm.mul %used, %eight : i64
    %bytesName = llvm.mlir.addressof @strict_metal_s_setBytes_length_atIndex : !llvm.ptr
    %setBytes = llvm.call @strict_metal_selector(%bytesName) : (!llvm.ptr) -> !llvm.ptr
    llvm.call %send(%encoder, %setBytes, %slots, %bytes, %zero) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %seven = llvm.mlir.constant(7 : i64) : i64
    %buffer = llvm.call @strict_metal_pointer(%seven) : (i64) -> !llvm.ptr
    %useName = llvm.mlir.addressof @strict_metal_s_useResource_usage : !llvm.ptr
    %use = llvm.call @strict_metal_selector(%useName) : (!llvm.ptr) -> !llvm.ptr
    %readWrite = llvm.mlir.constant(3 : i64) : i64
    llvm.call %send(%encoder, %use, %buffer, %readWrite) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
    %three = llvm.mlir.constant(3 : i64) : i64
    %grid = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    %group = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    llvm.call @strict_metal_size(%grid, %gridX, %gridY, %gridZ) : (!llvm.ptr, i32, i32, i32) -> ()
    llvm.call @strict_metal_size(%group, %groupX, %groupY, %groupZ) : (!llvm.ptr, i32, i32, i32) -> ()
    %dispatchName = llvm.mlir.addressof @strict_metal_s_dispatchThreadgroups_threadsPerThreadgroup : !llvm.ptr
    %dispatch = llvm.call @strict_metal_selector(%dispatchName) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @strict_metal_dispatch(%send, %encoder, %dispatch, %grid, %group) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %endName = llvm.mlir.addressof @strict_metal_s_endEncoding : !llvm.ptr
    %ended = llvm.call @strict_metal_message(%encoder, %endName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %commitName = llvm.mlir.addressof @strict_metal_s_commit : !llvm.ptr
    %committed = llvm.call @strict_metal_message(%commands, %commitName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.call @strict_metal_remember(%commands) : (!llvm.ptr) -> ()
    llvm.call @strict_metal_pop(%pool) : (!llvm.ptr) -> ()
    llvm.return %success : i32
  }
  llvm.func @strict_metal_size(%size: !llvm.ptr, %width: i32, %height: i32, %depth: i32) {
    %first = llvm.zext %width : i32 to i64
    %second = llvm.zext %height : i32 to i64
    %third = llvm.zext %depth : i32 to i64
    llvm.store %first, %size : i64, !llvm.ptr
    %heightSlot = llvm.getelementptr %size[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %second, %heightSlot : i64, !llvm.ptr
    %depthSlot = llvm.getelementptr %size[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %third, %depthSlot : i64, !llvm.ptr
    llvm.return
  }
  // keeps the command buffer (autoreleased) as the last one, releasing the one before
  llvm.func @strict_metal_remember(%commands: !llvm.ptr) {
    %lastIndex = llvm.mlir.constant(13 : i64) : i64
    %retainName = llvm.mlir.addressof @strict_metal_s_retain : !llvm.ptr
    %kept = llvm.call @strict_metal_message(%commands, %retainName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %old = llvm.call @strict_metal_pointer(%lastIndex) : (i64) -> !llvm.ptr
    %keptBits = llvm.ptrtoint %kept : !llvm.ptr to i64
    llvm.call @strict_metal_put(%lastIndex, %keptBits) : (i64, i64) -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    %hadOld = llvm.icmp "ne" %old, %null : !llvm.ptr
    llvm.cond_br %hadOld, ^release, ^done
  ^release:
    %releaseName = llvm.mlir.addressof @strict_metal_s_release : !llvm.ptr
    %released = llvm.call @strict_metal_message(%old, %releaseName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^done
  ^done:
    llvm.return
  }
  llvm.func @strict_metal_cuCtxSynchronize() -> i32 {
    %waited = llvm.call @strict_metal_wait() : () -> i32
    llvm.return %waited : i32
  }
  llvm.func @strict_metal_cuMemAllocHost_v2(%pointer: !llvm.ptr, %bytes: i64) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %outOfMemory = llvm.mlir.constant(2 : i32) : i32
    %host = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    llvm.store %host, %pointer : !llvm.ptr, !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %failed = llvm.icmp "eq" %host, %null : !llvm.ptr
    %result = llvm.select %failed, %outOfMemory, %success : i1, i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuMemFreeHost(%host: !llvm.ptr) -> i32 {
    llvm.call @free(%host) : (!llvm.ptr) -> ()
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuMemcpyHtoDAsync_v2(%device: i64, %host: !llvm.ptr, %bytes: i64, %stream: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_cuMemcpyHtoD_v2(%device, %host, %bytes) : (i64, !llvm.ptr, i64) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuMemcpyDtoDAsync_v2(%target: i64, %source: i64, %bytes: i64, %stream: !llvm.ptr) -> i32 {
    %waited = llvm.call @strict_metal_wait() : () -> i32
    %to = llvm.call @strict_metal_host(%target) : (i64) -> !llvm.ptr
    %from = llvm.call @strict_metal_host(%source) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%to, %from, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.return %waited : i32
  }
  llvm.func @strict_metal_handle(%slot: !llvm.ptr) -> i32 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %handle = llvm.inttoptr %one : i64 to !llvm.ptr
    llvm.store %handle, %slot : !llvm.ptr, !llvm.ptr
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuStreamCreate(%stream: !llvm.ptr, %flags: i32) -> i32 {
    %result = llvm.call @strict_metal_handle(%stream) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuStreamBeginCapture_v2(%stream: !llvm.ptr, %mode: i32) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuStreamEndCapture(%stream: !llvm.ptr, %graph: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_handle(%graph) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuGraphInstantiateWithFlags(%executable: !llvm.ptr, %graph: !llvm.ptr, %flags: i64) -> i32 {
    %result = llvm.call @strict_metal_handle(%executable) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuGraphExecUpdate_v2(%executable: !llvm.ptr, %graph: !llvm.ptr, %information: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuGraphLaunch(%executable: !llvm.ptr, %stream: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuGraphDestroy(%graph: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuGraphExecDestroy(%executable: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuMemGetInfo_v2(%free: !llvm.ptr, %total: !llvm.ptr) -> i32 {
    %ten = llvm.mlir.constant(10 : i64) : i64
    %eleven = llvm.mlir.constant(11 : i64) : i64
    %size = llvm.call @strict_metal_get(%ten) : (i64) -> i64
    %top = llvm.call @strict_metal_get(%eleven) : (i64) -> i64
    %left = llvm.sub %size, %top : i64
    llvm.store %left, %free : i64, !llvm.ptr
    llvm.store %size, %total : i64, !llvm.ptr
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuStreamQuery(%stream: !llvm.ptr) -> i32 {
    %success = llvm.mlir.constant(0 : i32) : i32
    %notReady = llvm.mlir.constant(600 : i32) : i32
    %lastIndex = llvm.mlir.constant(13 : i64) : i64
    %last = llvm.call @strict_metal_pointer(%lastIndex) : (i64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %none = llvm.icmp "eq" %last, %null : !llvm.ptr
    llvm.cond_br %none, ^done(%success : i32), ^status
  ^status:
    %statusName = llvm.mlir.addressof @strict_metal_s_status : !llvm.ptr
    %status = llvm.call @strict_metal_number(%last, %statusName) : (!llvm.ptr, !llvm.ptr) -> i64
    %completed = llvm.mlir.constant(4 : i64) : i64
    %finished = llvm.icmp "sge" %status, %completed : i64
    %result = llvm.select %finished, %success, %notReady : i1, i32
    llvm.br ^done(%result : i32)
  ^done(%answer: i32):
    llvm.return %answer : i32
  }
  // the one Metal queue runs every command buffer in order
  llvm.func @strict_metal_cuStreamWaitEvent(%stream: !llvm.ptr, %event: !llvm.ptr, %flags: i32) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuEventCreate(%event: !llvm.ptr, %flags: i32) -> i32 {
    %eight = llvm.mlir.constant(8 : i64) : i64
    %record = llvm.call @malloc(%eight) : (i64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.store %null, %record : !llvm.ptr, !llvm.ptr
    llvm.store %record, %event : !llvm.ptr, !llvm.ptr
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  // an event is the last command buffer when it was recorded
  llvm.func @strict_metal_cuEventRecord(%event: !llvm.ptr, %stream: !llvm.ptr) -> i32 {
    %lastIndex = llvm.mlir.constant(13 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %last = llvm.call @strict_metal_pointer(%lastIndex) : (i64) -> !llvm.ptr
    %old = llvm.load %event : !llvm.ptr -> !llvm.ptr
    %hasLast = llvm.icmp "ne" %last, %null : !llvm.ptr
    llvm.cond_br %hasLast, ^retain, ^store(%null : !llvm.ptr)
  ^retain:
    %retainName = llvm.mlir.addressof @strict_metal_s_retain : !llvm.ptr
    %kept = llvm.call @strict_metal_message(%last, %retainName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^store(%kept : !llvm.ptr)
  ^store(%recorded: !llvm.ptr):
    llvm.store %recorded, %event : !llvm.ptr, !llvm.ptr
    %hadOld = llvm.icmp "ne" %old, %null : !llvm.ptr
    llvm.cond_br %hadOld, ^release, ^done
  ^release:
    %releaseName = llvm.mlir.addressof @strict_metal_s_release : !llvm.ptr
    %released = llvm.call @strict_metal_message(%old, %releaseName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^done
  ^done:
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuEventSynchronize(%event: !llvm.ptr) -> i32 {
    %recorded = llvm.load %event : !llvm.ptr -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %none = llvm.icmp "eq" %recorded, %null : !llvm.ptr
    llvm.cond_br %none, ^done, ^wait
  ^wait:
    %waitName = llvm.mlir.addressof @strict_metal_s_waitUntilCompleted : !llvm.ptr
    %waited = llvm.call @strict_metal_message(%recorded, %waitName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^done
  ^done:
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuMemcpyDtoHAsync_v2(%host: !llvm.ptr, %device: i64, %bytes: i64, %stream: !llvm.ptr) -> i32 {
    %result = llvm.call @strict_metal_cuMemcpyDtoH_v2(%host, %device, %bytes) : (!llvm.ptr, i64, i64) -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuStreamSynchronize(%stream: !llvm.ptr) -> i32 {
    %waited = llvm.call @strict_metal_wait() : () -> i32
    llvm.return %waited : i32
  }
  llvm.func @strict_metal_cuFuncSetAttribute(%function: !llvm.ptr, %attribute: i32, %value: i32) -> i32 {
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  llvm.func @strict_metal_cuDeviceGetAttribute(%value: !llvm.ptr, %attribute: i32, %device: i32) -> i32 {
    %eight = llvm.mlir.constant(8 : i32) : i32
    llvm.store %eight, %value : i32, !llvm.ptr
    %result = llvm.call @strict_metal_succeeded() : () -> i32
    llvm.return %result : i32
  }
  // Metal and the Objective-C runtime it loads, the functions above take the place of the CUDA driver
  func.func private @strict_metal_resolve(%library: !llvm.ptr, %name: !llvm.ptr, %index: i64) -> i1 {
    %function = llvm.call @dlsym(%library, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %bits = llvm.ptrtoint %function : !llvm.ptr to i64
    llvm.call @strict_metal_put(%index, %bits) : (i64, i64) -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    %found = llvm.icmp "ne" %function, %null : !llvm.ptr
    return %found : i1
  }
  // the Metal function for a CUDA driver function name of strict_gpu_names
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %index = func.call @strict_metal_index(%name) : (!llvm.ptr) -> i64
    %function = llvm.call @strict_metal_function(%index) : (i64) -> !llvm.ptr
    return %function : !llvm.ptr
  }
  func.func private @strict_metal_index(%name: !llvm.ptr) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %count = arith.constant 34 : i64
    %none = arith.constant 0 : i32
    %names = llvm.mlir.addressof @strict_gpu_names : !llvm.ptr
    cf.br ^check(%zero, %names : i64, !llvm.ptr)
  ^check(%index: i64, %candidate: !llvm.ptr):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^compare, ^done(%count : i64)
  ^compare:
    %order = llvm.call @strcmp(%name, %candidate) : (!llvm.ptr, !llvm.ptr) -> i32
    %same = arith.cmpi eq, %order, %none : i32
    %length = llvm.call @strlen(%candidate) : (!llvm.ptr) -> i64
    %skipped = arith.addi %length, %one : i64
    %following = llvm.getelementptr %candidate[%skipped] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %next = arith.addi %index, %one : i64
    cf.cond_br %same, ^done(%index : i64), ^check(%next, %following : i64, !llvm.ptr)
  ^done(%found: i64):
    return %found : i64
  }
  llvm.func @strict_metal_function(%index: i64) -> !llvm.ptr {
    llvm.switch %index : i64, ^none [
      0: ^c0,
      1: ^c1,
      2: ^c2,
      3: ^c3,
      4: ^c4,
      5: ^c5,
      6: ^c6,
      7: ^c7,
      8: ^c8,
      9: ^c9,
      10: ^c10,
      11: ^c11,
      12: ^c12,
      13: ^c13,
      14: ^c14,
      15: ^c15,
      16: ^c16,
      17: ^c17,
      18: ^c18,
      19: ^c19,
      20: ^c20,
      21: ^c21,
      22: ^c22,
      23: ^c23,
      24: ^c24,
      25: ^c25,
      26: ^c26,
      27: ^c27,
      28: ^c28,
      29: ^c29,
      30: ^c30,
      31: ^c31,
      32: ^c32,
      33: ^c33
    ]
  ^none:
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  ^c0:
    %f0 = llvm.mlir.addressof @strict_metal_cuInit : !llvm.ptr
    llvm.return %f0 : !llvm.ptr
  ^c1:
    %f1 = llvm.mlir.addressof @strict_metal_cuDeviceGet : !llvm.ptr
    llvm.return %f1 : !llvm.ptr
  ^c2:
    %f2 = llvm.mlir.addressof @strict_metal_cuDevicePrimaryCtxRetain : !llvm.ptr
    llvm.return %f2 : !llvm.ptr
  ^c3:
    %f3 = llvm.mlir.addressof @strict_metal_cuCtxSetCurrent : !llvm.ptr
    llvm.return %f3 : !llvm.ptr
  ^c4:
    %f4 = llvm.mlir.addressof @strict_metal_cuModuleLoadData : !llvm.ptr
    llvm.return %f4 : !llvm.ptr
  ^c5:
    %f5 = llvm.mlir.addressof @strict_metal_cuModuleGetFunction : !llvm.ptr
    llvm.return %f5 : !llvm.ptr
  ^c6:
    %f6 = llvm.mlir.addressof @strict_metal_cuMemAlloc_v2 : !llvm.ptr
    llvm.return %f6 : !llvm.ptr
  ^c7:
    %f7 = llvm.mlir.addressof @strict_metal_cuMemFree_v2 : !llvm.ptr
    llvm.return %f7 : !llvm.ptr
  ^c8:
    %f8 = llvm.mlir.addressof @strict_metal_cuMemcpyHtoD_v2 : !llvm.ptr
    llvm.return %f8 : !llvm.ptr
  ^c9:
    %f9 = llvm.mlir.addressof @strict_metal_cuMemcpyDtoH_v2 : !llvm.ptr
    llvm.return %f9 : !llvm.ptr
  ^c10:
    %f10 = llvm.mlir.addressof @strict_metal_cuLaunchKernel : !llvm.ptr
    llvm.return %f10 : !llvm.ptr
  ^c11:
    %f11 = llvm.mlir.addressof @strict_metal_cuCtxSynchronize : !llvm.ptr
    llvm.return %f11 : !llvm.ptr
  ^c12:
    %f12 = llvm.mlir.addressof @strict_metal_cuMemAllocHost_v2 : !llvm.ptr
    llvm.return %f12 : !llvm.ptr
  ^c13:
    %f13 = llvm.mlir.addressof @strict_metal_cuMemFreeHost : !llvm.ptr
    llvm.return %f13 : !llvm.ptr
  ^c14:
    %f14 = llvm.mlir.addressof @strict_metal_cuMemcpyHtoDAsync_v2 : !llvm.ptr
    llvm.return %f14 : !llvm.ptr
  ^c15:
    %f15 = llvm.mlir.addressof @strict_metal_cuMemcpyDtoDAsync_v2 : !llvm.ptr
    llvm.return %f15 : !llvm.ptr
  ^c16:
    %f16 = llvm.mlir.addressof @strict_metal_cuStreamCreate : !llvm.ptr
    llvm.return %f16 : !llvm.ptr
  ^c17:
    %f17 = llvm.mlir.addressof @strict_metal_cuStreamBeginCapture_v2 : !llvm.ptr
    llvm.return %f17 : !llvm.ptr
  ^c18:
    %f18 = llvm.mlir.addressof @strict_metal_cuStreamEndCapture : !llvm.ptr
    llvm.return %f18 : !llvm.ptr
  ^c19:
    %f19 = llvm.mlir.addressof @strict_metal_cuGraphInstantiateWithFlags : !llvm.ptr
    llvm.return %f19 : !llvm.ptr
  ^c20:
    %f20 = llvm.mlir.addressof @strict_metal_cuGraphExecUpdate_v2 : !llvm.ptr
    llvm.return %f20 : !llvm.ptr
  ^c21:
    %f21 = llvm.mlir.addressof @strict_metal_cuGraphLaunch : !llvm.ptr
    llvm.return %f21 : !llvm.ptr
  ^c22:
    %f22 = llvm.mlir.addressof @strict_metal_cuGraphDestroy : !llvm.ptr
    llvm.return %f22 : !llvm.ptr
  ^c23:
    %f23 = llvm.mlir.addressof @strict_metal_cuGraphExecDestroy : !llvm.ptr
    llvm.return %f23 : !llvm.ptr
  ^c24:
    %f24 = llvm.mlir.addressof @strict_metal_cuMemGetInfo_v2 : !llvm.ptr
    llvm.return %f24 : !llvm.ptr
  ^c25:
    %f25 = llvm.mlir.addressof @strict_metal_cuStreamQuery : !llvm.ptr
    llvm.return %f25 : !llvm.ptr
  ^c26:
    %f26 = llvm.mlir.addressof @strict_metal_cuEventCreate : !llvm.ptr
    llvm.return %f26 : !llvm.ptr
  ^c27:
    %f27 = llvm.mlir.addressof @strict_metal_cuEventRecord : !llvm.ptr
    llvm.return %f27 : !llvm.ptr
  ^c28:
    %f28 = llvm.mlir.addressof @strict_metal_cuEventSynchronize : !llvm.ptr
    llvm.return %f28 : !llvm.ptr
  ^c29:
    %f29 = llvm.mlir.addressof @strict_metal_cuMemcpyDtoHAsync_v2 : !llvm.ptr
    llvm.return %f29 : !llvm.ptr
  ^c30:
    %f30 = llvm.mlir.addressof @strict_metal_cuStreamSynchronize : !llvm.ptr
    llvm.return %f30 : !llvm.ptr
  ^c31:
    %f31 = llvm.mlir.addressof @strict_metal_cuFuncSetAttribute : !llvm.ptr
    llvm.return %f31 : !llvm.ptr
  ^c32:
    %f32 = llvm.mlir.addressof @strict_metal_cuDeviceGetAttribute : !llvm.ptr
    llvm.return %f32 : !llvm.ptr
  ^c33:
    %f33 = llvm.mlir.addressof @strict_metal_cuStreamWaitEvent : !llvm.ptr
    llvm.return %f33 : !llvm.ptr
  }

  llvm.func @nanosleep(!llvm.ptr, !llvm.ptr) -> i32
  func.func private @strict_test_pause() {
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %millisecond = arith.constant 1000000 : i64
    %delay = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    llvm.store %zero, %delay : i64, !llvm.ptr
    %fraction = llvm.getelementptr %delay[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %millisecond, %fraction : i64, !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %paused = llvm.call @nanosleep(%delay, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    return
  }
  llvm.func @strict_test_thread(%argument: !llvm.ptr) -> !llvm.ptr {
    func.call @strict_test_watch() : () -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.return %null : !llvm.ptr
  }
  func.func private @strict_test_spawn() -> i1 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_test_thread : !llvm.ptr
    %handle = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %created = llvm.call @pthread_create(%handle, %null, %procedure, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %spawned = arith.cmpi eq, %created, %zero : i32
    return %spawned : i1
  }

  func.func private @strict_test_quit() {
    %failure = arith.constant 1 : i32
    llvm.call @_exit(%failure) : (i32) -> ()
    return
  }
