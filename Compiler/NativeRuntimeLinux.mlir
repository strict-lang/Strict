  // the heap the collector reserves (its page table and cards are an 8192th and a 512th of it), a pause while spinning
  llvm.mlir.global internal constant @strict_heap_bytes(68719476736 : i64) : i64
  llvm.func @strict_pause() attributes {passthrough = ["alwaysinline"]} {
    llvm.call_intrinsic "llvm.x86.sse2.pause"() : () -> ()
    llvm.return
  }
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
  func.func private @strict_registers_spilled() -> (i64, i64) {
    %none = arith.constant 0 : i64
    return %none, %none : i64, i64
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
  llvm.func @madvise(!llvm.ptr, i64, i32) -> i32
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    %readWrite = arith.constant 3 : i32
    %privateAnonymousNoReserve = arith.constant 16418 : i32
    %noFile = arith.constant -1 : i32
    %noOffset = arith.constant 0 : i64
    %failure = arith.constant -1 : i64
    %reserved = llvm.call @mmap(%null, %size, %readWrite, %privateAnonymousNoReserve, %noFile, %noOffset) : (!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr
    %address = llvm.ptrtoint %reserved : !llvm.ptr to i64
    %failed = arith.cmpi eq, %address, %failure : i64
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return %reserved : !llvm.ptr
  }
  llvm.func @getrlimit(i32, !llvm.ptr) -> i32
  llvm.func @setrlimit(i32, !llvm.ptr) -> i32
  // the main stack grows up to its limit (RLIMIT_STACK) below where it started, raised to 64 MB like on Windows and
  // macOS (8 MB is the usual default, the kernel keeps at least 128 MB free below the stack for growing)
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %none = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %success = arith.constant 0 : i32
    %stackResource = arith.constant 3 : i32
    %most = arith.constant 67108864 : i64
    %limits = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    llvm.store %none, %limits : i64, !llvm.ptr
    %got = llvm.call @getrlimit(%stackResource, %limits) : (i32, !llvm.ptr) -> i32
    %current = llvm.load %limits : !llvm.ptr -> i64
    %hardSlot = llvm.getelementptr %limits[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %hard = llvm.load %hardSlot : !llvm.ptr -> i64
    %allowed = arith.minui %most, %hard : i64
    %wanted = arith.maxui %current, %allowed : i64
    llvm.store %wanted, %limits : i64, !llvm.ptr
    %set = llvm.call @setrlimit(%stackResource, %limits) : (i32, !llvm.ptr) -> i32
    %isRaised = arith.cmpi eq, %set, %success : i32
    %limit = arith.select %isRaised, %wanted, %current : i64
    %size = arith.minui %limit, %most : i64
    %top = llvm.ptrtoint %stack : !llvm.ptr to i64
    %lowest = arith.subi %top, %size : i64
    return %lowest : i64
  }
  func.func private @strict_commit(%at: !llvm.ptr, %size: i64) {
    return
  }
  func.func private @strict_decommit(%at: !llvm.ptr, %size: i64) {
    %dontNeed = arith.constant 4 : i32
    %advised = llvm.call @madvise(%at, %size, %dontNeed) : (!llvm.ptr, i64, i32) -> i32
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
  llvm.func @readlink(!llvm.ptr, !llvm.ptr, i64) -> i64
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
  llvm.mlir.global internal constant @strict_linux("Linux\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @strict_path_variable("PATH\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_temp_template("/tmp/strictXXXXXX\00") : !llvm.array<18 x i8>
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
    %name = llvm.mlir.addressof @strict_linux : !llvm.ptr
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_self_executable("/proc/self/exe\00") : !llvm.array<15 x i8>
  func.func private @strict_process_current(%self: f64) -> f64 {
    %size = arith.constant 4096 : i64
    %limit = arith.constant 4095 : i64
    %zero = arith.constant 0 : i64
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %link = llvm.mlir.addressof @strict_self_executable : !llvm.ptr
    %length = llvm.call @readlink(%link, %path, %limit) : (!llvm.ptr, !llvm.ptr, i64) -> i64
    %failed = arith.cmpi slt, %length, %zero : i64
    %index = arith.select %failed, %zero, %length : i64
    %end = llvm.getelementptr %path[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %result : f64
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
  llvm.func @sigaction(i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @sigaltstack(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @write(i32, !llvm.ptr, i64) -> i64
  llvm.mlir.global internal constant @strict_overflow_text("Error: StackOverflow, methods called each other without end until the stack was used up\0A\00") : !llvm.array<89 x i8>
  llvm.func @strict_overflow_handler(%signal: i32, %info: !llvm.ptr, %context: !llvm.ptr) {
    %addressSlot = llvm.getelementptr %info[16] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %stackFlags = llvm.getelementptr %alternate[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %noFlags = arith.constant 0 : i32
    llvm.store %noFlags, %stackFlags : i32, !llvm.ptr
    %stackSize = llvm.getelementptr %alternate[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %stackBytes, %stackSize : i64, !llvm.ptr
    %alternated = llvm.call @sigaltstack(%alternate, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    %action = llvm.alloca %one x !llvm.array<152 x i8> : (i64) -> !llvm.ptr
    %actionBytes = arith.constant 152 : i64
    %cleared = llvm.call @memset(%action, %noFlags, %actionBytes) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    llvm.store %handler, %action : !llvm.ptr, !llvm.ptr
    %actionFlags = llvm.getelementptr %action[136] : (!llvm.ptr) -> !llvm.ptr, i8
    %resetOnStackWithInfo = arith.constant -2013265916 : i32
    llvm.store %resetOnStackWithInfo, %actionFlags : i32, !llvm.ptr
    %segmentation = arith.constant 11 : i32
    %installed = llvm.call @sigaction(%segmentation, %action, %null) : (i32, !llvm.ptr, !llvm.ptr) -> i32
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
    %secondsSlot = llvm.getelementptr %buffer[88] : (!llvm.ptr) -> !llvm.ptr, i8
    %seconds = llvm.load %secondsSlot : !llvm.ptr -> i64
    %nanosecondsSlot = llvm.getelementptr %buffer[96] : (!llvm.ptr) -> !llvm.ptr, i8
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
    %modified = llvm.getelementptr %buffer[88] : (!llvm.ptr) -> !llvm.ptr, i8
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
  func.func private @strict_temp_file() -> f64 {
    %size = arith.constant 18 : i64
    %template = llvm.mlir.addressof @strict_temp_template : !llvm.ptr
    %name = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %copied = llvm.call @memcpy(%name, %template, %size) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %descriptor = llvm.call @mkstemp(%name) : (!llvm.ptr) -> i32
    %closed = llvm.call @close(%descriptor) : (i32) -> i32
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
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
  llvm.func @pthread_create(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_mutex_lock(!llvm.ptr) -> i32
  llvm.func @pthread_mutex_unlock(!llvm.ptr) -> i32
  llvm.func @pthread_cond_wait(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_cond_broadcast(!llvm.ptr) -> i32
  llvm.mlir.global internal @strict_pool_lock(dense<0> : tensor<8xi64>) : !llvm.array<8 x i64>
  llvm.mlir.global internal @strict_pool_signal(dense<0> : tensor<8xi64>) : !llvm.array<8 x i64>
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
  func.func private @strict_pool_spawn() -> i64 {
    %online = arith.constant 84 : i32
    %processors = llvm.call @sysconf(%online) : (i32) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    // three quarters of the logical processors (24 of 32 on a 13900KF): a 4K blur 20% faster than with half, GPT-2 decoding
    // as fast as with half, with all of them GPT-2 is 20% slower
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %triple = arith.muli %processors, %three : i64
    %threeQuarters = arith.divsi %triple, %four : i64
    %standard = arith.maxsi %threeQuarters, %two : i64
    %threads = func.call @strict_pool_size(%standard) : (i64) -> i64
    %workers = arith.subi %threads, %one : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_pool_thread : !llvm.ptr
    %handle = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    cf.br ^check(%zero : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %workers : i64
    cf.cond_br %more, ^spawn, ^done
  ^spawn:
    %created = llvm.call @pthread_create(%handle, %null, %procedure, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
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
  llvm.mlir.global internal constant @strict_gpu_driver("libcuda.so.1\00") : !llvm.array<13 x i8>
  llvm.mlir.global internal constant @strict_gpu_amd_driver("libamdhip64.so\00") : !llvm.array<15 x i8>
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %nvidia = llvm.mlir.addressof @strict_gpu_driver : !llvm.ptr
    %amd = llvm.mlir.addressof @strict_gpu_amd_driver : !llvm.ptr
    %name = arith.select %isAmd, %amd, %nvidia : !llvm.ptr
    %now = arith.constant 2 : i32
    %library = llvm.call @dlopen(%name, %now) : (!llvm.ptr, i32) -> !llvm.ptr
    return %library : !llvm.ptr
  }
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %function = llvm.call @dlsym(%library, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    return %function : !llvm.ptr
  }
  llvm.mlir.global internal constant @strict_gpu_ptx_ending(".ptx\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_gpu_older_ending(".older.ptx\00") : !llvm.array<11 x i8>
