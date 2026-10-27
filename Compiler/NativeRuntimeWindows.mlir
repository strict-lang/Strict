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
  // snprintf with one or two numbers, its size is 32 bits in wasm32, newer Windows SDKs only have it inline in stdio.h
  llvm.linker_options ["/DEFAULTLIB:legacy_stdio_definitions.lib"]
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
  llvm.func @VirtualAlloc(!llvm.ptr, i64, i32, i32) -> !llvm.ptr
  llvm.func @VirtualFree(!llvm.ptr, i64, i32) -> i32
  func.func private @strict_reserve(%size: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    %reserve = arith.constant 8192 : i32
    %noAccess = arith.constant 1 : i32
    %reserved = llvm.call @VirtualAlloc(%null, %size, %reserve, %noAccess) : (!llvm.ptr, i64, i32, i32) -> !llvm.ptr
    %failed = llvm.icmp "eq" %reserved, %null : !llvm.ptr
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return %reserved : !llvm.ptr
  }
  func.func private @strict_commit(%at: !llvm.ptr, %size: i64) {
    %null = llvm.mlir.zero : !llvm.ptr
    %commit = arith.constant 4096 : i32
    %readWrite = arith.constant 4 : i32
    %committed = llvm.call @VirtualAlloc(%at, %size, %commit, %readWrite) : (!llvm.ptr, i64, i32, i32) -> !llvm.ptr
    %failed = llvm.icmp "eq" %committed, %null : !llvm.ptr
    cf.cond_br %failed, ^fail, ^done
  ^fail:
    func.call @strict_gc_fail() : () -> ()
    cf.br ^done
  ^done:
    return
  }
  func.func private @strict_decommit(%at: !llvm.ptr, %size: i64) {
    %decommit = arith.constant 16384 : i32
    %freed = llvm.call @VirtualFree(%at, %size, %decommit) : (!llvm.ptr, i64, i32) -> i32
    return
  }
  llvm.func @FindFirstFileA(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @FindNextFileA(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @FindClose(!llvm.ptr) -> i32
  llvm.func @GetFileAttributesA(!llvm.ptr) -> i32
  llvm.func @CreateDirectoryA(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @getenv(!llvm.ptr) -> !llvm.ptr
  llvm.func @SearchPathA(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @GetTempPathA(i32, !llvm.ptr) -> i32
  llvm.func @GetTempFileNameA(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
  llvm.func @CreateFileA(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @GetStdHandle(i32) -> !llvm.ptr
  llvm.func @CreateProcessA(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @WaitForSingleObject(!llvm.ptr, i32) -> i32
  llvm.func @TerminateProcess(!llvm.ptr, i32) -> i32
  llvm.func @GetExitCodeProcess(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @CloseHandle(!llvm.ptr) -> i32
  llvm.func @DeleteFileA(!llvm.ptr) -> i32
  llvm.func @GetModuleFileNameA(!llvm.ptr, !llvm.ptr, i32) -> i32
  llvm.func @GetFileAttributesExA(!llvm.ptr, i32, !llvm.ptr) -> i32
  llvm.func @_localtime64(!llvm.ptr) -> !llvm.ptr
  llvm.mlir.global internal constant @strict_separator_path("/\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_any_file("*\00") : !llvm.array<2 x i8>
  func.func private @strict_directory_exists(%self: f64, %path: f64) -> f64 {
    %invalid = arith.constant -1 : i32
    %directoryFlag = arith.constant 16 : i32
    %none = arith.constant 0 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %attributes = llvm.call @GetFileAttributesA(%name) : (!llvm.ptr) -> i32
    %valid = arith.cmpi ne, %attributes, %invalid : i32
    %masked = arith.andi %attributes, %directoryFlag : i32
    %isDirectory = arith.cmpi ne, %masked, %none : i32
    %exists = arith.andi %valid, %isDirectory : i1
    %result = func.call @strict_boolean(%exists) : (i1) -> f64
    return %result : f64
  }
  llvm.func @GetCurrentDirectoryA(i32, !llvm.ptr) -> i32
  llvm.func @_fdopen(i32, !llvm.ptr) -> !llvm.ptr
  func.func private @strict_standard_input() -> !llvm.ptr {
    %standardInput = arith.constant 0 : i32
    %mode = llvm.mlir.addressof @strict_read_mode : !llvm.ptr
    %stream = llvm.call @_fdopen(%standardInput, %mode) : (i32, !llvm.ptr) -> !llvm.ptr
    return %stream : !llvm.ptr
  }
  func.func private @strict_directory_current(%self: f64) -> f64 {
    %size = arith.constant 1024 : i64
    %limit = arith.constant 1023 : i32
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %length = llvm.call @GetCurrentDirectoryA(%limit, %path) : (i32, !llvm.ptr) -> i32
    %index = arith.extui %length : i32 to i64
    %end = llvm.getelementptr %path[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_directory_create(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %created = llvm.call @CreateDirectoryA(%name, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    %failure = arith.constant 0 : i32
    %made = arith.cmpi ne, %created, %failure : i32
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
  func.func private @strict_directory_files(%self: f64, %path: f64, %pattern: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %size = arith.constant 320 : i64
    %invalidHandle = arith.constant -1 : i64
    %directoryFlag = arith.constant 16 : i32
    %none = arith.constant 0 : i32
    %anyText = llvm.mlir.addressof @strict_any_file : !llvm.ptr
    %any = func.call @strict_bits(%anyText) : (!llvm.ptr) -> f64
    %isUnset = arith.cmpf oeq, %pattern, %zero : f64
    %safePattern = arith.select %isUnset, %any, %pattern : f64
    %patternLength = func.call @strict_text_length(%safePattern) : (f64) -> f64
    %isEmpty = arith.cmpf oeq, %patternLength, %zero : f64
    %chosen = arith.select %isEmpty, %any, %safePattern : f64
    %search = func.call @strict_path_join(%path, %chosen) : (f64, f64) -> f64
    %searchText = func.call @strict_ptr(%search) : (f64) -> !llvm.ptr
    %data = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %created = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    %handle = llvm.call @FindFirstFileA(%searchText, %data) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %handleValue = llvm.ptrtoint %handle : !llvm.ptr to i64
    %failed = arith.cmpi eq, %handleValue, %invalidHandle : i64
    cf.cond_br %failed, ^done(%created : f64), ^entry(%created : f64)
  ^entry(%found: f64):
    %attributes = llvm.load %data : !llvm.ptr -> i32
    %masked = arith.andi %attributes, %directoryFlag : i32
    %isFile = arith.cmpi eq, %masked, %none : i32
    cf.cond_br %isFile, ^add(%found : f64), ^next(%found : f64)
  ^add(%before: f64):
    %fileName = llvm.getelementptr %data[44] : (!llvm.ptr) -> !llvm.ptr, i8
    %nameBits = func.call @strict_bits(%fileName) : (!llvm.ptr) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %copied = func.call @strict_text_concat(%nameBits, %empty) : (f64, f64) -> f64
    %full = func.call @strict_path_join(%path, %copied) : (f64, f64) -> f64
    %added = func.call @strict_list_add(%before, %full) : (f64, f64) -> f64
    cf.br ^next(%added : f64)
  ^next(%current: f64):
    %more = llvm.call @FindNextFileA(%handle, %data) : (!llvm.ptr, !llvm.ptr) -> i32
    %hasMore = arith.cmpi ne, %more, %none : i32
    cf.cond_br %hasMore, ^entry(%current : f64), ^close(%current : f64)
  ^close(%all: f64):
    %closed = llvm.call @FindClose(%handle) : (!llvm.ptr) -> i32
    cf.br ^done(%all : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_directory_directories(%self: f64, %path: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %four = arith.constant 4.0 : f64
    %size = arith.constant 320 : i64
    %invalidHandle = arith.constant -1 : i64
    %directoryFlag = arith.constant 16 : i32
    %none = arith.constant 0 : i32
    %anyText = llvm.mlir.addressof @strict_any_file : !llvm.ptr
    %any = func.call @strict_bits(%anyText) : (!llvm.ptr) -> f64
    %search = func.call @strict_path_join(%path, %any) : (f64, f64) -> f64
    %searchText = func.call @strict_ptr(%search) : (f64) -> !llvm.ptr
    %data = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %created = func.call @strict_list_new(%four, %zero) : (f64, f64) -> f64
    %handle = llvm.call @FindFirstFileA(%searchText, %data) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %handleValue = llvm.ptrtoint %handle : !llvm.ptr to i64
    %failed = arith.cmpi eq, %handleValue, %invalidHandle : i64
    cf.cond_br %failed, ^done(%created : f64), ^entry(%created : f64)
  ^entry(%found: f64):
    %attributes = llvm.load %data : !llvm.ptr -> i32
    %masked = arith.andi %attributes, %directoryFlag : i32
    %isDirectory = arith.cmpi ne, %masked, %none : i32
    %fileName = llvm.getelementptr %data[44] : (!llvm.ptr) -> !llvm.ptr, i8
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
  ^next(%current: f64):
    %more = llvm.call @FindNextFileA(%handle, %data) : (!llvm.ptr, !llvm.ptr) -> i32
    %hasMore = arith.cmpi ne, %more, %none : i32
    cf.cond_br %hasMore, ^entry(%current : f64), ^close(%current : f64)
  ^close(%all: f64):
    %closed = llvm.call @FindClose(%handle) : (!llvm.ptr) -> i32
    cf.br ^done(%all : f64)
  ^done(%result: f64):
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_windows("Windows\00") : !llvm.array<8 x i8>
  llvm.mlir.global internal constant @strict_path_variable("PATH\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_exe_extension(".exe\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_temp_prefix("str\00") : !llvm.array<4 x i8>
  llvm.mlir.global internal constant @strict_quote_space("\22 \00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_start_failed("could not start \00") : !llvm.array<17 x i8>
  // tools and downloads stop after 10 minutes (30 s stopped clang in parallel builds and model downloads)
  llvm.mlir.global internal constant @strict_timed_out("timed out after 10 minutes\00") : !llvm.array<27 x i8>
  func.func private @strict_process_operatingsystem(%self: f64) -> f64 {
    %name = llvm.mlir.addressof @strict_windows : !llvm.ptr
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_process_current(%self: f64) -> f64 {
    %size = arith.constant 1024 : i64
    %limit = arith.constant 1023 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %path = func.call @strict_alloc_text(%size) : (i64) -> !llvm.ptr
    %length = llvm.call @GetModuleFileNameA(%null, %path, %limit) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %index = arith.extui %length : i32 to i64
    %end = llvm.getelementptr %path[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %result = func.call @strict_bits(%path) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.func @_time64(!llvm.ptr) -> i64
  llvm.func @QueryPerformanceCounter(!llvm.ptr) -> i32
  llvm.func @QueryPerformanceFrequency(!llvm.ptr) -> i32
  func.func private @strict_clock() -> f64 {
    %one = arith.constant 1 : i64
    %counter = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %frequency = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %counted = llvm.call @QueryPerformanceCounter(%counter) : (!llvm.ptr) -> i32
    %measured = llvm.call @QueryPerformanceFrequency(%frequency) : (!llvm.ptr) -> i32
    %ticks = llvm.load %counter : !llvm.ptr -> i64
    %perSecond = llvm.load %frequency : !llvm.ptr -> i64
    %ticksFloat = arith.sitofp %ticks : i64 to f64
    %perSecondFloat = arith.sitofp %perSecond : i64 to f64
    %seconds = arith.divf %ticksFloat, %perSecondFloat : f64
    return %seconds : f64
  }
  llvm.func @GetSystemTimePreciseAsFileTime(!llvm.ptr)
  func.func private @strict_time_now(%self: f64) -> f64 {
    %one = arith.constant 1 : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.call @GetSystemTimePreciseAsFileTime(%slot) : (!llvm.ptr) -> ()
    %fileTime = llvm.load %slot : !llvm.ptr -> i64
    %epoch = arith.constant 116444736000000000 : i64
    %since = arith.subi %fileTime, %epoch : i64
    %ticks = arith.sitofp %since : i64 to f64
    %perSecond = arith.constant 1.0e7 : f64
    %seconds = arith.divf %ticks, %perSecond : f64
    return %seconds : f64
  }
  func.func private @strict_random_seed() -> i64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %now = llvm.call @_time64(%null) : (!llvm.ptr) -> i64
    %one = arith.constant 1 : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %address = llvm.ptrtoint %slot : !llvm.ptr to i64
    %mixed = arith.xori %now, %address : i64
    return %mixed : i64
  }
  llvm.func @MoveFileExA(!llvm.ptr, !llvm.ptr, i32) -> i32
  func.func private @strict_file_rename(%path: f64, %newPath: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %replacing = arith.constant 1 : i32
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %newName = func.call @strict_ptr(%newPath) : (f64) -> !llvm.ptr
    %moved = llvm.call @MoveFileExA(%name, %newName, %replacing) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    return %zero : f64
  }
  llvm.mlir.global internal constant @strict_crash_suffix(".crash.txt\00") : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @strict_crash_mode("wb\00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_dialog_library("user32.dll\00") : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @strict_dialog_symbol("MessageBoxA\00") : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @strict_dialog_title("Application error\00") : !llvm.array<18 x i8>
  llvm.func @strict_error_dialog(%text: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    %libraryName = llvm.mlir.addressof @strict_dialog_library : !llvm.ptr
    %library = llvm.call @LoadLibraryA(%libraryName) : (!llvm.ptr) -> !llvm.ptr
    %symbol = llvm.mlir.addressof @strict_dialog_symbol : !llvm.ptr
    %function = llvm.call @GetProcAddress(%library, %symbol) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %missing = llvm.icmp "eq" %function, %null : !llvm.ptr
    llvm.cond_br %missing, ^done, ^show
  ^show:
    %title = llvm.mlir.addressof @strict_dialog_title : !llvm.ptr
    %flags = llvm.mlir.constant(16 : i32) : i32
    %shown = llvm.call %function(%null, %text, %title, %flags) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32
    llvm.br ^done
  ^done:
    llvm.return
  }
  llvm.func @strict_crash_log(%text: !llvm.ptr) {
    %output = llvm.mlir.constant(-11 : i32) : i32
    %handle = llvm.call @GetStdHandle(%output) : (i32) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %windowed = llvm.icmp "eq" %handle, %null : !llvm.ptr
    llvm.cond_br %windowed, ^write, ^done
  ^write:
    llvm.call @strict_error_dialog(%text) : (!llvm.ptr) -> ()
    %size = llvm.mlir.constant(1100 : i32) : i32
    %path = llvm.alloca %size x i8 : (i32) -> !llvm.ptr
    %limit = llvm.mlir.constant(1080 : i32) : i32
    %length = llvm.call @GetModuleFileNameA(%null, %path, %limit) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %offset = llvm.zext %length : i32 to i64
    %end = llvm.getelementptr %path[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %suffix = llvm.mlir.addressof @strict_crash_suffix : !llvm.ptr
    %suffixBytes = llvm.mlir.constant(11 : i64) : i64
    %appended = llvm.call @memcpy(%end, %suffix, %suffixBytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_crash_mode : !llvm.ptr
    %file = llvm.call @fopen(%path, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %unopened = llvm.icmp "eq" %file, %null : !llvm.ptr
    llvm.cond_br %unopened, ^done, ^fill
  ^fill:
    %bytes = llvm.call @strlen(%text) : (!llvm.ptr) -> i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %written = llvm.call @fwrite(%text, %one, %bytes, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    llvm.br ^done
  ^done:
    llvm.return
  }
  llvm.func @AddVectoredExceptionHandler(i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @SetThreadStackGuarantee(!llvm.ptr) -> i32
  llvm.func @ExitProcess(i32)
  llvm.mlir.global internal constant @strict_overflow_text("Error: StackOverflow, methods called each other without end until the stack was used up\0A\00") : !llvm.array<89 x i8>
  llvm.func @strict_overflow_handler(%pointers: !llvm.ptr) -> i32 {
    %record = llvm.load %pointers : !llvm.ptr -> !llvm.ptr
    %code = llvm.load %record : !llvm.ptr -> i32
    %overflow = llvm.mlir.constant(-1073741571 : i32) : i32
    %isOverflow = llvm.icmp "eq" %code, %overflow : i32
    llvm.cond_br %isOverflow, ^report, ^search
  ^report:
    %text = llvm.mlir.addressof @strict_overflow_text : !llvm.ptr
    %printed = llvm.call @printf(%text) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.call @strict_crash_log(%text) : (!llvm.ptr) -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    llvm.call @ExitProcess(%code) : (i32) -> ()
    llvm.br ^search
  ^search:
    %continueSearch = llvm.mlir.constant(0 : i32) : i32
    llvm.return %continueSearch : i32
  }
  llvm.func @SetUnhandledExceptionFilter(!llvm.ptr) -> !llvm.ptr
  llvm.mlir.global internal constant @strict_access_text("Error: AccessViolation, the program read or wrote memory it does not own (an instance used as a list or a missing value)\0A\00") : !llvm.array<122 x i8>
  // an access violation nothing handled (the driver handles its own first) is reported instead of ending silently
  llvm.func @strict_crash_handler(%pointers: !llvm.ptr) -> i32 {
    %record = llvm.load %pointers : !llvm.ptr -> !llvm.ptr
    %code = llvm.load %record : !llvm.ptr -> i32
    %text = llvm.mlir.addressof @strict_access_text : !llvm.ptr
    %printed = llvm.call @printf(%text) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.call @strict_crash_log(%text) : (!llvm.ptr) -> ()
    %null = llvm.mlir.zero : !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    llvm.call @ExitProcess(%code) : (i32) -> ()
    %handled = llvm.mlir.constant(1 : i32) : i32
    llvm.return %handled : i32
  }
  llvm.func @GetCurrentThreadStackLimits(!llvm.ptr, !llvm.ptr)
  func.func private @strict_stack_lowest(%stack: !llvm.ptr) -> i64 {
    %one = arith.constant 1 : i64
    %low = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %high = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.call @GetCurrentThreadStackLimits(%low, %high) : (!llvm.ptr, !llvm.ptr) -> ()
    %lowest = llvm.load %low : !llvm.ptr -> i64
    return %lowest : i64
  }
  func.func private @strict_guard() {
    %first = arith.constant 1 : i32
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    %added = llvm.call @AddVectoredExceptionHandler(%first, %handler) : (i32, !llvm.ptr) -> !llvm.ptr
    %crashHandler = llvm.mlir.addressof @strict_crash_handler : !llvm.ptr
    %previousFilter = llvm.call @SetUnhandledExceptionFilter(%crashHandler) : (!llvm.ptr) -> !llvm.ptr
    %one = arith.constant 1 : i64
    %reserve = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %bytes = arith.constant 65536 : i32
    llvm.store %bytes, %reserve : i32, !llvm.ptr
    %guaranteed = llvm.call @SetThreadStackGuarantee(%reserve) : (!llvm.ptr) -> i32
    return
  }
  func.func private @strict_file_size(%name: !llvm.ptr) -> i64 {
    %dataSize = arith.constant 40 : i64
    %data = llvm.alloca %dataSize x i8 : (i64) -> !llvm.ptr
    %standard = arith.constant 0 : i32
    %found = llvm.call @GetFileAttributesExA(%name, %standard, %data) : (!llvm.ptr, i32, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %missing = arith.cmpi eq, %found, %none : i32
    %zero = arith.constant 0 : i64
    cf.cond_br %missing, ^done(%zero : i64), ^size
  ^size:
    %highSlot = llvm.getelementptr %data[28] : (!llvm.ptr) -> !llvm.ptr, i8
    %high = llvm.load %highSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %lowSlot = llvm.getelementptr %data[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %low = llvm.load %lowSlot {alignment = 4 : i64} : !llvm.ptr -> i32
    %upper = arith.extui %high : i32 to i64
    %lower = arith.extui %low : i32 to i64
    %shift = arith.constant 32 : i64
    %shifted = arith.shli %upper, %shift : i64
    %bytes = arith.ori %shifted, %lower : i64
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
    %dataSize = arith.constant 40 : i64
    %data = llvm.alloca %dataSize x i8 : (i64) -> !llvm.ptr
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %standard = arith.constant 0 : i32
    %found = llvm.call @GetFileAttributesExA(%name, %standard, %data) : (!llvm.ptr, i32, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %missing = arith.cmpi eq, %found, %none : i32
    %zero = arith.constant 0.0 : f64
    cf.cond_br %missing, ^done(%zero : f64), ^time
  ^time:
    %writeSlot = llvm.getelementptr %data[20] : (!llvm.ptr) -> !llvm.ptr, i8
    %fileTime = llvm.load %writeSlot {alignment = 4 : i64} : !llvm.ptr -> i64
    %epoch = arith.constant 116444736000000000 : i64
    %since = arith.subi %fileTime, %epoch : i64
    %ticks = arith.sitofp %since : i64 to f64
    %perSecond = arith.constant 1.0e7 : f64
    %seconds = arith.divf %ticks, %perSecond : f64
    cf.br ^done(%seconds : f64)
  ^done(%result: f64):
    return %result : f64
  }
  func.func private @strict_process_built(%self: f64) -> f64 {
    %one = arith.constant 1 : i64
    %dataSize = arith.constant 40 : i64
    %data = llvm.alloca %dataSize x i8 : (i64) -> !llvm.ptr
    %seconds = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %path = func.call @strict_ptr(%self) : (f64) -> !llvm.ptr
    %standard = arith.constant 0 : i32
    %found = llvm.call @GetFileAttributesExA(%path, %standard, %data) : (!llvm.ptr, i32, !llvm.ptr) -> i32
    %none = arith.constant 0 : i32
    %missing = arith.cmpi eq, %found, %none : i32
    %null = llvm.mlir.zero : !llvm.ptr
    cf.cond_br %missing, ^done(%null : !llvm.ptr), ^time
  ^time:
    %writeSlot = llvm.getelementptr %data[20] : (!llvm.ptr) -> !llvm.ptr, i8
    %fileTime = llvm.load %writeSlot {alignment = 4 : i64} : !llvm.ptr -> i64
    %epoch = arith.constant 116444736000000000 : i64
    %since = arith.subi %fileTime, %epoch : i64
    %ticks = arith.constant 10000000 : i64
    %count = arith.divsi %since, %ticks : i64
    llvm.store %count, %seconds : i64, !llvm.ptr
    %local = llvm.call @_localtime64(%seconds) : (!llvm.ptr) -> !llvm.ptr
    cf.br ^done(%local : !llvm.ptr)
  ^done(%time: !llvm.ptr):
    %result = func.call @strict_time_text(%time) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_tool_path(%name: f64) -> f64 {
    %size = arith.constant 1024 : i32
    %bytes = arith.constant 1024 : i64
    %none = arith.constant 0 : i32
    %variable = llvm.mlir.addressof @strict_path_variable : !llvm.ptr
    %paths = llvm.call @getenv(%variable) : (!llvm.ptr) -> !llvm.ptr
    %extension = llvm.mlir.addressof @strict_exe_extension : !llvm.ptr
    %file = func.call @strict_ptr(%name) : (f64) -> !llvm.ptr
    %buffer = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %length = llvm.call @SearchPathA(%paths, %file, %extension, %size, %buffer, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
    %found = arith.cmpi ugt, %length, %none : i32
    %fits = arith.cmpi ult, %length, %size : i32
    %usable = arith.andi %found, %fits : i1
    %path = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %result = arith.select %usable, %path, %empty : f64
    return %result : f64
  }
  func.func private @strict_temp_file() -> f64 {
    %size = arith.constant 1024 : i32
    %bytes = arith.constant 1024 : i64
    %unique = arith.constant 0 : i32
    %directory = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %length = llvm.call @GetTempPathA(%size, %directory) : (i32, !llvm.ptr) -> i32
    %name = func.call @strict_alloc_text(%bytes) : (i64) -> !llvm.ptr
    %prefix = llvm.mlir.addressof @strict_temp_prefix : !llvm.ptr
    %created = llvm.call @GetTempFileNameA(%directory, %prefix, %unique, %name) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
    %result = func.call @strict_bits(%name) : (!llvm.ptr) -> f64
    return %result : f64
  }
  func.func private @strict_inherited_file(%path: f64) -> !llvm.ptr {
    %write = arith.constant 1073741824 : i32
    %share = arith.constant 3 : i32
    %always = arith.constant 2 : i32
    %normal = arith.constant 128 : i32
    %count = arith.constant 1 : i64
    %size = arith.constant 24 : i64
    %length = arith.constant 24 : i32
    %inherit = arith.constant 1 : i32
    %attributeWords = arith.constant 3 : i64
    %attributes = llvm.alloca %attributeWords x i64 : (i64) -> !llvm.ptr
    %attributesZero = arith.constant 0 : i32
    %attributesCleared = llvm.call @memset(%attributes, %attributesZero, %size) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    llvm.store %length, %attributes : i32, !llvm.ptr
    %inheritSlot = llvm.getelementptr %attributes[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %inherit, %inheritSlot : i32, !llvm.ptr
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %handle = llvm.call @CreateFileA(%name, %write, %share, %attributes, %always, %normal, %null) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
    return %handle : !llvm.ptr
  }
  // holds the file exclusively until this process ends, Windows deletes it then
  func.func private @strict_file_locked(%path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %access = arith.constant 1073807360 : i32
    %share = arith.constant 0 : i32
    %always = arith.constant 4 : i32
    %deleting = arith.constant 67108992 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %handle = llvm.call @CreateFileA(%name, %access, %share, %null, %always, %deleting, %null) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
    %invalid = arith.constant -1 : i64
    %address = llvm.ptrtoint %handle : !llvm.ptr to i64
    %held = arith.cmpi ne, %address, %invalid : i64
    %result = arith.uitofp %held : i1 to f64
    return %result : f64
  }
  func.func private @strict_removed_file(%path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %text = func.call @strict_file_text(%path) : (f64) -> f64
    %deleted = llvm.call @DeleteFileA(%name) : (!llvm.ptr) -> i32
    return %text : f64
  }
  // runs a program with the console of this one (its output appears right away) and returns its exit code
  func.func private @strict_process_started(%path: f64, %arguments: f64) -> f64 {
    %count = arith.constant 1 : i64
    %startupSize = arith.constant 104 : i64
    %startupLength = arith.constant 104 : i32
    %informationSize = arith.constant 24 : i64
    %codeSize = arith.constant 4 : i64
    %inherit = arith.constant 1 : i32
    %noFlags = arith.constant 0 : i32
    %infinite = arith.constant -1 : i32
    %quoteText = llvm.mlir.addressof @strict_quote : !llvm.ptr
    %quote = func.call @strict_bits(%quoteText) : (!llvm.ptr) -> f64
    %quoteSpaceText = llvm.mlir.addressof @strict_quote_space : !llvm.ptr
    %quoteSpace = func.call @strict_bits(%quoteSpaceText) : (!llvm.ptr) -> f64
    %quoted = func.call @strict_text_concat(%quote, %path) : (f64, f64) -> f64
    %program = func.call @strict_text_concat(%quoted, %quoteSpace) : (f64, f64) -> f64
    %command = func.call @strict_text_concat(%program, %arguments) : (f64, f64) -> f64
    %startupWords = arith.constant 13 : i64
    %startup = llvm.alloca %startupWords x i64 : (i64) -> !llvm.ptr
    %startupZero = arith.constant 0 : i32
    %startupCleared = llvm.call @memset(%startup, %startupZero, %startupSize) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    llvm.store %startupLength, %startup : i32, !llvm.ptr
    %informationWords = arith.constant 3 : i64
    %information = llvm.alloca %informationWords x i64 : (i64) -> !llvm.ptr
    %commandText = func.call @strict_ptr(%command) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    func.call @strict_flush() : () -> ()
    %started = llvm.call @CreateProcessA(%null, %commandText, %null, %null, %inherit, %noFlags, %null, %null, %startup, %information) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %failed = arith.cmpi eq, %started, %noFlags : i32
    cf.cond_br %failed, ^notStarted, ^running
  ^notStarted:
    %missing = arith.constant 127.0 : f64
    return %missing : f64
  ^running:
    %process = llvm.load %information : !llvm.ptr -> !llvm.ptr
    %threadSlot = llvm.getelementptr %information[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %thread = llvm.load %threadSlot : !llvm.ptr -> !llvm.ptr
    %waited = llvm.call @WaitForSingleObject(%process, %infinite) : (!llvm.ptr, i32) -> i32
    %codeSlot = llvm.alloca %count x i32 : (i64) -> !llvm.ptr
    %gotCode = llvm.call @GetExitCodeProcess(%process, %codeSlot) : (!llvm.ptr, !llvm.ptr) -> i32
    %exitCode = llvm.load %codeSlot : !llvm.ptr -> i32
    %exitNumber = arith.sitofp %exitCode : i32 to f64
    %closedProcess = llvm.call @CloseHandle(%process) : (!llvm.ptr) -> i32
    %closedThread = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
    return %exitNumber : f64
  }
  llvm.func @SetEnvironmentVariableA(!llvm.ptr, !llvm.ptr) -> i32
  // a child of a tool batch inherits STRICT_THREADS while it starts (a share above 0, set and removed under the lock)
  func.func private @strict_tool_environment(%share: i64, %value: !llvm.ptr) {
    %zero = arith.constant 0 : i64
    %sharing = arith.cmpi sgt, %share, %zero : i64
    cf.cond_br %sharing, ^set, ^done
  ^set:
    %name = llvm.mlir.addressof @strict_threads_name : !llvm.ptr
    %set = llvm.call @SetEnvironmentVariableA(%name, %value) : (!llvm.ptr, !llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return
  }
  // Process.RunTools starts several at once: a child must not inherit the output files of another one, an SRW lock lets
  // the threads waiting for it sleep while CreateProcessA takes milliseconds (spinning cost a core per waiting thread)
  llvm.mlir.global internal @strict_process_lock(0 : i64) : i64
  func.func private @strict_process_execute(%path: f64, %arguments: f64) -> f64 {
    %count = arith.constant 1 : i64
    %startupSize = arith.constant 104 : i64
    %startupLength = arith.constant 104 : i32
    %useHandles = arith.constant 256 : i32
    %inputHandle = arith.constant -10 : i32
    %informationSize = arith.constant 24 : i64
    %inherit = arith.constant 1 : i32
    %noFlags = arith.constant 0 : i32
    %quoteText = llvm.mlir.addressof @strict_quote : !llvm.ptr
    %quote = func.call @strict_bits(%quoteText) : (!llvm.ptr) -> f64
    %quoteSpaceText = llvm.mlir.addressof @strict_quote_space : !llvm.ptr
    %quoteSpace = func.call @strict_bits(%quoteSpaceText) : (!llvm.ptr) -> f64
    %quoted = func.call @strict_text_concat(%quote, %path) : (f64, f64) -> f64
    %program = func.call @strict_text_concat(%quoted, %quoteSpace) : (f64, f64) -> f64
    %command = func.call @strict_text_concat(%program, %arguments) : (f64, f64) -> f64
    %outputPath = func.call @strict_temp_file() : () -> f64
    %errorPath = func.call @strict_temp_file() : () -> f64
    %toolShareSlot = llvm.mlir.addressof @strict_tool_threads : !llvm.ptr
    %toolShare = llvm.load %toolShareSlot : !llvm.ptr -> i64
    %toolShareNumber = arith.sitofp %toolShare : i64 to f64
    %toolShareText = func.call @strict_number_text(%toolShareNumber) : (f64) -> f64
    %toolShareValue = func.call @strict_ptr(%toolShareText) : (f64) -> !llvm.ptr
    %lock = llvm.mlir.addressof @strict_process_lock : !llvm.ptr
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    %output = func.call @strict_inherited_file(%outputPath) : (f64) -> !llvm.ptr
    %error = func.call @strict_inherited_file(%errorPath) : (f64) -> !llvm.ptr
    %startupWords = arith.constant 13 : i64
    %startup = llvm.alloca %startupWords x i64 : (i64) -> !llvm.ptr
    %startupZero = arith.constant 0 : i32
    %startupCleared = llvm.call @memset(%startup, %startupZero, %startupSize) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    llvm.store %startupLength, %startup : i32, !llvm.ptr
    %flagsSlot = llvm.getelementptr %startup[60] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %useHandles, %flagsSlot : i32, !llvm.ptr
    %input = llvm.call @GetStdHandle(%inputHandle) : (i32) -> !llvm.ptr
    %inputSlot = llvm.getelementptr %startup[80] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %input, %inputSlot : !llvm.ptr, !llvm.ptr
    %outputSlot = llvm.getelementptr %startup[88] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %output, %outputSlot : !llvm.ptr, !llvm.ptr
    %errorSlot = llvm.getelementptr %startup[96] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %error, %errorSlot : !llvm.ptr, !llvm.ptr
    %informationWords = arith.constant 3 : i64
    %information = llvm.alloca %informationWords x i64 : (i64) -> !llvm.ptr
    %commandText = func.call @strict_ptr(%command) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    func.call @strict_tool_environment(%toolShare, %toolShareValue) : (i64, !llvm.ptr) -> ()
    %started = llvm.call @CreateProcessA(%null, %commandText, %null, %null, %inherit, %noFlags, %null, %null, %startup, %information) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    func.call @strict_tool_environment(%toolShare, %null) : (i64, !llvm.ptr) -> ()
    %closedOutput = llvm.call @CloseHandle(%output) : (!llvm.ptr) -> i32
    %closedError = llvm.call @CloseHandle(%error) : (!llvm.ptr) -> i32
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    %failed = arith.cmpi eq, %started, %noFlags : i32
    cf.cond_br %failed, ^notStarted, ^running
  ^notStarted:
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
  ^running:
    %timeout = arith.constant 600000 : i32
    %process = llvm.load %information : !llvm.ptr -> !llvm.ptr
    %threadSlot = llvm.getelementptr %information[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %thread = llvm.load %threadSlot : !llvm.ptr -> !llvm.ptr
    %waited = llvm.call @WaitForSingleObject(%process, %timeout) : (!llvm.ptr, i32) -> i32
    %timedOut = arith.cmpi ne, %waited, %noFlags : i32
    cf.cond_br %timedOut, ^kill, ^exited
  ^kill:
    %killed = llvm.call @TerminateProcess(%process, %inherit) : (!llvm.ptr, i32) -> i32
    %drain = arith.constant 2000 : i32
    %ended = llvm.call @WaitForSingleObject(%process, %drain) : (!llvm.ptr, i32) -> i32
    %timeoutCode = arith.constant 124.0 : f64
    cf.br ^collect(%timeoutCode : f64)
  ^exited:
    %codeSlot = llvm.alloca %count x i32 : (i64) -> !llvm.ptr
    %gotCode = llvm.call @GetExitCodeProcess(%process, %codeSlot) : (!llvm.ptr, !llvm.ptr) -> i32
    %exitCode = llvm.load %codeSlot : !llvm.ptr -> i32
    %exitNumber = arith.uitofp %exitCode : i32 to f64
    cf.br ^collect(%exitNumber : f64)
  ^collect(%code: f64):
    %closedProcess = llvm.call @CloseHandle(%process) : (!llvm.ptr) -> i32
    %closedThread = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
    %written = func.call @strict_removed_file(%outputPath) : (f64) -> f64
    %reported = func.call @strict_removed_file(%errorPath) : (f64) -> f64
    %timedOutText = llvm.mlir.addressof @strict_timed_out : !llvm.ptr
    %timedOutMessage = func.call @strict_bits(%timedOutText) : (!llvm.ptr) -> f64
    %message = arith.select %timedOut, %timedOutMessage, %reported : f64
    %result = func.call @strict_process_result(%code, %written, %message) : (f64, f64, f64) -> f64
    return %result : f64
  }
  llvm.func @LoadLibraryA(!llvm.ptr) -> !llvm.ptr
  llvm.func @GetProcAddress(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @GetFullPathNameA(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  func.func private @strict_library_call(%path: f64, %symbol: f64, %arguments: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %size = arith.constant 4096 : i32
    %full = llvm.alloca %size x i8 : (i32) -> !llvm.ptr
    %noPart = llvm.mlir.zero : !llvm.ptr
    %fullLength = llvm.call @GetFullPathNameA(%name, %size, %full, %noPart) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
    %library = llvm.call @LoadLibraryA(%full) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %isMissing = llvm.icmp "eq" %library, %null : !llvm.ptr
    cf.cond_br %isMissing, ^noLibrary, ^loaded
  ^noLibrary:
    %libraryMessage = llvm.mlir.addressof @strict_library_missing : !llvm.ptr
    %libraryError = func.call @strict_error_named(%libraryMessage) : (!llvm.ptr) -> f64
    return %libraryError : f64
  ^loaded:
    %symbolName = func.call @strict_ptr(%symbol) : (f64) -> !llvm.ptr
    %function = llvm.call @GetProcAddress(%library, %symbolName) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
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
  llvm.func @GetActiveProcessorCount(i16) -> i32
  llvm.func @GetPhysicallyInstalledSystemMemory(!llvm.ptr) -> i32
  llvm.func @CreateThread(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @AcquireSRWLockExclusive(!llvm.ptr)
  llvm.func @ReleaseSRWLockExclusive(!llvm.ptr)
  llvm.func @SleepConditionVariableSRW(!llvm.ptr, !llvm.ptr, i32, i32) -> i32
  llvm.func @WakeAllConditionVariable(!llvm.ptr)
  llvm.func @WakeConditionVariable(!llvm.ptr)
  llvm.mlir.global internal @strict_pool_lock(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_signal(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_finish(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_awaiting(0 : i64) : i64
  llvm.func @strict_pool_thread(%argument: !llvm.ptr) -> i32 {
    llvm.call @strict_pool_serve() : () -> ()
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
  llvm.func @strict_gpu_thread(%argument: !llvm.ptr) -> i32 {
    func.call @strict_gpu_warming() : () -> ()
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
  func.func private @strict_gpu_spawn() -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_gpu_thread : !llvm.ptr
    %defaultStack = arith.constant 0 : i64
    %noFlags = arith.constant 0 : i32
    %thread = llvm.call @CreateThread(%null, %defaultStack, %procedure, %null, %noFlags, %null) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %created = llvm.icmp "ne" %thread, %null : !llvm.ptr
    cf.cond_br %created, ^close, ^done
  ^close:
    %closed = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return %created : i1
  }
  // bytes of physical memory, 0 when unknown
  func.func private @strict_host_memory() -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %kilobyte = arith.constant 1024 : i64
    %installed = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    llvm.store %zero, %installed : i64, !llvm.ptr
    %known = llvm.call @GetPhysicallyInstalledSystemMemory(%installed) : (!llvm.ptr) -> i32
    %kilobytes = llvm.load %installed : !llvm.ptr -> i64
    %bytes = arith.muli %kilobytes, %kilobyte : i64
    return %bytes : i64
  }
  func.func private @strict_pool_spawn() -> i64 {
    %groups = arith.constant -1 : i16
    %processors = llvm.call @GetActiveProcessorCount(%groups) : (i16) -> i32
    %wide = arith.extui %processors : i32 to i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    // three quarters of the logical processors (24 of 32 on a 13900KF): a 4K blur 20% faster than with half, GPT-2 decoding
    // as fast as with half, with all of them GPT-2 is 20% slower
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %triple = arith.muli %wide, %three : i64
    %threeQuarters = arith.divsi %triple, %four : i64
    %standard = arith.maxsi %threeQuarters, %two : i64
    %threads = func.call @strict_pool_size(%standard) : (i64) -> i64
    %workers = arith.subi %threads, %one : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_pool_thread : !llvm.ptr
    // 64 MB reserved like the main thread (STACK_SIZE_PARAM_IS_A_RESERVATION), loop bodies recurse as deep as it
    %workerStack = arith.constant 67108864 : i64
    %reserved = arith.constant 65536 : i32
    cf.br ^check(%zero : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %workers : i64
    cf.cond_br %more, ^spawn, ^done
  ^spawn:
    %thread = llvm.call @CreateThread(%null, %workerStack, %procedure, %null, %reserved, %null) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %closed = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
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
    %forever = llvm.mlir.constant(-1 : i32) : i32
    %exclusive = llvm.mlir.constant(0 : i32) : i32
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    %sleeping = llvm.atomicrmw add %sleepersSlot, %one seq_cst : !llvm.ptr, i64
    llvm.br ^check
  ^check:
    %current = llvm.load %jobSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %same = llvm.icmp "eq" %current, %seen : i64
    llvm.cond_br %same, ^sleep, ^leave
  ^sleep:
    %slept = llvm.call @SleepConditionVariableSRW(%signal, %lock, %forever, %exclusive) : (!llvm.ptr, !llvm.ptr, i32, i32) -> i32
    llvm.br ^check
  ^leave:
    %awake = llvm.atomicrmw sub %sleepersSlot, %one seq_cst : !llvm.ptr, i64
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.return
  }
  // wakes as many sleeping workers as the job has chunks for besides the caller's: waking all 95 for a job of 4 items
  // cost the caller about 0.3 ms, they all took the lock again before the next job could wake anyone
  llvm.func @strict_pool_wake(%helpers: i64) {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_signal : !llvm.ptr
    %sleepersSlot = llvm.mlir.addressof @strict_pool_sleepers : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %sleepers = llvm.load %sleepersSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %anyone = llvm.icmp "ne" %sleepers, %zero : i64
    %needed = llvm.icmp "sgt" %helpers, %zero : i64
    %waking = llvm.and %anyone, %needed : i1
    llvm.cond_br %waking, ^wake, ^leave
  ^wake:
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    %everyone = llvm.icmp "sge" %helpers, %sleepers : i64
    llvm.cond_br %everyone, ^all, ^some(%zero : i64)
  ^all:
    llvm.call @WakeAllConditionVariable(%signal) : (!llvm.ptr) -> ()
    llvm.return
  ^some(%woken: i64):
    %more = llvm.icmp "slt" %woken, %helpers : i64
    llvm.cond_br %more, ^single, ^leave
  ^single:
    llvm.call @WakeConditionVariable(%signal) : (!llvm.ptr) -> ()
    %next = llvm.add %woken, %one : i64
    llvm.br ^some(%next : i64)
  ^leave:
    llvm.return
  }
  // the caller of a job sleeps here until its last chunk is done instead of spinning through a slow chunk
  // (a folder test's caller spun a core for the slowest child), strict_pool_finished of that chunk wakes it
  llvm.func @strict_pool_await(%count: i64) {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_finish : !llvm.ptr
    %doneSlot = llvm.mlir.addressof @strict_pool_done : !llvm.ptr
    %awaitingSlot = llvm.mlir.addressof @strict_pool_awaiting : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.store %one, %awaitingSlot atomic seq_cst {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^check
  ^check:
    %done = llvm.load %doneSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %finished = llvm.icmp "sge" %done, %count : i64
    llvm.cond_br %finished, ^leave, ^sleep
  ^sleep:
    %forever = llvm.mlir.constant(-1 : i32) : i32
    %exclusive = llvm.mlir.constant(0 : i32) : i32
    %slept = llvm.call @SleepConditionVariableSRW(%signal, %lock, %forever, %exclusive) : (!llvm.ptr, !llvm.ptr, i32, i32) -> i32
    llvm.br ^check
  ^leave:
    llvm.store %zero, %awaitingSlot atomic seq_cst {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @strict_pool_finished() {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_finish : !llvm.ptr
    %awaitingSlot = llvm.mlir.addressof @strict_pool_awaiting : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %awaiting = llvm.load %awaitingSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %asleep = llvm.icmp "ne" %awaiting, %zero : i64
    llvm.cond_br %asleep, ^wake, ^leave
  ^wake:
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.call @WakeConditionVariable(%signal) : (!llvm.ptr) -> ()
    llvm.return
  ^leave:
    llvm.return
  }
  llvm.mlir.global internal constant @strict_gpu_driver("nvcuda.dll\00") : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @strict_gpu_amd_driver("amdhip64.dll\00") : !llvm.array<13 x i8>
  func.func private @strict_gpu_library(%isAmd: i1) -> !llvm.ptr {
    %nvidia = llvm.mlir.addressof @strict_gpu_driver : !llvm.ptr
    %amd = llvm.mlir.addressof @strict_gpu_amd_driver : !llvm.ptr
    %name = arith.select %isAmd, %amd, %nvidia : !llvm.ptr
    %library = llvm.call @LoadLibraryA(%name) : (!llvm.ptr) -> !llvm.ptr
    return %library : !llvm.ptr
  }
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %function = llvm.call @GetProcAddress(%library, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    return %function : !llvm.ptr
  }
  llvm.mlir.global internal constant @strict_gpu_ptx_ending(".ptx\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @strict_gpu_older_ending(".older.ptx\00") : !llvm.array<11 x i8>

  llvm.func @Sleep(i32)
  // Process.Wait: blocks this thread for the given seconds
  func.func private @strict_process_wait(%self: f64, %seconds: f64) -> f64 {
    %thousand = arith.constant 1000.0 : f64
    %milliseconds = arith.mulf %seconds, %thousand : f64
    %wait = arith.fptosi %milliseconds : f64 to i32
    llvm.call @Sleep(%wait) : (i32) -> ()
    %none = arith.constant 0.0 : f64
    return %none : f64
  }
  func.func private @strict_test_pause() {
    %wait = arith.constant 1 : i32
    llvm.call @Sleep(%wait) : (i32) -> ()
    return
  }
  llvm.func @strict_test_thread(%argument: !llvm.ptr) -> i32 {
    func.call @strict_test_watch() : () -> ()
    %zero = arith.constant 0 : i32
    llvm.return %zero : i32
  }
  func.func private @strict_test_spawn() -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_test_thread : !llvm.ptr
    %stack = arith.constant 0 : i64
    %flags = arith.constant 0 : i32
    %thread = llvm.call @CreateThread(%null, %stack, %procedure, %null, %flags, %null) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %created = llvm.icmp "ne" %thread, %null : !llvm.ptr
    cf.cond_br %created, ^close, ^done
  ^close:
    %closed = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
    cf.br ^done
  ^done:
    return %created : i1
  }

  func.func private @strict_test_quit() {
    %current = arith.constant -1 : i64
    %process = llvm.inttoptr %current : i64 to !llvm.ptr
    %failure = arith.constant 1 : i32
    %killed = llvm.call @TerminateProcess(%process, %failure) : (!llvm.ptr, i32) -> i32
    return
  }
