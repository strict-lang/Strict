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
  llvm.func @calloc(i64, i64) -> !llvm.ptr
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
    return %zero : f64
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
  // ponytail: a tool or download running longer than 10 minutes is stopped, a timeout argument for Process.Run would let
  // callers choose (30 s stopped clang in parallel builds and the curl downloads of the GPT-2 and Qwen3 models)
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
    %null = llvm.mlir.zero : !llvm.ptr
    %flushed = llvm.call @fflush(%null) : (!llvm.ptr) -> i32
    llvm.call @ExitProcess(%code) : (i32) -> ()
    llvm.br ^search
  ^search:
    %continueSearch = llvm.mlir.constant(0 : i32) : i32
    llvm.return %continueSearch : i32
  }
  func.func private @strict_guard() {
    %first = arith.constant 1 : i32
    %handler = llvm.mlir.addressof @strict_overflow_handler : !llvm.ptr
    %added = llvm.call @AddVectoredExceptionHandler(%first, %handler) : (i32, !llvm.ptr) -> !llvm.ptr
    %one = arith.constant 1 : i64
    %reserve = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %bytes = arith.constant 65536 : i32
    llvm.store %bytes, %reserve : i32, !llvm.ptr
    %guaranteed = llvm.call @SetThreadStackGuarantee(%reserve) : (!llvm.ptr) -> i32
    return
  }
  func.func private @strict_file_modified(%path: f64) -> f64 {
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
    %attributes = llvm.call @calloc(%count, %size) : (i64, i64) -> !llvm.ptr
    llvm.store %length, %attributes : i32, !llvm.ptr
    %inheritSlot = llvm.getelementptr %attributes[16] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %inherit, %inheritSlot : i32, !llvm.ptr
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %handle = llvm.call @CreateFileA(%name, %write, %share, %attributes, %always, %normal, %null) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
    return %handle : !llvm.ptr
  }
  func.func private @strict_removed_file(%path: f64) -> f64 {
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %text = func.call @strict_file_text(%path) : (f64) -> f64
    %deleted = llvm.call @DeleteFileA(%name) : (!llvm.ptr) -> i32
    return %text : f64
  }
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
    %output = func.call @strict_inherited_file(%outputPath) : (f64) -> !llvm.ptr
    %error = func.call @strict_inherited_file(%errorPath) : (f64) -> !llvm.ptr
    %startup = llvm.call @calloc(%count, %startupSize) : (i64, i64) -> !llvm.ptr
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
    %information = llvm.call @calloc(%count, %informationSize) : (i64, i64) -> !llvm.ptr
    %commandText = func.call @strict_ptr(%command) : (f64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %started = llvm.call @CreateProcessA(%null, %commandText, %null, %null, %inherit, %noFlags, %null, %null, %startup, %information) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %failed = arith.cmpi eq, %started, %noFlags : i32
    cf.cond_br %failed, ^notStarted, ^running
  ^notStarted:
    %closedOutput = llvm.call @CloseHandle(%output) : (!llvm.ptr) -> i32
    %closedError = llvm.call @CloseHandle(%error) : (!llvm.ptr) -> i32
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
    %codeSlot = llvm.call @calloc(%count, %count) : (i64, i64) -> !llvm.ptr
    %gotCode = llvm.call @GetExitCodeProcess(%process, %codeSlot) : (!llvm.ptr, !llvm.ptr) -> i32
    %exitCode = llvm.load %codeSlot : !llvm.ptr -> i32
    %exitNumber = arith.uitofp %exitCode : i32 to f64
    cf.br ^collect(%exitNumber : f64)
  ^collect(%code: f64):
    %closedProcess = llvm.call @CloseHandle(%process) : (!llvm.ptr) -> i32
    %closedThread = llvm.call @CloseHandle(%thread) : (!llvm.ptr) -> i32
    %closedOut = llvm.call @CloseHandle(%output) : (!llvm.ptr) -> i32
    %closedErr = llvm.call @CloseHandle(%error) : (!llvm.ptr) -> i32
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
  llvm.func @CreateThread(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @AcquireSRWLockExclusive(!llvm.ptr)
  llvm.func @ReleaseSRWLockExclusive(!llvm.ptr)
  llvm.func @SleepConditionVariableSRW(!llvm.ptr, !llvm.ptr, i32, i32) -> i32
  llvm.func @WakeAllConditionVariable(!llvm.ptr)
  llvm.mlir.global internal @strict_pool_lock(0 : i64) : i64
  llvm.mlir.global internal @strict_pool_signal(0 : i64) : i64
  llvm.func @strict_pool_thread(%argument: !llvm.ptr) -> i32 {
    llvm.call @strict_pool_serve() : () -> ()
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
  func.func private @strict_pool_spawn() -> i64 {
    %groups = arith.constant -1 : i16
    %processors = llvm.call @GetActiveProcessorCount(%groups) : (i16) -> i32
    %wide = arith.extui %processors : i32 to i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    // ponytail: a quarter of the logical processors was fastest for memory bound GPT-2 decoding (13900KF), compute bound loops could use all
    %four = arith.constant 4 : i64
    %two = arith.constant 2 : i64
    %quarter = arith.divsi %wide, %four : i64
    %threads = arith.maxsi %quarter, %two : i64
    %workers = arith.subi %threads, %one : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %procedure = llvm.mlir.addressof @strict_pool_thread : !llvm.ptr
    %defaultStack = arith.constant 0 : i64
    %noFlags = arith.constant 0 : i32
    cf.br ^check(%zero : i64)
  ^check(%index: i64):
    %more = arith.cmpi slt, %index, %workers : i64
    cf.cond_br %more, ^spawn, ^done
  ^spawn:
    %thread = llvm.call @CreateThread(%null, %defaultStack, %procedure, %null, %noFlags, %null) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
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
  llvm.func @strict_pool_wake() {
    %lock = llvm.mlir.addressof @strict_pool_lock : !llvm.ptr
    %signal = llvm.mlir.addressof @strict_pool_signal : !llvm.ptr
    %sleepersSlot = llvm.mlir.addressof @strict_pool_sleepers : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %sleepers = llvm.load %sleepersSlot atomic seq_cst {alignment = 8 : i64} : !llvm.ptr -> i64
    %anyone = llvm.icmp "ne" %sleepers, %zero : i64
    llvm.cond_br %anyone, ^wake, ^leave
  ^wake:
    llvm.call @AcquireSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.call @ReleaseSRWLockExclusive(%lock) : (!llvm.ptr) -> ()
    llvm.call @WakeAllConditionVariable(%signal) : (!llvm.ptr) -> ()
    llvm.return
  ^leave:
    llvm.return
  }
  llvm.mlir.global internal constant @strict_gpu_driver("nvcuda.dll\00") : !llvm.array<11 x i8>
  func.func private @strict_gpu_library() -> !llvm.ptr {
    %name = llvm.mlir.addressof @strict_gpu_driver : !llvm.ptr
    %library = llvm.call @LoadLibraryA(%name) : (!llvm.ptr) -> !llvm.ptr
    return %library : !llvm.ptr
  }
  func.func private @strict_gpu_lookup(%library: !llvm.ptr, %name: !llvm.ptr) -> !llvm.ptr {
    %function = llvm.call @GetProcAddress(%library, %name) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    return %function : !llvm.ptr
  }
