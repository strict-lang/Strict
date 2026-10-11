  llvm.func @FindFirstFileA(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @FindNextFileA(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @FindClose(!llvm.ptr) -> i32
  llvm.func @GetFileAttributesA(!llvm.ptr) -> i32
  llvm.func @CreateDirectoryA(!llvm.ptr, !llvm.ptr) -> i32
  llvm.mlir.global internal constant @strict_separator_path("\5C\00") : !llvm.array<2 x i8>
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
    %data = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
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
