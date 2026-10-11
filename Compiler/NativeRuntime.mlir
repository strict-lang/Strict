  llvm.mlir.global internal @strict_heap_next(0 : i64) : i64
  llvm.mlir.global internal @strict_heap_end(0 : i64) : i64
  func.func private @strict_alloc(%size: i64) -> !llvm.ptr {
    %seven = arith.constant 7 : i64
    %mask = arith.constant -8 : i64
    %chunk = arith.constant 67108864 : i64
    %padded = arith.addi %size, %seven : i64
    %rounded = arith.andi %padded, %mask : i64
    %nextSlot = llvm.mlir.addressof @strict_heap_next : !llvm.ptr
    %endSlot = llvm.mlir.addressof @strict_heap_end : !llvm.ptr
    %next = llvm.load %nextSlot : !llvm.ptr -> i64
    %end = llvm.load %endSlot : !llvm.ptr -> i64
    %after = arith.addi %next, %rounded : i64
    %fits = arith.cmpi ule, %after, %end : i64
    cf.cond_br %fits, ^take(%next : i64), ^refill
  ^refill:
    %big = arith.cmpi ugt, %rounded, %chunk : i64
    %chunkSize = arith.select %big, %rounded, %chunk : i64
    %block = llvm.call @malloc(%chunkSize) : (i64) -> !llvm.ptr
    %start = llvm.ptrtoint %block : !llvm.ptr to i64
    %blockEnd = arith.addi %start, %chunkSize : i64
    llvm.store %blockEnd, %endSlot : i64, !llvm.ptr
    cf.br ^take(%start : i64)
  ^take(%at: i64):
    %taken = arith.addi %at, %rounded : i64
    llvm.store %taken, %nextSlot : i64, !llvm.ptr
    %result = llvm.inttoptr %at : i64 to !llvm.ptr
    return %result : !llvm.ptr
  }
  func.func private @strict_ptr(%bits: f64) -> !llvm.ptr {
    %i = arith.bitcast %bits : f64 to i64
    %p = llvm.inttoptr %i : i64 to !llvm.ptr
    return %p : !llvm.ptr
  }
  func.func private @strict_bits(%p: !llvm.ptr) -> f64 {
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    %f = arith.bitcast %i : i64 to f64
    return %f : f64
  }
  func.func private @strict_list_new(%capacity: f64, %length: f64) -> f64 {
    %c16 = arith.constant 16 : i64
    %c24 = arith.constant 24 : i64
    %eight = arith.constant 8.0 : f64
    %view = func.call @strict_alloc(%c16) : (i64) -> !llvm.ptr
    %holder = func.call @strict_alloc(%c24) : (i64) -> !llvm.ptr
    %bytesf = arith.mulf %capacity, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %items = func.call @strict_alloc(%bytes) : (i64) -> !llvm.ptr
    %itemsbits = func.call @strict_bits(%items) : (!llvm.ptr) -> f64
    llvm.store %itemsbits, %holder : f64, !llvm.ptr
    %used = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %length, %used : f64, !llvm.ptr
    %cap = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %capacity, %cap : f64, !llvm.ptr
    llvm.store %length, %view : f64, !llvm.ptr
    %holderbits = func.call @strict_bits(%holder) : (!llvm.ptr) -> f64
    %viewholder = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %holderbits, %viewholder : f64, !llvm.ptr
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
  func.func private @strict_list_copy(%list: f64, %extra: f64) -> f64 {
    %two = arith.constant 2.0 : f64
    %eight = arith.constant 8.0 : f64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %doubled = arith.mulf %length, %two : f64
    %capacity = arith.addf %doubled, %extra : f64
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
    cf.cond_br %newest, ^grow, ^copy
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
    %four = arith.constant 4.0 : f64
    %eight = arith.constant 8.0 : f64
    %leftLength = func.call @strict_list_length(%left) : (f64) -> f64
    %rightLength = func.call @strict_list_length(%right) : (f64) -> f64
    %extra = arith.addf %rightLength, %four : f64
    %copy = func.call @strict_list_copy(%left, %extra) : (f64, f64) -> f64
    %to = func.call @strict_list_items(%copy) : (f64) -> !llvm.ptr
    %offset = arith.fptosi %leftLength : f64 to i64
    %target = llvm.getelementptr %to[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %from = func.call @strict_list_items(%right) : (f64) -> !llvm.ptr
    %bytesf = arith.mulf %rightLength, %eight : f64
    %bytes = arith.fptosi %bytesf : f64 to i64
    %done = llvm.call @memcpy(%target, %from, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %total = arith.addf %leftLength, %rightLength : f64
    %view = func.call @strict_ptr(%copy) : (f64) -> !llvm.ptr
    llvm.store %total, %view : f64, !llvm.ptr
    %holderslot = llvm.getelementptr %view[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %holderbits = llvm.load %holderslot : !llvm.ptr -> f64
    %holder = func.call @strict_ptr(%holderbits) : (f64) -> !llvm.ptr
    %used = llvm.getelementptr %holder[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %total, %used : f64, !llvm.ptr
    return %copy : f64
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
    %a = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %offset = arith.fptosi %start : f64 to i64
    %count = arith.fptosi %length : f64 to i64
    %size = arith.addi %count, %one : i64
    %copy = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
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
    %copy = func.call @strict_alloc(%two) : (i64) -> !llvm.ptr
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
    %buffer = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
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
    %buffer = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %first = llvm.call @memcpy(%buffer, %a, %leftLength) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %end = llvm.getelementptr %buffer[%leftLength] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %rightSize = arith.addi %rightLength, %one : i64
    %second = llvm.call @memcpy(%end, %b, %rightSize) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %result = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %result : f64
  }
  llvm.mlir.global internal constant @strict_separator(", \00") : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @strict_open("(\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @strict_close(")\00") : !llvm.array<2 x i8>
  func.func private @strict_list_text(%list: f64, %isText: i1) -> f64 {
    %start = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %length = func.call @strict_list_length(%list) : (f64) -> f64
    %count = arith.fptosi %length : f64 to i64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %emptyText = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %empty = func.call @strict_bits(%emptyText) : (!llvm.ptr) -> f64
    %isSingle = arith.cmpi eq, %count, %step : i64
    %openText = llvm.mlir.addressof @strict_open : !llvm.ptr
    %open = func.call @strict_bits(%openText) : (!llvm.ptr) -> f64
    %prefix = arith.select %isSingle, %empty, %open : f64
    cf.br ^check(%start, %prefix : i64, f64)
  ^check(%index: i64, %text: f64):
    %more = arith.cmpi slt, %index, %count : i64
    cf.cond_br %more, ^append, ^finish(%text : f64)
  ^append:
    %slot = llvm.getelementptr %items[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %value = llvm.load %slot : !llvm.ptr -> f64
    %number = func.call @strict_number_text(%value) : (f64) -> f64
    %element = arith.select %isText, %value, %number : f64
    %isFirst = arith.cmpi eq, %index, %start : i64
    %separatorText = llvm.mlir.addressof @strict_separator : !llvm.ptr
    %separator = func.call @strict_bits(%separatorText) : (!llvm.ptr) -> f64
    %gap = arith.select %isFirst, %empty, %separator : f64
    %spaced = func.call @strict_text_concat(%text, %gap) : (f64, f64) -> f64
    %joined = func.call @strict_text_concat(%spaced, %element) : (f64, f64) -> f64
    %next = arith.addi %index, %step : i64
    cf.br ^check(%next, %joined : i64, f64)
  ^finish(%result: f64):
    %isMultiple = arith.cmpi sgt, %count, %step : i64
    %closeText = llvm.mlir.addressof @strict_close : !llvm.ptr
    %close = func.call @strict_bits(%closeText) : (!llvm.ptr) -> f64
    %suffix = arith.select %isMultiple, %close, %empty : f64
    %closed = func.call @strict_text_concat(%result, %suffix) : (f64, f64) -> f64
    return %closed : f64
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
    %buffer = func.call @strict_alloc(%capacity) : (i64) -> !llvm.ptr
    %count = llvm.call @fread(%buffer, %one, %size, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %end = llvm.getelementptr %buffer[%count] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
    %result = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    return %result : f64
  ^absent:
    %empty = llvm.mlir.addressof @strict_empty : !llvm.ptr
    %none = func.call @strict_bits(%empty) : (!llvm.ptr) -> f64
    return %none : f64
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
  func.func private @strict_file_write(%path: f64, %text: f64) -> f64 {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0.0 : f64
    %name = func.call @strict_ptr(%path) : (f64) -> !llvm.ptr
    %mode = llvm.mlir.addressof @strict_write_mode : !llvm.ptr
    %file = llvm.call @fopen(%name, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %content = func.call @strict_ptr(%text) : (f64) -> !llvm.ptr
    %length = llvm.call @strlen(%content) : (!llvm.ptr) -> i64
    %written = llvm.call @fwrite(%content, %one, %length, %file) : (!llvm.ptr, i64, i64, !llvm.ptr) -> i64
    %closed = llvm.call @fclose(%file) : (!llvm.ptr) -> i32
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
    %size = arith.constant 8 : i64
    %path = func.call @strict_tool_path(%name) : (f64) -> f64
    %block = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    llvm.store %path, %block : f64, !llvm.ptr
    %result = func.call @strict_bits(%block) : (!llvm.ptr) -> f64
    return %result : f64
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
