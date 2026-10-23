"""Check native row-cache compatibility; run from Strict with LLVM/MLIR installed."""
import re, subprocess, ctypes, tempfile
from pathlib import Path
temporary=tempfile.TemporaryDirectory(prefix='strict-cache-')
root=Path(temporary.name)
source=Path('Compiler/NativeRuntime.mlir').read_text(encoding='utf-8')
helpers=[]
for name in ['strict_gpu_active_has', 'strict_gpu_oldest']:
	match=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)
	assert match, 'Missing active-input lifetime protection for GPU cache eviction: '+name
	helpers.append(match[0])
cached=re.search(r'  func.func private @strict_gpu_cached\(.*?\n  \}',source,re.S)[0]
module='''module {
  llvm.mlir.global internal @testSlot(dense<0> : tensor<8xi64>) : !llvm.array<8 x i64>
  func.func private @strict_gpu_slot(%key: i64, %probe: i64) -> !llvm.ptr {
    %slot = llvm.mlir.addressof @testSlot : !llvm.ptr
    return %slot : !llvm.ptr
  }
  func.func private @strict_gpu_tick() -> i64 {
    %one = arith.constant 1 : i64
    return %one : i64
  }
  func.func private @strict_gpu_view(%out: !llvm.ptr, %data: i64, %length: i64, %bits: i64) {
    llvm.store %data, %out : i64, !llvm.ptr
    return
  }
'''+cached+'''
  llvm.func @cached_kind(%stored: i64, %requested: i64) -> i32 {
    %slot = llvm.mlir.addressof @testSlot : !llvm.ptr
    %key = arith.constant 42 : i64
    %count = arith.constant 100 : i64
    %three = arith.constant 3 : i64
    %out = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    %kindSlot = llvm.getelementptr %slot[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %countSlot = llvm.getelementptr %slot[6] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %key, %slot : i64, !llvm.ptr
    llvm.store %stored, %kindSlot : i64, !llvm.ptr
    llvm.store %count, %countSlot : i64, !llvm.ptr
    %list = arith.bitcast %key : i64 to f64
    %hit = func.call @strict_gpu_cached(%list, %requested, %count, %out) : (f64, i64, i64, !llvm.ptr) -> i1
    %result = arith.extui %hit : i1 to i32
    llvm.return %result : i32
  }
}
'''
pressure='  llvm.mlir.global internal @strict_gpu_active(0 : i64) : i64\n  llvm.mlir.global internal @strict_gpu_cache(dense<0> : tensor<524288xi64>) : !llvm.array<524288 x i64>\n  func.func private @strict_gpu_entry(%index: i64) -> !llvm.ptr {\n    %eight = arith.constant 8 : i64\n    %cache = llvm.mlir.addressof @strict_gpu_cache : !llvm.ptr\n    %offset = arith.muli %index, %eight : i64\n    %entry = llvm.getelementptr %cache[0, %offset] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<524288 x i64>\n    return %entry : !llvm.ptr\n  }\n__HELPERS__\n  llvm.func @scope_choice(%mode: i64) -> i64 {\n    %zero = arith.constant 0 : i64\n    %one = arith.constant 1 : i64\n    %two = arith.constant 2 : i64\n    %three = arith.constant 3 : i64\n    %four = arith.constant 4 : i64\n    %first = arith.constant 101 : i64\n    %second = arith.constant 202 : i64\n    %third = arith.constant 303 : i64\n    %slots = llvm.alloca %three x i64 : (i64) -> !llvm.ptr\n    llvm.store %first, %slots : i64, !llvm.ptr\n    %slot1 = llvm.getelementptr %slots[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    %slot2 = llvm.getelementptr %slots[2] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %second, %slot1 : i64, !llvm.ptr\n    llvm.store %third, %slot2 : i64, !llvm.ptr\n    %outer = llvm.alloca %three x i64 : (i64) -> !llvm.ptr\n    %inner = llvm.alloca %three x i64 : (i64) -> !llvm.ptr\n    %outerBits = llvm.ptrtoint %outer : !llvm.ptr to i64\n    %innerBits = llvm.ptrtoint %inner : !llvm.ptr to i64\n    %all = arith.cmpi eq, %mode, %three : i64\n    %outerCount = arith.select %all, %three, %one : i64\n    %slotBits = llvm.ptrtoint %slots : !llvm.ptr to i64\n    %lastBits = llvm.ptrtoint %slot2 : !llvm.ptr to i64\n    llvm.store %zero, %outer : i64, !llvm.ptr\n    %outerSlots = llvm.getelementptr %outer[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    %outerLength = llvm.getelementptr %outer[2] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %slotBits, %outerSlots : i64, !llvm.ptr\n    llvm.store %outerCount, %outerLength : i64, !llvm.ptr\n    llvm.store %outerBits, %inner : i64, !llvm.ptr\n    %innerSlots = llvm.getelementptr %inner[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    %innerLength = llvm.getelementptr %inner[2] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %lastBits, %innerSlots : i64, !llvm.ptr\n    llvm.store %one, %innerLength : i64, !llvm.ptr\n    %onlyOuter = arith.cmpi eq, %mode, %one : i64\n    %none = arith.cmpi eq, %mode, %two : i64\n    %partial = arith.select %onlyOuter, %outerBits, %innerBits : i64\n    %withAll = arith.select %all, %outerBits, %partial : i64\n    %chosenFrame = arith.select %none, %zero, %withAll : i64\n    %active = llvm.mlir.addressof @strict_gpu_active : !llvm.ptr\n    llvm.store %chosenFrame, %active : i64, !llvm.ptr\n    %idx0 = arith.constant 0 : i64\n    %entry0 = func.call @strict_gpu_entry(%idx0) : (i64) -> !llvm.ptr\n    %v0_0 = arith.constant 1 : i64\n    %s0_0 = llvm.getelementptr %entry0[0] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v0_0, %s0_0 : i64, !llvm.ptr\n    %v0_1 = arith.constant 101 : i64\n    %s0_1 = llvm.getelementptr %entry0[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v0_1, %s0_1 : i64, !llvm.ptr\n    %v0_4 = arith.constant 1 : i64\n    %s0_4 = llvm.getelementptr %entry0[4] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v0_4, %s0_4 : i64, !llvm.ptr\n    %v0_7 = arith.constant 0 : i64\n    %s0_7 = llvm.getelementptr %entry0[7] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v0_7, %s0_7 : i64, !llvm.ptr\n    %idx1 = arith.constant 1 : i64\n    %entry1 = func.call @strict_gpu_entry(%idx1) : (i64) -> !llvm.ptr\n    %v1_0 = arith.constant 2 : i64\n    %s1_0 = llvm.getelementptr %entry1[0] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v1_0, %s1_0 : i64, !llvm.ptr\n    %v1_1 = arith.constant 202 : i64\n    %s1_1 = llvm.getelementptr %entry1[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v1_1, %s1_1 : i64, !llvm.ptr\n    %v1_4 = arith.constant 2 : i64\n    %s1_4 = llvm.getelementptr %entry1[4] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v1_4, %s1_4 : i64, !llvm.ptr\n    %v1_7 = arith.constant 0 : i64\n    %s1_7 = llvm.getelementptr %entry1[7] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v1_7, %s1_7 : i64, !llvm.ptr\n    %idx2 = arith.constant 2 : i64\n    %entry2 = func.call @strict_gpu_entry(%idx2) : (i64) -> !llvm.ptr\n    %v2_0 = arith.constant 3 : i64\n    %s2_0 = llvm.getelementptr %entry2[0] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v2_0, %s2_0 : i64, !llvm.ptr\n    %v2_1 = arith.constant 303 : i64\n    %s2_1 = llvm.getelementptr %entry2[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v2_1, %s2_1 : i64, !llvm.ptr\n    %v2_4 = arith.constant 3 : i64\n    %s2_4 = llvm.getelementptr %entry2[4] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v2_4, %s2_4 : i64, !llvm.ptr\n    %v2_7 = arith.constant 0 : i64\n    %s2_7 = llvm.getelementptr %entry2[7] : (!llvm.ptr) -> !llvm.ptr, i64\n    llvm.store %v2_7, %s2_7 : i64, !llvm.ptr\n    %pooled = arith.cmpi eq, %mode, %four : i64\n    %poolValue = arith.select %pooled, %one, %zero : i64\n    llvm.store %poolValue, %s1_7 : i64, !llvm.ptr\n    %candidate = func.call @strict_gpu_oldest() : () -> !llvm.ptr\n    %null = llvm.mlir.zero : !llvm.ptr\n    %missing = llvm.icmp "eq" %candidate, %null : !llvm.ptr\n    llvm.store %zero, %active : i64, !llvm.ptr\n    cf.cond_br %missing, ^empty, ^found\n  ^empty:\n    llvm.return %zero : i64\n  ^found:\n    %deviceSlot = llvm.getelementptr %candidate[1] : (!llvm.ptr) -> !llvm.ptr, i64\n    %device = llvm.load %deviceSlot : !llvm.ptr -> i64\n    llvm.return %device : i64\n  }\n'
module=module[:-2]+pressure.replace('__HELPERS__','\n'.join(helpers))+'}\n'
(root/'cache-kind.mlir').write_text(module,encoding='utf-8')
subprocess.run(['C:/msys64/ucrt64/bin/mlir-opt.exe','--mlir-disable-threading',str(root/'cache-kind.mlir'),'--convert-arith-to-llvm','--convert-func-to-llvm','--convert-cf-to-llvm','--reconcile-unrealized-casts','-o',str(root/'cache-kind.llvm.mlir')],check=True)
subprocess.run(['C:/msys64/ucrt64/bin/mlir-translate.exe','--mlir-disable-threading','--mlir-to-llvmir',str(root/'cache-kind.llvm.mlir'),'-o',str(root/'cache-kind.ll')],check=True)
llvm_ir=root/'cache-kind.ll';llvm_ir.write_text(llvm_ir.read_text(encoding='utf-8')+'\n@_fltused = global i32 0\n',encoding='utf-8')
subprocess.run(['C:/Program Files/LLVM/bin/clang.exe','--target=x86_64-pc-windows-msvc','-Wno-override-module','-O2','-c',str(llvm_ir),'-o',str(root/'cache-kind.obj')],check=True)
subprocess.run(['C:/Program Files/LLVM/bin/lld-link.exe','/dll','/noentry','/export:cached_kind','/export:scope_choice','/out:'+str(root/'cache-kind.dll'),str(root/'cache-kind.obj')],check=True)
lib=ctypes.CDLL(str(root/'cache-kind.dll'));lib.cached_kind.argtypes=[ctypes.c_int64,ctypes.c_int64]
for stored,requested,expected in [(4,2,1),(4,4,1),(2,2,1),(2,4,0),(3,2,0),(1,1,1)]:
	actual=lib.cached_kind(stored,requested)
	print(stored,requested,actual,flush=True)
	assert actual==expected,(stored,requested,actual,expected)
lib.scope_choice.argtypes=[ctypes.c_int64]
lib.scope_choice.restype=ctypes.c_int64
for mode,expected in [(0,202),(1,202),(2,101),(3,0),(4,0)]:
	actual=lib.scope_choice(mode)
	assert actual==expected,('active input lifetime',mode,actual,expected)
print('Native row-cache and nested input lifetime contracts passed')

ctypes.windll.kernel32.FreeLibrary.argtypes=[ctypes.c_void_p]
assert ctypes.windll.kernel32.FreeLibrary(lib._handle)
temporary.cleanup()
