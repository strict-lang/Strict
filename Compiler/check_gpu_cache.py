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
for name in ['strict_gpu_evicted', 'strict_gpu_handed']:
	body=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)[0]
	assert 'call @strict_gpu_active_has' in body, 'Missing active-input lifetime protection for GPU cache eviction: '+name
rows=re.search(r'  func.func private @strict_gpu_is_rows\(.*?\n  \}',source,re.S)
assert rows, 'Missing strict_gpu_is_rows: spilled and handed rows include rows with one scale (kind 4)'
helpers.append(rows[0])
for name in ['strict_gpu_spill', 'strict_gpu_handed']:
	body=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)[0]
	assert 'call @strict_gpu_is_rows' in body, name+' skips rows with one scale (kind 4)'
for name in ['strict_gpu_spill', 'strict_gpu_unspilled']:
	body=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)[0]
	assert '%slot[6]' in body, name+' loses the kind of spilled rows'
obtain=re.search(r'  func.func private @strict_gpu_obtain\(.*?\n  \}',source,re.S)[0]
collected=obtain[obtain.index('  ^ownCollect:'):]
assert -1<collected.find('call @strict_gpu_lend')<collected.find('call @strict_gpu_alloc'),'A block of a larger class the full collection freed is lent before a driver allocation (else 256 and 512 MB blocks churn through the driver)'
for name in ['strict_gpu_alloc', 'strict_gpu_free', 'strict_gpu_free_memory', 'strict_gpu_driver_free', 'strict_gpu_sized', 'strict_gpu_size_slot']:
	helpers+=re.findall(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)
helpers+=re.findall(r'  llvm.mlir.global internal @strict_gpu_(?:budget|sizes)\(.*',source)
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
pressure='''
  llvm.mlir.global internal @strict_gpu_active(0 : i64) : i64
  llvm.mlir.global internal @strict_gpu_cache(dense<0> : tensor<524288xi64>) : !llvm.array<524288 x i64>
  func.func private @strict_gpu_entry(%index: i64) -> !llvm.ptr {
    %eight = arith.constant 8 : i64
    %cache = llvm.mlir.addressof @strict_gpu_cache : !llvm.ptr
    %offset = arith.muli %index, %eight : i64
    %entry = llvm.getelementptr %cache[0, %offset] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<524288 x i64>
    return %entry : !llvm.ptr
  }
__HELPERS__
  llvm.func @cache_address() -> !llvm.ptr {
    %cache = llvm.mlir.addressof @strict_gpu_cache : !llvm.ptr
    llvm.return %cache : !llvm.ptr
  }
  llvm.func @scope_choice(%frame: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %active = llvm.mlir.addressof @strict_gpu_active : !llvm.ptr
    llvm.store %frame, %active : i64, !llvm.ptr
    %candidate = func.call @strict_gpu_oldest() : () -> !llvm.ptr
    llvm.store %zero, %active : i64, !llvm.ptr
    %bits = llvm.ptrtoint %candidate : !llvm.ptr to i64
    llvm.return %bits : i64
  }
  llvm.mlir.global internal @testDriverFree(0 : i64) : i64
  llvm.mlir.global internal @testAllocations(0 : i64) : i64
  llvm.func @testAlloc(%slot: !llvm.ptr, %bytes: i64) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %shift = llvm.mlir.constant(40 : i64) : i64
    %countSlot = llvm.mlir.addressof @testAllocations : !llvm.ptr
    %count = llvm.load %countSlot : !llvm.ptr -> i64
    %next = llvm.add %count, %one : i64
    llvm.store %next, %countSlot : i64, !llvm.ptr
    %high = llvm.shl %next, %shift : i64
    %device = llvm.or %high, %bytes : i64
    llvm.store %device, %slot : i64, !llvm.ptr
    llvm.return %zero : i32
  }
  llvm.func @testRelease(%device: i64) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
  llvm.func @testInfo(%free: !llvm.ptr, %total: !llvm.ptr) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %slot = llvm.mlir.addressof @testDriverFree : !llvm.ptr
    %value = llvm.load %slot : !llvm.ptr -> i64
    llvm.store %value, %free : i64, !llvm.ptr
    llvm.store %value, %total : i64, !llvm.ptr
    llvm.return %zero : i32
  }
  func.func private @strict_gpu_call(%index: i64) -> !llvm.ptr {
    %alloc = llvm.mlir.addressof @testAlloc : !llvm.ptr
    %release = llvm.mlir.addressof @testRelease : !llvm.ptr
    %info = llvm.mlir.addressof @testInfo : !llvm.ptr
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %isAlloc = arith.cmpi eq, %index, %six : i64
    %isRelease = arith.cmpi eq, %index, %seven : i64
    %orRelease = llvm.select %isRelease, %release, %info : i1, !llvm.ptr
    %chosen = llvm.select %isAlloc, %alloc, %orRelease : i1, !llvm.ptr
    return %chosen : !llvm.ptr
  }
  func.func private @strict_gpu_starved(%needed: i64) -> i1 {
    %false = arith.constant false
    return %false : i1
  }
  llvm.func @budget_free(%driver: i64) -> i64 {
    %slot = llvm.mlir.addressof @testDriverFree : !llvm.ptr
    llvm.store %driver, %slot : i64, !llvm.ptr
    %free = func.call @strict_gpu_free_memory() : () -> i64
    llvm.return %free : i64
  }
  llvm.func @budget_alloc(%bytes: i64) -> i64 {
    %device = func.call @strict_gpu_alloc(%bytes) : (i64) -> i64
    llvm.return %device : i64
  }
  llvm.func @budget_release(%device: i64) {
    func.call @strict_gpu_free(%device) : (i64) -> ()
    llvm.return
  }
  llvm.func @rows_kind(%kind: i64) -> i32 {
    %rows = func.call @strict_gpu_is_rows(%kind) : (i64) -> i1
    %result = arith.extui %rows : i1 to i32
    llvm.return %result : i32
  }
'''
module=module[:-2]+pressure.replace('__HELPERS__','\n'.join(helpers))+'}\n'
(root/'cache-kind.mlir').write_text(module,encoding='utf-8')
subprocess.run(['C:/msys64/ucrt64/bin/mlir-opt.exe','--mlir-disable-threading',str(root/'cache-kind.mlir'),'--convert-arith-to-llvm','--convert-func-to-llvm','--convert-cf-to-llvm','--reconcile-unrealized-casts','-o',str(root/'cache-kind.llvm.mlir')],check=True)
subprocess.run(['C:/msys64/ucrt64/bin/mlir-translate.exe','--mlir-disable-threading','--mlir-to-llvmir',str(root/'cache-kind.llvm.mlir'),'-o',str(root/'cache-kind.ll')],check=True)
llvm_ir=root/'cache-kind.ll';llvm_ir.write_text(llvm_ir.read_text(encoding='utf-8')+'\n@_fltused = global i32 0\n',encoding='utf-8')
subprocess.run(['C:/Program Files/LLVM/bin/clang.exe','--target=x86_64-pc-windows-msvc','-Wno-override-module','-O2','-c',str(llvm_ir),'-o',str(root/'cache-kind.obj')],check=True)
subprocess.run(['C:/Program Files/LLVM/bin/lld-link.exe','/dll','/noentry','/export:cached_kind','/export:scope_choice','/export:cache_address','/export:rows_kind','/export:budget_free','/export:budget_alloc','/export:budget_release','/out:'+str(root/'cache-kind.dll'),str(root/'cache-kind.obj')],check=True)
lib=ctypes.CDLL(str(root/'cache-kind.dll'));lib.cached_kind.argtypes=[ctypes.c_int64,ctypes.c_int64]
for stored,requested,expected in [(4,2,1),(4,4,1),(2,2,1),(2,4,0),(3,2,0),(1,1,1)]:
	actual=lib.cached_kind(stored,requested)
	print(stored,requested,actual,flush=True)
	assert actual==expected,(stored,requested,actual,expected)
for kind,expected in [(2,1),(4,1),(3,0),(1,0)]:
	assert lib.rows_kind(kind)==expected,('rows kind',kind)
lib.cache_address.restype=ctypes.c_void_p
cache=(ctypes.c_int64*524288).from_address(lib.cache_address())
for index,device in enumerate([101,202,303]):
	cache[index*8]=index+1;cache[index*8+1]=device;cache[index*8+4]=index+1
outerSlots=(ctypes.c_int64*1)(101);innerSlots=(ctypes.c_int64*1)(303)
outer=(ctypes.c_int64*3)(0,ctypes.addressof(outerSlots),1)
inner=(ctypes.c_int64*3)(ctypes.addressof(outer),ctypes.addressof(innerSlots),1)
allSlots=(ctypes.c_int64*3)(101,202,303)
allFrame=(ctypes.c_int64*3)(0,ctypes.addressof(allSlots),3)
lib.scope_choice.argtypes=[ctypes.c_int64];lib.scope_choice.restype=ctypes.c_int64
def chosen(frame):
	address=lib.scope_choice(ctypes.addressof(frame) if frame is not None else 0)
	return (ctypes.c_int64*8).from_address(address)[1] if address else 0
for frame,expected in [(inner,202),(outer,202),(None,101),(allFrame,0)]:
	assert chosen(frame)==expected,('active input lifetime',expected)
cache[15]=1
assert chosen(inner)==0,'Pooled copies must remain outside dedicated-cache eviction'
lib.budget_free.argtypes=[ctypes.c_int64];lib.budget_free.restype=ctypes.c_int64
lib.budget_alloc.argtypes=[ctypes.c_int64];lib.budget_alloc.restype=ctypes.c_int64
lib.budget_release.argtypes=[ctypes.c_int64]
gigabyte=1<<30
start=lib.budget_free(10*gigabyte)
device=lib.budget_alloc(2*gigabyte)
assert lib.budget_free(10*gigabyte)==start-2*gigabyte,'An allocation counts against the GPU budget at once, the driver may report it later'
lib.budget_release(device)
assert lib.budget_free(9*gigabyte)==start,'A free gives its bytes back to the budget at once, whatever the driver reports'
print('Native row-cache and nested input lifetime contracts passed')

ctypes.windll.kernel32.FreeLibrary.argtypes=[ctypes.c_void_p]
assert ctypes.windll.kernel32.FreeLibrary(lib._handle)
temporary.cleanup()
