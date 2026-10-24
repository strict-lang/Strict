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
for name in ['strict_gpu_unspilled','strict_gpu_spilled','strict_gpu_spill_at','strict_gpu_put_async','strict_gpu_put_ahead','strict_gpu_ahead_join','strict_gpu_followed','strict_gpu_prefetch','strict_gpu_ahead_hit']:
	match=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)
	assert match,'Missing '+name+': the spilled rows that went up next the last time go up ahead on a copy stream'
	helpers.append(match[0])
helpers+=re.findall(r'  llvm.mlir.global internal @strict_gpu_(?:spills|ahead)\(.*',source)
for name,called in [('strict_gpu_rows','strict_gpu_ahead_hit'),('strict_gpu_replay','strict_gpu_ahead_join'),('strict_gpu_capture','strict_gpu_ahead_join')]:
	body=re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)[0]
	assert 'call @'+called in body,name+' must call '+called+': launches wait for uploads ahead at the first use of their rows and before a capture begins or ends'
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
    %put = llvm.mlir.addressof @testPut : !llvm.ptr
    %record = llvm.mlir.addressof @testRecord : !llvm.ptr
    %wait = llvm.mlir.addressof @testWait : !llvm.ptr
    %six = arith.constant 6 : i64
    %seven = arith.constant 7 : i64
    %fourteen = arith.constant 14 : i64
    %twentySeven = arith.constant 27 : i64
    %thirtyThree = arith.constant 33 : i64
    %isAlloc = arith.cmpi eq, %index, %six : i64
    %isRelease = arith.cmpi eq, %index, %seven : i64
    %isPut = arith.cmpi eq, %index, %fourteen : i64
    %isRecord = arith.cmpi eq, %index, %twentySeven : i64
    %isWait = arith.cmpi eq, %index, %thirtyThree : i64
    %orWait = llvm.select %isWait, %wait, %info : i1, !llvm.ptr
    %orRecord = llvm.select %isRecord, %record, %orWait : i1, !llvm.ptr
    %orPut = llvm.select %isPut, %put, %orRecord : i1, !llvm.ptr
    %orRelease = llvm.select %isRelease, %release, %orPut : i1, !llvm.ptr
    %chosen = llvm.select %isAlloc, %alloc, %orRelease : i1, !llvm.ptr
    return %chosen : !llvm.ptr
  }
  llvm.mlir.global internal @testCalls(dense<0> : tensor<64xi64>) : !llvm.array<64 x i64>
  llvm.mlir.global internal @testHanded(0 : i64) : i64
  func.func private @strict_gpu_handed(%total: i64) -> i64 {
    %one = arith.constant 1 : i64
    %slot = llvm.mlir.addressof @testHanded : !llvm.ptr
    %device = llvm.load %slot : !llvm.ptr -> i64
    %next = arith.addi %device, %one : i64
    llvm.store %next, %slot : i64, !llvm.ptr
    return %device : i64
  }
  func.func private @strict_gpu_release(%device: i64) { return }
  func.func private @strict_gpu_remember(%list: f64, %kind: i64, %count: i64, %out: !llvm.ptr, %pool: i64) { return }
  func.func private @strict_gpu_captured(%function: !llvm.ptr) { return }
  func.func private @strict_gpu_stream() -> !llvm.ptr {
    %main = llvm.mlir.constant(999 : i64) : i64
    %stream = llvm.inttoptr %main : i64 to !llvm.ptr
    return %stream : !llvm.ptr
  }
  llvm.func @testLogged(%kind: i64, %first: i64, %second: i64) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %three = llvm.mlir.constant(3 : i64) : i64
    %calls = llvm.mlir.addressof @testCalls : !llvm.ptr
    %count = llvm.load %calls : !llvm.ptr -> i64
    %offset = llvm.mul %count, %three : i64
    %start = llvm.add %offset, %one : i64
    %kindSlot = llvm.getelementptr %calls[%start] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %kind, %kindSlot : i64, !llvm.ptr
    %firstSlot = llvm.getelementptr %kindSlot[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %first, %firstSlot : i64, !llvm.ptr
    %secondSlot = llvm.getelementptr %kindSlot[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %second, %secondSlot : i64, !llvm.ptr
    %next = llvm.add %count, %one : i64
    llvm.store %next, %calls : i64, !llvm.ptr
    llvm.return %zero : i32
  }
  llvm.func @testPut(%device: i64, %host: !llvm.ptr, %bytes: i64, %stream: !llvm.ptr) -> i32 {
    %kind = llvm.mlir.constant(1 : i64) : i64
    %streamBits = llvm.ptrtoint %stream : !llvm.ptr to i64
    %logged = llvm.call @testLogged(%kind, %streamBits, %device) : (i64, i64, i64) -> i32
    llvm.return %logged : i32
  }
  llvm.func @testRecord(%event: !llvm.ptr, %stream: !llvm.ptr) -> i32 {
    %kind = llvm.mlir.constant(2 : i64) : i64
    %eventBits = llvm.ptrtoint %event : !llvm.ptr to i64
    %streamBits = llvm.ptrtoint %stream : !llvm.ptr to i64
    %logged = llvm.call @testLogged(%kind, %eventBits, %streamBits) : (i64, i64, i64) -> i32
    llvm.return %logged : i32
  }
  llvm.func @testWait(%stream: !llvm.ptr, %event: !llvm.ptr, %flags: i32) -> i32 {
    %kind = llvm.mlir.constant(3 : i64) : i64
    %streamBits = llvm.ptrtoint %stream : !llvm.ptr to i64
    %eventBits = llvm.ptrtoint %event : !llvm.ptr to i64
    %logged = llvm.call @testLogged(%kind, %streamBits, %eventBits) : (i64, i64, i64) -> i32
    llvm.return %logged : i32
  }
  llvm.func @spill_upload(%listBits: i64, %count: i64) -> i64 {
    %three = arith.constant 3 : i64
    %out = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    %rows = arith.bitcast %listBits : i64 to f64
    %uploaded = func.call @strict_gpu_unspilled(%rows, %count, %out) : (f64, i64, !llvm.ptr) -> i1
    %device = llvm.load %out : !llvm.ptr -> i64
    llvm.return %device : i64
  }
  llvm.func @ahead_used(%device: i64) {
    %three = arith.constant 3 : i64
    %out = llvm.alloca %three x i64 : (i64) -> !llvm.ptr
    llvm.store %device, %out : i64, !llvm.ptr
    func.call @strict_gpu_ahead_hit(%out) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @spills_address() -> !llvm.ptr {
    %spills = llvm.mlir.addressof @strict_gpu_spills : !llvm.ptr
    llvm.return %spills : !llvm.ptr
  }
  llvm.func @ahead_address() -> !llvm.ptr {
    %ahead = llvm.mlir.addressof @strict_gpu_ahead : !llvm.ptr
    llvm.return %ahead : !llvm.ptr
  }
  llvm.func @calls_address() -> !llvm.ptr {
    %calls = llvm.mlir.addressof @testCalls : !llvm.ptr
    llvm.return %calls : !llvm.ptr
  }
  llvm.func @handed_address() -> !llvm.ptr {
    %handed = llvm.mlir.addressof @testHanded : !llvm.ptr
    llvm.return %handed : !llvm.ptr
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
pool=[]
for name in ['strict_gpu_stack','strict_gpu_obtain','strict_gpu_lend','strict_gpu_lent_class','strict_gpu_pair_put','strict_gpu_pair_taken','strict_gpu_obtained']:
	pool.append(re.search(r'  func.func private @'+name+r'\(.*?\n  \}',source,re.S)[0])
release=re.search(r'  func.func private @strict_gpu_trim_unused\(.*?\n  \}',source,re.S)
assert release, 'Missing explicit phase-boundary release of unused GPU pool buffers'
pool.append(release[0])
trim=re.search(r'  func.func private @strict_gpu_trim\(.*?\n  \}',source,re.S)
if trim:
	pool.append(trim[0])
for name in ['strict_gpu_blocks','strict_gpu_chunk','strict_gpu_lent','strict_gpu_spilled_bytes']:
	pool.append(re.search(r'  llvm.mlir.global internal @'+name+r'[^\n]*',source)[0])
pool.append('''
  llvm.mlir.global internal @strict_gc_young_limit(0 : i64) : i64
  llvm.mlir.global internal @pool_counters(dense<0> : tensor<3xi64>) : !llvm.array<3 x i64>
  func.func private @strict_rc_reclaim() { return }
  func.func private @strict_gc_run(%full: i1) {
    %counters = llvm.mlir.addressof @pool_counters : !llvm.ptr
    %slot = llvm.getelementptr %counters[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %calls = llvm.load %slot : !llvm.ptr -> i64
    %one = arith.extui %full : i1 to i64
    %next = arith.addi %calls, %one : i64
    llvm.store %next, %slot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_free_memory() -> i64 {
    %counters = llvm.mlir.addressof @pool_counters : !llvm.ptr
    %freed = llvm.load %counters : !llvm.ptr -> i64
    %initial = arith.constant 536870912 : i64
    %free = arith.addi %initial, %freed : i64
    return %free : i64
  }
  func.func private @strict_gpu_free(%device: i64) {
    %counters = llvm.mlir.addressof @pool_counters : !llvm.ptr
    %freed = llvm.load %counters : !llvm.ptr -> i64
    %more = arith.addi %freed, %device : i64
    llvm.store %more, %counters : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_replay() {
    %counters = llvm.mlir.addressof @pool_counters : !llvm.ptr
    %waitsSlot = llvm.getelementptr %counters[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %waits = llvm.load %waitsSlot : !llvm.ptr -> i64
    %one = arith.constant 1 : i64
    %more = arith.addi %waits, %one : i64
    llvm.store %more, %waitsSlot : i64, !llvm.ptr
    return
  }
  func.func private @strict_gpu_synchronize() { return }
  func.func private @strict_gpu_alloc(%bytes: i64) -> i64 {
    %none = arith.constant 0 : i64
    return %none : i64
  }
  llvm.func @pool_address() -> !llvm.ptr {
    %blocks = llvm.mlir.addressof @strict_gpu_blocks : !llvm.ptr
    llvm.return %blocks : !llvm.ptr
  }
  llvm.func @pool_take(%class: i64) -> i64 {
    %device = func.call @strict_gpu_obtain(%class) : (i64) -> i64
    llvm.return %device : i64
  }
  llvm.func @pool_real(%device: i64, %class: i64) -> i64 {
    %real = func.call @strict_gpu_lent_class(%device, %class) : (i64, i64) -> i64
    llvm.return %real : i64
  }
  llvm.func @pool_clear() -> f64 {
    %none = arith.constant 0.0 : f64
    %released = func.call @strict_gpu_trim_unused(%none) : (f64) -> f64
    llvm.return %released : f64
  }
  llvm.func @pool_stats() -> !llvm.ptr {
    %counters = llvm.mlir.addressof @pool_counters : !llvm.ptr
    llvm.return %counters : !llvm.ptr
  }
  llvm.func @pool_spilled() -> !llvm.ptr {
    %spilled = llvm.mlir.addressof @strict_gpu_spilled_bytes : !llvm.ptr
    llvm.return %spilled : !llvm.ptr
  }
''')
def built(name,text,exports):
	(root/(name+'.mlir')).write_text(text,encoding='utf-8')
	subprocess.run(['C:/msys64/ucrt64/bin/mlir-opt.exe','--mlir-disable-threading',str(root/(name+'.mlir')),'--convert-arith-to-llvm','--convert-func-to-llvm','--convert-cf-to-llvm','--reconcile-unrealized-casts','-o',str(root/(name+'.llvm.mlir'))],check=True)
	subprocess.run(['C:/msys64/ucrt64/bin/mlir-translate.exe','--mlir-disable-threading','--mlir-to-llvmir',str(root/(name+'.llvm.mlir')),'-o',str(root/(name+'.ll'))],check=True)
	llvm_ir=root/(name+'.ll');llvm_ir.write_text(llvm_ir.read_text(encoding='utf-8')+'\n@_fltused = global i32 0\n',encoding='utf-8')
	subprocess.run(['C:/Program Files/LLVM/bin/clang.exe','--target=x86_64-pc-windows-msvc','-Wno-override-module','-O2','-c',str(llvm_ir),'-o',str(root/(name+'.obj'))],check=True)
	subprocess.run(['C:/Program Files/LLVM/bin/lld-link.exe','/dll','/noentry']+['/export:'+export for export in exports]+['/out:'+str(root/(name+'.dll')),str(root/(name+'.obj'))],check=True)
	return ctypes.CDLL(str(root/(name+'.dll')))
lib=built('cache-kind',module[:-2]+pressure.replace('__HELPERS__','\n'.join(helpers))+'}\n',['cached_kind','scope_choice','cache_address','rows_kind','budget_free','budget_alloc','budget_release','spill_upload','ahead_used','spills_address','ahead_address','calls_address','handed_address'])
pools=built('cache-pool','module {\n'+'\n'.join(pool)+'}\n',['pool_address','pool_take','pool_real','pool_stats','pool_spilled','pool_clear'])
lib.cached_kind.argtypes=[ctypes.c_int64,ctypes.c_int64]
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
for name in ['spills_address','ahead_address','calls_address','handed_address']:
	getattr(lib,name).restype=ctypes.c_void_p
lib.spill_upload.argtypes=[ctypes.c_int64,ctypes.c_int64];lib.spill_upload.restype=ctypes.c_int64
lib.ahead_used.argtypes=[ctypes.c_int64]
spills=(ctypes.c_int64*8192).from_address(lib.spills_address())
ahead=(ctypes.c_int64*7).from_address(lib.ahead_address())
calls=(ctypes.c_int64*64).from_address(lib.calls_address())
(ctypes.c_int64*1).from_address(lib.handed_address())[0]=7000
pinned=(ctypes.c_int64*4)()
first,second=0x5008,0x6008
for index,rows in enumerate([first,second]):
	spills[index*8:index*8+8]=[rows,4,ctypes.addressof(pinned),4096,4,8,2,0]
ahead[0:3]=[111,222,333]
def logged(call):
	calls[0]=0
	result=call()
	return result,[tuple(calls[1+3*index:4+3*index]) for index in range(calls[0])]
logged(lambda:lib.spill_upload(first,4));logged(lambda:lib.spill_upload(second,4))
assert spills[7]==lib.spills_address()+64,'Spilled rows going up after others become their next ones'
device,log=logged(lambda:lib.spill_upload(first,4))
assert log[0]==(1,999,device),'Spilled rows go up in order on the stream of the launches using them'
assert log[1:]==[(2,222,999),(3,111,222),(1,111,device+1),(2,333,111)],'Their next rows go up on the copy stream after the launches so far'
assert ahead[3]==device+1,'The launches wait for rows that went up ahead only when they use them'
_,log=logged(lambda:lib.ahead_used(device+1))
assert log[0]==(3,999,333) and ahead[3]==device+2,'The first use of rows that went up ahead waits for them and sends their next rows up'
print('Native row-cache and nested input lifetime contracts passed')

pools.pool_address.restype=ctypes.c_void_p
blocks=(ctypes.c_int64*144).from_address(pools.pool_address())
pools.pool_take.argtypes=[ctypes.c_int64];pools.pool_take.restype=ctypes.c_int64
pools.pool_real.argtypes=[ctypes.c_int64,ctypes.c_int64];pools.pool_real.restype=ctypes.c_int64
first=(ctypes.c_int64*1)(123);second=(ctypes.c_int64*1)(456)
blocks[17*3]=ctypes.addressof(first);blocks[17*3+1]=1;blocks[17*3+2]=1
blocks[18*3]=ctypes.addressof(second);blocks[18*3+1]=1;blocks[18*3+2]=1
assert pools.pool_take(16)==123,'Pool must borrow a larger free block under memory pressure'
assert blocks[17*3+1]==0 and blocks[18*3+1]==1,'Must borrow the nearest available larger class'
assert pools.pool_real(123,16)==17 and pools.pool_real(123,16)==16,'A borrowed block must return to its original class exactly once'
assert pools.pool_take(16)==456 and pools.pool_real(456,16)==18,'Must reuse the second larger class when the nearest is empty'
assert pools.pool_take(16)==0,'Empty pool must fall through to the driver allocation'
pools.pool_stats.restype=ctypes.c_void_p
counters=(ctypes.c_int64*3).from_address(pools.pool_stats())
assert counters[1]==0,'An empty pool must not flush GPU work'
small=(ctypes.c_int64*1)(256<<20);large=(ctypes.c_int64*1)(512<<20)
blocks[20*3]=ctypes.addressof(small);blocks[20*3+1]=1;blocks[20*3+2]=1
blocks[21*3]=ctypes.addressof(large);blocks[21*3+1]=1;blocks[21*3+2]=1
pools.pool_take(16)
assert counters[0]==768<<20,'Unused large pool blocks must go before an allocation exhausts VRAM headroom'
assert counters[1]==1,'Freeing several pool blocks must drain queued GPU work only once'
assert blocks[20*3+1]==0 and blocks[21*3+1]==0,'Trimmed blocks must leave their free lists'
pools.pool_spilled.restype=ctypes.c_void_p
spilled=(ctypes.c_int64*1).from_address(pools.pool_spilled())
spilled[0]=1;blocks[20*3+1]=1;blocks[21*3+1]=1;counters[0]=0
pools.pool_take(16)
assert counters[0]==0 and blocks[20*3+1]==1 and blocks[21*3+1]==1,'With streamed rows spilled an allocation evicts those (strict_gpu_starved) instead of trimming pool blocks the next layers allocate again'
spilled[0]=0;blocks[20*3+1]=0
print('GPU pool borrows correctly and trims unused large buffers before allocating under pressure')
firstPhase=(ctypes.c_int64*1)(64<<20);secondPhase=(ctypes.c_int64*1)(512<<20)
blocks[18*3]=ctypes.addressof(firstPhase);blocks[18*3+1]=1;blocks[18*3+2]=1
blocks[21*3]=ctypes.addressof(secondPhase);blocks[21*3+1]=1;blocks[21*3+2]=1
live=list(cache[:24]);collected=counters[2];waited=counters[1]
pools.pool_clear.restype=ctypes.c_double
assert pools.pool_clear()==576<<20,'An explicit workload boundary releases every unused large pool block'
assert blocks[18*3+1]==0 and blocks[21*3+1]==0,'Released blocks must leave the free lists'
assert counters[2]==collected+1 and counters[1]==waited+1,'Collect once and drain once for a release'
assert list(cache[:24])==live,'Live tensor cache entries must remain intact'
assert pools.pool_clear()==0 and counters[1]==waited+1,'An empty pool must not drain work'
print('Explicit phase-boundary release collects once and preserves live tensor copies')


ctypes.windll.kernel32.FreeLibrary.argtypes=[ctypes.c_void_p]
for loaded in [lib,pools]:
	assert ctypes.windll.kernel32.FreeLibrary(loaded._handle)
temporary.cleanup()
