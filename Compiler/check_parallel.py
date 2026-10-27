"""Run from Strict: python Compiler/check_parallel.py (Windows pool publication, retention and small job regressions)."""
from pathlib import Path
import os
import shutil
import subprocess
import json

root = Path(__file__).resolve().parent.parent
assert os.name == "nt", "This check uses Windows events to force the pool interleaving"
output = root / "Compiler/output/ParallelCheck"
output.mkdir(exist_ok=True)
runtime = (root / "Compiler/NativeRuntime.mlir").read_text(encoding="utf-8")
start = runtime.index("  func.func private @strict_parallel_from(")
end = runtime.index("  func.func private @strict_pool_started()", start)
section = runtime[start:end].replace("    llvm.store %count, %countSlot : i64, !llvm.ptr",
	"    llvm.store %count, %countSlot : i64, !llvm.ptr\n    llvm.call @test_publishing(%count) : (i64) -> ()")
runtime = runtime[:start] + section + runtime[end:]
start = runtime.index("  llvm.func @strict_pool_run(")
end = runtime.index("  llvm.func @strict_pool_chunk_run(", start)
section = runtime[start:end].replace("  ^check:\n", "  ^check:\n    llvm.call @test_claiming(%job) : (i64) -> ()\n", 1)
section = section.replace("  ^leave:\n", "  ^leave:\n    llvm.call @test_retired(%job) : (i64) -> ()\n", 1)
runtime = runtime[:start] + section + runtime[end:]
wrapper = '''
  func.func @test_pool_gpu() -> i64 {
    %inside = llvm.mlir.addressof @strict_pool_inside : !llvm.ptr
    %one = arith.constant 1 : i64
    llvm.store %one, %inside : i64, !llvm.ptr
    %check = llvm.mlir.addressof @strict_gpu_check : !llvm.ptr
    %unknown = arith.constant -1 : i64
    llvm.store %unknown, %check : i64, !llvm.ptr
    %unset = llvm.mlir.zero : !llvm.ptr
    %none = arith.constant 0 : i64
    %zero = arith.constant 0.0 : f64
    %result = func.call @strict_gpu_run(%unset, %unset, %unset, %none, %zero, %none, %unset, %none, %none, %none, %none, %none, %none, %none, %none) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, f64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> f64
    %after = llvm.load %check : !llvm.ptr -> i64
    llvm.store %none, %inside : i64, !llvm.ptr
    return %after : i64
  }
  func.func @test_scan_invalidation(%stack: !llvm.ptr) -> i64 {
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    %size = arith.constant 32768 : i64
    %buffer = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %small = arith.constant 96 : i64
    %first = func.call @strict_alloc(%small) : (i64) -> !llvm.ptr
    %second = func.call @strict_alloc(%small) : (i64) -> !llvm.ptr
    %firstBits = llvm.ptrtoint %first : !llvm.ptr to i64
    llvm.store %firstBits, %buffer : i64, !llvm.ptr
    func.call @strict_gc_written(%buffer) : (!llvm.ptr) -> ()
    %bufferBits = func.call @strict_bits(%buffer) : (!llvm.ptr) -> f64
    func.call @strict_pin(%bufferBits) : (f64) -> ()
    func.call @strict_rc_mark_all() : () -> ()
    %next = llvm.getelementptr %buffer[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %secondBits = llvm.ptrtoint %second : !llvm.ptr to i64
    llvm.store %secondBits, %next : i64, !llvm.ptr
    func.call @strict_gc_written(%next) : (!llvm.ptr) -> ()
    %scannedSlot = llvm.getelementptr %buffer[-12] : (!llvm.ptr) -> !llvm.ptr, i8
    %scanned = llvm.load %scannedSlot : !llvm.ptr -> i32
    %result = arith.extui %scanned : i32 to i64
    return %result : i64
  }
  func.func @test_reused_counts(%stack: !llvm.ptr, %same: i1) -> i64 {
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    func.call @strict_rc_counted() : () -> ()
    %capacity = arith.constant 4.0 : f64
    %length = arith.constant 1.0 : f64
    %zero = arith.constant 0.0 : f64
    %list = func.call @strict_list_new(%capacity, %length) : (f64, f64) -> f64
    %items = func.call @strict_list_items(%list) : (f64) -> !llvm.ptr
    %size = arith.constant 256 : i64
    %old = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %fresh = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %element = arith.select %same, %old, %fresh : !llvm.ptr
    %oldBits = llvm.ptrtoint %old : !llvm.ptr to i64
    llvm.store %oldBits, %items : i64, !llvm.ptr
    %tail = llvm.getelementptr %items[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %oldBits, %tail : i64, !llvm.ptr
    %oldLocation:5 = func.call @strict_gc_locate(%oldBits) : (i64) -> (!llvm.ptr, i64, i64, i1, i1)
    %oldCounter = func.call @strict_rc_counter(%oldLocation#0, %oldLocation#1, %oldLocation#3) : (!llvm.ptr, i64, i1) -> !llvm.ptr
    %none = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %held = arith.select %same, %none, %one : i32
    llvm.store %held, %oldCounter : i32, !llvm.ptr
    %reused = func.call @strict_list_into(%list, %capacity, %zero) : (f64, f64, f64) -> f64
    %value = func.call @strict_bits(%element) : (!llvm.ptr) -> f64
    %pushed = func.call @strict_list_push(%reused, %value) : (f64, f64) -> f64
    %elementBits = llvm.ptrtoint %element : !llvm.ptr to i64
    %location:5 = func.call @strict_gc_locate(%elementBits) : (i64) -> (!llvm.ptr, i64, i64, i1, i1)
    %counter = func.call @strict_rc_counter(%location#0, %location#1, %location#3) : (!llvm.ptr, i64, i1) -> !llvm.ptr
    %oldCount = llvm.load %oldCounter : !llvm.ptr -> i32
    %newCount = llvm.load %counter : !llvm.ptr -> i32
    %ten = arith.constant 10 : i32
    %tens = arith.muli %oldCount, %ten : i32
    %total = arith.addi %tens, %newCount : i32
    %width = arith.constant 3.0 : f64
    %flat = func.call @strict_flat_list_new(%capacity, %width) : (f64, f64) -> f64
    %flatItems = func.call @strict_list_items(%flat) : (f64) -> !llvm.ptr
    %inline = llvm.load %flatItems : !llvm.ptr -> f64
    %flatReused = func.call @strict_list_into(%flat, %capacity, %width) : (f64, f64, f64) -> f64
    %flatPushed = func.call @strict_list_push(%flatReused, %inline) : (f64, f64) -> f64
    %flatPointer = func.call @strict_ptr(%flat) : (f64) -> !llvm.ptr
    %flatAddress = llvm.ptrtoint %flatPointer : !llvm.ptr to i64
    %flatLocation:5 = func.call @strict_gc_locate(%flatAddress) : (i64) -> (!llvm.ptr, i64, i64, i1, i1)
    %flatCounter = func.call @strict_rc_counter(%flatLocation#0, %flatLocation#1, %flatLocation#3) : (!llvm.ptr, i64, i1) -> !llvm.ptr
    %flatCount = llvm.load %flatCounter : !llvm.ptr -> i32
    %flatCorrect = arith.cmpi eq, %flatCount, %one : i32
    %tailBits = llvm.load %tail : !llvm.ptr -> i64
    %empty = arith.constant 0 : i64
    %tailCorrect = arith.cmpi eq, %tailBits, %empty : i64
    %correct = arith.andi %flatCorrect, %tailCorrect : i1
    %failure = arith.constant 99 : i32
    %checked = arith.select %correct, %total, %failure : i32
    %result = arith.extui %checked : i32 to i64
    return %result : i64
  }
  func.func @test_pinned(%stack: !llvm.ptr) -> i64 {
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    func.call @strict_rc_counted() : () -> ()
    %size = arith.constant 48 : i64
    %object = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %body = func.constant @test_pin_body : (!llvm.ptr, i64, i64) -> f64
    %count = arith.constant 3 : i64
    %ran = func.call @strict_parallel(%body, %object, %count) : ((!llvm.ptr, i64, i64) -> f64, !llvm.ptr, i64) -> f64
    %address = llvm.ptrtoint %object : !llvm.ptr to i64
    %fate = func.call @strict_rc_fate(%address) : (i64) -> i64
    return %fate : i64
  }
  func.func private @test_pin_body(%object: !llvm.ptr, %from: i64, %to: i64) -> f64 {
    %address = llvm.ptrtoint %object : !llvm.ptr to i64
    func.call @strict_rc_pin(%address) : (i64) -> ()
    %zero = arith.constant 0.0 : f64
    return %zero : f64
  }
  func.func @test_carried(%stack: !llvm.ptr) -> i64 {
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    %size = arith.constant 48 : i64
    %before = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %slots = arith.constant 5 : i64
    %context = llvm.alloca %slots x i64 : (i64) -> !llvm.ptr
    %beforeAddress = llvm.ptrtoint %before : !llvm.ptr to i64
    llvm.store %beforeAddress, %context : i64, !llvm.ptr
    %body = func.constant @test_carried_body : (!llvm.ptr, i64, i64) -> f64
    %one = arith.constant 1 : i64
    %ran = func.call @strict_parallel(%body, %context, %one) : ((!llvm.ptr, i64, i64) -> f64, !llvm.ptr, i64) -> f64
    %insideSlot = llvm.getelementptr %context[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %inside = llvm.load %insideSlot : !llvm.ptr -> i64
    %next = arith.addi %beforeAddress, %size : i64
    %isNext = arith.cmpi eq, %inside, %next : i64
    %adjacent = arith.extui %isNext : i1 to i64
    %mineSlot = llvm.getelementptr %context[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %mine = llvm.load %mineSlot : !llvm.ptr -> i64
    %oldSlot = llvm.getelementptr %context[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %old = llvm.load %oldSlot : !llvm.ptr -> i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %mineBit = arith.muli %mine, %two : i64
    %oldBit = arith.muli %old, %four : i64
    %partial = arith.addi %adjacent, %mineBit : i64
    %withOld = arith.addi %partial, %oldBit : i64
    %sharedSlot = llvm.getelementptr %context[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %shared = llvm.load %sharedSlot : !llvm.ptr -> i64
    %eight = arith.constant 8 : i64
    %sharedBit = arith.muli %shared, %eight : i64
    %result = arith.addi %withOld, %sharedBit : i64
    return %result : i64
  }
  func.func private @test_carried_body(%context: !llvm.ptr, %from: i64, %to: i64) -> f64 {
    %size = arith.constant 48 : i64
    %object = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %address = llvm.ptrtoint %object : !llvm.ptr to i64
    %insideSlot = llvm.getelementptr %context[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %address, %insideSlot : i64, !llvm.ptr
    %mine = func.call @strict_gc_mine(%address) : (i64) -> i1
    %mineNumber = arith.extui %mine : i1 to i64
    %mineSlot = llvm.getelementptr %context[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %mineNumber, %mineSlot : i64, !llvm.ptr
    %before = llvm.load %context : !llvm.ptr -> i64
    %old = func.call @strict_gc_mine(%before) : (i64) -> i1
    %oldNumber = arith.extui %old : i1 to i64
    %oldSlot = llvm.getelementptr %context[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %oldNumber, %oldSlot : i64, !llvm.ptr
    %laneSlot = llvm.mlir.addressof @strict_gc_lane : !llvm.ptr
    %ownLane = llvm.load %laneSlot : !llvm.ptr -> i64
    %sharedLane = arith.constant 0 : i64
    llvm.store %sharedLane, %laneSlot : i64, !llvm.ptr
    %sharedObject = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %sharedAddress = llvm.ptrtoint %sharedObject : !llvm.ptr to i64
    %sharedMine = func.call @strict_gc_mine(%sharedAddress) : (i64) -> i1
    llvm.store %ownLane, %laneSlot : i64, !llvm.ptr
    %sharedNumber = arith.extui %sharedMine : i1 to i64
    %sharedSlot = llvm.getelementptr %context[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %sharedNumber, %sharedSlot : i64, !llvm.ptr
    %zero = arith.constant 0.0 : f64
    return %zero : f64
  }
  func.func @test_checked_collection(%stack: !llvm.ptr) -> i64 {
    func.call @strict_gc_init(%stack) : (!llvm.ptr) -> ()
    %size = arith.constant 48 : i64
    %holder = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %target = func.call @strict_alloc(%size) : (i64) -> !llvm.ptr
    %targetWord = llvm.ptrtoint %target : !llvm.ptr to i64
    %slot = llvm.getelementptr %holder[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %targetWord, %slot : i64, !llvm.ptr
    func.call @strict_rc_pin(%targetWord) : (i64) -> ()
    func.call @strict_rc_mark_all() : () -> ()
    %young = arith.constant false
    func.call @strict_gc_run(%young) : (i1) -> ()
    %full = arith.constant true
    func.call @strict_gc_run(%full) : (i1) -> ()
    %holderWord = llvm.ptrtoint %holder : !llvm.ptr to i64
    return %holderWord : i64
  }
  llvm.func @test_claiming(i64)
  llvm.func @test_publishing(i64)
  llvm.func @test_retired(i64)
  func.func @test_parallel(%body: !llvm.ptr, %context: !llvm.ptr, %count: i64) -> f64 {
    %callable = builtin.unrealized_conversion_cast %body : !llvm.ptr to (!llvm.ptr, i64, i64) -> f64
    %result = func.call @strict_parallel(%callable, %context, %count) : ((!llvm.ptr, i64, i64) -> f64, !llvm.ptr, i64) -> f64
    return %result : f64
  }
'''
module = output / "pool.mlir"
declaration = next(line for line in (root / "Compiler/MlirTexts.strict").read_text(encoding="utf-8").splitlines()
	if line.startswith("constant Declarations = "))
declarations = json.loads("[" + declaration.split(" = (", 1)[1][:-1] + "]")
cached = (root / "Compiler/MlirCached.strict").read_text(encoding="utf-8").splitlines()
pieces = [json.loads("[" + next(line for line in cached if line.startswith("constant " + name + " = ")).split(" = (", 1)[1][:-1] + "]") for name in ("Checked", "Stored")]
cache_wrapper = '\n'.join(['  llvm.func @test_compute() -> f64',
	'  llvm.mlir.global internal @test_cache(0.0 : f64) : f64',
	'  llvm.mlir.global internal @test_set(0 : i64) : i64',
	'  llvm.mlir.global internal thread_local @test_active(0 : i64) : i64',
	'  func.func @test_cached() -> f64 {',
	'    %cache = llvm.mlir.addressof @test_cache : !llvm.ptr',
	'    %set = llvm.mlir.addressof @test_set : !llvm.ptr',
	'    %active = llvm.mlir.addressof @test_active : !llvm.ptr', *pieces[0],
	'    %computed = llvm.call @test_compute() : () -> f64', *pieces[1]])
module.write_text("module {\n  llvm.func @printf(!llvm.ptr, ...) -> i32\n" + "\n".join(declarations) + "\n" + runtime
	+ (root / "Compiler/NativeRuntimeWindows.mlir").read_text(encoding="utf-8") + wrapper + cache_wrapper + "}\n", encoding="utf-8")
driver = output / "driver.c"
driver.write_text('''#include <windows.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
extern int64_t test_pinned(void* stack);
extern int64_t test_carried(void* stack);
extern int64_t test_checked_collection(void* stack);
extern int64_t test_reused_counts(void* stack, _Bool same);
extern int64_t test_scan_invalidation(void* stack);
extern int64_t test_pool_gpu(void);
extern double test_cached(void);
static atomic_int computed;
static int recursive;
static HANDLE cache_start;
double test_compute(void) {
	atomic_fetch_add(&computed, 1);
	Sleep(20);
	if (recursive && atomic_load(&computed) == 1) return test_cached();
	return 7;
}
static DWORD WINAPI cached_reader(void* unused) {
	WaitForSingleObject(cache_start, INFINITE);
	double result = test_cached();
	union { double value; uint64_t bits; } checked = { result };
	return recursive ? (checked.bits >> 48) != 32762 : result != 7;
}
static DWORD caller;
static HANDLE ready, published, retired;
static atomic_int publishing, premature;
extern double test_parallel(double (*body)(void*, int64_t, int64_t), void* context, int64_t count);
static void waited(HANDLE event) {
	if (WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) {
		fprintf(stderr, "Pool publication check timed out\\n");
		exit(3);
	}
}
void test_claiming(int64_t job) {
	if (job == 1 && GetCurrentThreadId() != caller) {
		SetEvent(ready);
		waited(published);
	}
}
void test_publishing(int64_t count) {
	if (count == 2) {
		atomic_store(&publishing, 1);
		SetEvent(published);
		waited(retired);
		atomic_store(&publishing, 0);
	}
}
void test_retired(int64_t job) {
	if (job == 1 && GetCurrentThreadId() != caller)
		SetEvent(retired);
}
static HANDLE begun;
static double tail(void* context, int64_t from, int64_t to) {
	if (GetCurrentThreadId() == caller)
		waited(begun);
	else {
		SetEvent(begun);
		Sleep(300);
	}
	return 0;
}
static double busy(FILETIME user, FILETIME kernel) {
	return ((double)(((ULONGLONG)user.dwHighDateTime << 32) | user.dwLowDateTime) +
		(double)(((ULONGLONG)kernel.dwHighDateTime << 32) | kernel.dwLowDateTime)) / 10000;
}
static double body(void* context, int64_t from, int64_t to) {
	if (*(int*)context == 1)
		waited(ready);
	else if (atomic_load(&publishing))
		atomic_fetch_add(&premature, 1);
	return 0;
}
int main(int argc, char** argv) {
	if (argc > 1 && (argv[1][0] == 'k' || argv[1][0] == 'j')) {
		recursive = argv[1][0] == 'j';
		if (recursive) test_scan_invalidation(&argc);
		HANDLE readers[16];
		cache_start = CreateEventA(0, TRUE, FALSE, 0);
		for (int index = 0; index < 16; index++) readers[index] = CreateThread(0, 0, cached_reader, 0, 0, 0);
		SetEvent(cache_start);
		if (WaitForMultipleObjects(16, readers, TRUE, 5000) != WAIT_OBJECT_0) return 2;
		for (int index = 0; index < 16; index++) {
			DWORD code;
			GetExitCodeThread(readers[index], &code);
			if (code != 0) return 3;
			CloseHandle(readers[index]);
		}
		CloseHandle(cache_start);
		printf("Constant computations: %d (expected 1)\\n", atomic_load(&computed));
		return atomic_load(&computed) != 1;
	}
	if (argc > 1 && argv[1][0] == 'g') {
		int64_t state = test_pool_gpu();
		printf("GPU check state after pool launch: %lld (expected -1)\\n", (long long)state);
		return state != -1;
	}
	if (argc > 1 && argv[1][0] == 's') {
		int64_t bound = test_scan_invalidation(&argc);
		printf("Scan bound after a reference write: %lld (expected 0)\\n", (long long)bound);
		return bound != 0;
	}
	if (argc > 1 && argv[1][0] == 'r') {
		int64_t counts = test_reused_counts(&argc, argv[1][1] == 's');
		printf("Reused list counts: %lld (expected 11)\\n", (long long)counts);
		return counts != 11;
	}
	if (argc > 1 && argv[1][0] == 'w') {
		_putenv_s("STRICT_THREADS", "64");
		published = CreateEventA(0, TRUE, TRUE, 0);
		int idle = 0;
		test_parallel(body, &idle, 64);
		LARGE_INTEGER frequency, started, ended;
		QueryPerformanceFrequency(&frequency);
		double spent = 0;
		for (int job = 0; job < 1000; job++) {
			QueryPerformanceCounter(&started);
			do QueryPerformanceCounter(&ended); while ((ended.QuadPart - started.QuadPart) * 10000 < 3 * frequency.QuadPart);
			QueryPerformanceCounter(&started);
			test_parallel(body, &idle, 4);
			QueryPerformanceCounter(&ended);
			spent += (double)(ended.QuadPart - started.QuadPart) / frequency.QuadPart;
		}
		printf("Job of 4 items while 63 workers sleep: %.1f us\\n", spent * 1000);
		return spent * 1000 > 50;
	}
	if (argc > 1 && argv[1][0] == 'c') {
		_putenv_s("STRICT_RC_CHECK", "1");
		volatile int64_t holder = test_checked_collection(&argc);
		return holder == 0;
	}
	if (argc > 1 && argv[1][0] == 'a') {
		_putenv_s("STRICT_THREADS", "2");
		published = CreateEventA(0, TRUE, TRUE, 0);
		begun = CreateEventA(0, TRUE, FALSE, 0);
		caller = GetCurrentThreadId();
		FILETIME created, ended, kernel, user, kernelAfter, userAfter;
		GetThreadTimes(GetCurrentThread(), &created, &ended, &kernel, &user);
		int idle = 0;
		test_parallel(tail, &idle, 3);
		GetThreadTimes(GetCurrentThread(), &created, &ended, &kernelAfter, &userAfter);
		double spent = busy(userAfter, kernelAfter) - busy(user, kernel);
		printf("Caller CPU while a worker finishes a 300 ms chunk: %.0f ms\\n", spent);
		return spent > 100;
	}
	if (argc > 1) {
		_putenv_s("STRICT_THREADS", "1");
		int64_t carried = test_carried(&argc);
		printf("Lane span carried into a pool job: %lld (3: kept, the job's new object is its own, the older one and one of the shared lane 0 not)", (long long)carried);
		return carried != 3;
	}
	_putenv_s("STRICT_THREADS", "2");
	caller = GetCurrentThreadId();
	ready = CreateEventA(0, TRUE, FALSE, 0);
	published = CreateEventA(0, TRUE, FALSE, 0);
	retired = CreateEventA(0, TRUE, FALSE, 0);
	int first = 1, second = 2;
	test_parallel(body, &first, 1);
	test_parallel(body, &second, 2);
	printf("New body executed under previous job: %d\\n", atomic_load(&premature));
	CloseHandle(ready);
	CloseHandle(published);
	CloseHandle(retired);
	int64_t freed = test_pinned(&first);
	printf("Pool-pinned object reclaimed: %lld\\n", (long long)freed);
	return atomic_load(&premature) != 0 || freed != 0;
}
''', encoding="utf-8")
def run(name, arguments):
	folders = (Path("C:/Program Files/LLVM/bin"), Path("C:/msys64/ucrt64/bin"))
	tool = shutil.which(name) or next(str(f / (name + ".exe")) for f in folders if (f / (name + ".exe")).exists())
	result = subprocess.run([tool, *map(str, arguments)], cwd=root, capture_output=True, text=True, timeout=60)
	assert result.returncode == 0, result.stdout + result.stderr
lowered = output / "pool.llvm.mlir"
llvm = output / "pool.ll"
run("mlir-opt", [module, "--symbol-privatize=exclude=test_parallel,test_pinned,test_carried,test_checked_collection,test_reused_counts,test_scan_invalidation,test_pool_gpu,test_cached", "--symbol-dce", "--convert-scf-to-cf",
	"--convert-arith-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", lowered])
run("mlir-translate", ["--mlir-to-llvmir", lowered, "-o", llvm])
executable = output / "check.exe"
run("clang", [llvm, driver, "-O2", "-march=native", "-Wno-override-module", "-fuse-ld=lld", "-Wl,clang_rt.builtins-x86_64.lib", "-o", executable])
for mode in ["j", "k", "gpu", "rsame", "rdifferent", "scan"]:
	reused = subprocess.run([str(executable), mode], capture_output=True, text=True, timeout=15)
	print(reused.stdout.strip(), flush=True)
	assert reused.returncode == 0, reused.stdout + reused.stderr
result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
assert result.returncode == 0, result.stdout + result.stderr
print(result.stdout.strip())
carried = subprocess.run([str(executable), "carried"], capture_output=True, text=True, timeout=15)
print(carried.stdout.strip())
assert carried.returncode == 0, carried.stdout + carried.stderr
woken = subprocess.run([str(executable), "wake"], capture_output=True, text=True, timeout=15)
print(woken.stdout.strip())
assert woken.returncode == 0, woken.stdout + woken.stderr
awaited = subprocess.run([str(executable), "await"], capture_output=True, text=True, timeout=15)
print(awaited.stdout.strip())
assert awaited.returncode == 0, awaited.stdout + awaited.stderr
checked = subprocess.run([str(executable), "checked"], capture_output=True, text=True, timeout=15)
print("Collection after a reference count check marking: " + checked.stdout.strip())
assert checked.returncode == 0 and "Reference counts checked: 0 too low" in checked.stdout, (
	"The full collection after a check marking and a young collection counted a pinned target:\n" + checked.stdout + checked.stderr)
