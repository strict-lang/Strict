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
module.write_text("module {\n  llvm.func @printf(!llvm.ptr, ...) -> i32\n" + "\n".join(declarations) + "\n" + runtime
	+ (root / "Compiler/NativeRuntimeWindows.mlir").read_text(encoding="utf-8") + wrapper + "}\n", encoding="utf-8")
driver = output / "driver.c"
driver.write_text('''#include <windows.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
extern int64_t test_pinned(void* stack);
extern int64_t test_carried(void* stack);
extern int64_t test_checked_collection(void* stack);
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
run("mlir-opt", [module, "--symbol-privatize=exclude=test_parallel,test_pinned,test_carried,test_checked_collection", "--symbol-dce", "--convert-scf-to-cf",
	"--convert-arith-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", lowered])
run("mlir-translate", ["--mlir-to-llvmir", lowered, "-o", llvm])
executable = output / "check.exe"
run("clang", [llvm, driver, "-O2", "-Wno-override-module", "-fuse-ld=lld", "-o", executable])
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
	"The collection after a check marking was young or counted a pinned target:\n" + checked.stdout + checked.stderr)
