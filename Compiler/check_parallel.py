"""Run from Strict: python Compiler/check_parallel.py (Windows pool publication and retention regressions)."""
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
static double body(void* context, int64_t from, int64_t to) {
	if (*(int*)context == 1)
		waited(ready);
	else if (atomic_load(&publishing))
		atomic_fetch_add(&premature, 1);
	return 0;
}
int main(void) {
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
	tool = shutil.which(name) or str(Path("C:/msys64/ucrt64/bin") / (name + ".exe"))
	result = subprocess.run([tool, *map(str, arguments)], cwd=root, capture_output=True, text=True, timeout=60)
	assert result.returncode == 0, result.stdout + result.stderr
lowered = output / "pool.llvm.mlir"
llvm = output / "pool.ll"
run("mlir-opt", [module, "--symbol-privatize=exclude=test_parallel,test_pinned", "--symbol-dce", "--convert-scf-to-cf",
	"--convert-arith-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", lowered])
run("mlir-translate", ["--mlir-to-llvmir", lowered, "-o", llvm])
executable = output / "check.exe"
run("clang", [llvm, driver, "-O2", "-Wno-override-module", "-fuse-ld=lld", "-o", executable])
result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
assert result.returncode == 0, result.stdout + result.stderr
print(result.stdout.strip())
