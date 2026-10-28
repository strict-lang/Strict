"""Run from Strict: python Compiler/check_audit.py [locks|replace]."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent


def run(command):
	result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout


def locks(folder):
	functions = []
	for name in ("Posix", "Darwin"):
		source = (root / f"Compiler/NativeRuntime{name}.mlir").read_text(encoding="utf-8")
		function = re.search(r"  func.func private @strict_file_locked\(.*?\n  }", source, re.S).group()
		function = function.replace("private @strict_file_locked", "@lock_" + name)
		for call in ("open", "flock", "close"):
			function = function.replace("@" + call + "(", "@test_" + call + "(")
		functions.append(function)
	module = folder / "locks.mlir"
	module.write_text('''module {
  llvm.func @test_open(!llvm.ptr, i32, ...) -> i32
  llvm.func @test_flock(i32, i32) -> i32
  llvm.func @test_close(i32) -> i32
  func.func private @strict_ptr(%value: f64) -> !llvm.ptr {
    %bits = arith.bitcast %value : f64 to i64
    %pointer = llvm.inttoptr %bits : i64 to !llvm.ptr
    return %pointer : !llvm.ptr
  }
''' + "\n".join(functions) + "\n}", encoding="utf-8")
	driver = folder / "locks.c"
	driver.write_text('''#include <assert.h>
#include <stdio.h>
extern double lock_Posix(double), lock_Darwin(double);
static int descriptors, blocked, invalid;
int test_open(void* name, int flags, ...) { return invalid || descriptors == 64 ? -1 : ++descriptors; }
int test_flock(int descriptor, int flags) { assert(descriptor >= 0); return blocked ? -1 : 0; }
int test_close(int descriptor) { assert(descriptor > 0); --descriptors; return 0; }
int main(void) {
    double (*locks[])(double) = {lock_Posix, lock_Darwin};
    for (int platform = 0; platform < 2; platform++) {
        descriptors = 0; blocked = 1;
        for (int attempt = 0; attempt < 3000; attempt++) assert(locks[platform](0) == 0);
        assert(descriptors == 0);
        blocked = 0;
        assert(locks[platform](0) == 1 && descriptors == 1);
        invalid = 1;
        assert(locks[platform](0) == 0 && descriptors == 1);
        invalid = 0;
    }
    puts("Failed lock attempts close descriptors; successful locks stay held");
}
''', encoding="utf-8")
	lowered, llvm = folder / "locks.llvm.mlir", folder / "locks.ll"
	run([shutil.which("mlir-opt"), str(module), "--mlir-disable-threading", "--convert-arith-to-llvm", "--convert-func-to-llvm",
		"--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", str(lowered)])
	run([shutil.which("mlir-translate"), "--mlir-disable-threading", "--mlir-to-llvmir", str(lowered), "-o", str(llvm)])
	executable = folder / ("locks.exe" if os.name == "nt" else "locks")
	run([shutil.which("clang"), str(llvm), str(driver), "-O2", "-Wno-override-module", "-o", str(executable)])
	print(run([str(executable)]).strip(), flush=True)


def replacement(folder):
	assert os.name == "nt", "Replacement check requires Windows file sharing"
	original, candidate = folder / "app.exe", folder / "app.new.exe"
	backup = folder / "app.old.exe"
	source = folder / "ReplaceCheck.strict"
	build = (root / "Compiler/NativeBuild.strict").read_text(encoding="utf-8")
	body = build.split("Linked(base Text, tools Toolchain) Slow(Text)\n", 1)[1].split("\nCrossBuild(", 1)[0]
	body = body[body.index("\tlet executable ="):]
	body = body.replace("tools.Executable(base)", '"' + original.as_posix() + '"').replace("tools.isWeb", "false")
	body = re.sub(r"\tlet built = .*", '\tlet built = "Built " + target', body)
	source.write_text('has build Compiler/NativeBuild\nhas logger\nRun\n\tlogger.Log(Attempt)\nAttempt Text\n' + body,
		encoding="utf-8")
	command = [str(root / "strict.exe"), str(source)]
	original.write_bytes(b"old"); candidate.write_bytes(b"new")
	with candidate.open("rb"):
		result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
		assert result.returncode == 0, result.stdout + result.stderr
		assert original.exists() and original.read_bytes() == b"old", "Failed install lost original executable"
		assert candidate.read_bytes() == b"new", "Failed install lost candidate"
	print(run(command).strip(), flush=True)
	assert original.read_bytes() == b"new" and not candidate.exists() and not backup.exists()
	print("Failed replacement restores original; successful replacement removes backup", flush=True)


def discovery(folder):
	for index in range(70):
		(folder / f"Hop{index}.strict").write_text(
			f"has next Hop{index + 1}" if index < 69 else "has number", encoding="utf-8")
	source = folder / "DiscoveryCheck.strict"
	source.write_text('has logger\nRun\n'
		f'\tlogger.Log("./{(folder / "Hop69.strict").relative_to(root).as_posix()}" is in '
		f'SourceFiles(".").Of("{(folder / "Hop0.strict").as_posix()}"))', encoding="utf-8")
	result = run([str(root / "strict.exe"), str(source)])
	assert result.strip() == "true", "Source discovery truncated a finite dependency chain: " + result
	print("Source discovery reaches all 70 dependency edges", flush=True)


def linking(folder):
	for index in range(70):
		(folder / f"Chain{index}.strict").write_text("Result Number\n\t" +
			(f"Chain{index + 1}.Result" if index < 69 else "7"), encoding="utf-8")
	source = folder / "LinkCheck.strict"
	source.write_text('has logger\nRun\n'
		f'\tlogger.Log(FileCompiler(Logger).Compiled("{(folder / "Chain0.strict").as_posix()}", ".", false).Length >= 70)',
		encoding="utf-8")
	result = run([str(root / "strict.exe"), str(source)])
	assert result.strip().endswith("true"), "Linking failed to close a finite dependency chain: " + result
	print("Linking closes a chain longer than 50 rounds", flush=True)


def values(folder):
	(folder / "CachePair.strict").write_text("has Left Number\nhas Right Number", encoding="utf-8")
	source = folder / "ValueCheck.strict"
	source.write_text('''has logger
constant Stored = Made(4)
Run Number
	let values = Results(1024)
	logger.Log(values.Sum)
	values.Sum is 1024 then 0 else 1
Results(count Number) Numbers
	for count
		Checked(index)
Checked(index Number) Number
	let literal = (1, 2, 3)
	let pair = CachePair(1, 2)
	let appended = Stored + index
	let mapped = Mapped(Stored)
	let correct = (Stored is (0, 1, 2, 3)) and (mapped is (1, 2, 3, 4))
	correct and (literal.Sum is 6) and (pair.Left + pair.Right is 3) and (appended.Length is 5) then 1 else 0
Made(count Number) Numbers
	for count
		index
Mapped(numbers Numbers) Numbers
	for numbers
		value + 1
''', encoding="utf-8")
	print(run([str(root / "strict.exe"), "build", str(source)]).strip(), flush=True)
	module = source.with_suffix(".mlir").read_text(encoding="utf-8")
	assert "func.call @strict_cache_claim(" in module and "func.call @strict_cache_publish(" in module
	assert "@ValueCheck_Results_1_loop" in module, "Cold cache check did not exercise parallel emission"
	executable = source.with_suffix(".exe")
	for threads in ("1", "4"):
		result = subprocess.run([str(executable)], cwd=root, capture_output=True, text=True, timeout=15,
			env=dict(os.environ, STRICT_THREADS=threads, STRICT_GPU="0"))
		assert result.returncode == 0 and result.stdout.strip() == "1024", result.stdout + result.stderr
	print("Native literal/constructor caches and constant list reuse preserve values on 1 and 4 threads", flush=True)


with tempfile.TemporaryDirectory(prefix="AuditCheck", dir=root / "Examples") as temporary:
	folder = Path(temporary)
	for name, check in (("locks", locks), ("replace", replacement), ("discovery", discovery), ("linking", linking), ("values", values)):
		if len(sys.argv) == 1 or name in sys.argv[1:]:
			check(folder)
