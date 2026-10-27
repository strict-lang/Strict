"""Run from Strict: python Compiler/check_build.py. Builds gate on unit tests and cache passing results."""
import subprocess
import sys
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parent.parent
assert "@NativeCompiler_Checked_" in (root / "Compiler/Bootstrap.mlir").read_text(), \
	"Bootstrap compiler bypasses the unit-test gate"
extension = ".exe" if sys.platform == "win32" else ""
with tempfile.TemporaryDirectory(prefix="BuildCheck", dir=root / "Examples") as folder:
	source = Path(folder) / "BuildCheck.strict"
	dependency = Path(folder) / "BuildValue.strict"
	dependency.write_text("Result Number\n\t0", encoding="utf-8")
	content = Path(folder) / "Content"
	content.mkdir()
	asset = content / "data.txt"
	asset.write_text("first")
	source.write_text("Run Number\n\t1 is 2\n\tBuildValue.Result", encoding="utf-8")
	os.utime(source, (1700000000, 1700000000))
	def run(*arguments):
		return subprocess.run([str(root / ("strict" + extension)), *arguments], cwd=root,
			capture_output=True, text=True, timeout=270)
	trait = Path(folder) / "TraitCheck.strict"
	(Path(folder) / "Service.strict").write_text('Result Number\n', encoding="utf-8")
	(Path(folder) / "ServiceValue.strict").write_text('has number\nResult Number\n\tnumber\n', encoding="utf-8")
	(Path(folder) / "ServiceOther.strict").write_text('has number\nResult Number\n\tnumber * 2\n', encoding="utf-8")
	(Path(folder) / "ServiceHolder.strict").write_text('has service Service\nResult Number\n\tservice.Result\n', encoding="utf-8")
	trait.write_text('has logger\nRun Number\n\tlet first = ServiceHolder(ServiceValue(7)).Result + Read(ServiceOther(7))\n'
		'\tlet second = ServiceHolder(ServiceOther(7)).Result + Read(ServiceValue(7))\n'
		'\tlet result = first + second + ServiceHolder(ServiceValue(7)).service.Result\n'
		'\tlogger.Log(result)\n\tresult\n'
		'Read(service Service) Number\n\tservice.Result\n', encoding="utf-8")
	trait_build = run("build", str(trait))
	assert trait_build.returncode == 0, trait_build.stdout + trait_build.stderr
	assert subprocess.run([str(trait.with_suffix(extension))], timeout=10).returncode == 49
	trait_run = run(str(trait))
	assert trait_run.returncode == 0 and trait_run.stdout.strip() == "49", trait_run.stdout + trait_run.stderr
	print("Trait members and parameters call their concrete implementation", flush=True)
	cleanup = Path(folder) / "CleanupCheck.strict"
	locked_base = (Path(folder) / "Locked").as_posix()
	cleanup.write_text('has build Compiler/NativeBuild\nRun Number\n'
		f'\tNativeBuild(Logger).Removed("{locked_base}") then 0 else 1', encoding="utf-8")
	with open(locked_base + ".ll", "wb") as locked:
		cleaned = run(str(cleanup))
		assert "Could not remove build intermediate: " + locked_base + ".ll" in cleaned.stdout, cleaned.stdout + cleaned.stderr
		compiled = run("build", str(cleanup))
		assert compiled.returncode == 0, compiled.stdout + compiled.stderr
		cleaned = subprocess.run([str(cleanup.with_suffix(extension))], capture_output=True, text=True, timeout=10)
		assert cleaned.returncode == 1 and "Could not remove build intermediate:" in cleaned.stdout, cleaned.stdout + cleaned.stderr
	assert subprocess.run([str(cleanup.with_suffix(extension))], timeout=10).returncode == 0
	print("Locked build intermediates report cleanup failure", flush=True)
	nested = Path(folder) / "NestedGpu.strict"
	nested.write_text('has numbers Numbers\nhas logger\nRun Number\n'
		'\tlet result = NestedGpu(Counted(256), logger).Totals(8192).Sum\n\tlogger.Log(result)\n\t0\n'
		'Totals(count Number) Numbers\n\tfor count\n\t\tSquared.Sum + index\n'
		'Squared Numbers\n\tfor numbers\n\t\tvalue * value\n'
		'Counted(count Number) Numbers\n\tfor count\n\t\tindex', encoding="utf-8")
	compiled = run("build", str(nested))
	assert compiled.returncode == 0, compiled.stdout + compiled.stderr
	module = nested.with_suffix(".mlir").read_text(encoding="utf-8")
	assert "func.call @strict_parallel(%t" in module and "func.call @strict_gpu_run(%t" in module, "Nested CPU/GPU paths missing"
	for threads in ("1", "4"):
		result = subprocess.run([str(nested.with_suffix(extension))], capture_output=True, text=True, timeout=10,
			env=dict(os.environ, STRICT_THREADS=threads, STRICT_GPU="all"))
		expected = str(sum(number * number for number in range(256)) * 8192 + sum(range(8192)))
		assert result.returncode == 0 and result.stdout.strip() == expected, result.stdout + result.stderr
	print("Nested GPU loops use correct CPU fallback inside pool jobs", flush=True)
	profile = Path(folder) / "ProfileCheck.strict"
	profile.write_text('has profile Compiler/BuildProfile\nRun Number\n'
		'\tBuildProfile(Logger).Reported("Fast phase", Process.Elapsed)\n'
		'\tBuildProfile(Logger).Reported("Slow phase", Process.Elapsed - 3.1)\n\t0', encoding="utf-8")
	def checked_profile(result):
		assert result.returncode == 0, result.stdout + result.stderr
		assert "Fast phase:" not in result.stdout, result.stdout
		assert "Slow phase:" in result.stdout, result.stdout
	checked_profile(run(str(profile)))
	profile_build = run("build", str(profile))
	assert profile_build.returncode == 0, profile_build.stdout + profile_build.stderr
	checked_profile(subprocess.run([str(profile.with_suffix(extension))], capture_output=True,
		text=True, timeout=10))
	built = run("build", str(source))
	assert built.returncode != 0 and "Test failed:" in built.stdout, built.stdout + built.stderr
	assert not source.with_suffix(extension).exists(), "Failed unit tests produced a program"
	source.write_text("Run Number\n\t1 is 1\n\tBuildValue.Result", encoding="utf-8")
	os.utime(source, (1700000000, 1700000000))
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert "Running Tool" not in built.stdout and "Compiling " not in built.stdout, built.stdout
	assert "Test passed:" not in built.stdout and "Test failed:" not in built.stdout, built.stdout
	assert not source.with_suffix(".test" + extension).exists(), "Build compiled unwanted test executable"
	assert subprocess.run([str(source.with_suffix(extension))], timeout=10).returncode == 0
	executable = source.with_suffix(extension)
	modified = executable.stat().st_mtime_ns
	source.with_suffix(".mlir").unlink()
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	changed = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
	last_build = datetime.fromtimestamp(executable.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
	message = f"Up to date: {source} (last changed {changed}) < {executable.name} (last build {last_build})"
	assert built.stdout.strip() == message, built.stdout
	direct = subprocess.run([str(root / "Compiler" / ("NativeCompiler" + extension)), "build", str(source), "."],
		cwd=root, capture_output=True, text=True, timeout=10)
	assert direct.returncode == 0 and direct.stdout.strip() == message, direct.stdout + direct.stderr
	message_source = Path(folder) / "MessageCheck.strict"
	message_source.write_text('has cache Compiler/NativeCache\n'
		f'constant Source = {json.dumps(str(source))}\nRun Number\n'
		'\tLogger.Log(NativeCache("build", (Source, ".")).Message)\n\t0', encoding="utf-8")
	interpreted = run(str(message_source))
	assert interpreted.returncode == 0 and interpreted.stdout.strip() == message, interpreted.stdout + interpreted.stderr
	assert not source.with_suffix(".mlir").exists(), "Unchanged build regenerated MLIR"
	assert executable.stat().st_mtime_ns == modified, "Unchanged build relinked executable"
	asset.write_text("changed")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert source.with_suffix(".mlir").exists(), "Changed content was skipped"
	dependency.write_text("Result Number\n\t1", encoding="utf-8")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 1, "Dependency change was skipped"
	source.write_text("Run Number\n\t1 is 1\n\tBuildValue.Result + 1", encoding="utf-8")
	built = run("build", str(source), "small")
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 2, "Source change was skipped"
	assert source.with_suffix(".mlir").read_text().endswith("// -Os"), "Changed build options were skipped"
	executable.unlink()
	built = run("build", str(source), "small")
	assert built.returncode == 0 and executable.exists(), "Missing executable was not rebuilt: " + built.stdout
	built = run("build", str(source), "fast")
	assert built.returncode == 0, built.stdout + built.stderr
	assert source.with_suffix(".mlir").read_text().endswith("// -O0"), "Fast build still optimizes host code"
	assert subprocess.run([str(executable)], timeout=10).returncode == 2
	modified = executable.stat().st_mtime_ns
	source.with_suffix(".mlir").unlink()
	built = run("build", str(source), "fast")
	assert built.returncode == 0, built.stdout + built.stderr
	assert not source.with_suffix(".mlir").exists() and executable.stat().st_mtime_ns == modified, built.stdout
	built = run("build", str(source), "fast", "small")
	assert built.returncode != 0 and "Fast builds" in built.stdout, built.stdout + built.stderr
	source.write_text("Run Number\n\t1 is 2\n\tBuildValue.Result + 1", encoding="utf-8")
	tested = run("test", str(source))
	assert tested.returncode == 1 and "Test failed:" in tested.stdout, tested.stdout + tested.stderr
	tested = run("test", str(source), "native")
	assert source.with_suffix(".test" + extension).exists(), tested.stdout + tested.stderr
	assert tested.returncode == 1, tested.stdout + tested.stderr
	print("Build gates on unit tests; explicit VM and native failures surface")
	source = Path(folder) / "ReleaseCheck.strict"
	base = (Path(folder) / "Built").as_posix()
	release = (Path(folder) / "Release").as_posix()
	source.write_text('has release Compiler/Release\nRun Slow(Number)\n'
		f'\tReleaseCheck(Release(Logger, "{base}")).Retained is true\n\t0\nRetained Boolean\n'
		f'\tFile("{base + extension}").Write("built")\n'
		f'\tlet result = release.Refreshed("{release}", "", ("{extension}"))\n'
		f'\t(result is 0) and File("{base + extension}").Exists', encoding="utf-8")
	tested = run("test", str(source))
	assert tested.returncode == 0, tested.stdout + tested.stderr
	tested = run("test", str(source), "native")
	assert tested.returncode == 0, tested.stdout + tested.stderr
	assert (Path(release) / ("Built" + extension)).read_text() == "built"
	print("Release packaging retains executable for unchanged builds")
	source = Path(folder) / "CachedTests.strict"
	marker = Path(folder) / "Ran.exe"
	source.write_text('Run Number\n\tTouched is true\n\t0\nTouched Boolean\n'
		f'\tFile("{marker.as_posix()}").Write("tested")\n\ttrue', encoding="utf-8")
	tested = run("test", str(source), "native", "unit")
	assert tested.returncode == 0 and marker.exists(), tested.stdout + tested.stderr
	stamp = source.with_suffix(".test.nativecache")
	assert stamp.exists(), "Native test result was not saved: " + tested.stdout + tested.stderr
	before = stamp.read_text().splitlines()
	marker.unlink()
	source.with_suffix(".test.mlir").unlink()
	tested = run("test", str(source), "native", "unit")
	assert tested.returncode == 0, tested.stdout + tested.stderr
	after = stamp.read_text().splitlines()
	changed = [before[index] for index in range(1, len(before), 2)
		if index + 1 >= len(after) or before[index:index + 2] != after[index:index + 2]]
	assert not marker.exists(), "Unchanged native tests executed again: " + str(changed) + tested.stdout
	assert not source.with_suffix(".test.mlir").exists(), "Unchanged native tests regenerated MLIR"
	print("Unchanged builds and native tests skip compilation and execution")
	base = Path(folder) / "OldGpu"
	for suffix, content in [(".kernels.mlir", "module {\n}"), (".ptx", "// cached")]:
		artifact = base.with_suffix(suffix)
		artifact.write_text(content)
		os.utime(artifact, (1, 1))
	source = Path(folder) / "GpuCacheCheck.strict"
	source.write_text('has cache Compiler/PtxCache\nRun Number\n'
		f'\tGpuCacheCheck(PtxCache("{base.as_posix()}")).Previous is 1\n\t0\nPrevious Number\n'
		'\tcache.Previous("module {\\n}", false).Length', encoding="utf-8")
	tested = run("test", str(source))
	assert tested.returncode == 0, tested.stdout + tested.stderr
	print("Identical GPU input remains cached after compiler rebuild")
	cases = [(b"", ""), (b"one\ntwo\n", "one\ntwo\n"), (b"one\r\ntwo\r", "one\ntwo"),
		(b"one\rtwo", "one\rtwo"), (b"\xef\xbb\xbfhello", "hello"), ("ä\r\nö".encode(), "ä\nö")]
	checks = []
	for index, (content, expected) in enumerate(cases):
		data = Path(folder) / f"lines{index}.txt"
		data.write_bytes(content)
		checks.append(f'\tJoined({json.dumps(data.as_posix())}) is {json.dumps(expected, ensure_ascii=False)}')
	source = Path(folder) / "ReadTextCheck.strict"
	source.write_text('Joined(path Text) Text\n' + '\n'.join(checks) +
		'\n\tText.Combine(File(path).ReadLines, "\\n")', encoding="utf-8")
	for option in [(), ("native",)]:
		tested = run("test", str(source), *option)
		assert tested.returncode == 0, tested.stdout + tested.stderr
	assert 'func.call @strict_file_joined(' in source.with_suffix('.test.mlir').read_text(), \
		"Whole-file reads still allocate a list of lines before joining it"
	print("Whole-file read fusion preserves LF, CRLF, BOM, Unicode and empty files")

def check_parallel_refresh():
	"""Builds started together after a compiler change rebuild NativeCompiler once, the others wait for it."""
	os.utime(root / "Compiler/NativeCompiler.strict")
	programs = ["Examples/HelloLogger.strict", "Examples/FizzBuzz.strict", "Examples/Fibonacci.strict"]
	started = [subprocess.Popen([str(root / ("strict" + extension)), "build", program], cwd=root,
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for program in programs]
	outputs = [process.communicate(timeout=280)[0] for process in started]
	logs = "\n----\n".join(outputs)
	assert all(process.returncode == 0 for process in started), "A parallel build failed:\n" + logs
	assert sum("building it again first" in output for output in outputs) == 1, "Not exactly one rebuild:\n" + logs
	assert not (root / "Compiler/NativeCompiler.building").exists(), "The rebuild lock was left behind"
	print("Parallel builds after a compiler change rebuild NativeCompiler once")

check_parallel_refresh()
