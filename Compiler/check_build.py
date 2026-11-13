"""Run from Strict: python Compiler/check_build.py. Builds gate on unit tests and cache passing results."""
import subprocess
import sys
import tempfile
import os
import json
import time
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parent.parent
assert "@NativeCompiler_Checked_" in (root / "Compiler/Bootstrap.mlir").read_text(), \
	"Bootstrap compiler bypasses the unit-test gate"
runtime = (root / "Compiler/NativeRuntime.mlir").read_text(encoding="utf-8")
assert "strict_gc_watching(0 : i64)" in runtime and "strict_gc_watching(1 : i64)" not in runtime
bootstrap = (root / "Compiler/bootstrap.sh").read_text(encoding="utf-8")
assert "\nrelease=linux\n" in bootstrap and "build Strict.strict . $release\n" in bootstrap, \
	"Linux bootstrap does not build a release compiler"
assert "build Strict.strict . windows" in (root / "Compiler/Bootstrap.cmd").read_text(encoding="utf-8")
extension = ".exe" if sys.platform == "win32" else ""
def obj(source, suffix):
	"""Where a build keeps an intermediate file of source: obj/ next to it."""
	return source.parent / "obj" / (source.stem + suffix)
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
		'\tlogger.Log(result to Text)\n\tresult\n'
		'Read(service Service) Number\n\tservice.Result\n', encoding="utf-8")
	trait_build = run("build", str(trait))
	assert trait_build.returncode == 0, trait_build.stdout + trait_build.stderr
	assert subprocess.run([str(trait.with_suffix(extension))], timeout=10).returncode == 49
	trait_run = run(str(trait))
	assert trait_run.returncode == 49 and trait_run.stdout.strip().rsplit("\n", 1)[-1] == "49", trait_run.stdout + trait_run.stderr
	print("Trait members and parameters call their concrete implementation", flush=True)
	cleanup = Path(folder) / "CleanupCheck.strict"
	locked_base = (Path(folder) / "Locked").as_posix()
	cleanup.write_text('has build Compiler/NativeBuild\nRun Number\n'
		f'\tIntermediates(Logger).Removed("{locked_base}") is Error then 1 else 0', encoding="utf-8")
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
		'\tlet result = NestedGpu(Counted(256), logger).Totals(8192).Sum\n\tlogger.Log(result to Text)\n\t0\n'
		'Totals(count Number) Numbers\n\tfor count\n\t\tSquared.Sum + index\n'
		'Squared Numbers\n\tfor numbers\n\t\tvalue * value\n'
		'Counted(count Number) Numbers\n\tfor count\n\t\tindex', encoding="utf-8")
	compiled = run("build", str(nested))
	assert compiled.returncode == 0, compiled.stdout + compiled.stderr
	module = obj(nested, ".mlir").read_text(encoding="utf-8")
	assert "func.call @strict_parallel(%t" in module and "func.call @strict_gpu_run(%t" in module, "Nested CPU/GPU paths missing"
	expected = str(sum(number * number for number in range(256)) * 8192 + sum(range(8192)))
	automatic = {name: value for name, value in os.environ.items() if name != "STRICT_GPU"}
	for threads in ("1", "4"):
		result = subprocess.run([str(nested.with_suffix(extension))], capture_output=True, text=True, timeout=10,
			env=dict(automatic, STRICT_THREADS=threads))
		assert result.returncode == 0 and result.stdout.strip().endswith(expected), result.stdout + result.stderr
		forced = subprocess.run([str(nested.with_suffix(extension))], capture_output=True, text=True, timeout=10,
			env=dict(automatic, STRICT_THREADS=threads, STRICT_GPU="all"))
		assert forced.returncode == 1 and "GpuLoopMustRunOnGpu" in forced.stdout, forced.stdout + forced.stderr
	print("A small GPU loop inside pool jobs runs on the CPU, one forced onto the GPU stops the program", flush=True)
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
	assert not obj(source, ".test" + extension).exists(), "Build compiled unwanted test executable"
	assert subprocess.run([str(source.with_suffix(extension))], timeout=10).returncode == 0
	executable = source.with_suffix(extension)
	modified = executable.stat().st_mtime_ns
	obj(source, ".mlir").unlink()
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	changed = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
	last_build = datetime.fromtimestamp(executable.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
	message = f"Up to date: {source.as_posix()} (last changed {changed}) < {executable.name} (last build {last_build})"
	assert built.stdout.strip() == message, built.stdout
	direct = subprocess.run([str(root / ("strict" + extension)), "build", str(source), "."],
		cwd=root, capture_output=True, text=True, timeout=10)
	assert direct.returncode == 0 and direct.stdout.strip() == message, direct.stdout + direct.stderr
	message_source = Path(folder) / "MessageCheck.strict"
	message_source.write_text('has cache Compiler/NativeCache\n'
		f'constant Source = {json.dumps(source.as_posix())}\nRun Number\n'
		'\tLogger.Log(NativeCache("build", (Source, ".")).Message)\n\t0', encoding="utf-8")
	interpreted = run(str(message_source))
	assert interpreted.returncode == 0 and interpreted.stdout.strip().endswith(message), interpreted.stdout + interpreted.stderr
	assert not obj(source, ".mlir").exists(), "Unchanged build regenerated MLIR"
	assert executable.stat().st_mtime_ns == modified, "Unchanged build relinked executable"
	asset.write_text("changed")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert obj(source, ".mlir").exists(), "Changed content was skipped"
	dependency.write_text("Result Number\n\t1", encoding="utf-8")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 1, "Dependency change was skipped"
	source.write_text("Run Number\n\t1 is 1\n\tBuildValue.Result + 1", encoding="utf-8")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 2, "Source change was skipped"
	executable.unlink()
	built = run("build", str(source))
	assert built.returncode == 0 and executable.exists(), "Missing executable was not rebuilt: " + built.stdout
	source.write_text("Run Number\n\t1 is 2\n\tBuildValue.Result + 1", encoding="utf-8")
	tested = run("test", str(source))
	assert tested.returncode == 1 and "Test failed:" in tested.stdout, tested.stdout + tested.stderr
	tested = run("test", str(source), "native")
	assert obj(source, ".test" + extension).exists(), tested.stdout + tested.stderr
	assert tested.returncode == 1, tested.stdout + tested.stderr
	print("Build gates on unit tests; explicit VM and native failures surface")
	source = Path(folder) / "ReleaseCheck.strict"
	base = (Path(folder) / "Built").as_posix()
	release = (Path(folder) / "Release").as_posix()
	source.write_text('has release Compiler/Release\nRun Number slow\n'
		f'\tReleaseCheck(Release(Logger, "{base}")).Retained is true\n\t0\nRetained Boolean\n'
		f'\tFile("{base + extension}").Write("built")\n'
		f'\trelease.Refreshed("{release}", "", "{base}", ("{extension}"))\n'
		f'\tFile("{base + extension}").Exists', encoding="utf-8")
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
	stamp = obj(source, ".test.nativecache")
	assert stamp.exists(), "Native test result was not saved: " + tested.stdout + tested.stderr
	before = stamp.read_text().splitlines()
	marker.unlink()
	obj(source, ".test.mlir").unlink()
	tested = run("test", str(source), "native", "unit")
	assert tested.returncode == 0, tested.stdout + tested.stderr
	after = stamp.read_text().splitlines()
	changed = [before[index] for index in range(1, len(before), 2)
		if index + 1 >= len(after) or before[index:index + 2] != after[index:index + 2]]
	assert not marker.exists(), "Unchanged native tests executed again: " + str(changed) + tested.stdout
	assert not obj(source, ".test.mlir").exists(), "Unchanged native tests regenerated MLIR"
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
	assert 'func.call @strict_file_joined(' in obj(source, ".test.mlir").read_text(), \
		"Whole-file reads still allocate a list of lines before joining it"
	print("Whole-file read fusion preserves LF, CRLF, BOM, Unicode and empty files")

def check_concurrent_builds():
	"""Builds of one target take turns on its lock, a killed build releases it and the next one recovers."""
	with tempfile.TemporaryDirectory(prefix="ConcurrentCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "ConcurrentCheck.strict"
		source.write_text("Run Number\n\t1 is 1\n\t3", encoding="utf-8")
		lock = source.parent / "obj" / (source.name + ".lock")
		executable = source.with_suffix(extension)
		def started(*options):
			return subprocess.Popen([str(root / ("strict" + extension)), "build", str(source), *options], cwd=root,
				stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=sys.platform != "win32")
		def holding(process):
			deadline = time.monotonic() + 60
			while not lock.exists():
				assert process.poll() is None and time.monotonic() < deadline, "No build lock appeared: " + str(lock)
				time.sleep(0.02)
		def waiting(*options):
			process = started(*options)
			for line in process.stdout:
				if line.startswith("Waiting for the other build"):
					return process
			assert False, "Build did not wait for the lock: " + process.communicate()[0]
		def finished(process):
			output = process.communicate(timeout=120)[0]
			assert process.returncode == 0, output
		first = started()
		holding(first)
		second = waiting()
		finished(first)
		finished(second)
		assert subprocess.run([str(executable)], timeout=10).returncode == 3
		lock.unlink(missing_ok=True)
		source.write_text("Run Number\n\t1 is 1\n\t4", encoding="utf-8")
		killed = started()
		holding(killed)
		recovering = waiting()
		if sys.platform == "win32":
			subprocess.run(["taskkill", "/T", "/F", "/PID", str(killed.pid)], capture_output=True)
		else:
			os.killpg(killed.pid, 9)
		killed.wait(timeout=10)
		finished(recovering)
		assert subprocess.run([str(executable)], timeout=10).returncode == 4
		leftovers = [path.name for path in Path(folder).iterdir() if path.name.endswith((".new.exe", ".old.exe"))]
		assert not leftovers, "Replacement executables were left behind: " + str(leftovers)
	print("Concurrent builds of one target take turns, a killed build releases its lock")

def check_link_workers():
	"""A program whose calls are found over several link rounds compiles the same with one worker and with all."""
	source = root / "Examples/LinkChain/LinkChain.strict"
	modules = []
	for threads in ("1", ""):
		cached = source.parent / "obj"
		for stale in [*cached.glob("*.strictbinary"), *cached.glob("*.nativecache")]:
			stale.unlink()
		environment = dict(os.environ, STRICT_THREADS=threads) if threads else os.environ
		tested = subprocess.run([str(root / ("strict" + extension)), "test", str(source), "native"], cwd=root,
			capture_output=True, text=True, timeout=200, env=environment)
		assert tested.returncode == 0, tested.stdout + tested.stderr
		modules.append(obj(source, ".test.mlir").read_text().rsplit("\n", 1)[0])
	assert modules[0] == modules[1], "Linking with one worker emitted a different program"
	print("Links found over several rounds compile the same with one worker and with all")

def check_moved_program():
	"""A program moved away from its obj folder stops with the PTX error, also for a loop only big enough to warm the GPU."""
	with tempfile.TemporaryDirectory(prefix="MovedCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "MovedCheck.strict"
		source.write_text("has logger\nhas weights Numbers with value.Bits is 32\nLoop(count Number) Numbers\n\tfor count\n"
			"\t\tweights(index % 4) * 3\nRun(count Number)\n\tlet results = MovedCheck(logger, (1.1, 2.2, 3.3, 5.5)).Loop(count)\n"
			"\tlogger.Log(results(0) to Text)\n", encoding="utf-8")
		built = subprocess.run([str(root / ("strict" + extension)), "build", str(source)], cwd=root,
			capture_output=True, text=True, timeout=120)
		assert built.returncode == 0, built.stdout + built.stderr
		with tempfile.TemporaryDirectory(prefix="MovedCheck") as elsewhere:
			moved = Path(elsewhere) / ("MovedCheck" + extension)
			moved.write_bytes(source.with_suffix(extension).read_bytes())
			moved.chmod(0o755)
			for count in ("32000", "128000", "5000000"):
				ran = subprocess.run([str(moved), count], cwd=elsewhere, capture_output=True, text=True, timeout=30)
				assert ran.returncode == 1 and "is missing or empty" in ran.stdout, ran.stdout + ran.stderr
	print("A program without its PTX stops with an error instead of hanging")

def check_native_folder():
	"""Native libraries open relative paths in the folder of the request, not where the resident server started."""
	strict = str(root / ("strict" + extension))
	subprocess.run([strict, "stop"], cwd=root, capture_output=True, text=True, timeout=60)
	started = subprocess.run([strict, "test", "Fibonacci.strict"], cwd=root / "Examples", capture_output=True, text=True, timeout=120)
	assert started.returncode == 0, started.stdout + started.stderr
	cached = root / "NeuralNetworks/obj/Layer.strictbinary"
	if cached.exists():
		cached.unlink()
	tested = subprocess.run([strict, "test", "NeuralNetworks/Layer.strict"], cwd=root, capture_output=True, text=True, timeout=120)
	assert "GgufFileNotReadable" not in tested.stdout + tested.stderr, tested.stdout + tested.stderr
	subprocess.run([strict, "stop"], cwd=root, capture_output=True, text=True, timeout=60)
	print("Native libraries read files relative to the request of a resident server started elsewhere")
	with tempfile.TemporaryDirectory(prefix="MovingCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "MovingCheck.strict"
		source.write_text('Moved Number\n\tDirectory.Change("Math") is not Error\n\t1\n', encoding="utf-8")
		built = subprocess.run([strict, "build", str(source)], cwd=root, capture_output=True, text=True, timeout=120)
		assert built.returncode == 0, built.stdout + built.stderr
	subprocess.run([strict, "stop"], cwd=root, capture_output=True, text=True, timeout=60)
	print("A test that changes the folder does not move the build")

def check_native_change():
	"""A changed native source or .strict file of a package in another folder rebuilds for strict build and strict <file>."""
	strict = str(root / ("strict" + extension))
	with tempfile.TemporaryDirectory(prefix="NativeChange", dir=root / "Examples") as folder:
		source = Path(folder) / "NativeChange.strict"
		source.write_text('has logger\nRun\n\tlogger.Log(Hmac.Sha256("key", "text"))\n', encoding="utf-8")
		built = subprocess.run([strict, "build", str(source)], cwd=root, capture_output=True, text=True, timeout=270)
		assert built.returncode == 0, built.stdout + built.stderr
		for changed in ("Networking/native/Hmac.c", "Networking/Hmac.strict"):
			for command in (["build"], []):
				os.utime(root / changed)
				result = subprocess.run([strict, *command, str(source)], cwd=root, capture_output=True, text=True, timeout=270)
				assert result.returncode == 0 and "Up to date" not in result.stdout, changed + ": " + result.stdout + result.stderr
	print("A changed native source or package of another folder rebuilds for strict build and strict <file>")

def check_compiler_change():
	"""A changed compiler compiles a used type again instead of linking the bytecode the old one stored for it."""
	strict = root / ("strict" + extension)
	with tempfile.TemporaryDirectory(prefix="CompilerChange", dir=root / "Examples") as folder:
		(Path(folder) / "CompilerUsed.strict").write_text("Doubled(number Number) Number\n\tDoubled(2) is 4\n\tnumber * 2\n",
			encoding="utf-8")
		source = Path(folder) / "CompilerChange.strict"
		source.write_text("Tripled(number Number) Number\n\tTripled(2) is 6\n\tCompilerUsed.Doubled(number) + number\n",
			encoding="utf-8")
		used = obj(source.with_name("CompilerUsed.strict"), ".strictbinary")
		def tested():
			result = subprocess.run([str(strict), "test", str(source), "native"], cwd=root, capture_output=True, text=True,
				timeout=270)
			assert result.returncode == 0 and "Up to date" not in result.stdout, result.stdout + result.stderr
			return used.stat().st_mtime_ns
		stored = tested()
		os.utime(source)
		assert tested() == stored, "An unchanged compiler did not link the stored bytecode of a used type"
		stamp = strict.stat()
		try:
			os.utime(strict, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
			assert tested() != stored, "A changed compiler linked the bytecode the old one stored for a used type"
		finally:
			os.utime(strict, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
	print("A changed compiler compiles used types again")

def check_named_failure():
	"""A build stopped by an Error of the front end names it instead of ending with exit 1 and no output."""
	with tempfile.TemporaryDirectory(prefix="NamedCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "NamedCheck.strict"
		source.write_text('has logger\nNext(number Number) Number\n\tnumber + 1\nRun\n\tNext(3)\n\tlogger.Log("done")\n', encoding="utf-8")
		built = subprocess.run([str(root / ("strict" + extension)), "build", str(source)], cwd=root,
			capture_output=True, text=True, timeout=120)
		assert built.returncode == 1 and "ResultMustBeUsed: Next(3)" in built.stdout, built.stdout + built.stderr
	print("A build stopped by a front end Error names it")

def check_held_copy():
	"""A rebuild succeeds while processes still run the executable and its previous old copy (Windows moves it to .old2)."""
	if sys.platform != "win32":
		return
	strict = str(root / "strict.exe")
	with tempfile.TemporaryDirectory(prefix="HeldCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "HeldCheck.strict"
		executable = source.with_suffix(".exe")
		running = []
		try:
			for attempt in range(3):
				source.write_text("Run(seconds Number)\n\tlet started = Process.Elapsed\n\tfor as long as Process.Elapsed - "
					"started < seconds\n\t\tProcess.Wait(0.1" + str(attempt) + ")\n", encoding="utf-8")
				built = subprocess.run([strict, "build", str(source)], cwd=root, capture_output=True, text=True, timeout=120)
				assert built.returncode == 0, built.stdout + built.stderr
				if attempt < 2:
					running.append(subprocess.Popen([str(executable), "60"]))
			assert source.with_suffix(".old2.exe").exists(), "The held old copy was not kept beside a new .old2 copy"
		finally:
			for process in running:
				process.kill()
				process.wait()
	print("A rebuild succeeds while the previous executable and its old copy still run")

def check_restarted_resident():
	"""A rebuild stops a running resident program (obj/strict.server* next to it) and starts it again with its arguments."""
	if sys.platform != "win32":
		return
	strict = str(root / "strict.exe")
	def listed(executable):
		query = ("Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq '" + str(executable) +
			"' } | ForEach-Object { '' + $_.ProcessId + ' ' + $_.CommandLine }")
		found = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=60)
		return [line.split(" ", 1) for line in found.stdout.splitlines() if line.strip()]
	with tempfile.TemporaryDirectory(prefix="ResidentCheck", dir=root / "Examples") as folder:
		source = Path(folder) / "ResidentCheck.strict"
		executable = source.with_suffix(".exe")
		started = None
		try:
			for attempt in range(2):
				source.write_text("Run(seconds Number)\n\tlet resident = Resident(Path(Process.Current.executable).PathOnly.text)\n"
					"\tlet server = resident.Opened\n\tlet started = Process.Elapsed\n\tfor as long as Process.Elapsed - started < "
					"seconds\n\t\tProcess.Wait(0.1" + str(attempt) + ")\n\tresident.Closed(server)\n", encoding="utf-8")
				built = subprocess.run([strict, "build", str(source)], cwd=root, capture_output=True, text=True, timeout=120)
				assert built.returncode == 0, built.stdout + built.stderr
				if attempt == 0:
					started = subprocess.Popen([str(executable), "60"])
					time.sleep(1)
			assert started.poll() is not None, "The build did not stop the running resident program\n" + built.stdout
			again = listed(executable)
			assert len(again) == 1 and again[0][1].split()[1:] == ["60"], str(again) + "\n" + built.stdout
			assert "Stopping " in built.stdout and "Started " in built.stdout, built.stdout
		finally:
			if started is not None and started.poll() is None:
				started.kill()
			for process in listed(executable):
				subprocess.run(["taskkill", "/F", "/PID", process[0]], capture_output=True)
			time.sleep(1)
	print("A rebuild stops a running resident program and starts it again with its arguments")

check_held_copy()
check_restarted_resident()
check_named_failure()
check_concurrent_builds()
check_link_workers()
check_moved_program()
check_native_folder()
check_native_change()
check_compiler_change()
