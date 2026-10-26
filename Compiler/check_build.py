"""Run from Strict: python Compiler/check_build.py. Builds must not execute tests."""
import subprocess
import sys
import tempfile
import os
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
extension = ".exe" if sys.platform == "win32" else ""
with tempfile.TemporaryDirectory(prefix="BuildCheck", dir=root / "Examples") as folder:
	source = Path(folder) / "BuildCheck.strict"
	dependency = Path(folder) / "BuildValue.strict"
	dependency.write_text("Result Number\n\t0", encoding="utf-8")
	source.write_text("Run Number\n\t1 is 2\n\tBuildValue.Result", encoding="utf-8")
	def run(*arguments):
		return subprocess.run([str(root / ("strict" + extension)), *arguments], cwd=root,
			capture_output=True, text=True, timeout=270)
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert "Test passed:" not in built.stdout and "Test failed:" not in built.stdout, built.stdout
	assert not source.with_suffix(".test" + extension).exists(), "Build compiled unwanted test executable"
	assert subprocess.run([str(source.with_suffix(extension))], timeout=10).returncode == 0
	executable = source.with_suffix(extension)
	modified = executable.stat().st_mtime_ns
	source.with_suffix(".mlir").unlink()
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert not source.with_suffix(".mlir").exists(), "Unchanged build regenerated MLIR"
	assert executable.stat().st_mtime_ns == modified, "Unchanged build relinked executable"
	dependency.write_text("Result Number\n\t1", encoding="utf-8")
	built = run("build", str(source))
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 1, "Dependency change was skipped"
	source.write_text("Run Number\n\t1 is 2\n\tBuildValue.Result + 1", encoding="utf-8")
	built = run("build", str(source), "small")
	assert built.returncode == 0, built.stdout + built.stderr
	assert subprocess.run([str(executable)], timeout=10).returncode == 2, "Source change was skipped"
	assert source.with_suffix(".mlir").read_text().endswith("// -Os"), "Changed build options were skipped"
	executable.unlink()
	built = run("build", str(source), "small")
	assert built.returncode == 0 and executable.exists(), "Missing executable was not rebuilt: " + built.stdout
	tested = run("test", str(source))
	assert tested.returncode == 1 and "Test failed:" in tested.stdout, tested.stdout + tested.stderr
	tested = run("test", str(source), "native")
	assert source.with_suffix(".test" + extension).exists(), tested.stdout + tested.stderr
	assert tested.returncode == 1, tested.stdout + tested.stderr
	print("Build skips tests; explicit VM and native tests still fail correctly")
	source = Path(folder) / "ReleaseCheck.strict"
	base = (Path(folder) / "Built").as_posix()
	release = (Path(folder) / "Release").as_posix()
	source.write_text('has release Compiler/Release\nRun Number\n'
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
	tested = run("test", str(source), "native")
	assert tested.returncode == 0 and marker.exists(), tested.stdout + tested.stderr
	stamp = source.with_suffix(".test.nativecache")
	assert stamp.exists(), "Native test result was not saved: " + tested.stdout + tested.stderr
	before = stamp.read_text().splitlines()
	marker.unlink()
	source.with_suffix(".test.mlir").unlink()
	tested = run("test", str(source), "native")
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
