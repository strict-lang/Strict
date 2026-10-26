"""Run from Strict: python Compiler/check_profile.py."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
extension = ".exe" if os.name == "nt" else ""
with tempfile.TemporaryDirectory(prefix="ProfileCheck", dir=root / "Examples") as temporary:
	folder = Path(temporary)
	probe = folder / "Probe.strict"
	probe.write_text('has profile Compiler/BuildProfile\nRun Number\n'
		'\tBuildProfile(Logger).Reported("Fast profile probe", Process.Elapsed)\n'
		'\t(BuildProfile.Fallback is BuildProfile.Detailed) then 0 else 1', encoding="utf-8")
	def run(arguments, detailed):
		environment = dict(os.environ, STRICT_BUILD_PROFILE=detailed, STRICT_THREADS="8")
		result = subprocess.run(arguments, cwd=root, env=environment, capture_output=True, text=True, timeout=90)
		assert result.returncode == 0, result.stdout + result.stderr
		return result.stdout
	command = str(root / ("strict" + extension))
	assert "Fast profile probe:" in run([command, str(probe)], "1"), "Detailed VM timings are missing"
	assert "Fast profile probe:" not in run([command, str(probe)], "0")
	run([command, "build", str(probe)], "1")
	assert "Fast profile probe:" in run([str(probe.with_suffix(extension))], "1")
	assert "Fast profile probe:" not in run([str(probe.with_suffix(extension))], "0")
	program = folder / "Program.strict"
	program.write_text("Run Number\n\t0", encoding="utf-8")
	output = run([command, "build", str(program)], "1")
	for stage in ["Dependency inventory", "Dependency snapshot", "Frontend / source discovery", "Frontend / type inference",
		"Frontend / dependency linking", "Frontend / syntax validation", "Frontend / name validation",
		"Emission / reachability", "Emission / safety", "Emission / function bodies", "GPU preparation",
		"Host runtime assembly", "mlir-opt", "mlir-translate", "Linking", "Packaging", "Build total"]:
		assert stage in output, stage + " missing:\n" + output
	assert "Test passed:" not in output and "Test failed:" not in output, output
	executable = program.with_suffix(extension)
	stamp = executable.stat().st_mtime_ns
	for detailed in ["1", "0"]:
		output = run([command, "build", str(program)], detailed)
		assert "Up to date:" in output and "Frontend /" not in output, output
		assert ("CLI build cache hit:" in output) == (detailed == "1"), output
		assert executable.stat().st_mtime_ns == stamp
	print("Detailed VM/native timings are opt-in; cache hits still skip all compilation and tests")
