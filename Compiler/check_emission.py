"""Run from Strict after rebuilding NativeCompiler: python Compiler/check_emission.py."""
from pathlib import Path
import os
import subprocess
import time

root = Path(__file__).resolve().parent.parent
compiler = root / "Compiler" / ("NativeCompiler.exe" if os.name == "nt" else "NativeCompiler")
started = time.perf_counter()
process = subprocess.Popen([str(compiler), "test", "Math/Matrix.strict", ".", "native"],
	cwd=root, start_new_session=os.name != "nt")
try:
	code = process.wait(timeout=240)
except subprocess.TimeoutExpired:
	if os.name == "nt":
		subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
	else:
		import signal
		os.killpg(process.pid, signal.SIGKILL)
	process.wait()
	raise
if os.name == "nt":
	import ctypes
	from ctypes import wintypes
	class Memory(ctypes.Structure):
		_fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
			(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
			"QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
			"QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
	memory = Memory()
	memory.cb = ctypes.sizeof(memory)
	assert ctypes.windll.psapi.GetProcessMemoryInfo(
		wintypes.HANDLE(int(process._handle)), ctypes.byref(memory), memory.cb)
	peak = memory.PeakWorkingSetSize
else:
	import resource
	peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)
assert code == 0, f"Matrix native tests failed: {code}"
print(f"Matrix: {peak / 2**20:.0f} MB peak, {time.perf_counter() - started:.2f} s")
assert peak < 1024 * 2**20, "Matrix compilation exceeded 1 GB"
