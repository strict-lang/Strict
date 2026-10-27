"""Run from Strict: python Compiler/check_memory.py. A failed host allocation ends with OutOfMemory (Windows only)."""
import ctypes
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

if sys.platform != "win32":
	sys.exit("check_memory.py limits the process memory with a Windows job object")
root = Path(__file__).resolve().parent.parent


class Limits(ctypes.Structure):
	_fields_ = [("perProcessTime", ctypes.c_int64), ("perJobTime", ctypes.c_int64), ("flags", wintypes.DWORD),
		("minimumWorkingSet", ctypes.c_size_t), ("maximumWorkingSet", ctypes.c_size_t),
		("activeProcesses", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
		("scheduling", wintypes.DWORD), ("io", ctypes.c_uint64 * 6), ("processMemory", ctypes.c_size_t),
		("jobMemory", ctypes.c_size_t), ("peakProcess", ctypes.c_size_t), ("peakJob", ctypes.c_size_t)]


def limited(executable, data, megabytes):
	kernel = ctypes.WinDLL("kernel32", use_last_error=True)
	kernel.CreateJobObjectW.restype = wintypes.HANDLE
	kernel.OpenProcess.restype = wintypes.HANDLE
	job = kernel.CreateJobObjectW(None, None)
	limits = Limits(flags=0x100, processMemory=megabytes << 20)
	assert kernel.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(limits), ctypes.sizeof(limits))
	process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT)
	handle = kernel.OpenProcess(0x1F0FFF, False, process.pid)
	assert kernel.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(handle)), ctypes.get_last_error()
	output, _ = process.communicate(data, timeout=60)
	return process.returncode, output.decode(errors="replace")


with tempfile.TemporaryDirectory(prefix="MemoryCheck", dir=root / "Examples") as folder:
	source = Path(folder) / "MemoryCheck.strict"
	source.write_text("Run Number\n\tProcess.Input.Length", encoding="utf-8")
	built = subprocess.run([str(root / "strict.exe"), "build", str(source)], cwd=root, capture_output=True,
		text=True, timeout=270)
	assert built.returncode == 0, built.stdout + built.stderr
	executable = source.with_suffix(".exe")
	code, output = limited(executable, b"x" * 1000, 128)
	assert code == 1000, (code, output)
	code, output = limited(executable, b"x" * (256 << 20), 128)
	assert "OutOfMemory" in output and "AccessViolation" not in output, (code, output)
print("check_memory.py passed")
