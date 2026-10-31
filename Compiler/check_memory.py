"""Run from Strict: python Compiler/check_memory.py. A failed host allocation ends with OutOfMemory (Windows only), a
program dropping 52 MB lists while keeping 260 MB runs in the 1 GB heap of the web and iOS (natively and in node)."""
import ctypes
import os
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


def built(folder, name, code, platform=""):
	source = Path(folder) / (name + ".strict")
	source.write_text(code, encoding="utf-8")
	result = subprocess.run([str(root / "strict.exe"), "build", str(source)] + ([platform] if platform else []),
		cwd=root, capture_output=True, text=True, timeout=270)
	assert result.returncode == 0, result.stdout + result.stderr
	return source.with_suffix(".js" if platform else ".exe")


kept = ("has logger\nFilled(count Number, offset Number) Numbers\n\tFilled(2, 1) is (1, 2)\n\tfor count\n\t\tindex + offset\n"
	"Run\n\tlet kept = (Filled(6500000, 1), Filled(6500000, 2), Filled(6500000, 3), Filled(6500000, 4), "
	"Filled(6500000, 5))\n")


with tempfile.TemporaryDirectory(prefix="MemoryCheck", dir=root / "Examples") as folder:
	executable = built(folder, "MemoryCheck", "Run Number\n\tProcess.Input.Length")
	code, output = limited(executable, b"x" * 1000, 128)
	assert code == 1000, (code, output)
	code, output = limited(executable, b"x" * (256 << 20), 128)
	assert "OutOfMemory" in output and "AccessViolation" not in output, (code, output)
	pressure = built(folder, "MemoryPressure", kept + "\tmutable total = 0\n\tfor 60\n"
		"\t\tlet temporary = Filled(6500000 + index % 2 * 8192, 0)\n"
		"\t\ttotal = total + temporary(index) + kept(index % 5)(index)\n\tlogger.Log(\"total \" + total)")
	ran = subprocess.run([str(pressure)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_HEAP": "1024"})
	assert ran.returncode == 0 and "total 3720" in ran.stdout, (ran.returncode, ran.stdout[-300:])
	script = built(folder, "WebPressure", kept + "\tmutable recent = kept(0)\n\tmutable seed = 0.7071067811865476\n"
		"\tfor 60\n\t\tlet temporary = Filled(6500000 + index % 2 * 8192, recent(index) + seed)\n"
		"\t\tseed = (seed * 9301.123 + 0.49297) % 1\n\t\trecent = temporary\n"
		"\tlogger.Log(\"total \" + (recent(59) + kept(4)(59)).Round)", "web")
	node = next((root / "Tools" / "emsdk" / "node").glob("*/node.exe"))
	ran = subprocess.run([str(node), str(script)], capture_output=True, text=True, timeout=60)
	assert ran.returncode == 0 and "total 1924" in ran.stdout, (ran.returncode, ran.stdout[-300:])
print("check_memory.py passed")
