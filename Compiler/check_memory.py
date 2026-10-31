"""Run from Strict: python Compiler/check_memory.py. A failed host allocation ends with OutOfMemory (Windows only), a
program dropping 52 MB lists while keeping 260 MB runs in the 1 GB heap of the web and iOS (natively and in node), one
dropping 3 MB lists and many small texts in 128 MB, a Realtime type collects at the end of its frames (Presented)."""
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


filled = "has logger\nFilled(count Number, offset Number) Numbers\n\tFilled(2, 1) is (1, 2)\n\tfor count\n\t\tindex + offset\n"
kept = filled + ("Run\n\tlet kept = (Filled(6500000, 1), Filled(6500000, 2), Filled(6500000, 3), Filled(6500000, 4), "
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
	small = built(folder, "SmallPressure", filled + "Words(count Number, offset Number) Texts\n"
		"\tWords(2, 1) is (\"w1\", \"w2\")\n\tfor count\n\t\t\"w\" + (index + offset)\nHeld(count Number) List(Numbers)\n"
		"\tHeld(0).Length is 0\n\tfor count\n\t\tFilled(2000000, index)\nRun\n\tlet held = Held(4)\n"
		"\tmutable recent = held(0)\n\tmutable total = 0\n\tfor 1500\n"
		"\t\tlet temporary = Filled(400000 + index % 3 * 8192, recent(index % 1000))\n\t\tlet texts = Words(3000, index)\n"
		"\t\ttotal = total + texts(index % 3000).Length + held(index % 4)(index)\n\t\trecent = temporary\n"
		"\tlogger.Log(\"total \" + (total + recent(5)).Round)")
	ran = subprocess.run([str(small)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_HEAP": "128", "STRICT_THREADS": "1", "STRICT_GPU": "0"})
	assert ran.returncode == 0 and "total 1757700" in ran.stdout, (ran.returncode, ran.stdout[-300:])
	ticker = built(folder, "FrameTicker", "has size Number\nhas frame Number = 0\nhas kept Numbers = List(Number)\n"
		"has logger\nhas Realtime\nPresented FrameTicker\n\tFrameTicker(3).Presented.kept is (0, 1, 2)\n"
		"\tFrameTicker(size, frame + 1, Filled(frame))\nFilled(seed Number) Numbers\n\tFrameTicker(2).Filled(1) is (1, 2)\n"
		"\tfor size\n\t\tindex + seed\nRun\n\tmutable ticker = FrameTicker(5000000)\n\tfor 10\n"
		"\t\tticker = ticker.Presented\n\tlogger.Log(\"frames \" + ticker.frame)")
	ran = subprocess.run([str(ticker)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_GC_LOG": "1", "STRICT_GPU": "0"})
	collections = [line for line in ran.stdout.splitlines() if line.startswith("Collection: ")]
	ended = [line.endswith(", frame end") for line in collections]
	assert ran.returncode == 0 and True in ended and all(ended[ended.index(True):]) and ended.count(True) >= 9, \
		ran.stdout[-500:]
print("check_memory.py passed")
