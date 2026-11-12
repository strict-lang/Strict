"""Run from Strict: python Compiler/check_memory.py. A failed host allocation ends with OutOfMemory (Windows only), a
program dropping 52 MB lists while keeping 260 MB runs in the 1 GB heap of the web and iOS (natively and in node), one
dropping 3 MB lists and many small texts in 128 MB, lists grown by Add free their items by counts alone (no collection),
so does a fresh list passed straight to a method (a grown one and a long one),
giving 8 GB of free pages back after a full collection is not counted as collecting (GarbageCollectionExceedsTenth),
GpuMemory.Released never collects and Process.Collected always does, a list a method named Validated hands back
is not refilled while the local it came from still reads it, a Realtime type collects at the end of its
frames (Presented), reading 1000 lines of 80 KB measures the 80 MB text once (its remembered length survives reclaims
that do not free it, measuring it again per line took 13 s)."""
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
	return source.parent / "Release" / "Web" / (name + ".js") if platform else source.with_suffix(".exe")


def frameEnds(folder, name, listType, item, filledTwo, size, frames):
	"""A Realtime type makes a list of size items per frame, gives for every collection whether it ran at a frame end."""
	executable = built(folder, name, f"has size Number\nhas frame Number = 0\nhas kept {listType} = List({listType[:-1]})\n"
		f"has logger\nhas Realtime\nPresented {name}\n\t{name}(3).Presented.kept.Length is 3\n"
		f"\t{name}(size, frame + 1, Filled(frame))\nFilled(seed Number) {listType}\n\t{name}(2).Filled(1) is {filledTwo}\n"
		f"\tfor size\n\t\t{item}\nRun\n\tmutable ticker = {name}({size})\n\tfor {frames}\n"
		f"\t\tticker = ticker.Presented\n\tlogger.Log(\"frames \" + ticker.frame)")
	ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_GC_LOG": "1", "STRICT_GPU": "0"})
	assert ran.returncode == 0 and f"frames {frames}" in ran.stdout, ran.stdout[-500:]
	return [line.endswith(", frame end") for line in ran.stdout.splitlines() if line.startswith("Collection: ")]


filled = "has logger\nFilled(count Number, offset Number) Numbers\n\tFilled(2, 1) is (1, 2)\n\tfor count\n\t\tindex + offset\n"
grower = ("Grown(count Number, offset Number) List(Numbers)\n\tName(Logger).Grown(13, 0)(12) is (12, 13)\n"
	"\tmutable grown = List(Mutable(Numbers))\n\tfor count\n"
	"\t\tgrown.Add(Filled(count is 13 then 2 else 8192, index + offset))\n\tgrown\n")


def uncollected(executable, total):
	"""The program logs its total and frees what it drops by counts alone: no collection runs."""
	ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_GC_LOG": "1", "STRICT_GPU": "0"})
	assert ran.returncode == 0 and f"total {total}" in ran.stdout and "Collection: " not in ran.stdout, ran.stdout[-500:]


kept = filled + ("Run\n\tlet kept =(Filled(6500000, 1), Filled(6500000, 2), Filled(6500000, 3), Filled(6500000, 4), "
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
	grown = built(folder, "GrownLists", filled + grower.replace("Name", "GrownLists") + "Run\n\tmutable total = 0\n"
		"\tfor 4000\n\t\ttotal = total + Grown(16, index)(15)(1)\n\tlogger.Log(\"total \" + total)")
	uncollected(grown, 8062000)
	uncollected(built(folder, "GrownPassed", filled + grower.replace("Name", "GrownPassed") +
		"Picked(lists List(Numbers)) Number\n\tGrownPassed(Logger).Picked(((1, 2), (3, 4))) is 4\n"
		"\tlists(lists.Length - 1)(1)\nRun\n\tmutable total = 0\n\tfor 4000\n"
		"\t\ttotal = total + Picked(Grown(16, index))\n\tlogger.Log(\"total \" + total)"), 8062000)
	uncollected(built(folder, "PassedLarge", filled + "Last(numbers Numbers) Number\n"
		"\tPassedLarge(Logger).Last((1, 2)) is 2\n\tnumbers(numbers.Length - 1)\nRun\n\tmutable total = 0\n"
		"\tfor 60000\n\t\ttotal = total + Last(Filled(8192, index))\n\tlogger.Log(\"total \" + total)"), 2291430000)
	trimmed = built(folder, "TrimmedPool", filled + "Held(count Number) Slow(List(Numbers))\n"
		"\tTrimmedPool(Logger).Held(1).Length is 1\n\tfor count\n\t\tFilled(16000000, index)\nLast(count Number) Slow(Number)\n"
		"\tTrimmedPool(Logger).Last(1) is 1\n\tHeld(count).Last(1)\nRun\n\tmutable total = 0\n\tfor 3\n"
		"\t\ttotal = total + Last(64)\n\t\tProcess.Collected\n\tlogger.Log(\"collected\")\n\tGpuMemory.Released\n"
		"\tlogger.Log(\"total \" + total)")
	ran = subprocess.run([str(trimmed)], capture_output=True, text=True, timeout=60,
		env=os.environ | {"STRICT_GC_LOG": "1", "STRICT_GPU": "0"})
	assert ran.returncode == 0 and "total 192" in ran.stdout, ran.stdout[-500:]
	collected, released = ran.stdout.split("collected")
	assert collected.count("Collection: full") >= 3 and "Collection: " not in released, ran.stdout[-500:]
	handed = built(folder, "HandedBack", "has logger\nconstant Plain = HandedBack(Logger)\n" + filled[11:] +
		"Validated(numbers Numbers) Numbers\n\tPlain.Validated((1, 2)) is (1, 2)\n\tnumbers\nRun\n\tmutable total = 0\n"
		"\tfor 1000\n\t\tlet made = Filled(300, index)\n\t\tmutable checked = Plain.Validated(made)\n"
		"\t\tchecked = Filled(300, index + 1)\n\t\ttotal = total + made(5) + checked(5)\n\tlogger.Log(\"total \" + total)")
	ran = subprocess.run([str(handed)], capture_output=True, text=True, timeout=60, env=os.environ | {"STRICT_GPU": "0"})
	assert ran.returncode == 0 and "total 1010000" in ran.stdout, ran.stdout[-500:]
	ended = frameEnds(folder, "FrameTicker", "Numbers", "index + seed", "(1, 2)", 5000000, 10)
	assert True in ended and all(ended[ended.index(True):]) and ended.count(True) >= 2, ended
	ended = frameEnds(folder, "WordTicker", "Texts", "\"w\" + (index + seed)", "(\"w1\", \"w2\")", 100000, 40)
	assert all(ended) and len(ended) <= 20, ended
	(Path(folder) / "long.txt").write_text(("x" * 80000 + "\n") * 1000, encoding="utf-8", newline="\n")
	long = built(folder, "LongLines", "has logger\nRun\n\tlet started = Process.Elapsed\n"
		f"\tlet lines = File(\"{Path(folder).as_posix()}/long.txt\").ReadLines\n"
		"\tlogger.Log(\"lines \" + lines.Length + \" slow \" + (Process.Elapsed - started > 3))")
	ran = subprocess.run([str(long)], capture_output=True, text=True, timeout=60, env=os.environ | {"STRICT_GPU": "0"})
	assert ran.returncode == 0 and "lines 1001 slow false" in ran.stdout, ran.stdout[-300:]
print("check_memory.py passed")
