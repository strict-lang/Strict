"""Run from Strict: python Compiler/check_throttling.py. A Strict program opts out of execution speed throttling, started
from a background console Windows ran it on efficiency cores (MossImage 13.9-15.2 instead of 9.0 s for one image)."""
import ctypes
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

if sys.platform != "win32":
	sys.exit("check_throttling.py reads the power throttling of a Windows process")
root = Path(__file__).resolve().parent.parent


class Throttling(ctypes.Structure):
	_fields_ = [("version", wintypes.ULONG), ("controlMask", wintypes.ULONG), ("stateMask", wintypes.ULONG)]


with tempfile.TemporaryDirectory(prefix="ThrottlingCheck", dir=root / "Examples") as folder:
	source = Path(folder) / "ThrottlingCheck.strict"
	source.write_text("Run Number\n\tProcess.Input.Length", encoding="utf-8")
	built = subprocess.run([str(root / "strict.exe"), "build", str(source)], cwd=root, capture_output=True,
		text=True, timeout=270)
	assert built.returncode == 0, built.stdout + built.stderr
	process = subprocess.Popen([str(source.with_suffix(".exe"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	kernel = ctypes.WinDLL("kernel32", use_last_error=True)
	kernel.OpenProcess.restype = wintypes.HANDLE
	handle = wintypes.HANDLE(kernel.OpenProcess(0x1000, False, process.pid))
	state = Throttling(1, 0, 0)
	started = time.time()
	while state.controlMask == 0 and time.time() - started < 2:
		assert kernel.GetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state)), ctypes.get_last_error()
		time.sleep(0.01)
	process.communicate(b"x", timeout=60)
	assert state.controlMask & 1 and not state.stateMask & 1, (state.controlMask, state.stateMask)
print("check_throttling.py passed")
