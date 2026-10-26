"""Run from Strict after building SocketChecks: python Networking/check_sockets.py."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import subprocess
import sys
import ctypes
import threading

root = Path(__file__).resolve().parent.parent
extension = ".exe" if sys.platform == "win32" else ""
library = ctypes.CDLL(str(root / "Networking/native" / ("Networking.dll" if extension else "Networking.so")))
listen, port, close = library.Socket_Listen, library.Socket_Port, library.Socket_Close
for method in [listen, port, close]:
	method.argtypes, method.restype = [ctypes.c_double] * 2, ctypes.c_double
barrier = threading.Barrier(16, timeout=10)
def started(index):
	barrier.wait()
	handle = listen(0, 0)
	assert handle > 0
	try:
		assigned = port(0, handle)
		assert 0 < assigned < 65536
		barrier.wait()
		return assigned
	finally:
		close(0, handle)
with ThreadPoolExecutor(max_workers=16) as workers:
	assert len(set(workers.map(started, range(16)))) == 16
with socket.socket() as occupied:
	if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
		occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
	occupied.bind(("0.0.0.0", 34567))
	occupied.listen()
	def checked(index):
		result = subprocess.run([str(root / "Networking" / ("SocketChecks.test" + extension))],
			cwd=root, capture_output=True, text=True, timeout=15)
		assert result.returncode == 0, result.stdout + result.stderr
	with ThreadPoolExecutor(max_workers=8) as workers:
		list(workers.map(checked, range(16)))
print("Concurrent loopbacks pass with the old fixed port occupied")
