"""Run from Strict: python NeuralNetworks/check_native_threads.py [GgufFile]."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import sys

root = Path(__file__).resolve().parent.parent
clang = shutil.which("clang")
provider = sys.argv[1] if len(sys.argv) > 1 else "TensorFile"
if "--worker" not in sys.argv:
	result = subprocess.run([sys.executable, str(Path(__file__).resolve()), provider, "--worker"], cwd=root, timeout=90)
	sys.exit(result.returncode)
def encoded(text):
	buffer = ctypes.create_string_buffer(text.encode())
	return buffer, struct.unpack("d", struct.pack("Q", ctypes.addressof(buffer)))[0]
def pointer(value):
	bits = struct.unpack("Q", struct.pack("d", value))[0]
	assert bits >> 48 != 0x7ffc, ctypes.string_at(bits & 0xffffffffffff).decode()
	return bits & 0xffffffffffff
with tempfile.TemporaryDirectory(prefix="NativeThreads", dir=root / "Examples") as folder:
	folder = Path(folder)
	dll = root / "Compiler/output" / (provider + "Threads" + (".dll" if os.name == "nt" else ".so"))
	command = [clang, "-shared", "-O2", "-DSTRICT_LIBRARY", "-I" + str(root / "Compiler"),
		str(root / ("NeuralNetworks/native/" + provider + ".c")), "-o", str(dll)]
	if provider == "GgufFile":
		command.extend((root / "NeuralNetworks/native/Windows.flags").read_text(encoding="utf-8").split())
	if os.name != "nt":
		command.extend(["-fPIC", "-lpthread", "-lm"])
	subprocess.run(command, check=True, capture_output=True, timeout=60)
	library = ctypes.CDLL(str(dll))
	rows = getattr(library, provider + "_Rows")
	rows.argtypes = [ctypes.c_double] * 4
	rows.restype = ctypes.c_double
	paths = []
	header = {"matrix": {"dtype": "F32", "shape": [128, 64], "data_offsets": [0, 128 * 64 * 4]}}
	entry, entry_value = encoded(json.dumps(header["matrix"], separators=(",", ":"))[1:-1])
	if provider == "GgufFile":
		entry, entry_value = encoded("matrix")
	for index, value in enumerate([1.25, 9.5]):
		path = folder / (str(index) + ".safetensors")
		text = json.dumps(header, separators=(",", ":")).encode()
		path.write_bytes(struct.pack("Q", len(text)) + text + struct.pack("f", value) * (128 * 64))
		if provider == "GgufFile":
			text = struct.pack("<IIQQ", 0x46554747, 3, 1, 0)
			text += struct.pack("<Q", 6) + b"matrix" + struct.pack("<IQQIQ", 2, 64, 128, 0, 0)
			path.write_bytes(text + bytes((-len(text)) % 32) + struct.pack("f", value) * (128 * 64))
		paths.append(encoded(str(path)))
	first_ready = threading.Event()
	second_done = threading.Event()
	def checked(result, expected):
		assert ctypes.c_double.from_address(result).value == 128
		for index in [0, 64, 127]:
			row = pointer(ctypes.c_double.from_address(result + (index + 1) * 8).value)
			assert ctypes.c_double.from_address(row).value == 64
			assert list((ctypes.c_float * 64).from_address(row + 16)) == [expected] * 64
	def first():
		result = pointer(rows(0, paths[0][1], entry_value, 32))
		row = pointer(ctypes.c_double.from_address(result + 8).value)
		checked(result, 1.25)
		first_ready.set()
		assert second_done.wait(10)
		assert ctypes.c_float.from_address(row + 16).value == 1.25, "Another thread overwrote a returned matrix"
	def second():
		assert first_ready.wait(10)
		try:
			pointer(rows(0, paths[1][1], entry_value, 32))
		finally:
			second_done.set()
	with ThreadPoolExecutor(max_workers=2) as workers:
		futures = [workers.submit(first), workers.submit(second)]
		for future in futures:
			future.result(timeout=15)
	def repeated(index):
		for repeat in range(20):
			result = pointer(rows(0, paths[index % 2][1], entry_value, 32))
			checked(result, [1.25, 9.5][index % 2])
	with ThreadPoolExecutor(max_workers=8) as workers:
		list(workers.map(repeated, range(8)))
	if provider == "GgufFile":
		values = library.GgufFile_Values
		values.argtypes, values.restype = [ctypes.c_double] * 3, ctypes.c_double
		experts = library.GgufFile_ExpertOutputs
		experts.argtypes, experts.restype = [ctypes.c_double] * 5, ctypes.c_double
		model, model_value = encoded(str(root / "NeuralNetworks/experts.gguf"))
		name, name_value = encoded("blk.0.ffn_gate_inp.weight")
		prefixes = [encoded("blk.0."), encoded("blk.1.")]
		chosen = (ctypes.c_double * 3)(2, 1, 0)
		chosen_value = struct.unpack("d", struct.pack("Q", ctypes.addressof(chosen)))[0]
		def evaluated(index):
			input = pointer(values(0, model_value, name_value))
			routed = (ctypes.c_double * 257)(256, *(ctypes.c_double * 256).from_address(input + 8))
			input_value = struct.unpack("d", struct.pack("Q", ctypes.addressof(routed)))[0]
			for repeat in range(5):
				result = pointer(experts(0, model_value, prefixes[index % 2][1], chosen_value, input_value))
				assert ctypes.c_double.from_address(result).value == 512
				actual = ctypes.c_double.from_address(result + 8 + 261 * 8).value
				assert abs(actual - [12.111, 100.175][index % 2]) < 0.05, actual
		with ThreadPoolExecutor(max_workers=8) as workers:
			list(workers.map(evaluated, range(8)))
	print("Returned matrices remain thread-owned; overlapping matrix loads pass")
