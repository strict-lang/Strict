"""Run from Strict with an NVIDIA GPU (sm_80 or newer): rows of Safetensors.Quantized(name, 32) keep their tokens to
about 15 bits (strict_gpu_tokens codes them and what the codes leave, 254 times finer than 8 bit tokens)."""
import os
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy

root = Path(__file__).resolve().parent.parent
program = '''has logger
Run
	let file = Safetensors("kept.safetensors", 32)
	let tokens = file.Values("tokens")
	let plain = Batch(file.Quantized("w"), Layer.Unbiased).Sums(tokens)
	let kept = Batch(file.Quantized("w", 32), Layer.Unbiased).Sums(tokens)
	logger.Log(Safetensors("plain.safetensors").Written("sums", plain) + Safetensors("kept.out.safetensors").Written("sums", kept))
'''
rows, width = 64, 256
generator = numpy.random.default_rng(7)
weights = generator.integers(-127, 128, (rows, width), dtype=numpy.int8)
scales = generator.uniform(0.005, 0.02, (rows, 1)).astype(numpy.float32)


def tokens_of(count):
	values = generator.normal(0, 1, (count, width)).astype(numpy.float32)
	values[numpy.arange(count), generator.integers(0, width, count)] = 100
	return values


def written(path, named):
	header, data, offset = {}, b'', 0
	for name, (dtype, values) in named.items():
		raw = values.tobytes()
		header[name] = {'dtype': dtype, 'shape': list(values.shape), 'data_offsets': [offset, offset + len(raw)]}
		data, offset = data + raw, offset + len(raw)
	text = json.dumps(header, separators=(',', ':')).encode()
	text += b' ' * (-len(text) % 8)
	path.write_bytes(struct.pack('<Q', len(text)) + text + data)


def read(path, tokens):
	data = path.read_bytes()
	length = struct.unpack('<Q', data[:8])[0]
	entry = json.loads(data[8:8 + length])['sums']
	start, end = entry['data_offsets']
	return numpy.frombuffer(data[8 + length + start:8 + length + end], dtype=numpy.float32).reshape(tokens, rows)


with tempfile.TemporaryDirectory(prefix='strict-kept-') as directory:
	folder = Path(directory)
	(folder / 'KeptTokens.strict').write_bytes(program.replace('\n', '\r\n').encode())
	built = subprocess.run([str(root / 'strict.exe'), 'build', str(folder / 'KeptTokens.strict')], cwd=root, capture_output=True, text=True, timeout=180)
	assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
	for tokens in (96, 20):
		inputs = tokens_of(tokens)
		written(folder / 'kept.safetensors', {'w.weight': ('I8', weights), 'w.weight_scale': ('F32', scales), 'tokens': ('F32', inputs.reshape(-1))})
		ran = subprocess.run([str(folder / 'KeptTokens.exe')], cwd=folder, env=dict(os.environ, STRICT_GPU='all'), capture_output=True, text=True, timeout=60)
		assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
		exact = inputs.astype(numpy.float64) @ (weights.astype(numpy.float64) * scales).T
		error = lambda sums: float(numpy.sqrt(numpy.mean((sums - exact) ** 2) / numpy.mean(exact ** 2)))
		plain, kept = error(read(folder / 'plain.safetensors', tokens)), error(read(folder / 'kept.out.safetensors', tokens))
		print(f'{tokens} tokens: relative error of 8 bit tokens {plain:.2e}, of kept tokens {kept:.2e}')
		if tokens >= 64:
			assert plain > 1e-3, 'the 8 bit token path no longer quantizes, this check can not tell them apart'
			assert kept < plain / 100, 'rows of Quantized(name, 32) multiply 8 bit tokens'
		assert kept < 1e-3, 'rows of Quantized(name, 32) lose more than 15 bit tokens'
