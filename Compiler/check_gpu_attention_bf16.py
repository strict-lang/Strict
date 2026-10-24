"""Build and run bf16 prompt attention; require native bf16 tensor operands."""
import os
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
program = '''has logger
Brains(input Numbers, factor Number) BFloat16s
	AttentionBrains(Logger).Brains((0, 1), 131072) is (0, 131072)
	for input.Length
		input(index) * factor
Run
	let table = Safetensors("attention.safetensors", 32).Values("input")
	let numbers = Brains(table, 1)
	let queries = Brains(table, 131072)
	let attended = Attention(queries, queries, numbers, 1, 1, 128, 128, false).Merged
	logger.Log("attention " + attended.Length + " " + attended.First + " " + attended.Last)
'''
with tempfile.TemporaryDirectory(prefix='strict-attention-bf16-') as directory:
	folder = Path(directory)
	count = 129 * 128 * 128
	header = json.dumps({'input': {'dtype': 'F32', 'shape': [count], 'data_offsets': [0, count * 4]}}, separators=(',', ':')).encode()
	header += b' ' * (-len(header) % 8)
	(folder / 'attention.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<ff', 0, 1) * (count // 2))
	(folder / 'AttentionBrains.strict').write_bytes(program.replace('\n', '\r\n').encode())
	built = subprocess.run([str(root / 'strict.exe'), 'build', str(folder / 'AttentionBrains.strict')], cwd=root, capture_output=True, text=True, timeout=120)
	assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
	ptx = (folder / 'AttentionBrains.runtime.ptx').read_text()
	assert re.search(r'mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32', ptx), 'bf16 prompt attention still converts operands to fp16'
	ran = subprocess.run([str(folder / 'AttentionBrains.exe')], cwd=folder, env=dict(os.environ, STRICT_GPU='all'), capture_output=True, text=True, timeout=60)
	assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
	assert 'attention 2113536 0 1' in ran.stdout, ran.stdout
	print(ran.stdout, end='')
