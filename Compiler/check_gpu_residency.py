import os,subprocess,sys,tempfile
from pathlib import Path
# a table a GPU loop computes from numbers alone and only GPU loops read stays on the GPU: no download, no upload again;
# one needing a 64 bit Power (not on the GPU) is computed on the CPU right away instead of failing there first
root=Path(__file__).resolve().parent.parent
folder=Path(tempfile.mkdtemp(prefix='strict-residency-'))
program='''has logger
Table(count Number) Numbers
	GpuResidency(Logger).Table(2) is (1, 1.125)
	for count
		index / 8 + 1
Squares(count Number) Numbers
	GpuResidency(Logger).Squares(2) is (0, 0.25)
	for count
		(index / 2) ^ 2
Codes(count Number) Bytes
	GpuResidency(Logger).Codes(2) is (0, 1)
	for count
		index % 200
Mixed(codes Bytes, table Numbers) Numbers
	GpuResidency(Logger).Mixed((1, 2, 3, 4), (3, 4, 5, 6)) is (4)
	for codes.Length / 4
		codes(index) + table(index * 4)
Folded(table Numbers) Numbers
	GpuResidency(Logger).Folded((1, 2, 3, 4, 5, 6, 7, 8)) is (2, 6)
	for table.Length / 4
		table(index * 4) + 1
Run
	let folded = Folded(Table(16384))
	let summed = Folded(Squares(8192))
	let mixed = Mixed(Codes(16000), Table(16384))
	let read = Table(1000)
	logger.Log("folded " + folded.Length + " " + folded(1))
	logger.Log("summed " + summed.Length + " " + summed(1))
	logger.Log("mixed " + mixed.Length + " " + mixed(1))
	logger.Log("read " + read.Length + " " + read(3))
'''
(folder/'GpuResidency.strict').write_bytes(program.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','GpuResidency.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
environment=dict(os.environ,STRICT_GPU='all',STRICT_GPU_DOWNLOADS='1')
ran=subprocess.run([str(folder/'GpuResidency.exe')],cwd=folder,capture_output=True,text=True,env=environment)
print(ran.stdout,flush=True)
assert ran.returncode==0,ran.stderr[-2000:]
assert 'folded 4096 2.5' in ran.stdout and 'summed 2048 5' in ran.stdout and 'mixed 4000 2.5' in ran.stdout and 'read 1000 1.375' in ran.stdout,'wrong result'
assert 'Strict GPU download of 4096 numbers' in ran.stdout,'the downloaded result was not named, STRICT_GPU_DOWNLOADS broken'
assert 'Strict GPU download of 1000 numbers in 64 bits' in ran.stdout,'the table the CPU reads did not stay on the GPU until then'
assert 'Strict GPU download of 16384 numbers' not in ran.stdout,'the table went from the GPU to the host and back'
assert 'Strict GPU download of 8192 numbers' not in ran.stdout,'the squares failed on the GPU first and came back'
print('A list computed on the GPU from numbers alone stays there until the CPU reads it')
