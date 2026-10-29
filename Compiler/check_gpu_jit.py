import os,subprocess,tempfile
from pathlib import Path
# loading a module the driver has to compile first (a new build or runtime, or its JIT cache disabled) says so with the
# seconds it took (the attention kernels of the runtime take seconds), a load from the driver's cache says nothing
root=Path(__file__).resolve().parent.parent
folder=Path(tempfile.mkdtemp(prefix='strict-jit-'))
program='''has logger
Table(count Number) Numbers
	GpuJit(Logger).Table(2) is (0, 0.001)
	for count
		index * 0.001
Brains(input Numbers) BFloat16s
	GpuJit(Logger).Brains((0, 1)) is (0, 1)
	for input.Length
		input(index)
Run
	let numbers = Brains(Table(129 * 128 * 128))
	logger.Log("attention " + Attention(numbers, numbers, numbers, 1, 1, 128, 128, false).Merged.Length)
'''
(folder/'GpuJit.strict').write_bytes(program.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','GpuJit.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
def ran(**settings):
	result=subprocess.run([str(folder/'GpuJit.exe')],cwd=folder,capture_output=True,text=True,env=dict(os.environ,STRICT_GPU='all',**settings))
	assert result.returncode==0 and 'attention 2113536' in result.stdout,result.stdout[-2000:]+result.stderr[-2000:]
	return result.stdout
compiled=ran(CUDA_CACHE_DISABLE='1')
print(compiled,flush=True)
assert 'The driver compiled the GPU' in compiled,'a load the driver compiled for seconds was not mentioned'
ran()
cached=ran()
assert 'The driver compiled' not in cached,'a load from the driver cache was called compiled: '+cached
print('A module the driver compiles is mentioned with its seconds, a cached one is not')
