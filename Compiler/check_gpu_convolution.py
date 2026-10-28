import os,re,subprocess,tempfile
from pathlib import Path
# Convolved.Summed of 3 and of 1024 channels takes the tensor core convolution like a multiple of 32 up to 512 does:
# right against Tapped and fast (the generic GPU loop it fell back to took 1.0 s and 29 s for these, the calls with
# their list copies now about 0.1 s)
root=Path(__file__).resolve().parent.parent
folder=Path(tempfile.mkdtemp(prefix='strict-convolution-'))
program='''has logger
Filled(count Number) Numbers
	GpuConvolution(Logger).Filled(2) is (-0.5, -0.375)
	for count
		(index % 8) / 8 - 0.5
Checked(name Text, convolved Convolved) Number
	GpuConvolution(Logger).Checked("tiny", Convolved((1, 2, 3, 4), (1, 0), (0, 0, 0, 0, 1, 0, 0, 0, 0), (2, 2, 1, 1, 0, 1))) is 0
	let warm = convolved.Summed(0)
	let before = Process.Elapsed
	let summed = convolved.Summed
	let last = summed(summed.Length - 1)
	let milliseconds = ((Process.Elapsed - before) * 1000).Round(1)
	let pixel = convolved.shape(0) * convolved.shape(1) - 1
	let expected = convolved.Tapped(pixel, convolved.shape(3) - 1)
	let difference = (last - expected).Absolute.Round(4)
	logger.Log(name + " " + milliseconds + " ms, difference " + difference + " of " + expected.Absolute.Round(4))
	0
Run
	let few = Convolved(Filled(3 * 256 * 256), Filled(6), Filled(27 * 128), (256, 256, 3, 128, 1, 1))
	let wide = Convolved(Filled(1024 * 32 * 32), Filled(2048), Filled(9216 * 1024), (32, 32, 1024, 1024, 1, 1))
	let middle = Convolved(Filled(512 * 32 * 32), Filled(1024), Filled(4608 * 512), (32, 32, 512, 512, 1, 1))
	Checked("few", few) + Checked("wide", wide) + Checked("middle", middle)
'''
(folder/'GpuConvolution.strict').write_bytes(program.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','GpuConvolution.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
ran=subprocess.run([str(folder/'GpuConvolution.exe')],cwd=folder,capture_output=True,text=True,env=dict(os.environ,STRICT_GPU='all'))
print(ran.stdout,flush=True)
assert ran.returncode==0,ran.stderr[-2000:]
results={name:(float(time),float(difference),float(size)) for name,time,difference,size in re.findall(r'(\w+) ([\d.]+) ms, difference ([\d.]+) of ([\d.]+)',ran.stdout)}
assert set(results)=={'few','wide','middle'},'a convolution did not report'
for name,(time,difference,size) in results.items():
	assert difference<=0.002*max(size,1),name+' differs from Tapped by '+str(difference)+' of '+str(size)
	assert time<400,name+' took '+str(time)+' ms, not the tensor core convolution'
print('Convolutions of 3 and 1024 channels run on the tensor cores like 512 channels')
