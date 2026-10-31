import os,subprocess,sys,tempfile
from pathlib import Path
# a table a GPU loop computes from numbers alone and only GPU loops read stays on the GPU: no download, no upload again;
# one needing a 64 bit Power (not on the GPU) is computed on the CPU right away instead of failing there first; Bytes
# stay too, also when the method of the GPU loop reading them hoists them for its CPU loop; a host list joined after
# 32 bit floats on the GPU moves up instead of both going to the host, also 32 bit floats the CPU read; bytes the GPU
# computed go into a file as they are; the doubles of a loop reading lists stay there once its run is checked (in a
# block of their own, results in the arena still come back with the flag)
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
Written(path Text) Number
	GpuResidency(Logger).Written("written.bin") is 20000
	File(path).Write(Codes(20000))
	File(path).Length
Averaged(first Numbers, second Numbers) Numbers
	GpuResidency(Logger).Averaged((1, 2), (3, 5)) is (2, 3.5)
	for first.Length
		(first(index) + second(index)) / 2
Folded(table Numbers) Numbers
	GpuResidency(Logger).Folded((1, 2, 3, 4, 5, 6, 7, 8)) is (2, 6)
	for table.Length / 4
		table(index * 4) + 1
Run
	let folded = Folded(Table(16384))
	let summed = Folded(Squares(8192))
	let mixed = Mixed(Codes(16000), Table(16384))
	let read = Table(1000)
	let joined = Folded(Mixed(Codes(49152), Table(49152)) + (7, 0, 0, 0))
	logger.Log("folded " + folded.Length + " " + folded(1))
	logger.Log("summed " + summed.Length + " " + summed(1))
	logger.Log("mixed " + mixed.Length + " " + mixed(1))
	logger.Log("read " + read.Length + " " + read(3))
	logger.Log("joined " + joined.Length + " " + joined(3072))
	let averaged = Averaged(Averaged(Table(4194304), Table(4194304)), Table(4194304))
	logger.Log("averaged " + averaged.Length + " " + averaged(8))
	let packed = Folded(Mixed(Codes(24576), Table(24576)) + mixed)
	logger.Log("packed " + packed.Length + " " + packed(1536) + ", written " + Written("codes.bin"))
'''
(folder/'GpuResidency.strict').write_bytes(program.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','GpuResidency.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
environment=dict(os.environ,STRICT_GPU='all',STRICT_GPU_DOWNLOADS='1')
ran=subprocess.run([str(folder/'GpuResidency.exe')],cwd=folder,capture_output=True,text=True,env=environment)
print(ran.stdout,flush=True)
assert ran.returncode==0,ran.stderr[-2000:]
assert 'folded 4096 2.5' in ran.stdout and 'summed 2048 5' in ran.stdout and 'mixed 4000 2.5' in ran.stdout and 'read 1000 1.375' in ran.stdout and 'joined 3073 8' in ran.stdout and 'averaged 4194304 2' in ran.stdout and 'packed 2536 2, written 20000' in ran.stdout,'wrong result'
assert (folder/'codes.bin').read_bytes()==bytes(index%200 for index in range(20000)),'the bytes the GPU computed went into the file wrong'
assert 'Strict GPU download of 4096 numbers' in ran.stdout,'the downloaded result was not named, STRICT_GPU_DOWNLOADS broken'
assert 'Strict GPU download of 1000 numbers in 64 bits' in ran.stdout,'the table the CPU reads did not stay on the GPU until then'
assert 'Strict GPU download of 16384 numbers' not in ran.stdout,'the table went from the GPU to the host and back'
assert 'Strict GPU download of 8192 numbers' not in ran.stdout,'the squares failed on the GPU first and came back'
assert 'Strict GPU download of 16000 numbers' not in ran.stdout,'the bytes went from the GPU to the host before a GPU loop read them'
assert ran.stdout.count('Strict GPU download of 4194304 numbers') == 1,'a result of a loop reading lists went to the host before the next GPU loop'
assert 'Strict GPU download of 12288 numbers' not in ran.stdout,'the GPU list joined with a host list went to the host'
assert 'Strict GPU download of 6144 numbers' not in ran.stdout,'the GPU list joined with read 32 bit floats went to the host'
small='''has logger
Table(count Number) Numbers
	SmallGpu(Logger).Table(2) is (1, 1.125)
	for count
		index / 8 + 1
Doubled(table Numbers) Numbers
	SmallGpu(Logger).Doubled((1, 2)) is (2, 4)
	for table.Length
		table(index) * 2
Run
	let opened = Table(2000000)
	let before = Process.Executed
	let doubled = Doubled(Table(200000))
	let after = Process.Executed
	logger.Log("gpu kernels " + (after(3) - before(3)) + ", doubled " + doubled(16) + ", opened " + opened(8))
'''
(folder/'SmallGpu.strict').write_bytes(small.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','SmallGpu.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
ran=subprocess.run([str(folder/'SmallGpu.exe')],cwd=folder,capture_output=True,text=True,env={key: value for key, value in os.environ.items() if key!='STRICT_GPU'})
print(ran.stdout,flush=True)
assert 'gpu kernels 2, doubled 6, opened 2' in ran.stdout,'loops over more than 1 MB did not run on the open GPU'
pair='''has real Number
has imaginary Number
+(other) Pair
	Pair(1, 2) + Pair(3, -1) is Pair(4, 1)
	Pair(real + other.real, imaginary + other.imaginary)
Squared Pair
	Pair(1, 2).Squared is Pair(-3, 4)
	Pair(real * real - imaginary * imaginary, 2 * real * imaginary)
SquaredLength Number
	Pair(3, 4).SquaredLength is 25
	real * real + imaginary * imaginary
'''
escape='''has logger
Escaped(count Number) Numbers
	PairEscape(Logger).Escaped(2) is (3, 64)
	for count
		let c = Pair(index / count * 2 - 1.5, 0.5)
		mutable z = Pair(0, 0)
		mutable steps = 0
		for 64
			if z.SquaredLength < 4
				z = z.Squared + c
				steps = steps + 1
		steps
Run
	let before = Process.Executed
	let escaped = Escaped(2000000)
	let after = Process.Executed
	logger.Log("gpu kernels " + (after(3) - before(3)) + ", " + escaped(1000000) + " " + escaped(0))
'''
(folder/'Pair.strict').write_bytes(pair.replace('\n','\r\n').encode('utf-8'))
(folder/'PairEscape.strict').write_bytes(escape.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','PairEscape.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
ran=subprocess.run([str(folder/'PairEscape.exe')],cwd=folder,capture_output=True,text=True,env={key: value for key, value in os.environ.items() if key!='STRICT_GPU'})
print(ran.stdout,flush=True)
assert 'gpu kernels 1, 64 3' in ran.stdout,'a loop with Pair values and a mutable Pair did not run as one GPU kernel'
reading='''has logger
Table(count Number) Numbers
	SimpleRead(Logger).Table(2) is (0, 0.5)
	for count
		index / count
Read(table Numbers, count Number) Numbers
	SimpleRead(Logger).Read((1, 2), 3) is (2, 4, 2)
	for count
		table(index % table.Length) * 2
Run
	let read = Read(Table(1000), 2000000)
	logger.Log("read " + read(0) + " " + read(1999999))
'''
(folder/'SimpleRead.strict').write_bytes(reading.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','SimpleRead.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
ran=subprocess.run([str(folder/'SimpleRead.exe')],cwd=folder,capture_output=True,text=True,env={key: value for key, value in os.environ.items() if key!='STRICT_GPU'})
print(ran.stdout,flush=True)
assert 'read 0 1.998' in ran.stdout,'the result of a GPU loop reading a host list crashed or was wrong when the CPU read it'
orbiting='''has orbit List(Pair)
has shift Pair
Escaped(count Number) Numbers
	Orbiting((Pair(0, 0), Pair(0.5, 0.5)), Pair(-1.5, 0)).Escaped(2) is (32, 3)
	for count
		let reference = orbit(index % orbit.Length)
		Steps(reference, Pair(index / count, reference.imaginary) + shift)
Steps(start Pair, c Pair) Number
	Orbiting((Pair(0, 0)), Pair(0, 0)).Steps(Pair(0, 0), Pair(2, 0)) is 1
	mutable z = start
	for 32
		if z.SquaredLength >= 4
			return index
		z = z.Squared + c
	32
'''
orbits='''has logger
Orbit(count Number) List(Pair)
	OrbitGpu(Logger).Orbit(2) is (Pair(0, 0.5), Pair(0.5, 0.25))
	for count
		Pair(index / count, 0.5 - index / count / 2)
Run
	let before = Process.Executed
	let escaped = Orbiting(Orbit(1000), Pair(-1.5, 0)).Escaped(2000000)
	let after = Process.Executed
	logger.Log("gpu kernels " + (after(3) - before(3)) + ", " + escaped(0) + " " + escaped(1000000) + " " + escaped(1999999))
'''
(folder/'Orbiting.strict').write_bytes(orbiting.replace('\n','\r\n').encode('utf-8'))
(folder/'OrbitGpu.strict').write_bytes(orbits.replace('\n','\r\n').encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','OrbitGpu.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
ran=subprocess.run([str(folder/'OrbitGpu.exe')],cwd=folder,capture_output=True,text=True,env={key: value for key, value in os.environ.items() if key!='STRICT_GPU'})
print(ran.stdout,flush=True)
assert 'gpu kernels 1, 3 6 32' in ran.stdout,'a loop reading a List(Pair) and a Pair member, calling a method with Pair parameters, did not run as one GPU kernel'
viewing='''has zoom Number
has seed Pair = Pair.Zero
has center Pair
Zoomed(factor Number) Viewing
	Viewing(2, Pair(1, 1), Pair(3, 4)).Zoomed(2) is Viewing(4, Pair(1, 1), Pair(3, 4))
	Viewing(2, center = Pair(3, 4)).Zoomed(2) is Viewing(4, Pair.Zero, Pair(3, 4))
	Viewing(zoom * factor, seed, center)
'''
(folder/'Pair.strict').write_bytes(pair.replace('has imaginary Number' + chr(10), 'has imaginary Number' + chr(10) + 'constant Zero = Pair(0, 0)' + chr(10)).replace(chr(10), chr(13) + chr(10)).encode('utf-8'))
(folder/'Viewing.strict').write_bytes(viewing.replace(chr(10), chr(13) + chr(10)).encode('utf-8'))
built=subprocess.run([str(root/'strict.exe'),'build','Viewing.strict'],cwd=folder,capture_output=True,text=True)
assert built.returncode==0,built.stdout[-2000:]+built.stderr[-2000:]
tested=subprocess.run([str(folder/'Viewing.test.exe')],cwd=folder,capture_output=True,text=True)
assert tested.returncode==0,'a type with Pair members (one with a default) compared with is failed natively: '+tested.stdout[-2000:]
print('A list computed on the GPU from numbers alone stays there until the CPU reads it')
