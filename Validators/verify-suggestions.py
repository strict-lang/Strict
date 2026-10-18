from pathlib import Path
import subprocess, sys, tempfile, time

root = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='strict-native-suggest-') as folder:
	Path(folder, 'Bogus.strict').write_text('Doubled(number Number) Number\n'
		'\tDoubled(3) is 6\n\tDoubled(0) is 0\n\tnumber + number\n')
	cases = [('Suggesting', 'Doubled', 'has number\nDoubled Number\n'
		'\tSuggesting(3).Doubled is 6\n\tSuggesting(0).Doubled is 0\n', 'number + number'),
		('Temperature', 'Clamped', 'has celsius Number\nClamped(minimum Number, maximum Number) Number\n'
		'\tTemperature(30).Clamped(0, 20) is 20\n\tTemperature(10).Clamped(0, 20) is 10\n',
		'celsius.Clamp(minimum, maximum)')]
	for name, method, source, expected in cases:
		file = Path(folder, name + '.strict')
		file.write_text(source)
		started = time.perf_counter()
		run = subprocess.run([str(root / 'strict.exe'), 'suggest', file.name, root.as_posix(), method, '-'],
			input=source, cwd=folder, text=True, capture_output=True, timeout=15)
		assert run.returncode == 0, run.stdout + run.stderr
		bodies = [line[12:] for line in run.stdout.splitlines() if line.startswith('Suggestion: ')]
		assert expected in bodies, run.stdout
		assert not any(body in ('number', 'celsius', 'minimum', 'maximum') for body in bodies), bodies
		assert not any('Bogus.' in body or '.Epsilon' in body or '.Pi' in body for body in bodies), bodies
		assert all(line.startswith('Suggestion: ') for line in run.stdout.splitlines()), run.stdout
		assert file.read_text() == source
		assert not list(Path(folder).glob('*.strictbinary'))
		print(f'{name}.{method}: {bodies} ({(time.perf_counter()-started)*1000:.0f} ms)')
	source = cases[1][2] + 'Doubled Number\n\tTemperature(2).Doubled is 5\n\tcelsius * 2\n' + \
		'Other Number\n\tTemperature(2).Other is 2\n'
	file = Path(folder, 'Temperature.strict')
	before = file.read_text()
	run = subprocess.run([str(root / 'strict.exe'), 'suggest', file.name, root.as_posix(), 'Clamped', '-'],
		input=source, cwd=folder, text=True, capture_output=True, timeout=15)
	assert run.returncode == 0 and 'Suggestion: celsius.Clamp(minimum, maximum)' in run.stdout, run.stdout + run.stderr
	assert all(line.startswith('Suggestion: ') for line in run.stdout.splitlines()), run.stdout
	assert file.read_text() == before
	print('Unrelated failing/pending methods and unsaved source: passed')
	Path(folder, 'Wrapper.strict').write_text('has dispatcher Dispatcher\nClamped Number\n'
		'\tWrapper(Dispatcher(1)).Clamped is 1\n')
	Path(folder, 'Dispatcher.strict').write_text('has number\nClamped Number\n'
		'\tWrapper(Dispatcher(number)).Clamped\n')
	run = subprocess.run([str(root / 'strict.exe'), 'suggest', 'Wrapper.strict', root.as_posix(), 'Clamped'],
		cwd=folder, text=True, capture_output=True, timeout=15)
	assert run.returncode == 0 and not run.stdout.strip(), run.stdout + run.stderr
	assert not list(Path(folder).glob('*.strictbinary'))
	print('Recursive dispatcher: no passing suggestions')
	Path(folder, 'Dual.strict').write_text('has number\nPick(other Number) Number\n'
		'\tDual(2).Pick(3) is 5\n\tnumber + other\nPick Number\n'
		'\tDual(3).Pick is 6\n\tDual(0).Pick is 0\n')
	run = subprocess.run([str(root / 'strict.exe'), 'suggest', 'Dual.strict', root.as_posix(), '5'],
		cwd=folder, text=True, capture_output=True, timeout=15)
	assert run.returncode == 0 and 'Suggestion: number + number' in run.stdout, run.stdout + run.stderr
	print('Overloaded method selected by header line: passed')
	run = subprocess.run([str(root / 'strict.exe'), 'suggest', 'Suggesting.strict', root.as_posix(), 'Doubled', '-'],
		input='has number\nDoubled Number\n', cwd=folder, text=True, capture_output=True, timeout=15)
	assert run.returncode == 0 and not run.stdout.strip(), run.stdout + run.stderr
	run = subprocess.run([str(root / 'strict.exe'), 'suggest', 'Suggesting.strict', root.as_posix(), 'Missing'],
		cwd=folder, text=True, capture_output=True, timeout=15)
	assert run.returncode == 2 and 'MethodNotFound' in run.stdout, run.stdout + run.stderr
	print('No tests and unknown method: passed')
