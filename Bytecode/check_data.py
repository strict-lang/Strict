"""Run from Strict: python Bytecode/check_data.py [native]."""
from pathlib import Path
import subprocess
import tempfile
import sys
import json
import uuid

root = Path(__file__).resolve().parent.parent
extension = ".exe" if sys.platform == "win32" else ""
with tempfile.TemporaryDirectory(prefix="DataCheck", dir=root / "Examples") as folder:
	repository = Path(folder)
	subprocess.run(["git", "init", "-q", str(repository)], check=True, timeout=10)
	(repository / ".gitignore").write_text("/Tools/\n/ignored/\n/images/cache/\n*.dat\n!payload.dat\n")
	for name in ["Tools", "ignored", "images/cache"]:
		(repository / name).mkdir(parents=True)
	for name in ["Tools/sdk.h", "ignored/noise.dat", "ignored/explicit.dat", "images/pixel.data", "images/cache/noise.data"]:
		(repository / name).write_text("fixture")
	remote = "\\sc" + uuid.uuid4().hex[:8] + "\\fixture"
	(repository / "Probe.strict").write_text('constant Remote = ' + json.dumps(remote) + '\n'
		'constant Tool = "Tools"\nconstant Git = ".git"\n'
		'constant Noise = "ignored"\nconstant Exact = "ignored/explicit.dat"\nconstant Assets = "images"')
	program = repository / "DataCheck.strict"
	program.write_text('has files Bytecode/DataFiles\nhas binary Bytecode/BinaryFile\nRun Number\n'
		'\tlet started = Process.Elapsed\n'
		'\tlet ignored = DataFiles.Ignored((Directory.Current, TypeSource.Root))\n'
		'\tlet found = DataFiles(ignored).Selected(List(Text) + "Probe.strict")\n'
		'\tLogger.Log(Text.Combine(found, "\\n"))\n'
		'\tLogger.Log("Discovery seconds: " + (Process.Elapsed - started))\n\t0', encoding="utf-8")
	command = [str(root / ("strict" + extension)), str(program)]
	if "native" in sys.argv:
		built = subprocess.run([str(root / ("Compiler/NativeCompiler" + extension)), "build", str(program), str(root)], cwd=root,
			capture_output=True, text=True, timeout=240)
		assert built.returncode == 0, built.stdout + built.stderr
		command = [str(program.with_suffix(extension))]
	result = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=30)
	assert result.returncode == 0, result.stdout + result.stderr
	paths = result.stdout.splitlines()
	assert "ignored/explicit.dat" in paths and "images/pixel.data" in paths, result.stdout
	assert str(repository).replace("\\", "/") + "/.gitignore" in paths, result.stdout
	assert not any(path.startswith(("Tools/", ".git/", "images/cache/")) for path in paths), result.stdout
	assert "ignored/noise.dat" not in paths, result.stdout
	seconds = float(next(line.removeprefix("Discovery seconds: ") for line in paths if line.startswith("Discovery seconds: ")))
	assert seconds < 1, "Quoted strings triggered network discovery: " + str(seconds) + " s"
	print(f"Fixture discovery {seconds * 1000:.1f} ms; ignored trees excluded, explicit files retained, rules tracked")
