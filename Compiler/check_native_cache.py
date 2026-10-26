"""Run from Strict: python Compiler/check_native_cache.py."""
from pathlib import Path
import subprocess
import tempfile
import sys

root = Path(__file__).resolve().parent.parent
extension = ".exe" if sys.platform == "win32" else ""
with tempfile.TemporaryDirectory(prefix="ObjectCheck", dir=root / "Examples") as folder:
	folder = Path(folder)
	native = folder / "native"
	native.mkdir()
	(native / "Flags.c").write_text('#include "strict.h"\nSTRICT_EXPORT double Flags_Mode(double self) {\n'
		'#ifdef __OPTIMIZE_SIZE__\nreturn 1;\n#else\nreturn 2;\n#endif\n}', encoding="utf-8")
	(folder / "Flags.strict").write_text("Mode Number", encoding="utf-8")
	program = folder / "ObjectCheck.strict"
	program.write_text('has logger\nhas flags Flags\nRun Number\n\tlogger.Log(Flags.Mode)\n\t0', encoding="utf-8")
	for options, expected in [([], "2"), (["small"], "1"), ([], "2")]:
		result = subprocess.run([str(root / ("Compiler/NativeCompiler" + extension)), "build", str(program),
			str(root), *options], cwd=root, capture_output=True, text=True, timeout=60)
		assert result.returncode == 0, result.stdout + result.stderr
		result = subprocess.run([str(program.with_suffix(extension))], capture_output=True, text=True, timeout=10)
		assert result.returncode == 0 and result.stdout.strip() == expected, (options, result.stdout, result.stderr)
	print("Native C flags follow default/small/default build changes")
