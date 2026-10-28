"""Run from Strict: python Runtime/check_tests.py."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
command = str(root / ("strict.exe" if os.name == "nt" else "strict"))
environment = dict(os.environ, STRICT_THREADS="8")
with tempfile.TemporaryDirectory(prefix="TestPolicy", dir=root / "Examples") as temporary:
	folder = Path(temporary)
	work = folder / "TestWork.strict"
	work.write_text('Pause Number\n\tlet started = Process.Elapsed\n'
		'\tmutable count = 0\n\tfor as long as Process.Elapsed - started < 0.15\n'
		'\t\tcount = count + 1\n\tcount > 0 then 1 else 0', encoding="utf-8")
	source = folder / "OverBudget.strict"
	source.write_text('Delay Number\n\tOverBudget.Delay is 1\n\tTestWork.Pause', encoding="utf-8")
	result = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert result.returncode != 0 and "TestDurationMustNotExceedHundredMilliseconds" in result.stdout + result.stderr, (
		"An over-budget VM unit test passed:\n" + result.stdout + result.stderr)
	assert not source.with_suffix(".strictbinary").exists(), "Failed unit tests were cached as passed"
	for arguments in [["build", str(source)], ["test", str(source), "unit", "native"]]:
		checked = subprocess.run([command, *arguments], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90)
		assert checked.returncode != 0 and "TestDurationMustNotExceedHundredMilliseconds" in checked.stdout + checked.stderr, (
			str(arguments) + " accepted an over-budget unit test:\n" + checked.stdout + checked.stderr)

	source = folder / "SharedBudget.strict"
	source.write_text('constant Shared = TestWork.Pause\nCheck Number\n\tSharedBudget.Shared is 1\n\t1', encoding="utf-8")
	for options in [["unit"], ["unit", "native"]]:
		shared = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90 if "native" in options else 10)
		assert shared.returncode == 0, "A shared constant's first initialization counted for its test:\n" + shared.stdout + shared.stderr

	source = folder / "LazyFixture.strict"
	source.write_text('has logger\nconstant Shared = Created\nCreated Number\n\tlogger.Log("fixture computed")\n\t7\nCheck Number\n\tLazyFixture.Check is 7\n\tShared is 7\n\tShared + Shared is 14\n\tfalse then Shared else 7', encoding="utf-8")
	lazy = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert lazy.returncode == 0 and lazy.stdout.count("fixture computed") == 1, lazy.stdout + lazy.stderr
	assert lazy.stdout.index("Test passed: LazyFixture.Check is 7") < lazy.stdout.index("fixture computed"), lazy.stdout

	source = folder / "SetupBudget.strict"
	source.write_text('Check Number\n\tconstant sample = TestWork.Pause\n\tsample is 1\n\t1', encoding="utf-8")
	setup = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert setup.returncode != 0 and "TestDurationMustNotExceedHundredMilliseconds" in setup.stdout + setup.stderr, setup.stdout + setup.stderr

	source = folder / "Hangs.strict"
	source.write_text('Spin Number\n\tHangs.Spin is 1\n\tmutable count = 0\n'
		'\tfor as long as Process.Elapsed >= 0\n\t\tcount = count + 1\n\tcount', encoding="utf-8")
	for options in [["unit"], ["unit", "native"]]:
		hung = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90 if "native" in options else 10)
		assert hung.returncode != 0 and "TestDurationMustNotExceedHundredMilliseconds" in hung.stdout + hung.stderr, hung.stdout + hung.stderr
	validated = subprocess.run([command, "check", str(folder)], cwd=root, env=environment,
		capture_output=True, text=True, timeout=30)
	assert validated.returncode != 0 and "TestDurationMustNotExceedHundredMilliseconds" in validated.stdout, validated.stdout + validated.stderr

	source = folder / "Policy.strict"
	source.write_text('has logger\nconstant Fixture = Policy.Value + 1\nRun Number\n\tlogger.Log("application")\n\t0\n'
		'Value Slow(Number)\n\tPolicy.Value is 7\n\tlet started = Process.Elapsed\n'
		'\tmutable count = 0\n\tfor as long as Process.Elapsed - started < 0.04\n\t\tcount = count + 1\n'
		'\tlogger.Log("manual fixture")\n\tcount > 0 then 7 else 0\n'
		'Quick Number\n\tPolicy.Quick is 3\n\tSlow(3) + 0 is 3\n\tSlow((1, 2)).First is 1\n\t3\n'
		'FixtureUsed Slow(Number)\n\tPolicy.Fixture is 8\n\tFixture', encoding="utf-8")
	unit = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert unit.returncode == 0 and "manual fixture" not in unit.stdout, unit.stdout + unit.stderr
	manual = subprocess.run([command, "test", str(source)], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert manual.returncode == 0 and "manual fixture" in manual.stdout, "Explicit file run skipped Slow tests:\n" + manual.stdout + manual.stderr

	selected = subprocess.run([command, "test", str(source), "Quick"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert selected.returncode == 0 and "manual fixture" not in selected.stdout, selected.stdout + selected.stderr
	run = subprocess.run([command, str(source)], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert run.returncode == 0 and run.stdout.index("manual fixture") < run.stdout.index("application"), run.stdout + run.stderr

	for options, include_slow in [(["unit", "native"], False), (["native", "Value"], True)]:
		native = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90)
		assert native.returncode == 0 and ("manual fixture" in native.stdout) == include_slow, native.stdout + native.stderr

	waiting = folder / "Waiting.strict"
	waiting.write_text('Delay Number\n\tWaiting.Delay is 1\n\tlet started = Process.Elapsed\n'
		'\tmutable count = 0\n\tfor as long as Process.Elapsed - started < 0.04\n'
		'\t\tcount = count + 1\n\tcount > 0 then 1 else 0', encoding="utf-8")
	for options in [["unit"], ["unit", "native"]]:
		allowed = subprocess.run([command, "test", str(waiting), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90)
		assert allowed.returncode == 0, "A 40 ms unit test failed the 100 ms limit:\n" + allowed.stdout + allowed.stderr
	source = root / "Bytecode" / "PolicyProbe.strict"
	try:
		source.write_text(waiting.read_text(encoding="utf-8").replace("Waiting", "PolicyProbe").replace("0.04", "0.15"),
			encoding="utf-8")
		for options in [["unit"], ["unit", "native"]]:
			skipped = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
				capture_output=True, text=True, timeout=90)
			assert skipped.returncode == 0 and "Test passed" not in skipped.stdout, (
				"An automatic run tested an internal package:\n" + skipped.stdout + skipped.stderr)
		explicit = subprocess.run([command, "test", str(source)], cwd=root, env=environment,
			capture_output=True, text=True, timeout=10)
		assert explicit.returncode == 0 and "Test passed: PolicyProbe.Delay" in explicit.stdout, (
			"An explicit run of an internal package was timed or skipped:\n" + explicit.stdout + explicit.stderr)
	finally:
		source.unlink(missing_ok=True)
		source.with_suffix(".strictbinary").unlink(missing_ok=True)

	source = folder / "Shares.strict"
	tool, line = ("cmd", "/c echo %STRICT_THREADS%") if os.name == "nt" else ("sh", "-c 'echo $STRICT_THREADS'")
	source.write_text('has logger\nRun Number\n\tfor Process.RunTools("' + tool + '", ("' + line + '", "' + line + '"))\n'
		'\t\tlogger.Log("threads " + value.Output.Trim)\n\t0', encoding="utf-8")
	unset = dict(environment)
	unset.pop("STRICT_THREADS")
	shared = subprocess.run([command, str(source)], cwd=root, env=unset, capture_output=True, text=True, timeout=10)
	shares = [line.split()[-1] for line in shared.stdout.splitlines() if line.startswith("threads")]
	assert len(shares) == 2 and all(share.isdigit() and int(share) >= 2 for share in shares), (
		"Tools started together did not get a share of the threads:\n" + shared.stdout + shared.stderr)
	kept = subprocess.run([command, str(source)], cwd=root, env=environment, capture_output=True, text=True, timeout=10)
	assert kept.stdout.count("threads 8") == 2, "STRICT_THREADS of the caller was not kept:\n" + kept.stdout + kept.stderr

	print("Unit deadlines, native/build failures, Slow selection, internal packages, transparent values, manual Run order and tool thread shares passed")
