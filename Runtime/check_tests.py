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
		'\tmutable count = 0\n\tfor as long as Process.Elapsed - started < 0.04\n'
		'\t\tcount = count + 1\n\tcount > 0 then 1 else 0', encoding="utf-8")
	source = folder / "OverBudget.strict"
	source.write_text('Delay Number\n\tOverBudget.Delay is 1\n\tTestWork.Pause', encoding="utf-8")
	result = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert result.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in result.stdout + result.stderr, (
		"An over-budget VM unit test passed:\n" + result.stdout + result.stderr)
	assert not source.with_suffix(".strictbinary").exists(), "Failed unit tests were cached as passed"
	for arguments in [["build", str(source)], ["test", str(source), "unit", "native"]]:
		checked = subprocess.run([command, *arguments], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90)
		assert checked.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in checked.stdout + checked.stderr, (
			str(arguments) + " accepted an over-budget unit test:\n" + checked.stdout + checked.stderr)

	source = folder / "SharedBudget.strict"
	source.write_text('constant Shared = TestWork.Pause\nCheck Number\n\tSharedBudget.Shared is 1\n\t1', encoding="utf-8")
	for options in [["unit"], ["unit", "native"]]:
		shared = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90 if "native" in options else 10)
		assert shared.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in shared.stdout + shared.stderr, (
			"An over-budget shared fixture passed:\n" + shared.stdout + shared.stderr)

	source = folder / "LazyFixture.strict"
	source.write_text('has logger\nconstant Shared = LazyFixture.Created\nCreated Number\n\tlogger.Log("fixture computed")\n\t7\nCheck Number\n\tLazyFixture.Check is 7\n\tShared is 7\n\tShared + Shared is 14\n\tfalse then Shared else 7', encoding="utf-8")
	lazy = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert lazy.returncode == 0 and lazy.stdout.count("fixture computed") == 1, lazy.stdout + lazy.stderr
	assert lazy.stdout.index("Test passed: LazyFixture.Check is 7") < lazy.stdout.index("fixture computed"), lazy.stdout

	source = folder / "SetupBudget.strict"
	source.write_text('Check Number\n\tconstant sample = TestWork.Pause\n\tsample is 1\n\t1', encoding="utf-8")
	setup = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert setup.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in setup.stdout + setup.stderr, setup.stdout + setup.stderr

	source = folder / "Hangs.strict"
	source.write_text('Spin Number\n\tHangs.Spin is 1\n\tmutable count = 0\n'
		'\tfor as long as Process.Elapsed >= 0\n\t\tcount = count + 1\n\tcount', encoding="utf-8")
	for options in [["unit"], ["unit", "native"]]:
		hung = subprocess.run([command, "test", str(source), *options], cwd=root, env=environment,
			capture_output=True, text=True, timeout=90 if "native" in options else 10)
		assert hung.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in hung.stdout + hung.stderr, hung.stdout + hung.stderr
	validated = subprocess.run([command, "check", str(folder)], cwd=root, env=environment,
		capture_output=True, text=True, timeout=30)
	assert validated.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in validated.stdout, validated.stdout + validated.stderr

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

	package = folder / "Bytecode"
	package.mkdir()
	source = package / "ValueCodegen.strict"
	def vm_budget(seconds):
		source.write_text('Delay Number\n\tValueCodegen.Delay is 1\n\tlet started = Process.Elapsed\n'
			'\tmutable count = 0\n\tfor as long as Process.Elapsed - started < ' + str(seconds) + '\n'
			'\t\tcount = count + 1\n\tcount > 0 then 1 else 0', encoding="utf-8")
		return subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
			capture_output=True, text=True, timeout=10)
	extended = vm_budget(0.04)
	assert extended.returncode == 0, "A bounded VM self-test allowance is missing:\n" + extended.stdout + extended.stderr
	native = subprocess.run([command, "test", str(source), "unit", "native"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=90)
	assert native.returncode != 0 and "TestDurationMustNotExceedTwentyMilliseconds" in native.stdout + native.stderr, native.stdout + native.stderr
	over_budget = vm_budget(0.15)
	assert over_budget.returncode != 0 and "TestDurationMustNotExceedVmBudget" in over_budget.stdout + over_budget.stderr, over_budget.stdout + over_budget.stderr
	source.write_text('Delay Number\n\tValueCodegen.Delay is 1\n\tmutable count = 0\n'
		'\tfor as long as Process.Elapsed >= 0\n\t\tcount = count + 1\n\tcount', encoding="utf-8")
	hung_vm = subprocess.run([command, "test", str(source), "unit"], cwd=root, env=environment,
		capture_output=True, text=True, timeout=10)
	assert hung_vm.returncode != 0 and "TestDurationMustNotExceedVmBudget" in hung_vm.stdout + hung_vm.stderr, hung_vm.stdout + hung_vm.stderr

	print("Unit deadlines, native/build failures, Slow selection, transparent values and manual Run order passed")
