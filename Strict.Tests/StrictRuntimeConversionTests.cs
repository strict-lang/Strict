using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.Tests;

/// <summary>
/// Guards for converting Strict VirtualMachine/Runner C# types to .strict under Runtime/.
/// </summary>
public sealed class StrictRuntimeConversionTests
{
	[Test]
	public void RuntimePackageHasExpectedCoreTypes()
	{
		var path = GetRuntimePath();
		foreach (var typeName in new[]
		{
			"VmValue", "RegisterBank", "CallFrame", "VmMemory", "VmInstruction", "InstrBuilder",
			"InstrList", "VmState", "ArithmeticExec", "InstructionExec", "VirtualMachine",
			"RunnerPipeline"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public void RuntimePackageHasDemoAndTestEntryPoints()
	{
		var path = GetRuntimePath();
		foreach (var typeName in new[]
		{
			"VmDemo", "RegisterTests", "FrameTests", "VmTests", "CompareTests"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadVirtualMachineFromRuntimePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Runtime");
		var vm = package.GetType("VirtualMachine");
		Assert.That(vm.Methods.Any(method => method.Name == "Run"), Is.True);
		Assert.That(vm.Methods.Any(method => method.Name == "RunFrom"), Is.True);
		Assert.That(vm.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	[Test]
	public async Task LoadRegisterBankAndCallFrame()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Runtime");
		var bank = package.GetType("RegisterBank");
		Assert.That(bank.Methods.Any(method => method.Name == "Get"), Is.True);
		Assert.That(bank.Methods.Any(method => method.Name == "Set"), Is.True);
		Assert.That(bank.Members.Any(member => member.Name == "Capacity"), Is.True);
		var frame = package.GetType("CallFrame");
		Assert.That(frame.Methods.Any(method => method.Name == "Get"), Is.True);
		Assert.That(frame.Methods.Any(method => method.Name == "Set"), Is.True);
		Assert.That(frame.Methods.Any(method => method.Name == "Has"), Is.True);
	}

	[Test]
	public async Task LoadInstructionExecAndPipeline()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Runtime");
		var exec = package.GetType("InstructionExec");
		Assert.That(exec.Methods.Any(method => method.Name == "Execute"), Is.True);
		var pipeline = package.GetType("RunnerPipeline");
		Assert.That(pipeline.Methods.Any(method => method.Name == "RunSimple"), Is.True);
		Assert.That(pipeline.Methods.Any(method => method.Name == "RunExpression"), Is.True);
		Assert.That(pipeline.Methods.Any(method => method.Name == "RunStored"), Is.True);
	}

	[Test]
	public async Task LoadVmValueAndArithmetic()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Runtime");
		var value = package.GetType("VmValue");
		Assert.That(value.Methods.Any(method => method.Name == "FromNumber"), Is.True);
		Assert.That(value.Methods.Any(method => method.Name == "EqualsValue"), Is.True);
		var arith = package.GetType("ArithmeticExec");
		Assert.That(arith.Methods.Any(method => method.Name == "Compute"), Is.True);
		Assert.That(arith.Methods.Any(method => method.Name == "Add"), Is.True);
	}

	private static string GetRuntimePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Runtime");
}
