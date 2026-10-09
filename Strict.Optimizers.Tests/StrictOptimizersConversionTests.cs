using Strict.Expressions;

namespace Strict.Optimizers.Tests;

/// <summary>
/// Guards for converting Strict.Optimizers C# types to .strict files under Optimizers/.
/// </summary>
public sealed class StrictOptimizersConversionTests
{
	[Test]
	public void OptimizersPackageHasExpectedCoreTypes()
	{
		var path = GetOptimizersPath();
		foreach (var typeName in new[]
			{
				"OptimizerStats", "ConstantFolder", "StrengthReduce",
				"DeadStore", "RedundantLoad", "JumpThread", "UnreachableCode", "TestCodeRemove",
				"AllOptimizers"
			})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public void OptimizersPackageHasDemoAndTestEntryPoints()
	{
		var path = GetOptimizersPath();
		foreach (var typeName in new[]
			{
				"OptimizerDemo", "FolderTests", "StrengthTests", "DeadStoreTests", "UnreachableTests",
				"PipelineTests"
			})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadAllOptimizersFromPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Optimizers");
		var all = package.GetType("AllOptimizers");
		Assert.That(all.Methods.Any(method => method.Name == "Optimize"), Is.True);
		Assert.That(all.Methods.Any(method => method.Name == "OptimizeWithStats"), Is.True);
		Assert.That(all.Members.Any(member => member.Name == "OptimizerCount"), Is.True);
	}

	[Test]
	public async Task LoadConstantFolderAndStrengthReduce()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Optimizers");
		var folder = package.GetType("ConstantFolder");
		Assert.That(folder.Methods.Any(method => method.Name == "Optimize"), Is.True);
		Assert.That(folder.Methods.Any(method => method.Name == "FoldFrom"), Is.True);
		Assert.That(folder.Methods.Any(method => method.Name == "Propagated"), Is.True);
		var strength = package.GetType("StrengthReduce");
		Assert.That(strength.Methods.Any(method => method.Name == "Optimize"), Is.True);
		Assert.That(strength.Methods.Any(method => method.Name == "IsIdentity"), Is.True);
	}

	[Test]
	public async Task LoadDeadStoreAndUnreachable()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Optimizers");
		var dead = package.GetType("DeadStore");
		Assert.That(dead.Methods.Any(method => method.Name == "Optimize"), Is.True);
		Assert.That(dead.Methods.Any(method => method.Name == "IsLoaded"), Is.True);
		var unreachable = package.GetType("UnreachableCode");
		Assert.That(unreachable.Methods.Any(method => method.Name == "Optimize"), Is.True);
		Assert.That(unreachable.Methods.Any(method => method.Name == "Optimize"), Is.True);
	}

	[Test]
	public async Task OptimizersUseSharedBytecodeInstructionList()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var list = package.GetType("InstructionList");
		Assert.That(list.Methods.Any(method => method.Name == "Skip"), Is.True);
		Assert.That(list.Methods.Any(method => method.Name == "Prepend"), Is.True);
		Assert.That(list.Methods.Any(method => method.Name == "Empty"), Is.True);
		var instruction = package.GetType("BytecodeInstruction");
		Assert.That(instruction.Methods.Any(method => method.Name == "IsBinary"), Is.True);
		Assert.That(instruction.Methods.Any(method => method.Name == "IsReturn"), Is.True);
		Assert.That(instruction.Methods.Any(method => method.Name == "LoadConstant"), Is.True);
	}

	private static string GetOptimizersPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Optimizers");
}