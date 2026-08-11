namespace Strict.Bytecode.Tests;

/// <summary>
/// Guards for converting Strict.Bytecode C# types to .strict files under Bytecode/.
/// </summary>
public sealed class StrictBytecodeConversionTests
{
	[Test]
	public void BytecodePackageHasExpectedCoreTypes()
	{
		var path = GetBytecodePath();
		foreach (var typeName in new[]
		{
			"InstructionType", "InstructionNames", "Register", "Registry", "ValueKind",
			"ExpressionKind", "BytecodeValue", "BytecodeInstruction", "InstructionBuilder",
			"InstructionText", "InvokeInfo", "NameTable", "BinaryMember", "BinaryMethod",
			"BinaryTypeData", "BinaryExecutable", "GenerationResult", "ExpressionCodegen",
			"LineGenerator", "NumberLiteral", "Decompiler", "InstructionList"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public void BytecodePackageHasDemoAndTestEntryPoints()
	{
		var path = GetBytecodePath();
		foreach (var typeName in new[]
		{
			"BytecodeDemo", "RegistryTests", "InstructionTests", "NameTableTests",
			"GeneratorTests", "DecompilerTests", "ValueTests", "ExecutableTests"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadRegistryFromBytecodePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var registry = package.GetType("Registry");
		Assert.That(registry.Methods.Any(method => method.Name == "Allocate"), Is.True);
		Assert.That(registry.Methods.Any(method => method.Name == "Advance"), Is.True);
		Assert.That(registry.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	[Test]
	public async Task LoadInstructionBuilderFromBytecodePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var builder = package.GetType("InstructionBuilder");
		foreach (var name in new[]
		{
			"SetNumber", "LoadConstant", "LoadVariable", "StoreConstant", "StoreRegister",
			"BinaryOp", "ReturnOp", "JumpOp", "InvokeOp", "PrintOp"
		})
			Assert.That(builder.Methods.Any(method => method.Name == name), Is.True, name);
	}

	[Test]
	public async Task LoadLineGeneratorFromBytecodePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var generator = package.GetType("LineGenerator");
		Assert.That(generator.Methods.Any(method => method.Name == "GenerateLine"), Is.True);
		Assert.That(generator.Methods.Any(method => method.Name == "GenerateBody"), Is.True);
		Assert.That(generator.Methods.Any(method => method.Name == "GenerateMethod"), Is.True);
		Assert.That(generator.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	[Test]
	public async Task LoadDecompilerAndNameTableFromBytecodePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var decompiler = package.GetType("Decompiler");
		Assert.That(decompiler.Methods.Any(method => method.Name == "ReconstructInstruction"),
			Is.True);
		Assert.That(decompiler.Methods.Any(method => method.Name == "ReconstructMethod"), Is.True);
		var table = package.GetType("NameTable");
		Assert.That(table.Methods.Any(method => method.Name == "Add"), Is.True);
		Assert.That(table.Methods.Any(method => method.Name == "BuiltIn"), Is.True);
	}

	[Test]
	public async Task LoadBinaryExecutableSurfaceFromBytecodePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var executable = package.GetType("BinaryExecutable");
		Assert.That(executable.Methods.Any(method => method.Name == "AddType"), Is.True);
		Assert.That(executable.Methods.Any(method => method.Name == "SetEntry"), Is.True);
		Assert.That(executable.Methods.Any(method => method.Name == "Empty"), Is.True);
		var typeData = package.GetType("BinaryTypeData");
		Assert.That(typeData.Methods.Any(method => method.Name == "Create"), Is.True);
		Assert.That(typeData.Methods.Any(method => method.Name == "FindMethod"), Is.True);
	}

	[Test]
	public async Task LoadInstructionTypeAndRegisterConstants()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Bytecode");
		var instructionType = package.GetType("InstructionType");
		Assert.That(instructionType.Members.Any(member => member.Name == "Add"), Is.True);
		Assert.That(instructionType.Members.Any(member => member.Name == "Return"), Is.True);
		Assert.That(instructionType.Members.Any(member => member.Name == "Equal"), Is.True);
		var register = package.GetType("Register");
		Assert.That(register.Members.Any(member => member.Name == "Count"), Is.True);
		Assert.That(register.Methods.Any(method => method.Name == "IsValid"), Is.True);
		Assert.That(register.Methods.Any(method => method.Name == "NameOf"), Is.True);
	}

	private static string GetBytecodePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Bytecode");
}
