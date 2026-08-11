using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.Compiler.Assembly.Tests;

/// <summary>
/// Guards for converting Strict.Compiler / Assembly C# types to .strict under Compiler/.
/// </summary>
public sealed class StrictCompilerConversionTests
{
	[Test]
	public void CompilerPackageHasExpectedCoreTypes()
	{
		var path = GetCompilerPath();
		foreach (var typeName in new[]
		{
			"Platform", "ToolInfo", "RegisterMap", "CompInstruction", "CompBuilder", "CompList",
			"AsmText", "InstrToAsm", "EntryPoint", "NasmFormat", "LinkerPlan",
			"InstructionsToNasm", "CompilerPipeline"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public void CompilerPackageHasDemoAndTestEntryPoints()
	{
		var path = GetCompilerPath();
		foreach (var typeName in new[]
		{
			"CompilerDemo", "PlatformTests", "EmitTests", "LinkerTests"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadPlatformAndNasmFormat()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Compiler");
		var platform = package.GetType("Platform");
		Assert.That(platform.Members.Any(member => member.Name == "Windows"), Is.True);
		Assert.That(platform.Members.Any(member => member.Name == "Linux"), Is.True);
		Assert.That(platform.Methods.Any(method => method.Name == "NameOf"), Is.True);
		var format = package.GetType("NasmFormat");
		Assert.That(format.Methods.Any(method => method.Name == "FormatFor"), Is.True);
	}

	[Test]
	public async Task LoadInstructionsToNasmAndInstrToAsm()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Compiler");
		var nasm = package.GetType("InstructionsToNasm");
		Assert.That(nasm.Methods.Any(method => method.Name == "Compile"), Is.True);
		Assert.That(nasm.Methods.Any(method => method.Name == "CompileSimpleAdd"), Is.True);
		Assert.That(nasm.Members.Any(member => member.Name == "Extension"), Is.True);
		var emit = package.GetType("InstrToAsm");
		Assert.That(emit.Methods.Any(method => method.Name == "Emit"), Is.True);
		Assert.That(emit.Methods.Any(method => method.Name == "EmitBinary"), Is.True);
	}

	[Test]
	public async Task LoadLinkerPlanAndEntryPoint()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Compiler");
		var plan = package.GetType("LinkerPlan");
		Assert.That(plan.Methods.Any(method => method.Name == "ForAsm"), Is.True);
		Assert.That(plan.Methods.Any(method => method.Name == "ForWindows"), Is.True);
		var entry = package.GetType("EntryPoint");
		Assert.That(entry.Methods.Any(method => method.Name == "Build"), Is.True);
		Assert.That(entry.Methods.Any(method => method.Name == "WindowsMain"), Is.True);
	}

	[Test]
	public async Task LoadToolInfoAndRegisterMap()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Compiler");
		var tools = package.GetType("ToolInfo");
		Assert.That(tools.Methods.Any(method => method.Name == "NasmName"), Is.True);
		Assert.That(tools.Methods.Any(method => method.Name == "LinkerName"), Is.True);
		var map = package.GetType("RegisterMap");
		Assert.That(map.Methods.Any(method => method.Name == "XmmName"), Is.True);
		Assert.That(map.Methods.Any(method => method.Name == "IsValidReg"), Is.True);
	}

	private static string GetCompilerPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Compiler");
}
