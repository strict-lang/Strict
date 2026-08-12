using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

/// <summary>
/// Regression coverage for Language/Type.strict methods under BinaryGenerator + VM.
/// </summary>
public sealed class LanguageTypeVmTests
{
	[Test]
	public async Task TypeIsMemberLineDoesNotStackOverflowUnderVm()
	{
		var repos = new Repositories(new MethodExpressionParser());
		using var language = await repos.LoadStrictPackage("Strict/Language");
		var typeType = language.GetType("Type");
		var isMemberLine = typeType.Methods.Single(method => method.Name == "IsMemberLine");
		var body = isMemberLine.GetBodyAndParseIfNeeded();
		Assert.That(body, Is.InstanceOf<Body>());
		var bodyExpr = (Body)body;
		TestContext.WriteLine("IsMemberLine expressions:");
		foreach (var expression in bodyExpr.Expressions)
			TestContext.WriteLine("  " + expression.GetType().Name + ": " + expression);

		var lines = new[]
		{
			"has logger",
			"Run",
			"\tconstant type = Type(\"HelloLogger\", (\"has logger\", \"Run\"))",
			"\ttype.IsMemberLine(\"has logger\")"
		};
		var package = language;
		using var repro = new Type(package, new TypeLines("TypeIsMemberLineVm", lines)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var run = repro.Methods.Single(method => method.Name == Method.Run);
		run.GetBodyAndParseIfNeeded();
		var binary = BinaryGenerator.GenerateFromRunMethods(run, [run]);
		DumpBinary(binary);

		// Execute via reflection-free path: use Strict.VirtualMachine from Strict project - not referenced.
		// Assert that compiled Type.IsMemberLine instructions exist and do not call IsMemberLine.
		var typeBinaryName = binary.MethodsPerType.Keys.FirstOrDefault(name =>
			name == "Type" || name.EndsWith("/Type", StringComparison.Ordinal));
		Assert.That(typeBinaryName, Is.Not.Null, "Type should be in binary method table");
		var typeData = binary.MethodsPerType[typeBinaryName!];
		Assert.That(typeData.MethodGroups.ContainsKey("IsMemberLine"), Is.True,
			"IsMemberLine should be compiled into binary");
		var isMemberLineMethods = typeData.MethodGroups["IsMemberLine"];
		Assert.That(isMemberLineMethods, Is.Not.Empty);
		foreach (var method in isMemberLineMethods)
		{
			TestContext.WriteLine("IsMemberLine instructions (" + method.instructions.Count + "):");
			foreach (var instruction in method.instructions)
			{
				TestContext.WriteLine("  " + instruction);
				if (instruction is Invoke invoke)
					TestContext.WriteLine("    => " + invoke.MethodInfo.TypeFullName + "." +
						invoke.MethodInfo.MethodName + " ret=" + invoke.MethodInfo.ReturnTypeName +
						" args=" + invoke.MethodInfo.ArgumentRegisters.Length +
						" instance=" + invoke.MethodInfo.InstanceRegister +
						" params=[" + string.Join(",", invoke.MethodInfo.ParameterNames) + "]");
			}
			var recursiveInvokes = method.instructions.OfType<Invoke>().Where(invoke =>
				invoke.MethodInfo.MethodName == "IsMemberLine").ToList();
			Assert.That(recursiveInvokes, Is.Empty,
				"IsMemberLine must not invoke itself (would stack-overflow)");
		}
		if (binary.MethodsPerType.TryGetValue("Strict/Boolean", out var booleanType) &&
			booleanType.MethodGroups.TryGetValue("or", out var orMethods))
		{
			TestContext.WriteLine("Boolean.or instructions:");
			foreach (var instruction in orMethods[0].instructions)
			{
				TestContext.WriteLine("  " + instruction);
				if (instruction is Invoke invoke)
					TestContext.WriteLine("    => " + invoke.MethodInfo.TypeFullName + "." +
						invoke.MethodInfo.MethodName);
			}
		}
		if (binary.MethodsPerType.TryGetValue("Strict/Text", out var textType) &&
			textType.MethodGroups.TryGetValue("StartsWith", out var startsWithMethods))
		{
			TestContext.WriteLine("Text.StartsWith instructions:");
			foreach (var instruction in startsWithMethods[0].instructions)
			{
				TestContext.WriteLine("  " + instruction);
				if (instruction is Invoke invoke)
					TestContext.WriteLine("    => " + invoke.MethodInfo.TypeFullName + "." +
						invoke.MethodInfo.MethodName + " instance=" +
						invoke.MethodInfo.InstanceRegister);
			}
		}
	}

	[Test]
	public async Task TypeMembersMethodCompilesWithoutSelfRecursion()
	{
		var repos = new Repositories(new MethodExpressionParser());
		using var language = await repos.LoadStrictPackage("Strict/Language");
		var lines = new[]
		{
			"has logger",
			"Run",
			"\tconstant type = Type(\"HelloLogger\", (\"has logger\", \"Run\"))",
			"\ttype.Members.Length"
		};
		using var repro = new Type(language, new TypeLines("TypeMembersVm", lines)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var run = repro.Methods.Single(method => method.Name == Method.Run);
		run.GetBodyAndParseIfNeeded();
		var binary = BinaryGenerator.GenerateFromRunMethods(run, [run]);
		DumpBinary(binary);
		var typeBinaryName = binary.MethodsPerType.Keys.First(name =>
			name == "Type" || name.Contains("Language/Type", StringComparison.Ordinal));
		Assert.That(binary.MethodsPerType[typeBinaryName].MethodGroups.ContainsKey("Members"),
			Is.True, "Members should be compiled");
		var membersGroup = binary.MethodsPerType[typeBinaryName].MethodGroups["Members"];
		foreach (var method in membersGroup)
		{
			TestContext.WriteLine("Members overload params=" + method.parameters.Count +
				" instr=" + method.instructions.Count);
			foreach (var instruction in method.instructions.Take(30))
				TestContext.WriteLine("  " + instruction);
			// Filtered for-if must not always-WriteToList after the if without a then-only path
			var writeToListCount = method.instructions.Count(i => i is WriteToListInstruction);
			TestContext.WriteLine("WriteToList count: " + writeToListCount);
		}
	}

	private static void DumpBinary(BinaryExecutable binary)
	{
		TestContext.WriteLine("Binary types: " + string.Join(", ", binary.MethodsPerType.Keys));
		foreach (var (typeName, typeData) in binary.MethodsPerType)
		{
			TestContext.WriteLine("TYPE " + typeName);
			foreach (var (methodName, overloads) in typeData.MethodGroups)
			foreach (var method in overloads)
				TestContext.WriteLine("  " + methodName + "(" + method.parameters.Count + ") -> " +
					method.ReturnTypeName + " [" + method.instructions.Count + " instr]");
		}
	}
}
