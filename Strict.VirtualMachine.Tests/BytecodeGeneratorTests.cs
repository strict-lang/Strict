using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

public sealed class BytecodeGeneratorTests
{
	private readonly BytecodeGenerator generator = new();
	private static readonly Package Package = new TestPackage();

	[TestCase("SetVariable 0\nSetVariable 0\nLoad first R0\nLoad second R1\nAdd R0, R1, R2",
		"Add",
		"has Number",
		"Calculate(first Number, second Number) Number",
		"\tfirst + second")]
	public void Generate(string expectedBytecode, string programName, params string[] code)
	{
		var program =
			new Type(Package, new TypeLines(programName, code)).ParseMembersAndMethods(
				new MethodExpressionParser());
		var statements = BytecodeGenerator.Generate(program.Methods[0]);
		var statementsString = "";
		statements.ForEach(statement => statementsString += $"{statement}\n");
		Assert.That(statementsString.Trim(), Is.EqualTo(expectedBytecode));
	}
}