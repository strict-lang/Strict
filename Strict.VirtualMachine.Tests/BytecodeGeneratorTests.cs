using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

public sealed class ByteCodeGeneratorTests : TestExpressions
{
	[TestCase(
		"Add(10, 5).Calculate",
		"Add",
		"SetVariable 10\nSetVariable 5\nLoad R0\nLoad R1\nAdd R0, R1, R2",
		"has First Number",
		"has Second Number",
		"from(first Number, second Number)",
		"\tFirst = first",
		"\tSecond = second",
		"Calculate Number",
		"\tAdd(10, 5).Calculate is 15",
		"\tFirst + Second")]
	[TestCase(
		"Multiply(10).By(2)",
		"Multiply",
		"SetVariable 10\nSetVariable 2\nLoad R0\nLoad R1\nMultiply R0, R1, R2",
		"has number",
		"By(multiplyBy Number) Number",
		"\tMultiply(10).By(2) is 20",
		"\tnumber * multiplyBy")]
	// ReSharper disable once TooManyArguments
	public void Generate(string methodCall, string programName, string expectedByteCode, params string[] code)
	{
		new Type(type.Package, new TypeLines(programName, code)).ParseMembersAndMethods(
			new MethodExpressionParser());
		var expression = (MethodCall)ParseExpression(methodCall);
		var statements = new BytecodeGenerator(expression).Generate();
		var statementsString = "";
		statements?.ForEach(statement => statementsString += $"{statement}\n");
		Assert.That(statementsString.Trim(), Is.EqualTo(expectedByteCode));
	}
}