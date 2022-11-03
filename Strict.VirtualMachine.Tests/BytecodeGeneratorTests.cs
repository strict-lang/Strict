using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

public sealed class ByteCodeGeneratorTests : TestExpressions
{
	private static readonly Type TextType = new TestPackage().FindType(Base.Text)!;
	private static readonly Type NumberType = new TestPackage().FindType(Base.Number)!;
	private static object[] bytecodeCases =
	{
		// @formatter:off
		new object[]
		{
			"Add(10, 5).Calculate",
			"Add",
			new Statement[]{
				new StoreStatement(new Instance(NumberType, 10), "First"),
				new StoreStatement(new Instance(NumberType, 5), "Second"),
				new LoadVariableStatement(Register.R1, "First"),
				new LoadVariableStatement(Register.R0, "Second"),
				new(Instruction.Add, Register.R1, Register.R0, Register.R2)
				},
			new[]{
			"has First Number",
			"has Second Number",
			"Calculate Number",
			"\tAdd(10, 5).Calculate is 15",
			"\tFirst + Second"

			}
		},
				new object[]
		{
			"Multiply(10).By(2)",
			"Multiply",
			new Statement[]{
				new StoreStatement(new Instance(NumberType, 10), "number"),
				new StoreStatement(new Instance(NumberType, 2), "multiplyBy"),
				new LoadVariableStatement(Register.R1, "number"),
				new LoadVariableStatement(Register.R0, "multiplyBy"),
				new(Instruction.Multiply, Register.R0, Register.R1, Register.R2)
				},
			new[]{
			"has number",
			"By(multiplyBy Number) Number",
			"\tMultiply(10).By(2) is 20",
			"\tnumber * multiplyBy"
				}
		}
	};

	// @formatter:on
	[TestCaseSource(nameof(bytecodeCases))]
	// ReSharper disable once TooManyArguments
	public void Generate(string methodCall, string programName, Statement[] expectedByteCode,
		params string[] code)
	{
		new Type(type.Package, new TypeLines(programName, code)).ParseMembersAndMethods(
			new MethodExpressionParser());
		var expression = (MethodCall)ParseExpression(methodCall);
		var statements = new BytecodeGenerator(expression).Generate();
		Assert.That(statements.ConvertAll(x => x.ToString()),
			Is.EqualTo(expectedByteCode.ToList().ConvertAll(x => x.ToString())));
	}
}