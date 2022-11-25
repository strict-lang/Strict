using NUnit.Framework;

namespace Strict.VirtualMachine.Tests;

public sealed class ByteCodeGeneratorTests : BaseVirtualMachineTests
{
	//ncrunch: no coverage start
	private static IEnumerable<TestCaseData> ByteCodeCases
	{
		get
		{
			yield return new TestCaseData("Test(5).Assign", "Test", new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 5), "number"),
					new StoreStatement(new Instance(NumberType, 5), "bla"),
					new LoadVariableStatement(Register.R0, "bla"),
					new LoadConstantStatement(Register.R1, new Instance(NumberType, 5)),
					new(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new StoreFromRegisterStatement(Register.R2, "something"),
					new LoadVariableStatement(Register.R3, "something"),
					new LoadConstantStatement(Register.R4, new Instance(NumberType, 5)),
					new(Instruction.Add, Register.R3, Register.R4, Register.R5),
					new ReturnStatement(Register.R5)
				},
				new[]
				{
					"has number",
					"Assign Number",
					"\tlet bla = 5",
					"\tlet something = bla + 5",
					"\tsomething + 10"
				});
			yield return new TestCaseData("Add(10, 5).Calculate", "Add",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "First"),
					new StoreStatement(new Instance(NumberType, 5), "Second"),
					new LoadVariableStatement(Register.R0, "First"),
					new LoadVariableStatement(Register.R1, "Second"),
					new(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new ReturnStatement(Register.R2)
				},
				new[]
				{
					"has First Number",
					"has Second Number",
					"Calculate Number",
					"\tAdd(10, 5).Calculate is 15",
					"\tFirst + Second"
				});
			yield return new TestCaseData("Multiply(10).By(2)", "Multiply",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "number"),
					new StoreStatement(new Instance(NumberType, 2), "multiplyBy"),
					new LoadVariableStatement(Register.R0, "number"),
					new LoadVariableStatement(Register.R1, "multiplyBy"),
					new(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
					new ReturnStatement(Register.R2)
				},
				new[]
				{
					"has number", "By(multiplyBy Number) Number", "\tMultiply(10).By(2) is 20",
					"\tnumber * multiplyBy"
				});
			yield return new TestCaseData("Bla(10).SomeFunction", "Bla",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "number"),
					new StoreStatement(new Instance(NumberType, 5), "bla"),
					new LoadVariableStatement(Register.R0, "bla"),
					new LoadVariableStatement(Register.R1, "number"),
					new(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new ReturnStatement(Register.R2)
				}, new[] { "has number", "SomeFunction Number", "\tlet bla = 5", "\tbla + number" });
			yield return new TestCaseData("SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"SimpleLoopExample",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "number"),
					new StoreStatement(new Instance(NumberType, 1), "result"),
					new StoreStatement(new Instance(NumberType, 2), "multiplier"),
					new LoadConstantStatement(Register.R0, new Instance(NumberType, 10)),
					new LoadConstantStatement(Register.R1, new Instance(NumberType, 1)),
					new InitLoopStatement("number"), new LoadVariableStatement(Register.R2, "result"),
					new LoadVariableStatement(Register.R3, "multiplier"),
					new(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
					new StoreFromRegisterStatement(Register.R4, "result"),
					new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
					new JumpStatement(Instruction.JumpIfNotZero, -7),
					new LoadVariableStatement(Register.R5, "result"),
					new ReturnStatement(Register.R5)
				}, SimpleLoopExample);
			yield return new TestCaseData("RemoveParentheses(\"some(thing)\").Remove",
				"RemoveParentheses",
				ExpectedStatementsOfRemoveParanthesesKata,
				RemoveParenthesesKata);
			yield return new TestCaseData("ArithmeticFunction(10, 5).Calculate(\"add\")",
				"ArithmeticFunction", ExpectedStatementsOfArithmeticFunctionExample,
				ArithmeticFunctionExample);
		}
	}
	//ncrunch: no coverage end

	// @formatter:on
	[TestCaseSource(nameof(ByteCodeCases))]
	// ReSharper disable once TooManyArguments
	public void Generate(string methodCall, string programName, Statement[] expectedByteCode,
		params string[] code)
	{
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource(programName, methodCall, code)).
				Generate();
		Assert.That(statements.ConvertAll(x => x.ToString()),
			Is.EqualTo(expectedByteCode.ToList().ConvertAll(x => x.ToString())));
	}
}