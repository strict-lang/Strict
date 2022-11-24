using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine.Tests;

public sealed class ByteCodeGeneratorTests : BaseVirtualMachineTests
{
	//ncrunch: no coverage start
	private static IEnumerable<TestCaseData> ByteCodeCases
	{
		get
		{
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
					"has First Number", "has Second Number", "Calculate Number",
					"\tAdd(10, 5).Calculate is 15", "\tFirst + Second"
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
				},
				new[]
				{
					"has number", "SomeFunction Number", "\tlet bla = 5", "\tbla + number"
				});
			yield return new TestCaseData("SimpleLoopExample(10).GetMultiplicationOfNumbers", "SimpleLoopExample",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "number"),
					new StoreStatement(new Instance(NumberType, 1), "result"),
					new StoreStatement(new Instance(NumberType, 2), "multiplier"),
					new LoadConstantStatement(Register.R0, new Instance(NumberType, 10)),
					new LoadConstantStatement(Register.R1, new Instance(NumberType, 1)),
					new InitLoopStatement("number"),
					new LoadVariableStatement(Register.R2, "result"),
					new LoadVariableStatement(Register.R3, "multiplier"),
					new(Instruction.Multiply, Register.R2, Register.R3, Register.R2),
					new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
					new JumpStatement(Instruction.JumpIfNotZero, -5),
					new ReturnStatement(Register.R2)
				}, SimpleLoopExample);
			yield return new TestCaseData("RemoveParentheses(\"some(thing)\").Remove",
				"RemoveParentheses",
				new Statement[]
				{
					new StoreStatement(new Instance(TextType, "some(thing)"), "text"),
					new StoreStatement(new Instance(TextType, "\"\""), "result"),
					new StoreStatement(new Instance(NumberType, 0), "count"),
					new LoadConstantStatement(Register.R0, new Instance(NumberType, 11)),
					new LoadConstantStatement(Register.R1, new Instance(NumberType, 1)),
					new InitLoopStatement("text"),
					new LoadVariableStatement(Register.R2, "value"),
					new LoadConstantStatement(Register.R3, new Instance(TextType, "(")),
					new(Instruction.Equal, Register.R2, Register.R3),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 0),
					new LoadVariableStatement(Register.R2, "count"),
					new LoadConstantStatement(Register.R3, new Instance(NumberType, 1)),
					new(Instruction.Add, Register.R2, Register.R3, Register.R2),
					new JumpViaIdStatement(Instruction.JumpEnd, 0),
					new LoadVariableStatement(Register.R3, "value"),
					new LoadConstantStatement(Register.R2, new Instance(TextType, ")")),
					new(Instruction.Equal, Register.R3, Register.R2),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 1),
					new LoadVariableStatement(Register.R3, "count"),
					new LoadConstantStatement(Register.R2, new Instance(NumberType, 1)),
					new (Instruction.Subtract, Register.R3, Register.R2, Register.R3),
					new JumpViaIdStatement(Instruction.JumpEnd, 1),
					new LoadVariableStatement(Register.R2, "count"),
					new LoadConstantStatement(Register.R3, new Instance(NumberType, 0)),
					new(Instruction.Equal, Register.R2, Register.R3),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 2),
					new LoadVariableStatement(Register.R2, "result"),
					new LoadVariableStatement(Register.R3, "value"),
					new (Instruction.Add, Register.R2, Register.R3, Register.R2),
					new JumpViaIdStatement(Instruction.JumpEnd, 2),
					new (Instruction.Subtract, Register.R0, Register.R1, Register.R0),
					new JumpStatement(Instruction.JumpIfNotZero, -26),
					new ReturnStatement(Register.R2),
				},
				RemoveParenthesesKata);
			yield return new TestCaseData("ArithmeticFunction(10, 5).Calculate(\"add\")",
				"ArithmeticFunction",
				new Statement[]
				{
					new StoreStatement(new Instance(NumberType, 10), "First"),
					new StoreStatement(new Instance(NumberType, 5), "Second"),
					new StoreStatement(new Instance(TextType, "add"), "operation"),
					new LoadVariableStatement(Register.R0, "operation"),
					new LoadConstantStatement(Register.R1, new Instance(TextType, "add")),
					new(Instruction.Equal, Register.R0, Register.R1),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 0),
					new LoadVariableStatement(Register.R2, "First"),
					new LoadVariableStatement(Register.R3, "Second"),
					new(Instruction.Add, Register.R2, Register.R3, Register.R0),
					new ReturnStatement(Register.R0),
					new JumpViaIdStatement(Instruction.JumpEnd, 0),
					new LoadVariableStatement(Register.R1, "operation"),
					new LoadConstantStatement(Register.R2, new Instance(TextType, "subtract")),
					new(Instruction.Equal, Register.R1, Register.R2),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 1),
					new LoadVariableStatement(Register.R3, "First"),
					new LoadVariableStatement(Register.R0, "Second"),
					new(Instruction.Subtract, Register.R3, Register.R0, Register.R1),
					new ReturnStatement(Register.R1),
					new JumpViaIdStatement(Instruction.JumpEnd, 1),
					new LoadVariableStatement(Register.R2, "operation"),
					new LoadConstantStatement(Register.R3, new Instance(TextType, "multiply")),
					new(Instruction.Equal, Register.R2, Register.R3),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 2),
					new LoadVariableStatement(Register.R0, "First"),
					new LoadVariableStatement(Register.R1, "Second"),
					new(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
					new ReturnStatement(Register.R2),
					new JumpViaIdStatement(Instruction.JumpEnd, 2),
					new LoadVariableStatement(Register.R3, "operation"),
					new LoadConstantStatement(Register.R0, new Instance(TextType, "divide")),
					new(Instruction.Equal, Register.R3, Register.R0),
					new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 3),
					new LoadVariableStatement(Register.R1, "First"),
					new LoadVariableStatement(Register.R2, "Second"),
					new(Instruction.Divide, Register.R1, Register.R2, Register.R3),
					new ReturnStatement(Register.R3),
					new JumpViaIdStatement(Instruction.JumpEnd, 3)
				}, ArithmeticFunctionExample);
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