using NUnit.Framework;

namespace Strict.VirtualMachine.Tests;

public sealed class ByteCodeGeneratorTests : BaseVirtualMachineTests
{
	private static object[] bytecodeCases = //ncrunch: no coverage
	{
		// @formatter:off
		new object[]
		{
			"Add(10, 5).Calculate",
			"Add",
			new Statement[]{
				new StoreStatement(new Instance(NumberType, 10), "First"),
				new StoreStatement(new Instance(NumberType, 5), "Second"),
				new LoadVariableStatement(Register.R0, "First"),
				new LoadVariableStatement(Register.R1, "Second"),
				new(Instruction.Add, Register.R0, Register.R1, Register.R0)
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
				new LoadVariableStatement(Register.R0, "number"),
				new LoadVariableStatement(Register.R1, "multiplyBy"),
				new(Instruction.Multiply, Register.R0, Register.R1, Register.R0)
				},
			new[]{
			"has number",
			"By(multiplyBy Number) Number",
			"\tMultiply(10).By(2) is 20",
			"\tnumber * multiplyBy"
				}
		},
			new object[]
		{
			"ArithmeticFunction(10, 5).Calculate(\"add\")",
			"ArithmeticFunction",
			new Statement[]{
				new StoreStatement(new Instance(NumberType, 10), "First"),
				new StoreStatement(new Instance(NumberType, 5), "Second"),
				new StoreStatement(new Instance(TextType, "add"), "operation"),
				new LoadVariableStatement(Register.R0, "operation"),
				new LoadConstantStatement(Register.R1, new Instance(TextType, "add")),
				new (Instruction.Equal, Register.R0, Register.R1),
				new JumpStatement(Instruction.JumpIfFalse, 4),
				new LoadVariableStatement(Register.R2, "First"),
				new LoadVariableStatement(Register.R3, "Second"),
				new(Instruction.Add, Register.R2, Register.R3, Register.R2),
				new ReturnStatement( Register.R2),
				new LoadVariableStatement(Register.R0, "operation"),
				new LoadConstantStatement(Register.R1, new Instance(TextType, "subtract")),
				new (Instruction.Equal, Register.R0, Register.R1),
				new JumpStatement(Instruction.JumpIfFalse, 4),
				new LoadVariableStatement(Register.R2, "First"),
				new LoadVariableStatement(Register.R3, "Second"),
				new(Instruction.Subtract, Register.R2, Register.R3, Register.R2),
				new ReturnStatement( Register.R2),
				new LoadVariableStatement(Register.R0, "operation"),
				new LoadConstantStatement(Register.R1, new Instance(TextType, "multiply")),
				new (Instruction.Equal, Register.R0, Register.R1),
				new JumpStatement(Instruction.JumpIfFalse, 4),
				new LoadVariableStatement(Register.R2, "First"),
				new LoadVariableStatement(Register.R3, "Second"),
				new(Instruction.Multiply, Register.R2, Register.R3, Register.R2),
				new ReturnStatement( Register.R2),
				new LoadVariableStatement(Register.R0, "operation"),
				new LoadConstantStatement(Register.R1, new Instance(TextType, "divide")),
				new (Instruction.Equal, Register.R0, Register.R1),
				new JumpStatement(Instruction.JumpIfFalse, 4),
				new LoadVariableStatement(Register.R2, "First"),
				new LoadVariableStatement(Register.R3, "Second"),
				new(Instruction.Divide, Register.R2, Register.R3, Register.R2),
				new ReturnStatement( Register.R2)
			},
			ArithmeticFunctionExample
		}
	};

	// @formatter:on
	[TestCaseSource(nameof(bytecodeCases))]
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