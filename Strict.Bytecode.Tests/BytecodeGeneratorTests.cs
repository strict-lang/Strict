using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeGeneratorTests : TestBytecode
{
	[TestCaseSource(nameof(ByteCodeCases))]
	public void Generate(string methodCall, string programName, Instruction[] expectedByteCode,
		params string[] code)
	{
		var instructions =
			new BytecodeGenerator(GenerateMethodCallFromSource(programName, methodCall, code)).
				Generate();
		Assert.That(instructions.ConvertAll(x => x.ToString()),
			Is.EqualTo(expectedByteCode.ToList().ConvertAll(x => x.ToString())));
	}

	//ncrunch: no coverage start
	private static IEnumerable<TestCaseData> ByteCodeCases
	{
		get
		{
			yield return new TestCaseData("Test(5).Assign", "Test", new Instruction[]
				{
					new StoreVariableInstruction(Number(5), "number"),
					new StoreVariableInstruction(Number(5), "five"),
					new LoadVariableToRegister(Register.R0, "five"),
					new LoadConstantInstruction(Register.R1, Number(5)),
					new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
					new StoreFromRegisterInstruction(Register.R2, "something"),
					new LoadVariableToRegister(Register.R3, "something"),
					new LoadConstantInstruction(Register.R4, Number(10)),
					new BinaryInstruction(InstructionType.Add, Register.R3, Register.R4, Register.R5),
					new ReturnInstruction(Register.R5)
				},
				new[]
				{
					"has number",
					"Assign Number",
					"\tconstant five = 5",
					"\tconstant something = five + 5",
					"\tsomething + 10"
				});
			yield return new TestCaseData("Add(10, 5).Calculate", "Add",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(10), "First"),
					new StoreVariableInstruction(Number(5), "Second"),
					new LoadVariableToRegister(Register.R0, "First"),
					new LoadVariableToRegister(Register.R1, "Second"),
					new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
					new ReturnInstruction(Register.R2)
				},
				new[]
				{
					"has First Number",
					"has Second Number",
					"Calculate Number",
					"\tAdd(10, 5).Calculate is 15",
					"\tFirst + Second"
				});
			yield return new TestCaseData("AddOne(10, 5).Calculate", "AddOne",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(10), "First"),
					new StoreVariableInstruction(Number(5), "Second"),
					new LoadVariableToRegister(Register.R0, "First"),
					new LoadVariableToRegister(Register.R1, "Second"),
					new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
					new LoadConstantInstruction(Register.R3, Number(1.0)),
					new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3, Register.R4),
					new ReturnInstruction(Register.R4)
				},
				new[]
				{
					"has First Number",
					"has Second Number",
					"Calculate Number",
					"\tAddOne(10, 5).Calculate is 15",
					"\tFirst + Second + 1"
				});
			yield return new TestCaseData("Multiply(10).By(2)", "Multiply",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(10), "number"),
					new StoreVariableInstruction(Number(2), "multiplyBy"),
					new LoadVariableToRegister(Register.R0, "number"),
					new LoadVariableToRegister(Register.R1, "multiplyBy"),
					new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
					new ReturnInstruction(Register.R2)
				},
				new[]
				{
					"has number", "By(multiplyBy Number) Number", "\tMultiply(10).By(2) is 20",
					"\tnumber * multiplyBy"
				});
			yield return new TestCaseData("Bla(10).SomeFunction", "Bla",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(10), "number"),
					new StoreVariableInstruction(Number(5), "blaa"),
					new LoadVariableToRegister(Register.R0, "blaa"),
					new LoadVariableToRegister(Register.R1, "number"),
					new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
					new ReturnInstruction(Register.R2)
				}, new[] { "has number", "SomeFunction Number", "\tconstant blaa = 5", "\tblaa + number" });
			yield return new TestCaseData("SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"SimpleLoopExample",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(10), "number"),
					new StoreVariableInstruction(Number(1), "result"),
					new StoreVariableInstruction(Number(2), "multiplier"),
					new LoadVariableToRegister(Register.R0, "number"),
					new LoopBeginInstruction(Register.R0),
					new LoadVariableToRegister(Register.R1, "result"),
					new LoadVariableToRegister(Register.R2, "multiplier"),
					new BinaryInstruction(InstructionType.Multiply, Register.R1, Register.R2, Register.R3),
					new StoreFromRegisterInstruction(Register.R3, "result"),
					new LoopEndInstruction(7),
					new LoadVariableToRegister(Register.R4, "result"),
					new ReturnInstruction(Register.R4)
				}, (string[])
				[
					"has number",
					"GetMultiplicationOfNumbers Number",
					"\tmutable result = 1",
					"\tconstant multiplier = 2",
					"\tfor number",
					"\t\tresult = result * multiplier",
					"\tresult"
				]);
			yield return new TestCaseData("RemoveParentheses(\"some(thing)\").Remove",
				"RemoveParentheses",
				(Instruction[])
				[
					new StoreVariableInstruction(Text("some(thing)"), "text"),
					new StoreVariableInstruction(Text(""), "result"),
					new StoreVariableInstruction(Number(0), "count"),
					new LoadVariableToRegister(Register.R0, "text"),
					new LoopBeginInstruction(Register.R0),
					new LoadVariableToRegister(Register.R1, "value"),
					new LoadConstantInstruction(Register.R2, Text("(")),
					new BinaryInstruction(InstructionType.Equal, Register.R1, Register.R2),
					new JumpToId(InstructionType.JumpToIdIfFalse, 0),
					new LoadVariableToRegister(Register.R3, "count"),
					new LoadConstantInstruction(Register.R4, Number(1)),
					new BinaryInstruction(InstructionType.Add, Register.R3, Register.R4, Register.R5),
					new StoreFromRegisterInstruction(Register.R5, "count"),
					new JumpToId(InstructionType.JumpEnd, 0),
					new LoadVariableToRegister(Register.R6, "count"),
					new LoadConstantInstruction(Register.R7, Number(0)),
					new BinaryInstruction(InstructionType.Equal, Register.R6, Register.R7),
					new JumpToId(InstructionType.JumpToIdIfFalse, 1),
					new LoadVariableToRegister(Register.R8, "result"),
					new LoadVariableToRegister(Register.R9, "value"),
					new BinaryInstruction(InstructionType.Add, Register.R8, Register.R9, Register.R10),
					new StoreFromRegisterInstruction(Register.R10, "result"),
					new JumpToId(InstructionType.JumpEnd, 1),
					new LoadVariableToRegister(Register.R11, "value"),
					new LoadConstantInstruction(Register.R12, Text(")")),
					new BinaryInstruction(InstructionType.Equal, Register.R11, Register.R12),
					new JumpToId(InstructionType.JumpToIdIfFalse, 2),
					new LoadVariableToRegister(Register.R13, "count"),
					new LoadConstantInstruction(Register.R14, Number(1)),
					new BinaryInstruction(InstructionType.Subtract, Register.R13, Register.R14, Register.R15),
					new StoreFromRegisterInstruction(Register.R15, "count"),
					new JumpToId(InstructionType.JumpEnd, 2),
					new LoopEndInstruction(29),
					new LoadVariableToRegister(Register.R0, "result"),
					new ReturnInstruction(Register.R0)
				],
				(string[])
				[
					"has text",
					"Remove Text",
					"\tmutable result = \"\"",
					"\tmutable count = 0",
					"\tfor text",
					"\t\tif value is \"(\"",
					"\t\t\tcount = count + 1",
					"\t\tif count is 0",
					"\t\t\tresult = result + value",
					"\t\tif value is \")\"",
					"\t\t\tcount = count - 1",
					"\tresult"
				]);
			yield return new TestCaseData("ArithmeticFunction(10, 5).Calculate(\"add\")",
				"ArithmeticFunction", (Instruction[])
				[
					new StoreVariableInstruction(Number(10), "First"),
					new StoreVariableInstruction(Number(5), "Second"),
					new StoreVariableInstruction(Text("add"), "operation"),
					new LoadVariableToRegister(Register.R0, "operation"),
					new LoadConstantInstruction(Register.R1, Text("add")),
					new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
					new JumpToId(InstructionType.JumpToIdIfFalse, 0),
					new LoadVariableToRegister(Register.R2, "First"),
					new LoadVariableToRegister(Register.R3, "Second"),
					new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3, Register.R4),
					new ReturnInstruction(Register.R4), new JumpToId(InstructionType.JumpEnd, 0),
					new LoadVariableToRegister(Register.R5, "operation"),
					new LoadConstantInstruction(Register.R6, Text("subtract")),
					new BinaryInstruction(InstructionType.Equal, Register.R5, Register.R6),
					new JumpToId(InstructionType.JumpToIdIfFalse, 1),
					new LoadVariableToRegister(Register.R7, "First"),
					new LoadVariableToRegister(Register.R8, "Second"),
					new BinaryInstruction(InstructionType.Subtract, Register.R7, Register.R8, Register.R9),
					new ReturnInstruction(Register.R9), new JumpToId(InstructionType.JumpEnd, 1),
					new LoadVariableToRegister(Register.R10, "operation"),
					new LoadConstantInstruction(Register.R11, Text("multiply")),
					new BinaryInstruction(InstructionType.Equal, Register.R10, Register.R11),
					new JumpToId(InstructionType.JumpToIdIfFalse, 2),
					new LoadVariableToRegister(Register.R12, "First"),
					new LoadVariableToRegister(Register.R13, "Second"),
					new BinaryInstruction(InstructionType.Multiply, Register.R12, Register.R13, Register.R14),
					new ReturnInstruction(Register.R14), new JumpToId(InstructionType.JumpEnd, 2),
					new LoadVariableToRegister(Register.R15, "operation"),
					new LoadConstantInstruction(Register.R0, Text("divide")),
					new BinaryInstruction(InstructionType.Equal, Register.R15, Register.R0),
					new JumpToId(InstructionType.JumpToIdIfFalse, 3),
					new LoadVariableToRegister(Register.R1, "First"),
					new LoadVariableToRegister(Register.R2, "Second"),
					new BinaryInstruction(InstructionType.Divide, Register.R1, Register.R2, Register.R3),
					new ReturnInstruction(Register.R3), new JumpToId(InstructionType.JumpEnd, 3)
				],
				(string[])
				[
					"has First Number",
					"has Second Number",
					"Calculate(operation Text) Number",
					"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
					"\tArithmeticFunction(10, 5).Calculate(\"subtract\") is 5",
					"\tArithmeticFunction(10, 5).Calculate(\"multiply\") is 50",
					"\tif operation is \"add\"",
					"\t\treturn First + Second",
					"\tif operation is \"subtract\"",
					"\t\treturn First - Second",
					"\tif operation is \"multiply\"",
					"\t\treturn First * Second",
					"\tif operation is \"divide\"",
					"\t\treturn First / Second"
				]);
			yield return new TestCaseData("SimpleListDeclaration(5).Declare",
				"SimpleListDeclaration", (Instruction[])
				[
					new StoreVariableInstruction(Number(5), "number"),
					new LoadConstantInstruction(Register.R0,
						new ValueInstance(ListType.GetGenericImplementation(NumberType),
							[Number(1), Number(2), Number(3), Number(4), Number(5)])),
					new ReturnInstruction(Register.R0)
				],
				(string[])
				[
					"has number", "Declare Numbers", "\t(1, 2, 3, 4, 5)"
				]);
			yield return new TestCaseData("Invertor((1, 2, 3, 4)).Invert",
				"Invertor", (Instruction[])
				[
					new StoreVariableInstruction(
						new ValueInstance(ListType.GetGenericImplementation(NumberType),
							[Number(1), Number(2), Number(3), Number(4)]), "numbers"),
					new StoreVariableInstruction(Text(""), "result"),
					new LoadVariableToRegister(Register.R0, "numbers"),
					new LoopBeginInstruction(Register.R0),
					new LoadVariableToRegister(Register.R1, "value"),
					new LoadConstantInstruction(Register.R2, Number(-1)),
					new BinaryInstruction(InstructionType.Multiply, Register.R1, Register.R2, Register.R3),
					new LoadVariableToRegister(Register.R4, "result"),
					new BinaryInstruction(InstructionType.Add, Register.R4, Register.R3, Register.R5),
					new StoreFromRegisterInstruction(Register.R5, "result"),
					new LoopEndInstruction(8),
					new LoadVariableToRegister(Register.R6, "result"),
					new ReturnInstruction(Register.R6)
				],
				(string[])
				[
					"has numbers",
					"Invert Text",
					"\tmutable result = \"\"",
					"\tfor numbers",
					"\t\tresult = result + value * -1",
					"\tresult"
				]);
			yield return new TestCaseData("AddNumbers(2, 5).GetSum",
				"AddNumbers", (Instruction[])
				[
					new StoreVariableInstruction(Number(2), "firstNumber"),
					new StoreVariableInstruction(Number(5), "secondNumber"),
					new Invoke(Register.R0, null!, null!),
					new ReturnInstruction(Register.R0)
				],
				(string[])
				[
					"has firstNumber Number",
					"has secondNumber Number",
					"GetSum Number",
					"\tSumNumbers(firstNumber, secondNumber)",
					"SumNumbers(fNumber Number, sNumber Number) Number",
					"\tfNumber + sNumber"
				]);
			yield return new TestCaseData("IfWithMethodCallLeft(5).Check", "IfWithMethodCallLeft",
				new Instruction[]
				{
					new StoreVariableInstruction(Number(5), "number"),
					new Invoke(Register.R0, null!, null!),
					new LoadConstantInstruction(Register.R1, Number(0)),
					new BinaryInstruction(InstructionType.GreaterThan, Register.R0, Register.R1),
					new JumpToId(InstructionType.JumpToIdIfFalse, 0),
					new LoadConstantInstruction(Register.R2,
						new ValueInstance(TestPackage.Instance.GetType(Type.Boolean), 1)),
					new ReturnInstruction(Register.R2), new JumpToId(InstructionType.JumpEnd, 0),
					new LoadConstantInstruction(Register.R3,
						new ValueInstance(TestPackage.Instance.GetType(Type.Boolean), 0)),
					new ReturnInstruction(Register.R3)
				},
				new[]
				{
					"has number", "Check Boolean", "\tif GetValue > 0", "\t\treturn true", "\tfalse",
					"GetValue Number", "\tnumber + 1"
				});
		}
	}
}