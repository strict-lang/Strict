using Strict.Runtime.Instructions;

namespace Strict.Runtime.Tests;

public sealed class ByteCodeGeneratorTests : BaseVirtualMachineTests
{
	[TestCaseSource(nameof(ByteCodeCases))]
	public void Generate(string methodCall, string programName, Instruction[] expectedByteCode,
		params string[] code)
	{
		var instructions =
			new ByteCodeGenerator(GenerateMethodCallFromSource(programName, methodCall, code)).
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
				}, SimpleLoopExample);
			yield return new TestCaseData("RemoveParentheses(\"some(thing)\").Remove",
				"RemoveParentheses",
				ExpectedInstructionsOfRemoveParenthesesKata,
				RemoveParenthesesKata);
			yield return new TestCaseData("ArithmeticFunction(10, 5).Calculate(\"add\")",
				"ArithmeticFunction", ExpectedInstructionsOfArithmeticFunctionExample,
				ArithmeticFunctionExample);
			yield return new TestCaseData("SimpleListDeclaration(5).Declare",
				"SimpleListDeclaration", ExpectedInstructionsOfSimpleListDeclaration,
				SimpleListDeclarationExample);
			yield return new TestCaseData("Invertor((1, 2, 3, 4)).Invert",
				"Invertor", ExpectedInstructionsOfInvertValueKata,
				InvertValueKata);
			yield return new TestCaseData("AddNumbers(2, 5).GetSum",
				"AddNumbers", ExpectedSimpleMethodCallCode,
				SimpleMethodCallCode);
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