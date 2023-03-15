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
					new StoreVariableStatement(new Instance(NumberType, 5), "number"),
					new StoreVariableStatement(new Instance(NumberType, 5), "bla"),
					new LoadVariableStatement(Register.R0, "bla"),
					new LoadConstantStatement(Register.R1, new Instance(NumberType, 5)),
					new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new StoreFromRegisterStatement(Register.R2, "something"),
					new LoadVariableStatement(Register.R3, "something"),
					new LoadConstantStatement(Register.R4, new Instance(NumberType, 5)),
					new BinaryStatement(Instruction.Add, Register.R3, Register.R4, Register.R5),
					new ReturnStatement(Register.R5)
				},
				new[]
				{
					"has number",
					"Assign Number",
					"\tconstant bla = 5",
					"\tconstant something = bla + 5",
					"\tsomething + 10"
				});
			yield return new TestCaseData("Add(10, 5).Calculate", "Add",
				new Statement[]
				{
					new StoreVariableStatement(new Instance(NumberType, 10), "First"),
					new StoreVariableStatement(new Instance(NumberType, 5), "Second"),
					new LoadVariableStatement(Register.R0, "First"),
					new LoadVariableStatement(Register.R1, "Second"),
					new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2),
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
			yield return new TestCaseData("AddOne(10, 5).Calculate", "AddOne",
				new Statement[]
				{
					new StoreVariableStatement(new Instance(NumberType, 10), "First"),
					new StoreVariableStatement(new Instance(NumberType, 5), "Second"),
					new LoadVariableStatement(Register.R0, "First"),
					new LoadVariableStatement(Register.R1, "Second"),
					new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new BinaryStatement(Instruction.Add, Register.R2, Register.R3, Register.R4),
					new ReturnStatement(Register.R4)
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
				new Statement[]
				{
					new StoreVariableStatement(new Instance(NumberType, 10), "number"),
					new StoreVariableStatement(new Instance(NumberType, 2), "multiplyBy"),
					new LoadVariableStatement(Register.R0, "number"),
					new LoadVariableStatement(Register.R1, "multiplyBy"),
					new BinaryStatement(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
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
					new StoreVariableStatement(new Instance(NumberType, 10), "number"),
					new StoreVariableStatement(new Instance(NumberType, 5), "blaa"),
					new LoadVariableStatement(Register.R0, "blaa"),
					new LoadVariableStatement(Register.R1, "number"),
					new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2),
					new ReturnStatement(Register.R2)
				}, new[] { "has number", "SomeFunction Number", "\tconstant blaa = 5", "\tblaa + number" });
			yield return new TestCaseData("SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"SimpleLoopExample",
				new Statement[]
				{
					new StoreVariableStatement(new Instance(NumberType, 10), "number"),
					new StoreVariableStatement(new Instance(NumberType, 1), "result"),
					new StoreVariableStatement(new Instance(NumberType, 2), "multiplier"),
					new LoopBeginStatement("number"),
					new LoadVariableStatement(Register.R0, "result"),
					new LoadVariableStatement(Register.R1, "multiplier"),
					new BinaryStatement(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
					new StoreFromRegisterStatement(Register.R2, "result"),
					new IterationEndStatement(7),
					new LoadVariableStatement(Register.R3, "result"),
					new ReturnStatement(Register.R3)
				}, SimpleLoopExample);
			yield return new TestCaseData("RemoveParentheses(\"some(thing)\").Remove",
				"RemoveParentheses",
				ExpectedStatementsOfRemoveParanthesesKata,
				RemoveParenthesesKata);
			yield return new TestCaseData("ArithmeticFunction(10, 5).Calculate(\"add\")",
				"ArithmeticFunction", ExpectedStatementsOfArithmeticFunctionExample,
				ArithmeticFunctionExample);
			yield return new TestCaseData("SimpleListDeclaration(5).Declare",
				"SimpleListDeclaration", ExpectedStatementsOfSimpleListDeclaration,
				SimpleListDeclarationExample);
			yield return new TestCaseData("Invertor((1, 2, 3, 4)).Invert",
				"Invertor", ExpectedStatementsOfInvertValueKata,
				InvertValueKata);
			yield return new TestCaseData("AddNumbers(2, 5).GetSum",
				"AddNumbers", ExpectedSimpleMethodCallCode,
				SimpleMethodCallCode);
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