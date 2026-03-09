global using Type = Strict.Language.Type;
using Strict.Runtime.Instructions;

namespace Strict.Runtime.Tests;

public class BaseVirtualMachineTests : TestExpressions
{
	//ncrunch: no coverage start
	protected static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);
	protected static readonly Type ListType = TestPackage.Instance.GetType(Type.List);
	protected static ValueInstance Number(double value) => new(NumberType, value);
	protected static ValueInstance Text(string value) => new(value);
	protected static readonly string[] ArithmeticFunctionExample =
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
	];
	protected static readonly Instruction[] ExpectedInstructionsOfArithmeticFunctionExample =
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
	];
	protected static readonly string[] SimpleLoopExample =
	[
		"has number",
		"GetMultiplicationOfNumbers Number",
		"\tmutable result = 1",
		"\tconstant multiplier = 2",
		"\tfor number",
		"\t\tresult = result * multiplier",
		"\tresult"
	];
	protected static readonly string[] RemoveParenthesesKata =
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
	];
	protected static readonly string[] IfAndElseTestCode =
	[
		"has number",
		"IsEven Text",
		"\tmutable result = \"\"",
		"\tif number > 10",
		"\t\tresult = \"Number is more than 10\"",
		"\t\treturn result", "\telse",
		"\t\tresult = \"Number is less or equal than 10\"",
		"\t\treturn result"
	];
	protected static readonly string[] SimpleMethodCallCode =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tSumNumbers(firstNumber, secondNumber)",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tfNumber + sNumber"
	];
	protected static readonly string[] CurrentlyFailingTest =
	[
		"has number",
		"SumEvenNumbers Number",
		"\tComputeSum",
		"ComputeSum Number",
		"\tmutable sum = 0",
		"\tfor number",
		"\t\tif index % 2 is 0",
		"\t\t\tsum = sum + index",
		"\tsum"
	];
	protected static readonly string[] MethodCallWithConstantValues =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tSumNumbers(5, 1)",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tfNumber + sNumber"
	];
	protected static readonly string[] MethodCallWithLocalVariables =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tconstant five = 5",
		"\tconstant six = 6",
		"\tSumNumbers(five, six)",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tfNumber + sNumber"
	];
	protected static readonly string[] MethodCallWithLocalWithNoArguments =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tSumNumbers",
		"SumNumbers Number",
		"\t10 + 532"
	];
	protected static readonly Instruction[] ExpectedSimpleMethodCallCode =
	[
		new StoreVariableInstruction(Number(2), "firstNumber"),
		new StoreVariableInstruction(Number(5), "secondNumber"),
		new Invoke(Register.R0, null!, null!),
		new ReturnInstruction(Register.R0)
	];
	protected static readonly Instruction[] ExpectedInstructionsOfRemoveParenthesesKata =
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
	];
	protected static readonly string[] SimpleListDeclarationExample =
	[
		"has number", "Declare Numbers", "\t(1, 2, 3, 4, 5)"
	];
	protected static readonly Instruction[] ExpectedInstructionsOfSimpleListDeclaration =
	[
		new StoreVariableInstruction(Number(5), "number"),
		new LoadConstantInstruction(Register.R0,
			new ValueInstance(ListType.GetGenericImplementation(NumberType),
				[Number(1), Number(2), Number(3), Number(4), Number(5)])),
		new ReturnInstruction(Register.R0)
	];
	protected static readonly string[] InvertValueKata =
	[
		"has numbers",
		"Invert Text",
		"\tmutable result = \"\"",
		"\tfor numbers",
		"\t\tresult = result + value * -1",
		"\tresult"
	];
	//TODO: remove if unused!
	protected static readonly Instruction[] ExpectedInstructionsOfInvertValueKata =
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
	];

	protected MethodCall GenerateMethodCallFromSource(string programName, string methodCall,
		params string[] source)
	{
		if (type.Package.FindDirectType(programName) == null)
			new Type(type.Package, new TypeLines(programName, source)).ParseMembersAndMethods(
				new MethodExpressionParser());
		return (MethodCall)ParseExpression(methodCall);
	}
}