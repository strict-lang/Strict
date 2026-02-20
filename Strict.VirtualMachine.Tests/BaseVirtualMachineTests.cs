global using Type = Strict.Language.Type;

namespace Strict.Runtime.Tests;

public class BaseVirtualMachineTests : TestExpressions
{
	//ncrunch: no coverage start
	protected static readonly Type NumberType = TestPackage.Instance.FindType(Base.Number)!;
	protected static readonly Type TextType = TestPackage.Instance.FindType(Base.Text)!;
	protected static readonly Type ListType = TestPackage.Instance.FindType(Base.List)!;
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
	protected static readonly Statement[] ExpectedStatementsOfArithmeticFunctionExample =
	[
		new StoreVariableStatement(new Instance(NumberType, 10), "First"),
		new StoreVariableStatement(new Instance(NumberType, 5), "Second"),
		new StoreVariableStatement(new Instance(TextType, "add"), "operation"),
		new LoadVariableToRegister(Register.R0, "operation"),
		new LoadConstantStatement(Register.R1, new Instance(TextType, "add")),
		new Binary(Instruction.Equal, Register.R0, Register.R1),
		new JumpToId(Instruction.JumpToIdIfFalse, 0),
		new LoadVariableToRegister(Register.R2, "First"),
		new LoadVariableToRegister(Register.R3, "Second"),
		new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
		new Return(Register.R4), new JumpToId(Instruction.JumpEnd, 0),
		new LoadVariableToRegister(Register.R5, "operation"),
		new LoadConstantStatement(Register.R6, new Instance(TextType, "subtract")),
		new Binary(Instruction.Equal, Register.R5, Register.R6),
		new JumpToId(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableToRegister(Register.R7, "First"),
		new LoadVariableToRegister(Register.R8, "Second"),
		new Binary(Instruction.Subtract, Register.R7, Register.R8, Register.R9),
		new Return(Register.R9), new JumpToId(Instruction.JumpEnd, 1),
		new LoadVariableToRegister(Register.R10, "operation"),
		new LoadConstantStatement(Register.R11, new Instance(TextType, "multiply")),
		new Binary(Instruction.Equal, Register.R10, Register.R11),
		new JumpToId(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableToRegister(Register.R12, "First"),
		new LoadVariableToRegister(Register.R13, "Second"),
		new Binary(Instruction.Multiply, Register.R12, Register.R13, Register.R14),
		new Return(Register.R14), new JumpToId(Instruction.JumpEnd, 2),
		new LoadVariableToRegister(Register.R15, "operation"),
		new LoadConstantStatement(Register.R0, new Instance(TextType, "divide")),
		new Binary(Instruction.Equal, Register.R15, Register.R0),
		new JumpToId(Instruction.JumpToIdIfFalse, 3),
		new LoadVariableToRegister(Register.R1, "First"),
		new LoadVariableToRegister(Register.R2, "Second"),
		new Binary(Instruction.Divide, Register.R1, Register.R2, Register.R3),
		new Return(Register.R3), new JumpToId(Instruction.JumpEnd, 3)
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
	protected static readonly Statement[] ExpectedSimpleMethodCallCode =
	[
		new StoreVariableStatement(new Instance(NumberType, 2), "firstNumber"),
		new StoreVariableStatement(new Instance(NumberType, 5), "secondNumber"),
		new Invoke("SumNumbers(firstNumber, secondNumber)", Register.R0),
		new Return(Register.R0)
	];
	protected static readonly Statement[] ExpectedStatementsOfRemoveParenthesesKata =
	[
		new StoreVariableStatement(new Instance(TextType, "some(thing)"), "text"),
		new StoreVariableStatement(new Instance(TextType, ""), "result"),
		new StoreVariableStatement(new Instance(NumberType, 0), "count"),
		new LoadVariableToRegister(Register.R0, "text"),
		new LoopBeginStatement(Register.R0),
		new LoadVariableToRegister(Register.R1, "value"),
		new LoadConstantStatement(Register.R2, new Instance(TextType, "(")),
		new Binary(Instruction.Equal, Register.R1, Register.R2),
		new JumpToId(Instruction.JumpToIdIfFalse, 0),
		new LoadVariableToRegister(Register.R3, "count"),
		new LoadConstantStatement(Register.R4, new Instance(NumberType, 1)),
		new Binary(Instruction.Add, Register.R3, Register.R4, Register.R5),
		new StoreFromRegisterStatement(Register.R5, "count"),
		new JumpToId(Instruction.JumpEnd, 0),
		new LoadVariableToRegister(Register.R6, "count"),
		new LoadConstantStatement(Register.R7, new Instance(NumberType, 0)),
		new Binary(Instruction.Equal, Register.R6, Register.R7),
		new JumpToId(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableToRegister(Register.R8, "result"),
		new LoadVariableToRegister(Register.R9, "value"),
		new Binary(Instruction.Add, Register.R8, Register.R9, Register.R10),
		new StoreFromRegisterStatement(Register.R10, "result"),
		new JumpToId(Instruction.JumpEnd, 1),
		new LoadVariableToRegister(Register.R11, "value"),
		new LoadConstantStatement(Register.R12, new Instance(TextType, ")")),
		new Binary(Instruction.Equal, Register.R11, Register.R12),
		new JumpToId(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableToRegister(Register.R13, "count"),
		new LoadConstantStatement(Register.R14, new Instance(NumberType, 1)),
		new Binary(Instruction.Subtract, Register.R13, Register.R14, Register.R15),
		new StoreFromRegisterStatement(Register.R15, "count"),
		new JumpToId(Instruction.JumpEnd, 2),
		new IterationEnd(29),
		new LoadVariableToRegister(Register.R0, "result"),
		new Return(Register.R0)
	];
	protected static readonly string[] SimpleListDeclarationExample =
	[
		"has number", "Declare Numbers", "\t(1, 2, 3, 4, 5)"
	];
	protected static readonly Statement[] ExpectedStatementsOfSimpleListDeclaration =
	[
		new StoreVariableStatement(new Instance(NumberType, 5), "number"),
		new LoadConstantStatement(Register.R0, new Instance(ListType,
			new List<Expression>
			{
				new Value(NumberType, 1),
				new Value(NumberType, 2),
				new Value(NumberType, 3),
				new Value(NumberType, 4),
				new Value(NumberType, 5)
			})),
		new Return(Register.R0)
	];
	protected static readonly string[] InvertValueKata =
	[
		"has numbers",
		"Invert Text",
		"\tmutable result = \"\"",
		"\tfor numbers",
		"\t\tresult = result + \"-\" + value to Text",
		"\tresult"
	];
	protected static readonly Statement[] ExpectedStatementsOfInvertValueKata =
	[
		new StoreVariableStatement(
			new Instance(ListType,
				new List<Expression>
				{
					new Value(NumberType, 1.0),
					new Value(NumberType, 2.0),
					new Value(NumberType, 3.0),
					new Value(NumberType, 4.0)
				}), "numbers"),
		new StoreVariableStatement(new Instance(TextType, ""), "result"),
		new LoadVariableToRegister(Register.R0, "numbers"),
		new LoopBeginStatement(Register.R0),
		new LoadVariableToRegister(Register.R1, "result"),
		new LoadConstantStatement(Register.R2, new Instance(TextType, "-")),
		new Binary(Instruction.Add, Register.R1, Register.R2, Register.R3),
		new LoadVariableToRegister(Register.R4, "value"),
		new Conversion(Register.R4, Register.R5, TextType, Instruction.ToText),
		new Binary(Instruction.Add, Register.R3, Register.R5, Register.R6),
		new StoreFromRegisterStatement(Register.R6, "result"),
		new IterationEnd(9),
		new LoadVariableToRegister(Register.R7, "result"),
		new Return(Register.R7)
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