using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

public class BaseVirtualMachineTests : TestExpressions
{
	//ncrunch: no coverage start
	protected static readonly Type NumberType = new TestPackage().FindType(Base.Number)!;
	protected static readonly Type TextType = new TestPackage().FindType(Base.Text)!;
	protected static readonly Type ListType = new TestPackage().FindType(Base.List)!;
	protected static readonly string[] ArithmeticFunctionExample =
	[
		"has First Number", "has Second Number", "Calculate(operation Text) Number",
		"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
		"\tArithmeticFunction(10, 5).Calculate(\"subtract\") is 5",
		"\tArithmeticFunction(10, 5).Calculate(\"multiply\") is 50", "\tif operation is \"add\"",
		"\t\treturn First + Second", "\tif operation is \"subtract\"", "\t\treturn First - Second",
		"\tif operation is \"multiply\"", "\t\treturn First * Second",
		"\tif operation is \"divide\"", "\t\treturn First / Second"
	];
	protected static readonly Statement[] ExpectedStatementsOfArithmeticFunctionExample =
	[
		new StoreVariableStatement(new Instance(NumberType, 10), "First"),
		new StoreVariableStatement(new Instance(NumberType, 5), "Second"),
		new StoreVariableStatement(new Instance(TextType, "add"), "operation"),
		new LoadVariableStatement(Register.R0, "operation"),
		new LoadConstantStatement(Register.R1, new Instance(TextType, "add")),
		new BinaryStatement(Instruction.Equal, Register.R0, Register.R1),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 0),
		new LoadVariableStatement(Register.R2, "First"),
		new LoadVariableStatement(Register.R3, "Second"),
		new BinaryStatement(Instruction.Add, Register.R2, Register.R3, Register.R4),
		new ReturnStatement(Register.R4), new JumpToIdStatement(Instruction.JumpEnd, 0),
		new LoadVariableStatement(Register.R5, "operation"),
		new LoadConstantStatement(Register.R6, new Instance(TextType, "subtract")),
		new BinaryStatement(Instruction.Equal, Register.R5, Register.R6),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableStatement(Register.R7, "First"),
		new LoadVariableStatement(Register.R8, "Second"),
		new BinaryStatement(Instruction.Subtract, Register.R7, Register.R8, Register.R9),
		new ReturnStatement(Register.R9), new JumpToIdStatement(Instruction.JumpEnd, 1),
		new LoadVariableStatement(Register.R0, "operation"),
		new LoadConstantStatement(Register.R1, new Instance(TextType, "multiply")),
		new BinaryStatement(Instruction.Equal, Register.R0, Register.R1),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableStatement(Register.R2, "First"),
		new LoadVariableStatement(Register.R3, "Second"),
		new BinaryStatement(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
		new ReturnStatement(Register.R4), new JumpToIdStatement(Instruction.JumpEnd, 2),
		new LoadVariableStatement(Register.R5, "operation"),
		new LoadConstantStatement(Register.R6, new Instance(TextType, "divide")),
		new BinaryStatement(Instruction.Equal, Register.R5, Register.R6),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 3),
		new LoadVariableStatement(Register.R7, "First"),
		new LoadVariableStatement(Register.R8, "Second"),
		new BinaryStatement(Instruction.Divide, Register.R7, Register.R8, Register.R9),
		new ReturnStatement(Register.R9), new JumpToIdStatement(Instruction.JumpEnd, 3)
	];
	protected static readonly string[] SimpleLoopExample =
	[
		"has number", "GetMultiplicationOfNumbers Number",
		"\tmutable result = 1",
		"\tconstant multiplier = 2",
		"\tfor number",
		"\t\tresult = result * multiplier",
		"\tresult"
	];
	protected static readonly string[] RemoveParenthesesKata =
	[
		"has text", "Remove Text",
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
		"has number", "IsEven Text", "\tmutable result = \"\"", "\tif number > 10",
		"\t\tresult = \"Number is more than 10\"", "\t\treturn result", "\telse",
		"\t\tresult = \"Number is less or equal than 10\"", "\t\treturn result"
	];
	protected static readonly string[] SimpleMethodCallCode =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tconstant result = SumNumbers(firstNumber, secondNumber)",
		"\tresult",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tconstant result = fNumber + sNumber",
		"\tresult"
	];
	protected static readonly string[] CurrentlyFailingTest =
	[
		"has number",
		"SumEvenNumbers Number",
		"\tconstant result = ComputeSum",
		"\tresult",
		"ComputeSum Number",
		"\tmutable sum = 0",
		"\tfor number",
		"\t\tif (index % 2) is 0",
		"\t\t\tsum = sum + index",
		"\tsum"
	];
	protected static readonly string[] MethodCallWithConstantValues =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tconstant result = SumNumbers(5, 1)",
		"\tresult",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tconstant result = fNumber + sNumber",
		"\tresult"
	];
	protected static readonly string[] MethodCallWithLocalVariables =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tconstant five = 5",
		"\tconstant six = 6",
		"\tconstant result = SumNumbers(five, six)",
		"\tresult",
		"SumNumbers(fNumber Number, sNumber Number) Number",
		"\tconstant result = fNumber + sNumber",
		"\tresult"
	];
	protected static readonly string[] MethodCallWithLocalWithNoArguments =
	[
		"has firstNumber Number",
		"has secondNumber Number",
		"GetSum Number",
		"\tconstant result = SumNumbers",
		"\tresult",
		"SumNumbers Number",
		"\tconstant result = 10 + 532",
		"\tresult"
	];
	protected static readonly Statement[] ExpectedSimpleMethodCallCode =
	[
		new StoreVariableStatement(new Instance(NumberType, 2), "firstNumber"),
		new StoreVariableStatement(new Instance(NumberType, 5), "secondNumber"),
		new InvokeStatement("SumNumbers(firstNumber, secondNumber)", Register.R0),
		new StoreFromRegisterStatement(Register.R0, "result"),
		new LoadVariableStatement(Register.R1, "result"),
		new ReturnStatement(Register.R1)
	];
	protected static readonly Statement[] ExpectedStatementsOfRemoveParanthesesKata =
	[
		new StoreVariableStatement(new Instance(TextType, "some(thing)"), "text"),
		new StoreVariableStatement(new Instance(TextType, ""), "result"),
		new StoreVariableStatement(new Instance(NumberType, 0), "count"),
		new LoadVariableStatement(Register.R0, "text"),
		new LoopBeginStatement(Register.R0),
		new LoadVariableStatement(Register.R1, "value"),
		new LoadConstantStatement(Register.R2, new Instance(TextType, "(")),
		new BinaryStatement(Instruction.Equal, Register.R1, Register.R2),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 0),
		new LoadVariableStatement(Register.R3, "count"),
		new LoadConstantStatement(Register.R4, new Instance(NumberType, 1)),
		new BinaryStatement(Instruction.Add, Register.R3, Register.R4, Register.R5),
		new StoreFromRegisterStatement(Register.R5, "count"),
		new JumpToIdStatement(Instruction.JumpEnd, 0),
		new LoadVariableStatement(Register.R6, "count"),
		new LoadConstantStatement(Register.R7, new Instance(TextType, ")")),
		new BinaryStatement(Instruction.Equal, Register.R6, Register.R7),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableStatement(Register.R8, "result"),
		new LoadVariableStatement(Register.R9, "value"),
		new BinaryStatement(Instruction.Add, Register.R7, Register.R8, Register.R9),
		new StoreFromRegisterStatement(Register.R0, "result"),
		new JumpToIdStatement(Instruction.JumpEnd, 1),
		new LoadVariableStatement(Register.R1, "value"),
		new LoadConstantStatement(Register.R2, new Instance(NumberType, 0)),
		new BinaryStatement(Instruction.Equal, Register.R1, Register.R2),
		new JumpToIdStatement(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableStatement(Register.R3, "count"),
		new LoadConstantStatement(Register.R4, new Instance(NumberType, 0)),
		new BinaryStatement(Instruction.Subtract, Register.R3, Register.R4, Register.R5),
		new StoreFromRegisterStatement(Register.R5, "count"),
		new JumpToIdStatement(Instruction.JumpEnd, 2),
		new IterationEndStatement(29),
		new LoadVariableStatement(Register.R6, "result"),
		new ReturnStatement(Register.R6)
	];
	protected static readonly string[] SimpleListDeclarationExample =
	[
		"has number", "Declare Numbers", "\tconstant myList = (1, 2, 3, 4 ,5)", "\tmyList"
	];
	protected static readonly Statement[] ExpectedStatementsOfSimpleListDeclaration =
	[
		new StoreVariableStatement(new Instance(NumberType, 5), "number"),
		new StoreVariableStatement(
			new Instance(ListType,
				new List<Expression>
				{
					new Value(NumberType, 1),
					new Value(NumberType, 2),
					new Value(NumberType, 3),
					new Value(NumberType, 4),
					new Value(NumberType, 5)
				}), "myList"),
		new LoadVariableStatement(Register.R0, "myList"), new ReturnStatement(Register.R0)
	];
	protected static readonly string[] InvertValueKata =
	[
		"has numbers", "Invert Text",
		"\tmutable result = \"\"",
		"\tfor numbers",
		"\t\tresult = result + (0 - value)",
		"\tresult"
	];
	protected static readonly Statement[] ExpectedStatementsOfInvertValueKata =
	[
		new StoreVariableStatement(
			new Instance(ListType,
				new List<Expression>
				{
					new Value(NumberType, 1),
					new Value(NumberType, 2),
					new Value(NumberType, 3),
					new Value(NumberType, 4),
					new Value(NumberType, 5)
				}), "numbers"),
		new StoreVariableStatement(new Instance(TextType, ""), "result"),
		new LoadVariableStatement(Register.R0, "numbers"),
		new LoopBeginStatement(Register.R0),
		new LoadConstantStatement(Register.R1, new Instance(NumberType, 0)),
		new LoadVariableStatement(Register.R2, "value"),
		new BinaryStatement(Instruction.Subtract, Register.R1, Register.R2, Register.R3),
		new LoadVariableStatement(Register.R4, "result"),
		new BinaryStatement(Instruction.Add, Register.R4, Register.R3, Register.R5),
		new StoreFromRegisterStatement(Register.R5, "result"),
		new IterationEndStatement(8),
		new LoadVariableStatement(Register.R6, "result"),
		new ReturnStatement(Register.R6)
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