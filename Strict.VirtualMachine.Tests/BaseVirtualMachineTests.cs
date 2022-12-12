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
	{
		"has First Number", "has Second Number", "Calculate(operation Text) Number",
		"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
		"\tArithmeticFunction(10, 5).Calculate(\"subtract\") is 5",
		"\tArithmeticFunction(10, 5).Calculate(\"multiply\") is 50", "\tif operation is \"add\"",
		"\t\treturn First + Second", "\tif operation is \"subtract\"", "\t\treturn First - Second",
		"\tif operation is \"multiply\"", "\t\treturn First * Second",
		"\tif operation is \"divide\"", "\t\treturn First / Second"
	};
	protected static readonly Statement[] ExpectedStatementsOfArithmeticFunctionExample =
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
		new(Instruction.Add, Register.R2, Register.R3, Register.R4),
		new ReturnStatement(Register.R4), new JumpViaIdStatement(Instruction.JumpEnd, 0),
		new LoadVariableStatement(Register.R5, "operation"),
		new LoadConstantStatement(Register.R6, new Instance(TextType, "subtract")),
		new(Instruction.Equal, Register.R5, Register.R6),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableStatement(Register.R7, "First"),
		new LoadVariableStatement(Register.R8, "Second"),
		new(Instruction.Subtract, Register.R7, Register.R8, Register.R9),
		new ReturnStatement(Register.R9), new JumpViaIdStatement(Instruction.JumpEnd, 1),
		new LoadVariableStatement(Register.R0, "operation"),
		new LoadConstantStatement(Register.R1, new Instance(TextType, "multiply")),
		new(Instruction.Equal, Register.R0, Register.R1),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableStatement(Register.R2, "First"),
		new LoadVariableStatement(Register.R3, "Second"),
		new(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
		new ReturnStatement(Register.R4), new JumpViaIdStatement(Instruction.JumpEnd, 2),
		new LoadVariableStatement(Register.R5, "operation"),
		new LoadConstantStatement(Register.R6, new Instance(TextType, "divide")),
		new(Instruction.Equal, Register.R5, Register.R6),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 3),
		new LoadVariableStatement(Register.R7, "First"),
		new LoadVariableStatement(Register.R8, "Second"),
		new(Instruction.Divide, Register.R7, Register.R8, Register.R9),
		new ReturnStatement(Register.R9), new JumpViaIdStatement(Instruction.JumpEnd, 3)
	};
	protected static readonly string[] SimpleLoopExample =
	{
		"has number",
		"GetMultiplicationOfNumbers Number",
		"\tmutable result = 1",
		"\tconstant multiplier = 2",
		"\tfor number",
		"\t\tresult = result * multiplier",
		"\tresult"
	};
	protected static readonly string[] RemoveParenthesesKata =
	{
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
	};
	protected static readonly Statement[] ExpectedStatementsOfRemoveParanthesesKata =
	{
		new StoreStatement(new Instance(TextType, "some(thing)"), "text"),
		new StoreStatement(new Instance(TextType, ""), "result"),
		new StoreStatement(new Instance(NumberType, 0), "count"),
		new LoadConstantStatement(Register.R0, new Instance(NumberType, 11)),
		new LoadConstantStatement(Register.R1, new Instance(NumberType, 1)),
		new InitLoopStatement("text"),
		new LoadVariableStatement(Register.R2, "value"),
		new LoadConstantStatement(Register.R3, new Instance(TextType, "(")),
		new(Instruction.Equal, Register.R2, Register.R3),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 0),
		new LoadVariableStatement(Register.R4, "count"),
		new LoadConstantStatement(Register.R5, new Instance(NumberType, 1)),
		new(Instruction.Add, Register.R4, Register.R5, Register.R6),
		new StoreFromRegisterStatement(Register.R6, "count"),
		new JumpViaIdStatement(Instruction.JumpEnd, 0),
		new LoadVariableStatement(Register.R7, "count"),
		new LoadConstantStatement(Register.R8, new Instance(TextType, ")")),
		new(Instruction.Equal, Register.R7, Register.R8),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 1),
		new LoadVariableStatement(Register.R9, "result"),
		new LoadVariableStatement(Register.R2, "value"),
		new(Instruction.Add, Register.R9, Register.R2, Register.R3),
		new StoreFromRegisterStatement(Register.R3, "result"),
		new JumpViaIdStatement(Instruction.JumpEnd, 1),
		new LoadVariableStatement(Register.R4, "value"),
		new LoadConstantStatement(Register.R5, new Instance(NumberType, 0)),
		new(Instruction.Equal, Register.R4, Register.R5),
		new JumpViaIdStatement(Instruction.JumpToIdIfFalse, 2),
		new LoadVariableStatement(Register.R6, "count"),
		new LoadConstantStatement(Register.R7, new Instance(NumberType, 0)),
		new(Instruction.Subtract, Register.R6, Register.R7, Register.R8),
		new StoreFromRegisterStatement(Register.R8, "count"),
		new JumpViaIdStatement(Instruction.JumpEnd, 2),
		new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
		new JumpStatement(Instruction.JumpIfNotZero, -30),
		new LoadVariableStatement(Register.R9, "result"),
		new ReturnStatement(Register.R9)
	};
	protected static readonly string[] SimpleListDeclarationExample =
	{
		"has number",
		"Declare Numbers",
		"\tconstant myList = (1, 2, 3, 4 ,5)",
		"\tmyList"
	};
	protected static readonly Statement[] ExpectedStatementsOfSimpleListDeclaration =
	{
		new StoreStatement(new Instance(NumberType, 5), "number"),
		new StoreStatement(
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
	};
	protected static readonly string[] InvertValueKata =
	{
		"has numbers",
		"Invert Text",
		"\tmutable result = \"\"",
		"\tfor numbers",
		"\t\tresult = result + (0 - value)",
		"\tresult"
	};
	protected static readonly Statement[] ExpectedStatementsOfInvertValueKata =
	{
		new StoreStatement(new Instance(ListType, new List<Expression>
		{
			new Value(NumberType, 1),
			new Value(NumberType, 2),
			new Value(NumberType, 3),
			new Value(NumberType, 4),
			new Value(NumberType, 5)
		}), "numbers"),
		new StoreStatement(new Instance(TextType, ""), "result"),
		new LoadConstantStatement(Register.R0, new Instance(NumberType, 4)),
		new LoadConstantStatement(Register.R1, new Instance(NumberType, 1)),
		new InitLoopStatement("numbers"),
		new LoadConstantStatement(Register.R2, new Instance(NumberType, 0)),
		new LoadVariableStatement(Register.R3, "value"),
		new(Instruction.Subtract, Register.R2, Register.R3, Register.R4),
		new LoadVariableStatement(Register.R5, "result"),
		new(Instruction.Add, Register.R5, Register.R4, Register.R6),
		new StoreFromRegisterStatement(Register.R6, "result"),
		new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
		new JumpStatement(Instruction.JumpIfNotZero, -9),
		new LoadVariableStatement(Register.R7, "result"),
		new ReturnStatement(Register.R7)
	};

	protected MethodCall GenerateMethodCallFromSource(string programName, string methodCall,
		params string[] source)
	{
		if (type.Package.FindDirectType(programName) == null)
			new Type(type.Package, new TypeLines(programName, source)).ParseMembersAndMethods(
				new MethodExpressionParser());
		return (MethodCall)ParseExpression(methodCall);
	}
}