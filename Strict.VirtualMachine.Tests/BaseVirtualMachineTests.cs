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
	protected static readonly string[] ArithmeticFunctionExample =
	{
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
	};
	protected static readonly string[] SimpleLoopExample =
	{
		"has number",
		"GetMultiplicationOfNumbers Number",
		"\tlet result = Mutable(1)",
		"\tlet multiplier = 2",
		"\tfor number",
		"\t\tresult = result * multiplier",
		"\tresult"
	};

	protected static readonly string[] RemoveParenthesesKata =
	{
		"has text",
		"Remove Text",
		"\tlet result = Mutable(\"\")",
		"\tlet count = Mutable(0)",
		"\tfor text",
		"\t\tif value is \"(\"",
		"\t\t\tcount = count + 1",
		"\t\tif value is \")\"",
		"\t\t\tcount = count - 1",
		"\t\tif count is 0",
		"\t\t\tresult = result + value",
		"\tresult"
	};
	//ncrunch: no coverage end

	protected MethodCall GenerateMethodCallFromSource(string programName, string methodCall, params string[] source)
	{
		if (type.Package.FindDirectType(programName) == null)
			new Type(type.Package, new TypeLines(programName, source)).ParseMembersAndMethods(
				new MethodExpressionParser());
		return (MethodCall)ParseExpression(methodCall);
	}
}