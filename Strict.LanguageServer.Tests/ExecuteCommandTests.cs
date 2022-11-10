using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;

namespace Strict.LanguageServer.Tests;

public class ExecuteCommandTests : LanguageServerTests
{
	private static readonly DocumentUri ArithmeticFunction = new("", "", "ArithmeticFunction.strict", "", "");

	[SetUp]
	public void ArithmeticFunctionSetup() =>
		handler.Document.AddOrUpdate(ArithmeticFunction,
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
			"\t\treturn First / Second");
}

