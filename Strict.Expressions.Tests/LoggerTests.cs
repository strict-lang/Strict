using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class LoggerTests
{
	[Test]
	public void PrintHelloWorld()
	{
		new Type(TestPackage.Instance, new TypeLines(Base.App, "Run"));
		var type =
			new Type(TestPackage.Instance,
					new TypeLines("Program", "has App", "has logger", "Run", "\tlogger.Log(\"Hello\")")).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(Run(type.Methods[0]), Is.EqualTo("Hello"));
	}

	public string Run(Method method)
	{
		if (method.GetBodyAndParseIfNeeded() is MethodCall call && call.Method.Name == "Log")
		{
			var text = call.Arguments[0] as Text;
			if (text == null)
				return ""; //ncrunch: no coverage
			var quoted = text.Data.ToExpressionCodeString();
			return quoted.Length >= 2 ? quoted[1..^1] : quoted;
		}
		return "";
	}
}