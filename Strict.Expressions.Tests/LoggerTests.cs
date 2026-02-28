using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class LoggerTests
{
	[Test]
	public void PrintHelloWorld()
	{
		using var app = new Type(TestPackage.Instance, new TypeLines(Base.App, "Run"));
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(PrintHelloWorld), "has App", "has logger", "Run",
				"\tlogger.Log(\"Hello\")")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(Run(type.Methods[0]), Is.EqualTo("Hello"));
	}

	public string Run(Method method)
	{
		if (method.GetBodyAndParseIfNeeded() is not MethodCall call || call.Method.Name != "Log")
			return ""; //ncrunch: no coverage
		var text = (Text)call.Arguments[0];
		var quoted = text.Data.ToExpressionCodeString();
		return quoted.Length >= 2
			? quoted[1..^1]
			: quoted;
	}
}