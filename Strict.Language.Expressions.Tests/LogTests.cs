using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public sealed class LogTests
{
	[Test]
	public void PrintHelloWorld()
	{
		var package = new TestPackage();
		new Type(package, new TypeLines(Base.App, "Run"));
		var type =
			new Type(package,
					new TypeLines("Program", "implement App", "has log", "Run", "\tlog.Write(\"Hello\")")).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(Run(type.Methods[0]), Is.EqualTo("Hello"));
	}

	public string Run(Method method)
	{
		foreach (var expression in method.Body.Expressions)
			if (expression is MethodCall call && call.Method.Name == "Write")
				return ((Text)call.Arguments[0]).Data.ToString()!;
		return ""; //ncrunch: no coverage
	}
}