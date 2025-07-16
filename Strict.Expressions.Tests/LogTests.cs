using NUnit.Framework;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Expressions.Tests;

public sealed class LogTests
{
	[Test]
	public void PrintHelloWorld()
	{
		var package = new TestPackage();
		new Type(package, new TypeLines(Base.App, "Run"));
		var type =
			new Type(package,
					new TypeLines("Program", "has App", "has logger", "Run", "\tlogger.Log(\"Hello\")")).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(Run(type.Methods[0]), Is.EqualTo("Hello"));
	}

	public string Run(Method method)
	{
		if (method.GetBodyAndParseIfNeeded() is MethodCall call && call.Method.Name == "Write")
			return ((Text)call.Arguments[0]).Data.ToString()!;
		return ""; //ncrunch: no coverage
	}
}