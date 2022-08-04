using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public sealed class LogTests
{
	[Test]
	public void PrintHelloWorld()
	{
		var package = new TestPackage();
		new Type(package, new FileData(Base.App, new[] { "Run" }), null!);
		var type = new Type(package, new FileData("Program", @"implement App
has log
Run
	log.Write(""Hello"")".SplitLines()), new MethodExpressionParser());
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