using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public sealed class LogTests
{
	[Test]
	public void PrintHelloWorld()
	{
		var package = new TestPackage();
		new Type(package, Base.App, null!).Parse("Run");
		var type = new Type(package, "Program", new MethodExpressionParser()).Parse(@"implement App
has log
Run
	log.Write(""Hello"")");
		Assert.That(Run(type.Methods[0]), Is.EqualTo("Hello"));
	}

	public string Run(Method method)
	{
		foreach (var expression in method.Body.Expressions)
			if (expression is OneArgumentMethodCall call && call.Method.Name == "Write")
				return ((Text)call.Argument).Data.ToString()!;
		return ""; //ncrunch: no coverage
	}
}