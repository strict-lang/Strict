using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class TextTests : TestExpressions
{
	[Test]
	public void ParseText() => ParseAndCheckOutputMatchesInput("\"Hi\"", new Text(method, "Hi"));

	[Test]
	public void ParseTextToNumber()
	{
		var methodCall = (MethodCall)ParseExpression("\"5\" to Number");
		Assert.That(methodCall.ReturnType, Is.EqualTo(type.GetType(Base.Number)));
		Assert.That(methodCall.Method.Name, Is.EqualTo("to"));
		Assert.That(methodCall.Instance?.ToString(), Is.EqualTo("\"5\""));
	}
}