using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ForTests : TestExpressions
{
	[Test]
	public void MissingBody() =>
		Assert.That(() => ParseExpression("for Range(2, 5)"),
			Throws.InstanceOf<For.MissingInnerBody>());

	[Test]
	public void MissingExpression() =>
		Assert.That(() => ParseExpression("for"), Throws.InstanceOf<For.MissingExpression>());

	[Test]
	public void InvalidForExpression() =>
		Assert.That(() => ParseExpression("for gibberish", "\tlog.Write(\"Hi\")"),
			Throws.InstanceOf<IdentifierNotFound>());

	[Test]
	public void VariableOutOfScope() =>
		Assert.That(
			() => ParseExpression("for Range(2, 5)", "\tlet num = 5", "for Range(0, 10)",
				"\tlog.Write(num)"),
			Throws.InstanceOf<IdentifierNotFound>().With.Message.StartWith("num"));

	[Test]
	public void ValidExpressionType() =>
		Assert.That(() => ParseExpression("for Range(2, 5)", "\tlog.Write(\"Hi\")"),
			Is.TypeOf(typeof(For)));

	[Test]
	public void MatchingHashCode()
	{
		var forExpression = (For)ParseExpression("for Range(2, 5)", "\tlog.Write(\"Hi\")");
		Assert.That(forExpression.GetHashCode(), Is.EqualTo(forExpression.Value.GetHashCode()));
	}

	[Test]
	public void ParseForExpression()
	{
		var forExpression = (For)ParseExpression("for Range(2, 5)", "\tlog.Write(\"Hi\")");
		Assert.That(forExpression.Body, Is.EqualTo(ParseExpression("log.Write(\"Hi\")")));
		Assert.That(forExpression.Value, Is.EqualTo(ParseExpression("Range(2, 5)")));
		Assert.That(forExpression.ToString(), Is.EqualTo("for Range(2, 5)"));
	}

	[Test]
	public void ValidLoopProgram()
	{
		var programType = new Type(type.Package,
				new TypeLines(Base.App, "has n Number", "CountNumber Number", "\tlet result = Count(1)",
					"\tfor Range(0, n)", "\t\tresult.Increment", "\tresult")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		var forMethodCall = ((For)parsedExpression.Expressions[1]).Body as MethodCall;
		Assert.That(parsedExpression.ReturnType.Name, Is.EqualTo(Base.Number));
		Assert.That(parsedExpression.Expressions[1], Is.TypeOf(typeof(For)));
		Assert.That(((For)parsedExpression.Expressions[1]).Value.ToString(),
			Is.EqualTo("Range(0, n)"));
		Assert.That(((VariableCall?)forMethodCall?.Instance)?.Name, Is.EqualTo("result"));
		Assert.That(forMethodCall?.Method.Name, Is.EqualTo("Increment"));
	}

	[Test]
	public void ParseNestedFor()
	{
		var forExpression =
			(For)ParseExpression("for Range(2, 5)", "\tfor Range(0, 10)", "\t\tlog.Write(\"Hi\")");
		Assert.That(forExpression.ToString(), Is.EqualTo("for Range(2, 5)"));
		Assert.That(forExpression.Body.ToString(), Is.EqualTo("for Range(0, 10)"));
		Assert.That(((For)forExpression.Body).Body.ToString(), Is.EqualTo("log.Write(\"Hi\")"));
	}
}