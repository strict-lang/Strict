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
	public void UsageOfInferredVariable() =>
		Assert.That(() => ParseExpression("for index in Range(0, 5)", "\tlog.Write(index)"),
			Throws.InstanceOf<For.UsageOfInferredVariable>());

	[Test]
	public void ParseForRangeExpression()
	{
		var forExpression = (For)ParseExpression("for Range(2, 5)", "\tlog.Write(index)");
		Assert.That(forExpression.Body.ToString(), Is.EqualTo("log.Write(index)"));
		Assert.That(forExpression.Value, Is.EqualTo(ParseExpression("Range(2, 5)")));
		Assert.That(forExpression.ToString(), Is.EqualTo("for Range(2, 5)"));
	}

	[Test]
	public void ParseForInExpression()
	{
		var forExpression =
			((Body)ParseExpression("let myIndex = 0", "for myIndex in Range(0, 5)",
				"\tlog.Write(myIndex)")).Expressions[1] as For;
		Assert.That(forExpression?.Body.ToString(), Is.EqualTo("log.Write(myIndex)"));
		Assert.That(forExpression?.Value.ToString(), Is.EqualTo("myIndex in Range(0, 5)"));
		Assert.That(forExpression?.ToString(), Is.EqualTo("for myIndex in Range(0, 5)"));
	}

	[Test]
	public void ParseForInExpressionWithCustomVariableName()
	{
		var forExpression =
			(For)ParseExpression("for myIndex in Range(0, 5)", "\tlog.Write(myIndex)");
		Assert.That(forExpression.Body.ToString(), Is.EqualTo("log.Write(myIndex)"));
		Assert.That(forExpression.Value.ToString(), Is.EqualTo("myIndex in Range(0, 5)"));
		Assert.That(forExpression.ToString(), Is.EqualTo("for myIndex in Range(0, 5)"));
	}

	[Test]
	public void ParseForListExpression()
	{
		var forExpression =
			((Body)ParseExpression("let element = 0", "for element in (1, 2, 3)",
				"\tlog.Write(element)")).Expressions[1] as For;
		Assert.That(forExpression?.Body.ToString(), Is.EqualTo("log.Write(element)"));
		Assert.That(forExpression?.Value.ToString(), Is.EqualTo("element in (1, 2, 3)"));
		Assert.That(forExpression?.ToString(), Is.EqualTo("for element in (1, 2, 3)"));
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
			(For)ParseExpression("for Range(2, 5)", "\tfor Range(0, 10)", "\t\tlog.Write(index)");
		Assert.That(forExpression.ToString(), Is.EqualTo("for Range(2, 5)"));
		Assert.That(forExpression.Body.ToString(), Is.EqualTo("for Range(0, 10)"));
		Assert.That(((For)forExpression.Body).Body.ToString(), Is.EqualTo("log.Write(index)"));
	}
}