using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class ForTests : TestExpressions
{
	[Test]
	public void MissingBody() =>
		Assert.That(() => ParseExpression("for Range(2,5)"),
			Throws.InstanceOf<For.MissingInnerBody>());

	[Test]
	public void MissingExpression() =>
		Assert.That(() => ParseExpression("for"), Throws.InstanceOf<For.MissingExpression>());

	[Test]
	public void ParseForExpression()
	{
		var forExpression = (For)ParseExpression("for Range(2, 5)", "\tlog.Write(\"Hi\")");
		Assert.That(forExpression.Body, Is.EqualTo(ParseExpression("log.Write(\"Hi\")")));
		Assert.That(forExpression.Value, Is.EqualTo(ParseExpression("Range(2, 5)")));
		Assert.That(forExpression.ToString(), Is.EqualTo("for Range(2, 5)"));
	}

	[Test]
	public void InvalidForExpression() =>
		Assert.That(() => ParseExpression("for gibberish", "\tlet num = 5"),
			Throws.InstanceOf<IdentifierNotFound>());
}