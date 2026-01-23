namespace Strict.Expressions.Tests;

public sealed class IfTests : TestExpressions
{
	[Test]
	public void MissingCondition() =>
		Assert.That(() => ParseExpression("if"),
			Throws.InstanceOf<If.MissingCondition>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 2"));

	[Test]
	public void InvalidCondition() =>
		Assert.That(() => ParseExpression("if 5", "\treturn 0"),
			Throws.InstanceOf<If.InvalidCondition>());

	[Test]
	public void ReturnTypeOfThenAndElseMustHaveMatchingType() =>
		Assert.That(
			() => ParseExpression("if 5 is 6", "\treturn \"hello\"", "else", "\treturn true").
				ReturnType, Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());

	[Test]
	public void ReturnTypeOfThenAndElseIsNumberAndCharacterIsValid() =>
		Assert.That(new Method(type, 0, this, [
					// @formatter:off
					"ReturnMethod Number",
					"	if five is 5",
					"		return Character(5)",
					"	else",
					"		return 5"
				]).GetBodyAndParseIfNeeded().ReturnType, Is.EqualTo(type.GetType(Base.Number)));

	[Test]
	public void ParseInvalidSpaceAfterElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else "),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtEndOfLine>());

	[Test]
	public void ParseJustElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else"),
			Throws.InstanceOf<If.UnexpectedElse>().With.Message.Contains("at Run in "));

	[Test]
	public void ParseIncompleteThen() =>
		Assert.That(() => ParseExpression("if five is 5"), Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void MissingThen() =>
		Assert.That(() => ParseExpression("if five is 5", "Run"), Throws.InstanceOf<If.MissingThen>());

	[TestCase("n")]
	[TestCase("no")]
	[TestCase("nope")]
	[TestCase("nott")]
	[TestCase("note")]
	public void InvalidNotKeyword(string invalidKeyword) =>
		Assert.That(() => ParseExpression($"if five is {invalidKeyword} 5", "\tlogger.Log(\"Hey\")"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartsWith(invalidKeyword));

	[Test]
	public void InvalidSpacingInsteadOfNot() =>
		Assert.That(() => ParseExpression("if five is  5", "\tlogger.Log(\"Hey\")"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<PhraseTokenizer.InvalidSpacing>());

	[Test]
	public void InvalidIsNotUsageOnDifferentType() =>
		Assert.That(() => ParseExpression("if five is not \"blu\"", "\tlogger.Log(\"Hey\")"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.Contains("blu"));

	[Test]
	public void ParseMissingElseExpression() =>
		Assert.That(() => ParseExpression("if five is 5", "\tRun", "else"),
			Throws.InstanceOf<If.MissingElseExpression>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 4"));

	[Test]
	public void ReturnGetHashCode()
	{
		var ifExpression = (If)ParseExpression("if five is 5", "\tRun");
		Assert.That(ifExpression.GetHashCode(),
			Is.EqualTo(ifExpression.Condition.GetHashCode() ^ ifExpression.Then.GetHashCode()));
	}

	[Test]
	public void MissingElseExpression() =>
		Assert.That(() => ParseExpression("constant result = true ? true"),
			Throws.InstanceOf<If.MissingElseExpression>());

	[Test]
	public void InvalidConditionInConditionalExpression() =>
		Assert.That(() => ParseExpression("constant result = 5 ? true"),
			Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void ReturnTypeOfConditionalThenAndElseMustHaveMatchingType() =>
		Assert.That(() => ParseExpression("constant result = true ? true else 5"),
			Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());
}