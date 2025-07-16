using NUnit.Framework;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Expressions.Tests;

public sealed class TextTests : TestExpressions
{
	[Test]
	public void ParseText() => ParseAndCheckOutputMatchesInput("\"Hi\"", new Text(method, "Hi"));

	[Test]
	public void ParseTextToNumber()
	{
		var methodCall = (MethodCall)ParseExpression("\"5\" to Number");
		Assert.That(methodCall.ReturnType.Name, Is.EqualTo(Base.Number));
		Assert.That(methodCall.Method.Name, Is.EqualTo("to"));
		Assert.That(methodCall.Instance?.ToString(), Is.EqualTo("\"5\""));
	}

	[Test]
	public void TextExceededMaximumCharacterLimitUseMultiLine() =>
		Assert.That(
			() => new Type(new TestPackage(),
					new TypeLines(nameof(TextExceededMaximumCharacterLimitUseMultiLine), "has number", "Run",
						"\tconstant result = \"HiHelloHowAreYou\" +", "\t\"HelloHowAreYouHiHello\"")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<TypeParser.MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred>().With.
				Message.StartWith("Current length: 63, Minimum Length for Multi line expressions: 100"));

	[TestCase("Single",
		"constant result = \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\" + \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\"",
		"has number", "Run",
		"\tconstant result = \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\" +",
		"\t\"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\"")]
	[TestCase("Multiple",
		"constant result = \"ThisStringShouldGoMoreThanHundred\" + \"SecondLineToMakeItThanHundredCharacters\" + \"ThirdLineToMakeItThanHundredCharacters\" + \"FourthLine\"",
		"has number", "Run", "\tconstant result = \"ThisStringShouldGoMoreThanHundred\" +",
		"\t\"SecondLineToMakeItThanHundredCharacters\" +",
		"\t\"ThirdLineToMakeItThanHundredCharacters\" +", "\t\"FourthLine\"")]
	public void
		ParseMultiLineTextExpressions(string testName, string expectedOutput, params string[] code) =>
		Assert.That(
			new Type(new TestPackage(),
					new TypeLines(nameof(ParseMultiLineTextExpressions) + testName, code)).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded().
				ToString(), Is.EqualTo(expectedOutput));

	[TestCase("ParseNewLineTextExpression", "\"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine", "has logger", "Run Text",
		"	constant input = \"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine")]
	[TestCase("ParseMultiLineTextExpressionWithNewLine", "\"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine + \"Ending\" + \"This is the continuation of the previous text line\"", "has logger", "Run Text",
		"	constant input = \"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine + \"Ending\" +",
		"	\"This is the continuation of the previous text line\"")]
	public void ParseNewLineTextExpression(string testName, string expected, params string[] code)
	{
		var multiLineType = new Type(new TestPackage(),
				new TypeLines(testName, code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var constantDeclaration =
			(ConstantDeclaration)multiLineType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(constantDeclaration.Value, Is.InstanceOf<Binary>());
		Assert.That(constantDeclaration.Value.ToString(), Is.EqualTo(expected));
	}

	[TestCase("ParseMultiLineTextExpressionWithNewLine",
		"\"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine + \"This is the continuation of the previous text line\"",
		"has logger", "Run Text",
		"	constant input = \"FirstLine\" + Text.NewLine + \"ThirdLine\" + Text.NewLine +",
		"	\"This is the continuation of the previous text line\"")]
	public void ParseMultiLineTextEndsWithNewLine(string testName, string expected,
		params string[] code)
	{
		var multiLineType =
			new Type(new TestPackage(), new TypeLines(testName, code)).ParseMembersAndMethods(
				new MethodExpressionParser());
		var constantDeclaration =
			(ConstantDeclaration)multiLineType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(constantDeclaration.Value, Is.InstanceOf<Binary>());
		Assert.That(constantDeclaration.Value.ToString(), Is.EqualTo(expected));
	}
}