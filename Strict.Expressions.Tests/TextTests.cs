using Strict.Language.Tests;

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
			() => new Type(TestPackage.Instance,
					new TypeLines(nameof(TextExceededMaximumCharacterLimitUseMultiLine), "has number", "Run",
						"\tconstant result = \"HiHelloHowAreYou\" +", "\t\"HelloHowAreYouHiHello\"")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<TypeParser.MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred>().With.
				Message.StartWith("Current length: 63, Minimum Length for Multi line expressions: 100"));

	[TestCase("Single",
		"constant result = \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\" + \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\"",
		"has number", "Run",
		"\tconstant result = \"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\" +",
		"\t\"ThisStringShouldGoMoreThanHundredCharactersLongSoThatTheTestCanBePassed\"",
		"\tresult is Text")]
	[TestCase("Multiple",
		"constant result = \"ThisStringShouldGoMoreThanHundred\" + \"SecondLineToMakeItThanHundredCharacters\" + \"ThirdLineToMakeItThanHundredCharacters\" + \"FourthLine\"",
		"has number", "Run", "\tconstant result = \"ThisStringShouldGoMoreThanHundred\" +",
		"\t\"SecondLineToMakeItThanHundredCharacters\" +",
		"\t\"ThirdLineToMakeItThanHundredCharacters\" +", "\t\"FourthLine\"",
		"\tresult is Text")]
	public void
		ParseMultiLineTextExpressions(string testName, string expectedOutput,
			params string[] code) =>
		Assert.That(
			((Body)new Type(TestPackage.Instance,
					new TypeLines(nameof(ParseMultiLineTextExpressions) + testName, code)).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].
				GetBodyAndParseIfNeeded()).Expressions[0].ToString(),
			Is.EqualTo(expectedOutput));

	[TestCase("ParseNewLineTextExpression", "\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine", "has logger", "Run Text",
		"	\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine")]
	[TestCase("ParseMultiLineTextExpressionWithNewLine", "\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine + \"Ending\" + \"This is the continuation of the previous text line\"", "has logger", "Run Text",
		"	\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine + \"Ending\" +",
		"	\"This is the continuation of the previous text line\"")]
	public void ParseNewLineTextExpression(string testName, string expected, params string[] code)
	{
		using var multiLineType = new Type(TestPackage.Instance,
				new TypeLines(testName, code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var binary = (Binary)multiLineType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(binary.ToString(), Is.EqualTo(expected));
	}

	[TestCase("ParseMultiLineTextExpressionWithNewLine",
		"\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine + \"This is the continuation of the previous text line\"",
		"has logger", "Run Text",
		"	\"FirstLine\" + Character.NewLine + \"ThirdLine\" + Character.NewLine +",
		"	\"This is the continuation of the previous text line\"")]
	public void ParseMultiLineTextEndsWithNewLine(string testName, string expected,
		params string[] code)
	{
		using var multiLineType =
			new Type(TestPackage.Instance, new TypeLines(testName, code)).ParseMembersAndMethods(
				new MethodExpressionParser());
		var constantDeclaration = (Binary)multiLineType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(constantDeclaration.ToString(), Is.EqualTo(expected));
	}
}