using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class ErrorTests : TestExpressions
{
	[Test]
	public void ParseErrorExpression()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ParseErrorExpression),
					"has number",
					"CheckNumberInRangeTen Number",
					"\tmutable NotANumber = Error",
					"\tif number is Range(0, 10)",
					"\t\treturn number",
					"\telse",
					"\t\treturn NotANumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((MutableDeclaration)parsedExpression.Expressions[0]).Value.ReturnType,
			Is.EqualTo(type.GetType(Base.Error)));
		Assert.That(((If)parsedExpression.Expressions[1]).OptionalElse?.ToString(),
			Is.EqualTo("return NotANumber"));
	}
}