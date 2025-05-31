using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ErrorTests : TestExpressions
{
	[Test]
	public void ParseErrorExpression()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ParseErrorExpression),
					"has number",
					"CheckNumberInRangeTen Number",
					"\tconstant notANumber = Error",
					"\tif number in Range(0, 10)",
					"\t\treturn number",
					"\telse",
					"\t\treturn notANumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((ConstantDeclaration)parsedExpression.Expressions[0]).Value.ReturnType,
			Is.EqualTo(type.GetType(Base.Error)));
		Assert.That(((If)parsedExpression.Expressions[1]).OptionalElse?.ToString(),
			Is.EqualTo("return notANumber"));
	}

	[Test]
	public void TypeLevelErrorExpression()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(TypeLevelErrorExpression),
					"has number",
					"constant NotANumber Error",
					"CheckNumberInRangeTen Number",
					"\tif number in Range(0, 10)",
					"\t\treturn number",
					"\telse",
					"\t\treturn NotANumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var ifExpression = (If)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(programType.Members[1].Type,
			Is.EqualTo(type.GetType(Base.Error)));
		Assert.That(ifExpression.OptionalElse?.ToString(), Is.EqualTo("return NotANumber"));
	}
}