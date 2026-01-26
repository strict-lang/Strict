namespace Strict.Expressions.Tests;

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
					"\tif number is in Range(0, 10)",
					"\t\treturn number",
					"\telse",
					"\t\treturn notANumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		var declaration = ((Declaration)parsedExpression.Expressions[0]).Value;
		Assert.That(declaration.ReturnType, Is.EqualTo(type.GetType(Base.Error)));
		Assert.That(((If)parsedExpression.Expressions[1]).OptionalElse?.ToString(),
			Is.EqualTo("return notANumber"));
	}

	[Test]
	public void TypeLevelErrorExpression()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(TypeLevelErrorExpression),
					"has number",
					"constant NotANumber = Error",
					"CheckIfNumberIsInRangeTen Number",
					"\tif number is in Range(0, 10)",
					"\t\treturn number",
					"\telse",
					"\t\treturn NotANumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var ifExpression = (If)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(programType.Members[1].Type,
			Is.EqualTo(type.GetType(Base.Error)));
		Assert.That(ifExpression.OptionalElse?.ToString(), Is.EqualTo("return NotANumber"));
	}

	[Test]
	public void ErrorTextAndStacktraceIsFilledAutomatically()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ErrorTextAndStacktraceIsFilledAutomatically),
					"has number",
					"Run",
					"\tError")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var returnExpression = (MethodCall)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(returnExpression.Arguments, Has.Count.EqualTo(2));
		Assert.That(returnExpression.Arguments[0].ToString(), Is.EqualTo("\"Run\""));
		Assert.That(returnExpression.Arguments[1].ReturnType,
			Is.EqualTo(type.GetListImplementationType(type.GetType(Base.Stacktrace))));
	}
}