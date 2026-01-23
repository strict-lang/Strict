global using Type = Strict.Language.Type;

namespace Strict.Expressions.Tests;

public sealed class BinaryTests : TestExpressions
{
	[Test]
	public void ParseBinary() =>
		ParseAndCheckOutputMatchesInput("5 + 5", CreateBinary(numberFive, BinaryOperator.Plus, numberFive));

	[Test]
	public void AddFiveToNumber() =>
		ParseAndCheckOutputMatchesInput("five + 5",
			CreateBinary(new MemberCall(null, five), BinaryOperator.Plus, numberFive));

	[Test]
	public void InvalidLeftNestedExpression() =>
		Assert.That(() => ParseExpression("five.Unknown + 5"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void MissingLeftExpression() =>
		Assert.That(() => ParseExpression("unknown + 5"),
			Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void MissingRightExpression() =>
		Assert.That(() => ParseExpression("5 + unknown"),
			Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void ArgumentsDoNotMatchBinaryOperatorParameters() =>
		Assert.That(() => ParseExpression("5 / \"text\""),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void NoMatchingMethodFound() =>
		Assert.That(() => ParseExpression("true - \"text\""),
			Throws.Exception.InnerException.InstanceOf<Type.NoMatchingMethodFound>().With.InnerException.
				Message.Contains("not found for TestPackage.Boolean, available methods"));

	[Test]
	public void ConversionTypeNotFound() =>
		Assert.That(() => ParseExpression("5 to gibberish"),
			Throws.InstanceOf<To.ConversionTypeNotFound>());

	[TestCase("5 to Logger")]
	[TestCase("5 to Range")]
	[TestCase("5 to Boolean")]
	public void ConversionNotImplemented(string code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<To.ConversionTypeIsIncompatible>().With.Message.
				StartsWith("Conversion for Number"));

	[Test]
	public void InvalidUsageOfToOperator() =>
		Assert.That(() => ParseExpression("to(Text)"),
			Throws.InnerException.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseComparison() =>
		ParseAndCheckOutputMatchesInput("five is 5",
			CreateBinary(new MemberCall(null, five), BinaryOperator.Is, numberFive));

	[Test]
	public void NestedBinary() =>
		ParseAndCheckOutputMatchesInput("2 * 5 + 3",
			CreateBinary(
				CreateBinary(new Number(method, 2), BinaryOperator.Multiply, new Number(method, 5)),
				BinaryOperator.Plus, new Number(method, 3)));

	[TestCase("1 + 2")]
	[TestCase("1 is 1")]
	[TestCase("1 % 2 ^ 0")]
	[TestCase("true and true is true")]
	[TestCase("true xor true")]
	[TestCase("5 to Text")]
	[TestCase("1 + 2 * 3")]
	[TestCase("1 * 2 + 3")]
	[TestCase("5 * 2 ^ 2")]
	[TestCase("(1 + 2) * 3")]
	[TestCase("(1 + 2) * (3 + 4)")]
	[TestCase("(2 + 2) ^ (5 - 4)")]
	[TestCase("1 + 2 + (3 + 4) * (5 + 6)")]
	[TestCase("(1 + 2 + 3 + 4) * (5 + 6)")]
	[TestCase("((1 + 2) * (3 + 4) + 1) * (5 + 6)")]
	[TestCase("(1 + 2) * (2 + 5) + 3")]
	[TestCase("3 + (1 + 2) * (2 + 5)")]
	[TestCase("3 + (1 + 2) * 5 * (2 + 5)")]
	[TestCase("3 + (1 + 2) % 5 * (2 + 5)")]
	[TestCase("(1 + 5, 2, 3) + (5, 2 * 5)")]
	[TestCase("5 > 4 or 10 < 100.5 and 5 > 4 and 5 < 6")]
	public void ParseGroupExpressionProducesSameCode(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void ParseToOperator() =>
		Assert.That(((To)ParseExpression("5 to Text")).ConversionType.Name, Is.EqualTo(Base.Text));

	[Test]
	public void ParsePowerWithMultiplyOperator() =>
		ParseAndCheckOutputMatchesInput("(5 * 2) ^ 2",
			CreateBinary(CreateBinary(numberFive, BinaryOperator.Multiply, new Number(type, 2)),
				BinaryOperator.Power, new Number(type, 2)));

	[Test]
	public void NestedBinaryWithBrackets() =>
		ParseAndCheckOutputMatchesInput("2 * (5 + 3)",
			CreateBinary(new Number(method, 2), BinaryOperator.Multiply,
				CreateBinary(new Number(method, 5), BinaryOperator.Plus, new Number(method, 3))));

	[Test]
	public void NestedBinaryExpressionsWithGrouping() =>
		ParseAndCheckOutputMatchesInput("(2 + 5) * 3",
			CreateBinary(
				CreateBinary(new Number(method, 2), BinaryOperator.Plus, new Number(method, 5)),
				BinaryOperator.Multiply, new Number(method, 3)));

	[Test]
	public void NestedBinaryExpressionsSingleGroup() =>
		ParseAndCheckOutputMatchesInput("6 + (7 + 8) * 9",
			CreateBinary(new Number(method, 6), BinaryOperator.Plus,
				CreateBinary(
					CreateBinary(new Number(method, 7), BinaryOperator.Plus, new Number(method, 8)),
					BinaryOperator.Multiply, new Number(method, 9))));

	[Test]
	public void NestedBinaryExpressionsTwoGroups() =>
		ParseAndCheckOutputMatchesInput("(1 + 2) * (3 + 4) * 5",
			CreateBinary(
				CreateBinary(new Number(method, 1), BinaryOperator.Plus, new Number(method, 2)),
				BinaryOperator.Multiply,
				CreateBinary(
					CreateBinary(new Number(method, 3), BinaryOperator.Plus, new Number(method, 4)),
					BinaryOperator.Multiply, new Number(method, 5))));

	[Test]
	public void HasMatchingLeftAndRightExpressionTypes()
	{
		var expression = ParseExpression("(\"a\", \"b\") + \"5\"");
		Assert.That(expression, Is.InstanceOf<Binary>());
		Assert.That(((Binary)expression).ReturnType, Is.EqualTo(type.GetType(Base.Text.Pluralize())));
	}
}