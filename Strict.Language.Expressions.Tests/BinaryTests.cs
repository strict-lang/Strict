using NUnit.Framework;
using static Strict.Language.Type;

namespace Strict.Language.Expressions.Tests;

public sealed class BinaryTests : TestExpressions
{
	[Test]
	public void ParseBinary() =>
		ParseAndCheckOutputMatchesInput("5 + 5", CreateBinary(number, BinaryOperator.Plus, number));

	[Test]
	public void AddFiveToNumber() =>
		ParseAndCheckOutputMatchesInput("bla + 5",
			CreateBinary(new MemberCall(null, bla), BinaryOperator.Plus, number));

	[Test]
	public void InvalidLeftNestedExpression() =>
		Assert.That(() => ParseExpression("bla.Unknown + 5"),
			Throws.Exception.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void MissingLeftExpression() =>
		Assert.That(() => ParseExpression("unknown + 5"),
			Throws.Exception.InstanceOf<IdentifierNotFound>());

	[Test]
	public void MissingRightExpression() =>
		Assert.That(() => ParseExpression("5 + unknown"),
			Throws.Exception.InstanceOf<IdentifierNotFound>());

	[Test]
	public void ArgumentsDoNotMatchBinaryOperatorParameters() =>
		Assert.That(() => ParseExpression("5 / \"text\""),
			Throws.Exception.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.Contains(
				"Argument: TestPackage.Text \"text\" do not match:\n/(other TestPackage.Number) Number"));

	[Test]
	public void NoMatchingMethodFound() =>
		Assert.That(() => ParseExpression("\"text1\" - \"text\""),
			Throws.Exception.InstanceOf<NoMatchingMethodFound>().With.Message.Contains(
				"not found for TestPackage.Text, available methods"));

	[Test]
	public void ParseComparison() =>
		ParseAndCheckOutputMatchesInput("bla is 5",
			CreateBinary(new MemberCall(null, bla), BinaryOperator.Is, number));

	[Test]
	public void NestedBinary() =>
		ParseAndCheckOutputMatchesInput("2 * 5 + 3",
			CreateBinary(
				CreateBinary(new Number(method, 2), BinaryOperator.Multiply, new Number(method, 5)),
				BinaryOperator.Plus, new Number(method, 3)));

	[TestCase("1 + 2")]
	[TestCase("1 is 1")]
	[TestCase("true xor true")]
	//[TestCase("5 to Text")] //TODO: Need to support to Operator parsing
	[TestCase("1 * 2 + 1")]
	[TestCase("(1 + 2) * 3")]
	[TestCase("(1 + 2) * (3 + 4)")]
	[TestCase("(2 + 2) ^ (5 - 4)")]
	[TestCase("1 + 2 + (3 + 4) * (5 + 6)")]
	[TestCase("(1 + 2 + 3 + 4) * (5 + 6)")]
	[TestCase("((1 + 2) * (3 + 4) + 1) * (5 + 6)")]
	[TestCase("(1 + 2) * (2 + 5) + 3")]
	[TestCase("3 + (1 + 2) * (2 + 5)")]
	[TestCase("3 + (1 + 2) * 5 * (2 + 5)")]
	[TestCase("3 + (1 + 2) % (5 * (2 + 5))")]
	[TestCase("(1 + 5, 2, 3) + (5, 2 * 5)")]
	[TestCase("(5 > 4) or (10 < 100.5) and (5 >= 5) and (5 <= 6)")]
	public void ParseGroupExpressionProducesSameCode(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void NestedBinaryWithBrackets() =>
		Assert.That(ParseExpression("2 * (5 + 3)"),
			Is.EqualTo(CreateBinary(new Number(method, 2), BinaryOperator.Multiply,
				CreateBinary(new Number(method, 5), BinaryOperator.Plus, new Number(method, 3)))));

	[Test]
	public void NestedBinaryExpressionsWithGrouping() =>
		Assert.That(ParseExpression("(2 + 5) * 3"),
			Is.EqualTo(CreateBinary(CreateBinary(new Number(method, 2), BinaryOperator.Plus, new Number(method, 5)), BinaryOperator.Multiply, new Number(method, 3))));

	[Test]
	public void NestedBinaryExpressionsSingleGroup() =>
		Assert.That(ParseExpression("6 + (2 + 5) * 3"),
			Is.EqualTo(CreateBinary(new Number(method, 6), BinaryOperator.Plus,
				CreateBinary(CreateBinary(new Number(method, 2), BinaryOperator.Plus, new Number(method, 5)), BinaryOperator.Multiply, new Number(method, 3)))));

	[Test]
	public void NestedBinaryExpressionsTwoGroups() =>
		Assert.That(ParseExpression("(6 * 3) + (2 + 5) * 3"),
			Is.EqualTo(CreateBinary(
				CreateBinary(new Number(method, 6), BinaryOperator.Multiply, new Number(method, 3)),
				BinaryOperator.Plus,
				CreateBinary(
					CreateBinary(new Number(method, 2), BinaryOperator.Plus, new Number(method, 5)),
					BinaryOperator.Multiply, new Number(method, 3)))));

	[Test]
	public void HasMatchingLeftAndRightExpressionTypes()
	{
		var expression = ParseExpression("(\"a\", \"b\") + Count(5)");
		Assert.That(expression, Is.InstanceOf<Binary>()!);
		Assert.That(((Binary)expression).ReturnType, Is.EqualTo(type.GetType(Base.Text)));
	}
}