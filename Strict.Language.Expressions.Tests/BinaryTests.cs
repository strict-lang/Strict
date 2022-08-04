using NUnit.Framework;

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
			Throws.Exception.InstanceOf<UnknownExpression>());

	[Test]
	public void MissingRightExpression() =>
		Assert.That(() => ParseExpression("5 + unknown"),
			Throws.Exception.InstanceOf<UnknownExpression>());

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
	[TestCase("1 * 2 + 1")]
	[TestCase("(1 + 2) * 3")]
	[TestCase("(1 + 2) * (3 + 4)")]
	[TestCase("1 + 2 + (3 + 4) * (5 + 6)")]
	[TestCase("(1 + 2 + 3 + 4) * (5 + 6)")]
	[TestCase("((1 + 2) * (3 + 4) + 1) * (5 + 6)")]
	[TestCase("(1 + 2) * (2 + 5) + 3")]
	[TestCase("3 + (1 + 2) * (2 + 5)")]
	[TestCase("3 + (1 + 2) * 5 * (2 + 5)")]
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
}