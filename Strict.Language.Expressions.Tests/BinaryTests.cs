using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class BinaryTests : TestExpressions
{
	[Test]
	public void ParseBinary() =>
		ParseAndCheckOutputMatchesInput("5 + 5",
			new Binary(number, number.ReturnType.Methods.First(m => m.Name == "+"), number));

	[Test]
	public void AddFiveToNumber() =>
		ParseAndCheckOutputMatchesInput("bla + 5",
			new Binary(new MemberCall(bla), number.ReturnType.Methods.First(m => m.Name == "+"),
				number));

	[Test]
	public void MissingLeftExpression() =>
		Assert.That(() => ParseExpression("unknown + 5"),
			Throws.Exception.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void MissingRightExpression() =>
		Assert.That(() => ParseExpression("5 + unknown"),
			Throws.Exception.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void ParseComparison() =>
		ParseAndCheckOutputMatchesInput("bla is 5",
			new Binary(new MemberCall(bla), binaryOperators.First(m => m.Name == BinaryOperator.Is),
				number));

	[Test]
	public void NestedBinary() =>
		ParseAndCheckOutputMatchesInput("2 * 5 + 3",
			new Binary(
				new Binary(new Number(method, 2),
					method.GetType(Base.Number).Methods.First(m => m.Name == "*"), new Number(method, 5)),
				method.GetType(Base.Number).Methods.First(m => m.Name == "+"), new Number(method, 3)));

	[TestCase("1 + 2")]
	[TestCase("(1 is 1)")]
	[TestCase("(1 * 1)")]
	[TestCase("(1 + 2 + 3)")]
	[TestCase("(1 + 2) + (3 + 4)")]
	[TestCase("(1 + 2) + (3 + 4) * (5 + 6)")]
	[TestCase("((1 + 2) + (3 + 4)) * (5 + 6)")]
	[TestCase("(((1 + 2) + (3 + 4)) * (5 + 6))")]
	[TestCase("(1 + 2) * (2 + 5) + 3")]
	[TestCase("3 + (1 + 2) * (2 + 5)")]
	[TestCase("3 + (1 + 2) * 5 * (2 + 5)")]
	public void ParseGroupExpression(string code)
	{
		var expression = ParseExpression(code);
		Assert.That(expression.ToString(), Is.EqualTo(code.Replace("(", "").Replace(")", "")));
	}

	[Ignore("Complex case")]
	[Test]
	public void NestedBinaryWithBrackets() =>
		Assert.That(ParseExpression("2 * (5 + 3)"),
			Is.EqualTo(new Binary(new Number(method, 2),
				method.GetType(Base.Number).Methods.First(m => m.Name == "*"),
				new Binary(new Number(method, 5),
					method.GetType(Base.Number).Methods.First(m => m.Name == "+"),
					new Number(method, 3)))));

	[Ignore("Complex case")]
	[Test]
	public void NestedBinaryExpressionsWithGrouping() =>
		Assert.That(ParseExpression("(2 + 5) * 3"),
			Is.EqualTo(new Binary(new Number(method, 2),
				method.GetType(Base.Number).Methods.First(m => m.Name == "+"),
				new Binary(new Number(method, 5),
					method.GetType(Base.Number).Methods.First(m => m.Name == "*"),
					new Number(method, 3)))));

	[Ignore("Complex case")]
	[Test]
	public void NestedBinaryExpressionsSingleGroup() =>
		Assert.That(ParseExpression("6 + (2 + 5) * 3"),
			Is.EqualTo(new Binary(new Number(method, 6),
				method.GetType(Base.Number).Methods.First(m => m.Name == "+"),
				new Binary(new Number(method, 2),
					method.GetType(Base.Number).Methods.First(m => m.Name == "+"),
					new Binary(new Number(method, 5),
						method.GetType(Base.Number).Methods.First(m => m.Name == "*"),
						new Number(method, 3))))));

	[Test]
	public void NestedBinaryExpressionsTwoGroups() =>
		Assert.That(ParseExpression("(6 * 3) + (2 + 5) * 3"),
			Is.EqualTo(new Binary(
				new Binary(new Number(method, 6),
					method.GetType(Base.Number).Methods.First(m => m.Name == "*"), new Number(method, 3)),
				method.GetType(Base.Number).Methods.First(m => m.Name == "+"),
				new Binary(
					new Binary(new Number(method, 2),
						method.GetType(Base.Number).Methods.First(m => m.Name == "+"), new Number(method, 5)),
					method.GetType(Base.Number).Methods.First(m => m.Name == "*"),
					new Number(method, 3)))));
}