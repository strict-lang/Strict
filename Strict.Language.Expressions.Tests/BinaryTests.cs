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

	[TestCase("")]
	[TestCase("1")]
	[TestCase("(1)")]
	[TestCase("(1, 3)")]
	[TestCase("(Run(1, 2) + 2)")]
	public void ParseNonBinaryExpressionInBracket(string code)
	{
		var expression = Group.TryParse(new Method.Line(type.Methods[0], 0, code, 0), code);
		Assert.That(expression, Is.Null);
	}

	[TestCase("(1 + 2)")]
	[TestCase("(1 is 1)")]
	[TestCase("(1 * 1)")]
	[TestCase("(1 + 2 + 3)")]
	[TestCase("(1 + 2) + 3")]
	//[TestCase("(1 + 2) * (2 + 5) + 3")]
	public void ParseGroupExpression(string code)
	{
		var expression = Group.TryParse(new Method.Line(type.Methods[0], 0, code, 0), code);
		Assert.That(expression, Is.InstanceOf<Binary>());
		Assert.That("(" + expression + ")", Is.EqualTo(code));
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

	[Ignore("Complex case")]
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