using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ListTests : TestExpressions
{
	[Test]
	public void EmptyListNotAllowed() =>
		Assert.That(() => ParseExpression("()"), Throws.InstanceOf<List.EmptyListNotAllowed>()!);

	[TestCase("(1)", 1)]
	[TestCase("(5.23)", 5.23)]
	public void ParseSingleElementLists(string input, double expectedElement) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(method, new List<Expression> { new Number(method, expectedElement) }));

	[TestCase("(true, false, true, false)", "true, false, true, false")]
	[TestCase("(\"1\", \"2\", \"3\", \"4\", \"5\")", "\"1\", \"2\", \"3\", \"4\", \"5\"")]
	public void ParseLists(string input, string expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(method, GetListExpressions(expected.Split(", "))));

	private List<Expression> GetListExpressions(IEnumerable<string> elements)
	{
		var expressions = new List<Expression>();
		foreach (var line in
			elements.Select(element => new Method.Line(method, 0, element.Trim(), 0)))
			new PhraseTokenizer(line.Text, new Range(0, line.Text.Length)).ProcessEachToken(
				tokenRange =>
				{
					if (line.Text[tokenRange.Start.Value] != ',')
						expressions.Add(method.ParseExpression(line, tokenRange));
				});
		return expressions;
	}

	[Test]
	public void ParseLists() =>
		ParseAndCheckOutputMatchesInput("((1, 3), (2, 4))",
			new List(method, GetListExpressions(new[] { "(1, 3)", "(2, 4)" })));

	[TestCase("(1, 2, 3, 4, 5) + \"4\"")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\")")]
	[TestCase("(1, 2, 3, 4, 5) + \"hello\" + 4")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\") + 4")]
	//[TestCase("(Any) + (\"hello\") + 4")] https://deltaengine.fogbugz.com/f/cases/24791/
	public void MismatchingTypeFound(string input) =>
		Assert.That(() => ParseExpression(input), Throws.InstanceOf<Binary.MismatchingTypeFound>()!);

	[TestCase("(1, 2, 3) * (1, 2)")]
	[TestCase("(1, 2, 3) * (1, 2, 3, 4)")]
	public void ListsHaveDifferentDimensionsIsNotAllowed(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<ListsHaveDifferentDimensions>()!);

	[TestCase("(1, 2, 3, 4, 5) + (6, 7, 8)", "1, 2, 3, 4, 5", "+", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) - (6, 7, 8)", "1, 2, 3, 4, 5", "-", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) is (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "is", "1, 2, 3, 4, 5")]
	[TestCase("(1, 2, 3, 4, 5) * (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "*", "1, 2, 3, 4, 5")]
	public void ParseListsBinaryOperation(string input, params string[] expected)
	{
		Console.WriteLine(ParseExpression(input).ToString());
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(method, GetListExpressions(expected[0].Split(","))), expected[1],
				new List(method, GetListExpressions(expected[2].Split(",")))));
	}

	[TestCase("(1, 2, 3, 4, 5) + 4", 4, "1, 2, 3, 4, 5", "+")]
	[TestCase("(1, 2, 3, 4, 5) - 4", 4, "1, 2, 3, 4, 5", "-")]
	public void
		ParseListsWithNumber(string input, double expectedRight, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(method, GetListExpressions(expected[0].Split(","))), expected[1],
				new Number(method, expectedRight)));

	[TestCase("(\"1\", \"2\", \"3\", \"4\") + \"5\"", "\"1\", \"2\", \"3\", \"4\"", "+", "5")]
	public void ParseListsWithString(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(method, GetListExpressions(expected[0].Split(","))), expected[1],
				new Text(method, expected[2])));

	[Test]
	public void LeftTypeShouldNotBeChanged()
	{
		var expression = ParseExpression("(\"1\", \"2\", \"3\", \"4\") + 5");
		Assert.That(expression, Is.InstanceOf<Binary>()!);
		Assert.That(((Binary)expression).ReturnType, Is.EqualTo(list.ReturnType));
	}

	[Test]
	public void LeftTypeShouldNotBeChangedUnlessRightIsList()
	{
		var expression = ParseExpression("5 + (\"1\", \"2\", \"3\", \"4\")");
		Assert.That(expression, Is.InstanceOf<Binary>()!);
		Assert.That(((Binary)expression).ReturnType, Is.EqualTo(number.ReturnType));
	}

	[Test]
	public void ParseMultipleListInBinary() =>
		ParseAndCheckOutputMatchesInput("(1, 2, 3, 4, 5) + (1) + 4",
			CreateBinary(new List(method, GetListExpressions("1, 2, 3, 4, 5".Split(", "))),
				BinaryOperator.Plus,
				CreateBinary(new List(method, GetListExpressions("1".Split(", "))), BinaryOperator.Plus,
					new Number(method, 4))));

	[Test]
	public void ParseNestedLists() =>
		ParseAndCheckOutputMatchesInput("((1, 2, 3) + (3, 4), (4))",
			new List(method,
				new List<Expression>
				{
					CreateBinary(new List(method, GetListExpressions("1, 2, 3".Split(", "))),
						BinaryOperator.Plus, new List(method, GetListExpressions("3, 4".Split(",")))),
					new List(method, GetListExpressions("4".Split(", ")))
				}));

	[Test]
	public void ParseComplexLists() =>
		Assert.That(
			ParseExpression("((\"Hello, World\", \"Yoyo (it is my secret+1)\"), (\"4\")) + 7"),
			Is.EqualTo(CreateBinary(
				new List(method,
					GetListExpressions(new[]
					{
						"((\"Hello, World\", \"Yoyo (it is my secret+1)\"), (\"4\"))"
					})), BinaryOperator.Plus, number)));
}