using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ListTests : TestExpressions
{
	[TestCase("(1, 2, 3, 4, 5) + \"4\"")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\")")]
	[TestCase("(1, 2, 3, 4, 5) + \"hello\" + 4")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\") + 4")]
	//[TestCase("(Any) + (\"hello\") + 4")] https://deltaengine.fogbugz.com/f/cases/24791/
	public void MismatchingTypeFound(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<Binary.MismatchingTypeFound>()!);

	[TestCase("(1, 2, 3) * (1, 2)")]
	[TestCase("(1, 2, 3) * (1, 2, 3, 4)")]
	public void ListsHaveDifferentDimensions(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<List.ListsHaveDifferentDimensions>()!);

	[TestCase("(1)", "1")]
	[TestCase("(\"1\", \"2\", \"3\", \"4\", \"5\")", "\"1\", \"2\", \"3\", \"4\", \"5\"")]
	[TestCase("(true, false, true, false)", "true, false, true, false")]
	public void ParseLists(string input, string expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(method,
				new List<Expression>(GetListExpressions(expected.Split(",")))));

	private IEnumerable<Expression> GetListExpressions(IEnumerable<string> elements) =>
		elements.Select(element =>
				method.TryParseExpression(new Method.Line(method, 0, "", 0), element.Trim())).
			Where(foundExpression => foundExpression != null).ToList()!;

	[TestCase("(1, 2, 3, 4, 5) + (6, 7, 8)", "1, 2, 3, 4, 5", "+", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) - (6, 7, 8)", "1, 2, 3, 4, 5", "-", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) is (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "is", "1, 2, 3, 4, 5")]
	[TestCase("(1, 2, 3, 4, 5) * (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "*", "1, 2, 3, 4, 5")]
	public void ParseListsBinaryOperation(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new Binary(
				new List(method, new List<Expression>(GetListExpressions(expected[0].Split(",")))),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new List(method, new List<Expression>(GetListExpressions(expected[2].Split(","))))));

	[TestCase("(1, 2, 3, 4, 5) + 4", 4, "1, 2, 3, 4, 5", "+")]
	[TestCase("(1, 2, 3, 4, 5) - 4", 4, "1, 2, 3, 4, 5", "-")]
	public void ParseListsWithNumber(string input, double expectedRight, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new Binary(
				new List(method, new List<Expression>(GetListExpressions(expected[0].Split(",")))),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new Number(method, expectedRight)));

	[TestCase("(\"1\", \"2\", \"3\", \"4\") + \"5\"", "\"1\", \"2\", \"3\", \"4\"", "+", "5")]
	public void ParseListsWithString(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new Binary(
				new List(method, new List<Expression>(GetListExpressions(expected[0].Split(",")))),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new Text(method, expected[2])));

	[TestCase("(1, 2, 3, 4, 5) + (1) + 4", 4, "1, 2, 3, 4, 5", "+", "1")]
	public void ParseMultipleListExpression(string input, double expectedRight, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new Binary(
				new List(method, new List<Expression>(GetListExpressions(expected[0].Split(",")))),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new Binary(
					new List(method, new List<Expression>(GetListExpressions(expected[2].Split(",")))),
					list.ReturnType.Methods.First(m => m.Name == expected[1]),
					new Number(method, expectedRight))));
}