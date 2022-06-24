using System;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ListTests : TestExpressions
{
	[TestCase("(1, 2, 3, 4, 5) + \"4\"")]
	public void MismatchingTypeFound(string input) =>
		Assert.That(() => ParseExpression(input), Throws.InstanceOf<Binary.MismatchingTypeFound>()!);

	[TestCase("(1, 2, 3, 4, 5)", "1, 2, 3, 4, 5")]
	[TestCase("(\"1\", \"2\", \"3\", \"4\", \"5\")", "\"1\", \"2\", \"3\", \"4\", \"5\"")]
	[TestCase("(true, false, true, false)", "true, false, true, false")]
	public void ParseLists(string input, string expected) =>
		ParseAndCheckOutputMatchesInput(input, new List(method, expected));

	[TestCase("(1, 2, 3, 4, 5) + (6, 7, 8)", "1, 2, 3, 4, 5", "+", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) - (6, 7, 8)", "1, 2, 3, 4, 5", "-", "6, 7, 8")]
	public void ParseListsBinaryOperation(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new Binary(new List(method, expected[0]),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new List(method, expected[2])));

	[TestCase("(1, 2, 3, 4, 5) + 4", 4, "1, 2, 3, 4, 5", "+")]
	[TestCase("(1, 2, 3, 4, 5) - 4", 4, "1, 2, 3, 4, 5", "-")]
	public void ParseListsAdditionWithNumber(string input, double expectedRight,
		params string[] expected)
	{
		foreach (var returnTypeMethod in number.ReturnType.Methods)
			Console.WriteLine(returnTypeMethod);
		ParseAndCheckOutputMatchesInput(input,
			new Binary(new List(method, expected[0]),
				list.ReturnType.Methods.First(m => m.Name == expected[1]),
				new Number(method, expectedRight)));
	}
}