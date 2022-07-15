using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ShuntingYardTests
{
	[TestCase(new string[0], new string[0])]
	[TestCase(new[] { "a" }, new[] { "a" })]
	[TestCase(new[] { "a", "+", "b" }, new[] { "a", "b", "+" })]
	[TestCase(new[] { "(", "2", "+", "(", "3", "+", "5", ")", "*", "5", ")", "*", "2" }, new[] { "2", "3", "5", "+", "5", "*", "+", "2", "*" })]
	[TestCase(new[] { "a", "+", "b", "*", "c", "-", "d" }, new[] { "a", "b", "c", "*", "+", "d", "-" })]
	[TestCase(new[] { "(", "a", "+", "b", ")", "/", "2" }, new[] { "a", "b", "+", "2", "/" })]
	[TestCase(new[] { "2", "*", "(", "a", "+", "b", ")" }, new[] { "2", "a", "b", "+", "*" })]
	[TestCase(new[] { "(", "(", "a", "+", "b", ")", "/", "2", ")" }, new[] { "a", "b", "+", "2", "/" })]
	[TestCase(new[] { "(", "a", "+", "b", ")", "*", "(", "c", "+", "d", ")" }, new[] { "a", "b", "+", "c", "d", "+", "*" })]
	[TestCase(new[] { "(", "a", "*", "a", "+", "b", ")", "/", "(", "c", "+", "d", "*", "e", ")" }, new[] { "a", "a", "*", "b", "+", "c", "d", "e", "*", "+", "/" })]
	public void EmptyOrSingleToken(string[] tokens, string[] expected) =>
		Assert.That(new ShuntingYard(tokens).Output.Reverse(), Is.EqualTo(expected), string.Join("\", \"", new ShuntingYard(tokens).Output.Reverse().ToArray()));
}