using System.Linq;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;

namespace Strict.Compiler.Tests;

public sealed class CSharpExpressionVisitorTests : TestExpressions
{
	[SetUp]
	public void CreateVisitor() => visitor = new CSharpExpressionVisitor();

	private CSharpExpressionVisitor visitor = null!;

	[Test]
	public void GenerateAssignment() =>
		Assert.That(
			visitor.Visit(new Assignment(new Identifier(nameof(number), number.ReturnType), number)),
			Is.EqualTo("var number = 5;"));

	[Test]
	public void GenerateBinary() =>
		Assert.That(
			visitor.Visit(new Binary(number, number.ReturnType.Methods.First(m => m.Name == "+"),
				number)), Is.EqualTo("5 + 5"));

	[TestCase("let other = 3", "var other = 3;")]
	[TestCase("5 + 5", "5 + 5")]
	public void ConvertStrictToCSharp(string strictCode, string expectedCSharpCode) =>
		Assert.That(visitor.Visit(ParseExpression(method, strictCode)),
			Is.EqualTo(expectedCSharpCode));
}