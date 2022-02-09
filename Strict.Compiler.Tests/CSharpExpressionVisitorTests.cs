using System.Linq;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;
using Strict.Language.Tests;

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
			Is.EqualTo("var number = 5"));

	[Test]
	public void GenerateBinary() =>
		Assert.That(
			visitor.Visit(new Binary(number, number.ReturnType.Methods.First(m => m.Name == "+"),
				number)), Is.EqualTo("5 + 5"));

	[Test]
	public void GenerateBoolean() =>
		Assert.That(visitor.Visit(new Boolean(method, false)), Is.EqualTo("false"));

	[Test]
	public void GenerateMemberCall() =>
		Assert.That(
			visitor.Visit(new MemberCall(new MemberCall(member),
				member.Type.Members.First(m => m.Name == "Text"))), Is.EqualTo("log.Text"));

	[Test]
	public void GenerateMethodCall() =>
		Assert.That(
			visitor.Visit(new MethodCall(new MemberCall(member), member.Type.Methods[0],
				new Text(type, "Hi"))), Is.EqualTo("Console.WriteLine(\"Hi\")"));

	[Test]
	public void GenerateNumber() =>
		Assert.That(
			visitor.Visit(new Number(method, 77)), Is.EqualTo("77"));

	[Test]
	public void GenerateText() =>
		Assert.That(visitor.Visit(new Text(method, "Hey")), Is.EqualTo("\"Hey\""));

	[TestCase("let other = 3", "var other = 3")]
	[TestCase("5 + 5", "5 + 5")]
	[TestCase("true", "true")]
	[TestCase("\"Hey\"", "\"Hey\"")]
	[TestCase("42", "42")]
	[TestCase("log.WriteLine(\"Hey\")", "Console.WriteLine(\"Hey\")")]
	[TestCase("log.Text", "log.Text")]
	public void ConvertStrictToCSharp(string strictCode, string expectedCSharpCode) =>
		Assert.That(visitor.Visit(ParseExpression(strictCode)), Is.EqualTo(expectedCSharpCode));

	[Test]
	public void GenerateInterfaceMethodBody() =>
		Assert.That(visitor.VisitBlock(method.Body)[0], Is.EqualTo("void Run();"));

	[Test]
	public void GenerateMultilineMethodBody()
	{
		var multilineMethod = new Method(type, this, MethodTests.NestedMethodLines);
		Assert.That(visitor.VisitBlock(multilineMethod.Body),
			Is.EqualTo(new[]
			{
				"public bool IsBla5()", "{", "	var number = 5;", "	if (bla == 5)", "		return true;",
				"	return false;", "}"
			}));
	}
}