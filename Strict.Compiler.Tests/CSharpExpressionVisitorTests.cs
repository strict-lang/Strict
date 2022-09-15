using System.Linq;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Expressions.Tests;
using Strict.Language.Tests;

namespace Strict.Compiler.Tests;

//[Ignore("TODO: Not yet done")]
public sealed class CSharpExpressionVisitorTests : TestExpressions
{
	[SetUp]
	public void CreateVisitor() => visitor = new CSharpExpressionVisitor();

	private CSharpExpressionVisitor visitor = null!;

	[Ignore("TODO: Not yet done")]
	[Test]
	public void ShouldCallVisitBlockForBlockExpressions() =>
		Assert.That(() => visitor.Visit(method.GetBodyAndParseIfNeeded())[0],
			Throws.InstanceOf<ExpressionVisitor.UseVisitBody>());

	[Ignore("TODO: Not yet done")]
	[Test]
	public void GenerateAssignment() =>
		Assert.That(visitor.Visit(new Assignment((Body)method.GetBodyAndParseIfNeeded(), nameof(number), number)),
			Is.EqualTo("var number = 5"));

	[Test]
	public void GenerateBinary() =>
		Assert.That(visitor.Visit(CreateBinary(number, BinaryOperator.Plus, number)),
			Is.EqualTo("5 + 5"));

	[Test]
	public void GenerateBoolean() =>
		Assert.That(visitor.Visit(new Boolean(method, false)), Is.EqualTo("false"));

	[Test]
	public void GenerateMemberCall() =>
		Assert.That(
			visitor.Visit(new MemberCall(new MemberCall(null, member),
				member.Type.Members.First(m => m.Name == "Text"))), Is.EqualTo("log.Text"));

	[Test]
	public void GenerateMethodCall() =>
		Assert.That(
			visitor.Visit(new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				new Expression[] { new Text(type, "Hi") })), Is.EqualTo("Console.WriteLine(\"Hi\")"));

	[Test]
	public void GenerateNumber() =>
		Assert.That(visitor.Visit(new Number(method, 77)), Is.EqualTo("77"));

	[Test]
	public void GenerateText() =>
		Assert.That(visitor.Visit(new Text(method, "Hey")), Is.EqualTo("\"Hey\""));

	[TestCase("let other = 3", "var other = 3")]
	[TestCase("5 + 5", "5 + 5")]
	[TestCase("true", "true")]
	[TestCase("\"Hey\"", "\"Hey\"")]
	[TestCase("42", "42")]
	[TestCase("log.Write(\"Hey\")", "Console.WriteLine(\"Hey\")")]
	[TestCase("log.Text", "log.Text")]
	public void ConvertStrictToCSharp(string strictCode, string expectedCSharpCode) =>
		Assert.That(visitor.Visit(ParseExpression(strictCode)), Is.EqualTo(expectedCSharpCode));

	[Ignore("TODO: Not yet done")]
	[Test]
	public void GenerateInterfaceMethodBody() =>
		Assert.That(visitor.VisitBody(method.GetBodyAndParseIfNeeded())[0], Is.EqualTo("void Run();"));

	[Test]
	public void GenerateMultilineMethodBody()
	{
		var multilineMethod = new Method(type, 0, this, MethodTests.NestedMethodLines);
		Assert.That(visitor.VisitBody(multilineMethod.GetBodyAndParseIfNeeded()), Is.EqualTo(new[]
		{
			// @formatter:off
			"public bool IsBlaFive()",
			"{",
			"	var number = 5;",
			"	if (bla == 5)",
			"		return true;",
			"	false;",
			"}"
		}));
	}

	[Test]
	public void GenerateIfElse()
	{
		var multilineMethod = new Method(type, 0, this,
			new[]
			{
				"IsBlaFive Boolean",
				"	let value = 5",
				"	if value is 5",
				"		return true",
				"	else",
				"		return false"
			});
		Assert.That(visitor.VisitBody(multilineMethod.GetBodyAndParseIfNeeded()),
			Is.EqualTo(new[]
			{
				"public bool IsBlaFive()",
        "{",
				"	var value = 5;",
				"	if (value == 5)",
				"		return true;",
				"	else",
				"		return false;",
				"}"
			}));
	}
}