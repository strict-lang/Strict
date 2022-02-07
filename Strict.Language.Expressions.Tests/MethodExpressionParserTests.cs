using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public class MethodExpressionParserTests : TestExpressions
{
	[Test]
	public void GetSingleLine()
	{
		var lines = GetMainLines(LetNumber);
		Assert.That(lines, Has.Count.EqualTo(1));
		Assert.That(lines[0], Is.EqualTo(LetNumber));
	}

	private const string LetNumber = "let number = 5";

	[Test]
	public void ParseSingleLine()
	{
		var body = method.TryParse(LetNumber);
		Assert.That(body.ReturnType, Is.EqualTo(type.FindType(Base.None)));
		Assert.That(body.Expressions, Has.Length.EqualTo(1));
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(number.ReturnType));
		Assert.That(body.Expressions[0], Is.TypeOf<Assignment>());
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(LetNumber));
	}

	[Test]
	public void GetMultipleLines()
	{
		var lines = GetMainLines(LetNumber + "\r\n" + LetOther);
		Assert.That(lines, Has.Count.EqualTo(2));
		Assert.That(lines[0], Is.EqualTo(LetNumber));
		Assert.That(lines[1], Is.EqualTo(LetOther));
	}

	private const string LetOther = "let other = 3";

	[Test]
	public void ParseMultipleLines()
	{
		var body = Parse(method, LetNumber + "\r\n" + LetOther);
		Assert.That(body.Expressions, Has.Length.EqualTo(2));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(LetNumber));
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo(LetOther));
	}

	[Test]
	public void GetNestedLines()
	{
		var lines = GetMainLines(NestedLines);
		Assert.That(lines, Has.Count.EqualTo(3));
		Assert.That(lines[0], Is.EqualTo(LetNumber));
		Assert.That(lines[1], Is.EqualTo("if number is 5\n\treturn true"));
		Assert.That(lines[2], Is.EqualTo("return false"));
	}

	private const string NestedLines = @"let number = 5
if number is 5
	return true
return false";

	[Test]
	public void ParseNestedLines()
	{
		var body = Parse(method, NestedLines);
		Assert.That(body.Expressions, Has.Length.EqualTo(3));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(LetNumber));
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo(LetOther));
		Assert.That(body.Expressions[2].ToString(), Is.EqualTo("return false"));
	}

	[Test]
	public void LastExpressionShouldNotHaveNewLine()
	{
		var lines = GetMainLines(@"file.Write(""Hello"")
log.Write(file.Read())
file.Delete()
");
		Assert.That(lines, Has.Count.EqualTo(3));
		Assert.That(lines[2], Is.EqualTo("file.Delete()"));
	}
}