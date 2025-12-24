using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class MethodExpressionParserTests : TestExpressions
{
	[Test]
	public void CannotParseEmptyInputException() =>
		Assert.That(() => new MethodExpressionParser().ParseExpression(new Body(method), ""),
			Throws.InstanceOf<CannotParseEmptyInput>());

	[Test]
	public void ParseSingleLine()
	{
		var body =
			(Body)new Method(type, 0, this,
					[MethodTests.Run, MethodTests.ConstantNumber, "\tnumber is Number"]).
				GetBodyAndParseIfNeeded();
		var declaration = (Declaration)body.Expressions[0];
		Assert.That(declaration.ReturnType, Is.EqualTo(type.FindType(Base.Number)));
		Assert.That(declaration.ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
	}

	[Test]
	public void ParseMultipleLines()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run, MethodTests.ConstantNumber, MethodTests.ConstantOther
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(2));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo(MethodTests.ConstantOther[1..]));
	}

	[Test]
	public void ParseNestedLines()
	{
		var body = (Body)new Method(type, 0, this, MethodTests.NestedMethodLines).
			GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(3));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
		Assert.That(body.Expressions[1].ToString(),
			Is.EqualTo(MethodTests.NestedMethodLines[2][1..] + "\r\n" +
				MethodTests.NestedMethodLines[3][1..]));
		Assert.That(body.Expressions[2].ToString(), Is.EqualTo("false"));
	}

	[TestCase("\tError(errorMessage)")]
	[TestCase("\tError(\"error occurred\")")]
	[TestCase("\tError(\"error occurred: \" + errorMessage)")]
	[TestCase("\tError(\"error occurred: \" + errorMessage + \"at line\")")]
	[TestCase("\tError(\"error occurred: \" + errorMessage + \"at line\" + \"5\")")]
	public void ParseErrorExpression(string errorExpression)
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run, MethodTests.ConstantErrorMessage, errorExpression
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.ReturnType, Is.EqualTo(type.FindType(Base.None)));
		Assert.That(body.Expressions, Has.Count.EqualTo(2));
		Assert.That(body.Expressions[0].ReturnType.Name, Is.EqualTo(Base.Text));
		Assert.That(body.Expressions[1], Is.TypeOf<MethodCall>());
	}

	[Test]
	public void IsVariableMutatedInNestedBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tmutable result = 0",
			"\tfor Range(0, 10)",
			"\t\tresult = result + 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInIfThen()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\tresult = 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInIfElse()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\treturn 1",
			"\telse",
			"\t\tresult = 2",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInNestedIfBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\tif true",
			"\t\t\tresult = 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInNestedElseBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\treturn 1",
			"\telse",
			"\t\tif true",
			"\t\t\tresult = 2",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInListCall()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tmutable result = (1, 2)",
			"\tresult(0) = 0",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}
}