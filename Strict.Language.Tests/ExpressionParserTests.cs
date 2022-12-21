using System;
using System.Collections.Generic;
using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public class ExpressionParserTests : ExpressionParser
{
	[SetUp]
	public void CreateType() =>
		type = new Type(new TestPackage(), new MockRunTypeLines()).ParseMembersAndMethods(this);

	private Type type = null!;

	[Test]
	public void ParsingHappensAfterCallingGetBodyAndParseIfNeeded()
	{
		Assert.That(parseWasCalled, Is.False);
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
		Assert.That(parseWasCalled, Is.True);
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ReturnType, Is.EqualTo(type.Methods[0].ReturnType));
	}

	private bool parseWasCalled;

	public class TestExpression : Expression
	{
		public TestExpression(Type returnType) : base(returnType) { }
	}

	public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line)
	{
		parseWasCalled = true;
		return new Number(body.Method, 1);
	}

	//ncrunch: no coverage start, not the focus here
	public override Expression ParseExpression(Body body, ReadOnlySpan<char> text) =>
		new Value(type.GetType(Base.Number), int.Parse(text));

	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text) => null!;
	//ncrunch: no coverage end

	[Test]
	public void CompareExpressions()
	{
		var expression = new TestExpression(type);
		Assert.That(expression, Is.EqualTo(new TestExpression(type)));
		Assert.That(expression.GetHashCode(), Is.EqualTo(new TestExpression(type).GetHashCode()));
		Assert.That(new TestExpression(type.Methods[0].ReturnType),
			Is.Not.EqualTo(new TestExpression(type)));
		Assert.That(expression.Equals((object)new TestExpression(type)), Is.True);
	}

	[Test]
	public void EmptyLineIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, new[] { "Run", "" }),
			Throws.InstanceOf<Type.EmptyLineIsNotAllowed>());

	[Test]
	public void NoIndentationIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, new[] { "Run", "abc" }),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void TooMuchIndentationIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, new[] { "Run", new string('\t', 4) }),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void ExtraWhitespacesAtBeginningOfLineAreNotAllowed() =>
		Assert.That(() => new Method(type, 0, this, new[] { "Run", "\t constant abc = 3" }),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());

	[Test]
	public void ExtraWhitespacesAtEndOfLineAreNotAllowed() =>
		Assert.That(() => new Method(type, 0, this, new[] { "Run", "\tconstant abc = 3 " }),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());

	[Test]
	public void GetSingleLine()
	{
		var method = new Method(type, 0, this, new[] { "Run", MethodTests.LetNumber });
		Assert.That(method.lines, Has.Length.EqualTo(2));
		Assert.That(method.lines[0], Is.EqualTo("Run"));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.LetNumber));
	}

	[Test]
	public void GetMultipleLines()
	{
		var method = new Method(type, 0, this,
			new[] { "Run", MethodTests.LetNumber, MethodTests.LetOther });
		Assert.That(method.lines, Has.Length.EqualTo(3));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.LetNumber));
		Assert.That(method.lines[2], Is.EqualTo(MethodTests.LetOther));
	}

	[Test]
	public void GetNestedLines()
	{
		var method = new Method(type, 0, this, MethodTests.NestedMethodLines);
		Assert.That(method.lines, Has.Length.EqualTo(5));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.LetNumber));
		Assert.That(method.lines[2], Is.EqualTo("	if bla is 5"));
		Assert.That(method.lines[3], Is.EqualTo("		return true"));
		Assert.That(method.lines[4], Is.EqualTo("	false"));
	}
}