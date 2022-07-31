using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public abstract class TestExpressions : MethodExpressionParser
{
	protected TestExpressions()
	{
		type = new Type(new TestPackage(), "dummy", this);
		boolean = type.GetType(Base.Boolean);
		//use Boolean: binaryOperators = type.GetType(Base.BinaryOperator).Methods;
		//use Boolean: unaryOperators = type.GetType(Base.UnaryOperator).Methods;
		member = new Member("log", new From(type.GetType(Base.Log)));
		((List<Member>)type.Members).Add(member);
		method = new Method(type, 0, this, new[] { MethodTests.Run });
		((List<Method>)type.Methods).Add(method);
		number = new Number(type, 5);
		list = new List(type, new List<Expression> { new Number(type, 5) });
		bla = new Member("bla", number);
		((List<Member>)type.Members).Add(bla);
	}

	protected readonly Type type;
	protected readonly Type boolean;
	//protected readonly IReadOnlyList<Method> binaryOperators;
	//protected readonly IReadOnlyList<Method> unaryOperators;
	protected readonly Member member;
	protected readonly Method method;
	protected readonly Number number;
	protected readonly Member bla;
	protected readonly List list;

	public void ParseAndCheckOutputMatchesInput(string code, Expression expectedExpression)
	{
		var expression = ParseExpression(code);
		Assert.That(expression, Is.EqualTo(expectedExpression));
		Assert.That(expression.ToString(), Is.EqualTo(code));
	}

	public Expression ParseExpression(params string[] lines)
	{
		var methodLines = lines.Select(line => '\t' + line).ToList();
		methodLines.Insert(0, MethodTests.Run);
		var body = new Method(type, 0, this, methodLines).Body;
		return body.Expressions.Count == 1
			? body.Expressions[0]
			: throw new MultipleExpressionsGiven();
	}

	public sealed class MultipleExpressionsGiven : Exception { }
}