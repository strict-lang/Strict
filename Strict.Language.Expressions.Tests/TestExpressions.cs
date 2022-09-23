using System;
using System.Collections.Generic;
using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public abstract class TestExpressions : MethodExpressionParser
{
	protected TestExpressions()
	{
		type = new Type(new TestPackage(), new TypeLines("dummy", "Run")).ParseMembersAndMethods(this);
		boolean = type.GetType(Base.Boolean);
		member = new Member(type, "log", new From(type.GetType(Base.Log)));
		((List<Member>)type.Members).Add(member);
		method = new Method(type, 0, this, new[] { MethodTests.Run });
		((List<Method>)type.Methods).Add(method);
		number = new Number(type, 5);
		list = new List(new Body(method), new List<Expression> { new Number(type, 5) });
		bla = new Member(type, "bla", number);
		((List<Member>)type.Members).Add(bla);
	}

	protected readonly Type type;
	protected readonly Type boolean;
	protected readonly Member member;
	protected readonly Method method;
	protected readonly Number number;
	protected readonly Member bla;
	protected readonly List list;

	public void ParseAndCheckOutputMatchesInput(string singleLine, Expression expectedExpression) =>
		ParseAndCheckOutputMatchesInput(new[] { singleLine }, expectedExpression);

	public void ParseAndCheckOutputMatchesInput(string[] lines, Expression expectedExpression)
	{
		var expression = ParseExpression(lines);
		Assert.That(expression, Is.EqualTo(expectedExpression));
		Assert.That(string.Join(Environment.NewLine, lines), Does.StartWith(expression.ToString()));
	}

	public Expression ParseExpression(params string[] lines)
	{
		var methodLines = new string[lines.Length + 1];
		methodLines[0] = MethodTests.Run;
		for (var index = 0; index < lines.Length; index++)
			methodLines[index + 1] = '\t' + lines[index];
		return new Method(type, 0, this, methodLines).GetBodyAndParseIfNeeded();
	}

	protected static MethodCall CreateFromMethodCall(Type fromType, params Expression[] arguments) =>
		new(fromType.FindMethod(Method.From, arguments)!, new From(fromType), arguments);

	protected static Binary CreateBinary(Expression left, string operatorName, Expression right)
	{
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorName, arguments), arguments);
	}

	protected Binary GetCondition(bool isNot = false) =>
		CreateBinary(new MemberCall(null, bla), isNot
			? BinaryOperator.IsNot
			: BinaryOperator.Is, number);
}