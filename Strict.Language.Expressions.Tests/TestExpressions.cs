using System;
using System.Collections.Generic;
using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests
{
	public abstract class TestExpressions : MethodExpressionParser
	{
		protected TestExpressions()
		{
			type = new Type(new TestPackage(), "dummy", this);
			member = new Member(type, "log");
			((List<Member>)type.Members).Add(member);
			method = new Method(type, this, new[] { "Run" });
			((List<Method>)type.Methods).Add(method);
			number = new Number(method, 5);
		}

		protected readonly Type type;
		protected readonly Member member;
		protected readonly Method method;
		protected readonly Number number;

		public void ParseAndCheckOutputMatchesInput(string code, Expression expectedExpression)
		{
			var expression = ParseExpression(method, code);
			Assert.That(expression, Is.EqualTo(expectedExpression));
			Assert.That(expression.ToString(), Is.EqualTo(code));
		}
		
		public Expression ParseExpression(Method context, string lines)
		{
			var body = base.Parse(context, lines) as MethodBody;
			if (body.Expressions.Count == 1)
				return body.Expressions[0];
			throw new MultipleExpressionsGiven();
		}

		public class MultipleExpressionsGiven : Exception { }
	}
}