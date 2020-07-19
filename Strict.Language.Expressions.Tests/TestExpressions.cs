using System.Collections.Generic;
using NUnit.Framework;
using Strict.Language.Tests;
using Strict.Tokens;

namespace Strict.Language.Expressions.Tests
{
	public abstract class TestExpressions : AllExpressionParser
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
		
		//old: TODO: remove and simplify
		public override void ParseOldTODO(Method parseMethod, List<Token> tokens)
		{
			lastTokens = tokens;
			base.ParseOldTODO(parseMethod, tokens);
		}

		protected List<Token> lastTokens;

		protected List<Token> GetTokens(string code)
		{
			new MethodBody(method, this, ("Dummy\n\t" + code).SplitLines());
			return lastTokens;
		}

		public void ParseAndCheckOutputMatchesInput(string code, Expression expectedExpression)
		{
			ParseOldTODO(method, GetTokens(code));
			Assert.That(Expressions[0], Is.EqualTo(expectedExpression));
			Assert.That(Expressions[0].ToString(), Is.EqualTo(code));
		}
	}
}