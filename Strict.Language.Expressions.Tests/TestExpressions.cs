using System.Collections.Generic;
using NUnit.Framework;
using Strict.Language.Tests;

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
		
		/*old: TODO: remove and simplify
		public override void ParseOldTODO(Method parseMethod, List<DefinitionToken> tokens)
		{
			lastTokens = tokens;
			base.ParseOldTODO(parseMethod, tokens);
		}

		protected List<DefinitionToken> lastTokens;

		protected List<DefinitionToken> GetTokens(string code)
		{
			new MethodBody(method, this, ("Dummy\n\t" + code).SplitLines());
			return lastTokens;
		}
		
		*/
		public void ParseAndCheckOutputMatchesInput(string code, Expression expectedExpression)
		{
			var expression = Parse(method, code);
			Assert.That(expression, Is.EqualTo(expectedExpression));
			Assert.That(expression.ToString(), Is.EqualTo(code));
		}
	}
}