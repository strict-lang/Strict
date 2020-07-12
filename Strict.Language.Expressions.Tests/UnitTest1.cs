using System.Collections.Generic;
using System.Linq;
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
			method = new Method(type, this, new[] { "Run" });
			number = new Number(method, 5);
		}

		protected readonly Type type;
		protected readonly Method method;
		protected readonly Number number;
		
		public override void Parse(Method parseMethod, List<Token> tokens)
		{
			lastTokens = tokens;
			base.Parse(parseMethod, tokens);
		}

		protected List<Token> lastTokens;
	}

	public class BinaryTests : TestExpressions
	{
		[Test]
		public void ParseBinary()
		{
			var code = "5 + 5";
			Parse(method, GetTokens(code));
			var plus = number.ReturnType.Methods.First(m => m.Name == "+");
			Assert.That(Expressions[0], Is.EqualTo(new Binary(number, plus, number)));
			Assert.That(Expressions[0].ToString(), Is.EqualTo(code));
		}

		private List<Token> GetTokens(string code)
		{
			new MethodBody(method, this, ("Dummy\n\t" + code).SplitLines());
			return lastTokens;
		}
	}

	public class NumberTests : TestExpressions
	{
		[Test]
		public void TwoNumbersWithTheSameValueAreTheSame() =>
			Assert.That(new Number(method, 5), Is.EqualTo(new Number(method, 5)));
	}

	/*TODO: should be done in Expressions.Tests
	[Test]
	public void ParseBody()
	{
		var code = @"log.WriteLine(""Hey"")";
		var method = new Method(type, @"Run", new[] { "\t" + code });
		Assert.That(method.Body.Expressions, Has.Count.EqualTo(1));
		var methodCall = method.Body.Expressions[0] as MethodCall;
		Assert.That(methodCall.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		Assert.That(methodCall.Method.Type, Is.EqualTo(type.GetType(Base.Log)));
		Assert.That(methodCall.Method.Name, Is.EqualTo("WriteLine"));
		Assert.That(methodCall.Arguments.Count, Is.EqualTo(1));
		var text = methodCall.Arguments[0] as Value;
		Assert.That(text.Data, Is.EqualTo("Hey"));
		Assert.That(methodCall.ToString(), Is.EqualTo(code));
	}

	[Test]
	public void ParseValues()
	{
		var code = @"return 5 + true";
		var method = new Method(type, @"Run returns Number", new[] { "\t" + code });
		var returnExpression = method.Body.Expressions[0] as Return;
		Assert.That(returnExpression.ReturnType, Is.EqualTo(type.GetType(Base.Number)));
		var binary = returnExpression.Expression as Binary;
		Assert.That(binary.Left, Is.InstanceOf<Number>());
		Assert.That(binary.Right, Is.InstanceOf<Boolean>());
		Assert.That(returnExpression.ToString(), Is.EqualTo(code));
	}
	*/
}