using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class ExpressionParserTests : ExpressionParser
	{
		[SetUp]
		public void CreateType() =>
			type = new Type(new TestPackage(), nameof(TypeTests), this).Parse(@"has log
Run
	log.WriteLine");

		private Type type = null!;

		[Test]
		public void ParsingHappensAfterCallingBody()
		{
			Assert.That(parseWasCalled, Is.False);
			Assert.That(type.Methods[0].Body, Is.Not.Null);
			Assert.That(parseWasCalled, Is.True);
			Assert.That(type.Methods[0].Body.Expressions, Is.Not.Empty);
			Assert.That(type.Methods[0].Body.ReturnType, Is.EqualTo(type.Methods[0].ReturnType));
		}

		// unused
		// public override void ParseOld(Method method, List<DefinitionToken> tokens)
		// {
		//		if (tokens.Count == 3)
		//				tokens.Clear();
		//		expressions.Add(new TestExpression(method.ReturnType));
		// }
		// [Test]
		// public void ThereMustBeNoTokensLeft() =>
		//		Assert.Throws<MethodBody.UnprocessedTokensAtEndOfFile>(() =>
		//				new MethodBody(type.Methods[0], this, new[] { "Dummy", "\tdummy" }));

		public override Expression Parse(Method context, string lines)
		{
			parseWasCalled = true;
			return new MethodBody(context,
				new Expression[] { new TestExpression(type.Methods[0].ReturnType) });
		}

		private bool parseWasCalled;

		public class TestExpression : Expression
		{
			public TestExpression(Type returnType) : base(returnType) { }
		}

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
	}
}