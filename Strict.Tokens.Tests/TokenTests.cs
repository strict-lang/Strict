using NUnit.Framework;

namespace Strict.Tokens.Tests
{
	public class TokenTests
	{
		[Test]
		public void TokensWithTheSameTypeAreAlwaysTheSame()
		{
			Assert.That(DefinitionToken.Close, Is.EqualTo(DefinitionToken.Close));
			Assert.That(DefinitionToken.FromIdentifier("abc"),
				Is.EqualTo(DefinitionToken.FromIdentifier("abc")));
			Assert.That(DefinitionToken.FromNumber("123"),
				Is.EqualTo(DefinitionToken.FromNumber(123)));
		}

		[Test]
		public void TokenToString()
		{
			Assert.That(DefinitionToken.Open.ToString(), Is.EqualTo("("));
			Assert.That(DefinitionToken.FromNumber(123).ToString(), Is.EqualTo("123"));
			Assert.That(DefinitionToken.FromIdentifier("Hello").ToString(), Is.EqualTo("Hello"));
		}
		/*unused
		[Test]
		public void KeywordTokenMustBeValid() =>
			Assert.Throws<DefinitionToken.InvalidKeyword>(() =>
				DefinitionToken.FromKeyword(nameof(KeywordTokenMustBeValid)));

		[Test]
		public void OperatorTokenMustBeValid() =>
			Assert.Throws<DefinitionToken.InvalidOperator>(() =>
				DefinitionToken.FromOperator(nameof(OperatorTokenMustBeValid)));

		[Test]
		public void CheckIfIsBinaryOperator()
		{
			Assert.That(Operator.Open.IsBinaryOperator(), Is.False);
			Assert.That(Operator.Plus.IsBinaryOperator(), Is.True);
		}
		*/
		[Test]
		public void CheckTokenType()
		{
			Assert.That(DefinitionToken.Open.IsIdentifier, Is.False);
			Assert.That(DefinitionToken.Open.IsNumber, Is.False);
			Assert.That(DefinitionToken.Close.IsText, Is.False);
			Assert.That(DefinitionToken.FromIdentifier(nameof(CheckTokenType)).IsIdentifier, Is.True);
			Assert.That(DefinitionToken.FromNumber(5).IsNumber, Is.True);
			Assert.That(DefinitionToken.FromText(nameof(TokenTests)).IsText, Is.True);
		}
	}
}