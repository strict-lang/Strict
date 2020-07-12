using NUnit.Framework;

namespace Strict.Tokens.Tests
{
	public class TokenTests
	{
		[Test]
		public void TokensWithTheSameTypeAreAlwaysTheSame()
		{
			Assert.That(Token.Test, Is.EqualTo(Token.Test));
			Assert.That(Token.Is, Is.EqualTo(Token.Is));
			Assert.That(Token.FromKeyword(Keyword.From),
				Is.EqualTo(Token.FromKeyword(Keyword.From)));
			Assert.That(Token.FromNumber("123"), Is.EqualTo(Token.FromNumber(123)));
			Assert.That(Token.FromIdentifier("Hello"), Is.EqualTo(Token.FromIdentifier("Hello")));
		}

		[Test]
		public void TokenToString()
		{
			Assert.That(Token.Test.ToString(), Is.EqualTo(Keyword.Test));
			Assert.That(Token.Open.ToString(), Is.EqualTo(Operator.Open));
			Assert.That(Token.FromKeyword(Keyword.From).ToString(), Is.EqualTo(Keyword.From));
			Assert.That(Token.FromNumber(123).ToString(), Is.EqualTo("123"));
			Assert.That(Token.FromIdentifier("Hello").ToString(), Is.EqualTo("Hello"));
		}

		[Test]
		public void KeywordTokenMustBeValid() =>
			Assert.Throws<Token.InvalidKeyword>(() =>
				Token.FromKeyword(nameof(KeywordTokenMustBeValid)));

		[Test]
		public void OperatorTokenMustBeValid() =>
			Assert.Throws<Token.InvalidOperator>(() =>
				Token.FromOperator(nameof(OperatorTokenMustBeValid)));
		
		[Test]
		public void CheckTokenType()
		{
			Assert.That(Token.FromOperator(Operator.Assign).IsIdentifier, Is.False);
			Assert.That(Token.FromOperator(Operator.Open).IsNumber, Is.False);
			Assert.That(Token.FromOperator(Operator.Close).IsText, Is.False);
			Assert.That(Token.FromIdentifier(nameof(CheckTokenType)).IsIdentifier, Is.True);
			Assert.That(Token.FromNumber(5).IsNumber, Is.True);
			Assert.That(Token.FromText(nameof(TokenTests)).IsText, Is.True);
		}

		[Test]
		public void CheckIfIsBinaryOperator()
		{
			Assert.That(Operator.Open.IsBinaryOperator(), Is.False);
			Assert.That(Operator.Plus.IsBinaryOperator(), Is.True);
		}
	}
}