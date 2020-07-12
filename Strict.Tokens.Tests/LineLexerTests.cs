using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace Strict.Tokens.Tests
{
	public class LineLexerTests : Tokenizer
	{
		[SetUp]
		public void CreateLexer()
		{
			lineLexer = new LineLexer(this);
			tokens.Clear();
		}

		private LineLexer lineLexer;
		private readonly List<Token> tokens = new List<Token>();
		public void Add(Token token) => tokens.Add(token);

		[Test]
		public void EveryLineMustStartWithTab() =>
			Assert.Throws<LineLexer.LineMustStartWithTab>(() => lineLexer.Process("test"));
		
		[Test]
		public void MultipleSpacesAreNotAllowed() =>
			Assert.Throws<LineLexer.UnexpectedSpaceOrEmptyParenthesisDetected>(() => lineLexer.Process("	  "));

		[Test]
		public void FindSingleToken()
		{
			CheckSingleToken("	test(", Token.Open);
			CheckSingleToken("	test(number)", Token.Close);
			CheckSingleToken("	is", Token.Is);
			CheckSingleToken("	test", Token.Test);
			CheckSingleToken("	53", Token.FromNumber(53));
			CheckSingleToken("	number", Token.FromIdentifier("number"));
		}

		private void CheckSingleToken(string line, Token expectedLastToken)
		{
			tokens.Clear();
			lineLexer.Process(line);
			Assert.That(tokens.Last(), Is.EqualTo(expectedLastToken));
		}
		
		[Test]
		public void NumbersInIdentifiersAreNotAllowed() =>
			Assert.Throws<LineLexer.InvalidIdentifierName>(() => lineLexer.Process("	let abc1"));
		
		[Test]
		public void AllUpperCaseIdentifiersAreNotAllowed() =>
			Assert.Throws<LineLexer.InvalidIdentifierName>(() => lineLexer.Process("	let AAA"));

		[Test]
		public void ProcessLine()
		{
			lineLexer.Process("	test(1) is 2");
			Assert.That(lineLexer.Tabs, Is.EqualTo(1));
			Assert.That(tokens,
				Is.EqualTo(new List<Token>
				{
					Token.Test,
					Token.Open,
					Token.FromNumber(1),
					Token.Close,
					Token.Is,
					Token.FromNumber(2)
				}));
		}

		[Test]
		public void ProcessMultipleLines()
		{
			lineLexer.Process("	test(1) is 2");
			lineLexer.Process("	let doubled = number + number");
			lineLexer.Process("	return doubled");
			Assert.That(lineLexer.Tabs, Is.EqualTo(1));
			Assert.That(tokens,
				Is.EqualTo(new List<Token>
				{
					Token.Test,
					Token.Open,
					Token.FromNumber(1),
					Token.Close,
					Token.Is,
					Token.FromNumber(2),
					Token.Let,
					Token.FromIdentifier("doubled"),
					Token.Assign,
					Token.FromIdentifier("number"),
					Token.Plus,
					Token.FromIdentifier("number"),
					Token.Return,
					Token.FromIdentifier("doubled")
				}));
		}

		[Test]
		public void ProcessMethodCall()
		{
			lineLexer.Process("	log.WriteLine(\"Hey\")");
			Assert.That(tokens,
				Is.EqualTo(new List<Token>
				{
					Token.FromIdentifier("log"),
					Token.Dot,
					Token.FromIdentifier("WriteLine"),
					Token.Open,
					Token.FromText("Hey"),
					Token.Close
				}));
		}
	}
}