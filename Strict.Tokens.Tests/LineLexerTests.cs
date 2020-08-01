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
		private readonly List<DefinitionToken> tokens = new List<DefinitionToken>();
		public void Add(DefinitionToken token) => tokens.Add(token);

		[Test]
		public void EveryLineMustStartWithTab() =>
			Assert.Throws<LineLexer.LineMustStartWithTab>(() => lineLexer.Process("test"));
		
		[Test]
		public void MultipleSpacesAreNotAllowed() =>
			Assert.Throws<LineLexer.UnexpectedSpaceOrEmptyParenthesisDetected>(() => lineLexer.Process("	  "));

		[Test]
		public void FindSingleToken()
		{
			CheckSingleToken("	test(", DefinitionToken.Open);
			CheckSingleToken("	test(number)", DefinitionToken.Close);
			/*unused
			CheckSingleToken("	is", DefinitionToken.Is);
			CheckSingleToken("	test", DefinitionToken.Test);
			*/
			CheckSingleToken("	53", DefinitionToken.FromNumber(53));
			CheckSingleToken("	number", DefinitionToken.FromIdentifier("number"));
		}

		private void CheckSingleToken(string line, DefinitionToken expectedLastToken)
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
		/*nah
		[Test]
		public void ProcessLine()
		{
			lineLexer.Process("	test(1) is 2");
			Assert.That(lineLexer.Tabs, Is.EqualTo(1));
			Assert.That(tokens,
				Is.EqualTo(new List<DefinitionToken>
				{
					DefinitionToken.Test,
					DefinitionToken.Open,
					DefinitionToken.FromNumber(1),
					DefinitionToken.Close,
					DefinitionToken.Is,
					DefinitionToken.FromNumber(2)
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
				Is.EqualTo(new List<DefinitionToken>
				{
					DefinitionToken.Test,
					DefinitionToken.Open,
					DefinitionToken.FromNumber(1),
					DefinitionToken.Close,
					DefinitionToken.Is,
					DefinitionToken.FromNumber(2),
					DefinitionToken.Let,
					DefinitionToken.FromIdentifier("doubled"),
					DefinitionToken.Assign,
					DefinitionToken.FromIdentifier("number"),
					DefinitionToken.Plus,
					DefinitionToken.FromIdentifier("number"),
					DefinitionToken.Return,
					DefinitionToken.FromIdentifier("doubled")
				}));
		}

		[Test]
		public void ProcessMethodCall()
		{
			lineLexer.Process("	log.WriteLine(\"Hey\")");
			Assert.That(tokens,
				Is.EqualTo(new List<DefinitionToken>
				{
					DefinitionToken.FromIdentifier("log"),
					DefinitionToken.Dot,
					DefinitionToken.FromIdentifier("WriteLine"),
					DefinitionToken.Open,
					DefinitionToken.FromText("Hey"),
					DefinitionToken.Close
				}));
		}
		*/
	}
}