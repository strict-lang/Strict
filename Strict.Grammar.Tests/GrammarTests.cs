using System;
using System.IO;
using Eto.Parse.Grammars;
using NUnit.Framework;

namespace Strict.Grammar.Tests
{
	public class GrammarTests
	{
		[SetUp]
		public void CreateGrammarAndSource()
		{
			grammar = new EbnfGrammar(EbnfStyle.CharacterSets | EbnfStyle.CardinalityFlags |
				EbnfStyle.WhitespaceSeparator | EbnfStyle.EscapeTerminalStrings);
			source = File.ReadAllText("Strict.ebnf");
		}

		private EbnfGrammar grammar;
		private string source;

		[Test]
		public void InvalidGrammarShouldFail() =>
			Assert.That(() => grammar.Build(source, "unknown"), Throws.InstanceOf<ArgumentException>());

		[Test]
		public void StrictGrammarIsValid() => grammar.Build(source, Start);

		private const string Start = "file";

		[Test]
		public void StrictCodeDoesNotCrash() =>
			Assert.That(grammar.ToCode(source, Start), Is.Not.Empty);
		
		[Test]
		public void StrictCodeRuns()
		{
			var parsedMember = grammar.Build(source, Start).Match("has number");
			Assert.That(parsedMember.Errors, Is.Empty, parsedMember.GetErrorMessage(true));
		}

		[Test]
		public void CheckAllBaseFiles()
		{
			foreach (var file in Directory.GetFiles(Directory.Exists(DefaultStrictBasePath)
				? DefaultStrictBasePath
				: @"S:\Strict\Base"))
				Assert.That(grammar.Build(source, Start).Match(File.ReadAllText(file)), Is.Not.Empty);
		}

		private const string DefaultStrictBasePath= @"c:\code\GitHub\strict-lang\Strict\Base";
	}
}