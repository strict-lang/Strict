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

		private EbnfGrammar grammar = null!;
		private string source = null!;

		[Test]
		public void InvalidGrammarShouldFail() =>
			Assert.That(() => grammar.Build(source, "unknown"), Throws.InstanceOf<ArgumentException>());

		[Test]
		public void StrictGrammarIsValid() => grammar.Build(source, Start);

		private const string Start = "file";

		[Test]
		public void StrictCodeDoesNotCrash() =>
			Assert.That(grammar.ToCode(source, Start), Is.Not.Empty);
		/*this is pretty hard to use, not really sure if this is the thing we should use, lets write our own grammar parser instead!
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
				: Path.Combine(FindSolutionPath(), "..", "Strict", "Base")))
			{
				var checkedGrammar = grammar.Build(source, Start).Match(File.ReadAllText(file));
				Assert.That(checkedGrammar.Errors, Is.Empty,
					file + " " + string.Join(',',
						checkedGrammar.Errors.Select(e =>
							e + ": " + e.DescriptiveName + ", Children: " +
							string.Join(',', e.Children.Select(c => c.DescriptiveName)))));
			}
		}

		private static string FindSolutionPath() =>
			Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..");

		private static readonly string DefaultStrictBasePath =
			Path.Combine(Repositories.DevelopmentFolder, "Strict", "Base");
		*/
	}
}