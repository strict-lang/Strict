using System;
using System.IO;
using System.Linq;
using Eto.Parse;
using Eto.Parse.Grammars;
using NUnit.Framework;
using Strict.Language;

namespace Strict.Grammar.Tests;

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

	//ncrunch: no coverage start
	[Test]
	[Ignore("Not yet working")]
	public void StrictCodeRuns()
	{
		var parsedMember = grammar.Build(source, Start).Match("has number");
		Assert.That(parsedMember.Errors, Is.Empty, GetErrorDetails("", parsedMember));
	}

	[Test]
	[Ignore("Not yet working")]
	public void CheckAllBaseFiles()
	{
		foreach (var file in Directory.GetFiles(Directory.Exists(DefaultStrictBasePath)
			? DefaultStrictBasePath
			: Path.Combine(FindSolutionPath(), "..", "Strict", "Base")))
		{
			var checkedGrammar = grammar.Build(source, Start).Match(File.ReadAllText(file));
			Assert.That(checkedGrammar.Errors, Is.Empty,
				GetErrorDetails(file, checkedGrammar));
		}
	}

	private static string GetErrorDetails(string file, GrammarMatch checkedGrammar) =>
		file + " " + checkedGrammar.GetErrorMessage(true) + string.Join(',',
			checkedGrammar.Errors.Select(error =>
				"Error " + error + ": " + error.DescriptiveName + ", Children: " +
				string.Join(',', error.Children.Select(c => c.DescriptiveName))));

	private static string FindSolutionPath() =>
		Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..");

	private static readonly string DefaultStrictBasePath =
		Path.Combine(Repositories.StrictDevelopmentFolderPrefix, "Base");
}