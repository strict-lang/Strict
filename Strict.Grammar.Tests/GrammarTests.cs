using Eto.Parse;
using Eto.Parse.Grammars;
using NUnit.Framework;
using Strict.Language;

namespace Strict.Grammar.Tests;

public sealed class GrammarTests
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

	[Test]
	public void StrictCodeRuns()
	{
		var parsedMember = BuildStrictGrammar().Match("has number");
		Assert.That(parsedMember.Success, Is.True, GetErrorDetails("", parsedMember));
	}

	[Test]
	public void CheckAllBaseFiles()
	{
		var basePath = Directory.Exists(DefaultStrictBasePath)
			? DefaultStrictBasePath
			: Path.Combine(FindSolutionPath(), "..", "..", "Strict.Base");
		foreach (var file in Directory.GetFiles(basePath, "*.strict"))
		{
			var result = BuildStrictGrammar().Match(File.ReadAllText(file));
			Assert.That(result.Success, Is.True, file + ": " + GetErrorDetails("", result));
		}
	}

	/// <summary>
	/// WhitespaceSeparator consumes all whitespace including \n and \t, but the grammar needs
	/// those as explicit LF and TAB tokens for block structure. Override to only consume spaces
	/// and carriage returns, preserving newlines and tabs for the grammar rules.
	/// </summary>
	private Eto.Parse.Grammar BuildStrictGrammar()
	{
		var built = grammar.Build(source, Start);
		built.Separator = (Terminals.Set(' ') | Terminals.Set('\r')).Repeat(0);
		built.Optimizations = GrammarOptimizations.CharacterSetAlternations |
			GrammarOptimizations.TrimUnnamedUnaryParsers |
			GrammarOptimizations.TrimSingleItemSequencesOrAlterations;
		return built;
	}

	private const string DefaultStrictBasePath = Repositories.StrictDevelopmentFolderPrefix + "Base";

	private static string GetErrorDetails(string file, GrammarMatch checkedGrammar) =>
		file + " " + checkedGrammar.GetErrorMessage(true) + string.Join(',',
			checkedGrammar.Errors.Select(error =>
				"Error " + error + ": " + error.DescriptiveName + ", Children: " +
				string.Join(',', error.Children.Select(c => c.DescriptiveName))));

	private static string FindSolutionPath() =>
		Path.Combine(Directory.GetCurrentDirectory(), "..", "..", ".."); //ncrunch: no coverage
}