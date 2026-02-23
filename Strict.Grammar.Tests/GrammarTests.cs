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

	[TestCase("has number")]
	[TestCase("has number\n")]
	[TestCase("Run\n\tvalue.Length\n")]
	[TestCase("Run\n\ttrue is \"true\"\n")]
	[TestCase("Run\n\t\"true\"\n")]
	[TestCase("Run\n\ttrue to Text\n")]
	[TestCase("Run\n\ttrue to Text is \"true\"\n")]
	[TestCase("Run\n\t1 is \"one\"\n")]
	[TestCase("Run\n\t1 + \"two\"\n")]
	[TestCase("Run\n\ttrue to Text\n\tfalse to Text\n")]
	[TestCase("Run\n\tnot (true)\n")]
	[TestCase("Run\n\t\"hello\"\n")]
	[TestCase("not Boolean\n\tnot true is false\n\tvalue then false else true\n")]
	[TestCase("and(other) Boolean\n\ttrue and false is false\n\ttrue and true\n\tvalue and other then true else false\n")]
	[TestCase("or(other) Boolean\n\ttrue or false\n\tfalse or false is false\n\tvalue or other then false else true\n")]
	[TestCase("xor(other) Boolean\n\ttrue xor true is false\n\tfalse xor true\n\tnot (false xor false)\n\t(value and other) or (not value and not other) then false else true\n")]
	[TestCase("is(other) Boolean\n\tnot false\n\tfalse is not true\n\tvalue is other\n")]
	[TestCase("to Text\n\ttrue to Text is \"true\"\n\tnot true to Text is \"false\"\n\tvalue then \"true\" else \"false\"\n")]
	public void ParsesValidStrictCode(string code)
	{
		var result = BuildStrictGrammar().Match(code);
		Assert.That(result.Success, Is.True, GetErrorDetails(code, result));
	}

	[Test]
	public void CheckAllBaseFiles()
	{
		var basePath = Directory.Exists(DefaultStrictBasePath)
			? DefaultStrictBasePath
			: Path.Combine(FindSolutionPath(), "Strict.Base");
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
		Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", ".."); //ncrunch: no coverage
}