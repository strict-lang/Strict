using Eto.Parse;
using Eto.Parse.Grammars;
using NUnit.Framework;
using Strict.Language;

namespace Strict.Grammar.Tests;

public sealed class GrammarTests
{
	[TestCase("has number")]
	[TestCase("has some Text")]
	[TestCase("has number = 5")]
	[TestCase("has number = 5\nhas text = \"hi\"")]
	[TestCase("constant NewLine = Character(13)")]
	[TestCase("Run\n\t5")]
	[TestCase("Run\n\tnot true")]
	[TestCase("Run\n\ttrue to Text is \"true\"")]
	[TestCase("Run\n\tnot true to Text is \"false\"")]
	[TestCase("Run\n\tvalue then \"true\" else \"false\"")]
	[TestCase("Run\n\tvalue.Length")]
	[TestCase("Run\n\ttrue is not \"true\"")]
	[TestCase("Run\n\t\"hello\"")]
	[TestCase("Run\n\ttrue to Text")]
	[TestCase("Run\n\t1 is not \"one\"")]
	[TestCase("Run\n\t\"hello \" + \"world\"")]
	[TestCase("Run\n\ttrue to Text\n\tfalse to Text")]
	[TestCase("Run\n\tconstant notANumber = Error")]
	[TestCase("Run\n\tresult is in Range(0, 10) then result else notANumber(value)")]
	[TestCase("Run\n\tCharacter(\"A\") to Number is notANumber")]
	[TestCase("not Boolean\n\tnot true is false\n\tvalue then false else true")]
	[TestCase("and(other) Boolean\n\ttrue and false is false\n\ttrue and true\n\tvalue and other then true else false")]
	[TestCase("or(other) Boolean\n\ttrue or false\n\tfalse or false is false\n\tvalue or other then false else true")]
	[TestCase("xor(other) Boolean\n\ttrue xor true is false\n\tfalse xor true\n\tnot (false xor false)\n\t(value and other) or (not value and not other) then false else true")]
	[TestCase("is(other) Boolean\n\tnot false\n\tfalse is not true\n\tvalue is other")]
	[TestCase("to Text\n\ttrue to Text is \"true\"\n\tnot true to Text is \"false\"\n\tvalue then \"true\" else \"false\"")]
	[TestCase("method(arg Number, argTwo Number)")]
	[TestCase("has keysAndValues List(key Generic, mappedValue Generic)")]
	[TestCase("Run\n\tError(\"Key \" + key + \" not found\")")]
	[TestCase("from\nto Type\nto Text")]
	[TestCase("from\n\tDictionary(Number, Number).Length is 0")]
	[TestCase("Get(key Generic) Generic\n\tfor keysAndValues\n\t\tif value.Key is key\n" +
		"\t\t\treturn value.Value")]
	[TestCase("Add(key Generic, mappedValue Generic) Mutable(Dictionary)\n" +
		"\tconstant DuplicateKey = Error\n" +
		"\tDictionary((1, 1)).Add(1, 1) is DuplicateKey")]
	[TestCase("CompareTo(member) Number\n\tvalue > member.Value then 1 else value < member.Value then -1 else 0")]
	[TestCase("Run\n\tsomeList(1) = 5")]
	[TestCase("has text with Length > 1 and \" \" is not in value")]
	[TestCase("digits Numbers\n\t1.digits is (1)")]
	[TestCase(">=(other) Boolean\n\t0 >= 0\n\tvalue >= other")]
	[TestCase("Reverse Range\n\tLength > 0 then Range(ExclusiveEnd - 1, Start - 1) else Range(ExclusiveEnd + 1, Start + 1)")]
	[TestCase("to Text\n\tto Text is \"\tat Stacktrace.to in Base\\Stacktrace.strict:line 7\"")]
	[TestCase("+(other) Text\n\t+(\"more\") is \"more\"")]
	[TestCase("to Number\n\t\"1e10\" to Number is 1e10")]
	public void ParsesValidStrictCode(string code)
	{
		var result = BuildGrammar().Match(code + "\n");
		Assert.That(result.Success, Is.True, GetErrorDetails(code, result));
	}

	private static Eto.Parse.Grammar BuildGrammar()
	{
		var source = File.ReadAllText("Strict.ebnf");
		return new EbnfGrammar(EbnfStyle.CharacterSets | EbnfStyle.CardinalityFlags |
			EbnfStyle.EscapeTerminalStrings).Build(source, "file");
	}

	private static string GetErrorDetails(string file, GrammarMatch checkedGrammar) =>
		file + " " + checkedGrammar.GetErrorMessage(true) + string.Join(',',
			checkedGrammar.Errors.Select(error =>
				"Error " + error + ": " + error.DescriptiveName + ", Children: " +
				string.Join(',', error.Children.Select(c => c.DescriptiveName))));

	[Test]
	public void CheckAllBaseFiles()
	{
		var basePath = Directory.Exists(DefaultStrictBasePath)
			? DefaultStrictBasePath
			: Path.Combine(FindSolutionPath(), "Strict.Base");
		foreach (var file in Directory.GetFiles(basePath, "*.strict"))
		{
			var result = BuildGrammar().Match(File.ReadAllText(file).Replace("\r\n", "\n") + "\n");
			Assert.That(result.Success, Is.True, file + ": " + GetErrorDetails("", result));
		}
	}

	private const string DefaultStrictBasePath = Repositories.StrictDevelopmentFolderPrefix + "Base";

	private static string FindSolutionPath() =>
		Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", ".."); //ncrunch: no coverage
}