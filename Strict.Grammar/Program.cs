using Eto.Parse;
using Eto.Parse.Grammars;
using Strict.Language;

//ncrunch: no coverage start, just for testing and debugging if all grammar rules work
var g = new EbnfGrammar(EbnfStyle.CharacterSets | EbnfStyle.CardinalityFlags |
	EbnfStyle.EscapeTerminalStrings);
var s = File.ReadAllText("Strict.ebnf");
var built = g.Build(s, "file");
built.Separator = (Eto.Parse.Terminals.Set(' ') | Eto.Parse.Terminals.Set('\r')).Repeat(0);
ShowResult("has number", built.Match("has number"));
ShowResult("dot notation", built.Match("Run\n\tvalue.Length\n"));
ShowResult("dot simple", built.Match("value.Length"));
ShowResult("is string", built.Match("Run\n\ttrue is \"true\"\n"));
ShowResult("just string", built.Match("Run\n\t\"true\"\n"));
ShowResult("conv+is", built.Match("Run\n\ttrue to Text\n"));
ShowResult("conv+is+str", built.Match("Run\n\ttrue to Text is \"true\"\n"));
ShowResult("is+str only", built.Match("Run\n\t1 is \"one\"\n"));
ShowResult("binop+str", built.Match("Run\n\t1 + \"two\"\n"));
ShowResult("nested conv", built.Match("Run\n\ttrue to Text\n\tfalse to Text\n"));
ShowResult("group expr", built.Match("Run\n\tnot (true)\n"));
ShowResult("string", built.Match("Run\n\t\"hello\"\n"));
ShowResult("decimal", built.Match("has number\n"));
const string DefaultStrictBasePath = Repositories.StrictDevelopmentFolderPrefix + "Base";
var basePath = Directory.Exists(DefaultStrictBasePath)
	? DefaultStrictBasePath
	: Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "..", "..", "Strict.Base");
foreach (var file in Directory.GetFiles(basePath, "*.strict"))
	ShowResult(Path.GetFileName(file), built.Match(File.ReadAllText(file)));

static void ShowResult(string filename, GrammarMatch result)
{
	Console.WriteLine(
		$"{filename}: Success={result.Success}, Length={result.Length}, Errors={result.Errors.Count<object>()}, ErrorIndex={result.ErrorIndex}");
	//if (!result.Success)
	//	foreach (var error in result.Errors)
	//		Console.WriteLine(error.GetErrorMessage());
}