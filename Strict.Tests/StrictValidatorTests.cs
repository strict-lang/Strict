using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict.Tests;

/// <summary>
/// Validators/ValidateCheck.strict reports the same rule as the C# TypeValidator and
/// ConstantCollapser (rule names are the C# exception names) and nothing in all Strict folders.
/// </summary>
[Category("Slow")]
public sealed class StrictValidatorTests
{
	[SetUp]
	public void CaptureConsole()
	{
		consoleWriter = new StringWriter();
		rememberConsole = Console.Out;
		Console.SetOut(consoleWriter);
	}

	private StringWriter consoleWriter = null!;
	private TextWriter rememberConsole = null!;

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsole);

	[TestCase("UnusedMethodVariableMustBeRemoved", "has logger", "Run(number Number)",
		"\tlet unused = number + 1", "\tlogger.Log(number)")]
	[TestCase("VariableDeclaredAsMutableButValueNeverChanged", "has logger", "Run(number Number)",
		"\tmutable count = number", "\tlogger.Log(count)")]
	[TestCase("UnusedMethodParameterMustBeRemoved", "has logger", "Run(number Number)",
		"\tlogger.Log(1)")]
	[TestCase("ParameterHidesMemberUseDifferentName", "has number", "Run(number Number) Number",
		"\tnumber + 1")]
	[TestCase("VariableHidesMemberUseDifferentName", "has count Number", "Run(input Number) Number",
		"\tlet count = input + 1", "\tcount * 2")]
	[TestCase("UnusedMemberMustBeRemoved", "has unused Number", "has logger", "Run",
		"\tlogger.Log(1)")]
	[TestCase("UseConstantHere", "has number = 17 + 4", "Run Number", "	number")]
	[TestCase("ParameterDeclaredAsMutableButValueNeverChanged", "has logger",
		"Run(mutable count Number)", "\tlogger.Log(count)")]
	public async Task StrictValidatorReportsSameRuleAsCSharp(string rule, params string[] lines)
	{
		var typeName = rule[..Math.Min(rule.Length, 40)];
		Assert.That(CSharpRule(typeName, lines), Is.EqualTo(rule));
		var folder = Path.Combine(Path.GetTempPath(), nameof(StrictValidatorTests), rule);
		Directory.CreateDirectory(folder);
		await File.WriteAllLinesAsync(Path.Combine(folder, typeName + Type.Extension), lines);
		await RunValidateCheck(folder);
		Assert.That(consoleWriter.ToString(), Does.Contain(rule));
	}

	private static string CSharpRule(string typeName, string[] lines)
	{
		var parser = new MethodExpressionParser();
		using var type = new Type(TestPackage.Instance, new TypeLines(typeName, lines)).
			ParseMembersAndMethods(parser);
		try
		{
			foreach (var method in type.Methods)
				method.GetBodyAndParseIfNeeded();
			new TypeValidator().Visit(type);
			new ConstantCollapser().Visit(type, true);
			return "";
		}
		catch (ParsingFailed failed)
		{
			return failed.GetType().Name;
		}
	}

	private static Task RunValidateCheck(string folder)
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		return new Runner(Path.Combine(root, "Validators", "ValidateCheck" + Type.Extension),
			folder + " " + root).Run();
	}

	[TestCaseSource(typeof(RunnerTests), nameof(RunnerTests.StrictFolders))]
	public async Task StrictValidatesEveryFile(string folder)
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		await RunValidateCheck(Path.Combine(root, folder));
		Assert.That(consoleWriter.ToString(), Does.Contain("Validation issues: 0"));
	}
}