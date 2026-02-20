using System.Diagnostics;
using NUnit.Framework;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

//[Ignore("TODO: fix later")]
public sealed class SourceGeneratorTests : TestCSharpGenerator
{
	[Test]
	public void GenerateCSharpInterface()
	{
		var app = new Type(package, new TypeLines("DummyApp", "Run")).ParseMembersAndMethods(parser);
		var file = generator.Generate(app);
		Assert.That(file.ToString(), Is.EqualTo(@"namespace SourceGeneratorTests;

public interface DummyApp
{
	void Run();
}"));
	}

	[Test]
	public void GenerateCSharpClass()
	{
		using var program = CreateHelloWorldProgramType();
		var file = generator.Generate(program);
		Assert.That(file.ToString(), Is.EqualTo(@"namespace SourceGeneratorTests;

public class Program
{
	public App App;
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}"), file.ToString());
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void CreateFileAndWriteIntoIt()
	{
		using var program = new Type(package, new TypeLines(nameof(CreateFileAndWriteIntoIt),
			"has App", // App has run funtion so its used as a trait with implementation
			"has file = \"" + TemporaryFile + "\"", // component because its initialized
			"has output", //
			"has file", // means implementation? should error
			"has logger",
			"Run2",
			"\tfile.Write(\"Hello\")",
			"\toutput.Write(5)",
			"\tlogger.Log(6)",
			"\tfileSystem.Write(5)")).ParseMembersAndMethods(parser);
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode), Is.EqualTo(""));
		Assert.That(File.Exists(Path.Combine(ProjectFolder, TemporaryFile)), Is.True);
		Assert.That(File.ReadAllText(Path.Combine(ProjectFolder, TemporaryFile)),
			Is.EqualTo("Hello"));
	}

	private const string TemporaryFile = "temp.txt";

	[Test]
	[Category("Slow")]
	public void GenerateFileReadProgram()
	{
		if (!Directory.Exists(ProjectFolder))
			Directory.CreateDirectory(ProjectFolder);
		File.WriteAllText(Path.Combine(ProjectFolder, TestTxt), ExpectedText);
		using var program =
			new Type(package,
					new TypeLines(nameof(GenerateFileReadProgram), "has App",
						"has file = \"" + TestTxt + "\"", "has logger", "Run", "\tlogger.Log(file.Read)")).
				ParseMembersAndMethods(parser);
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo(ExpectedText + Environment.NewLine));
		Assert.That(File.Exists(Path.Combine(ProjectFolder, TestTxt)), Is.True);
	}

	private const string ProjectFolder = nameof(GenerateFileReadProgram);
	private const string ExpectedText = "Hello, World";
	private const string TestTxt = "test.txt";

	private static string GenerateNewConsoleAppAndReturnOutput(string folder, string generatedCode)
	{
		if (!Directory.Exists(folder))
			CreateFolderOnceByCreatingDotnetProject(folder, generatedCode);
		File.WriteAllText(Path.Combine(folder, "Program.cs"), generatedCode);
		var actualText = RunDotnetAndReturnOutput(folder, "run", out var error);
		return error.Length > 0
			? throw new CSharpCompilationFailed(error, actualText, generatedCode)
			: actualText;
	}

	private static void CreateFolderOnceByCreatingDotnetProject(string folder, string generatedCode)
	{
		var creationOutput =
			RunDotnetAndReturnOutput("", "new console --force --name " + folder, out var creationError);
		if (!creationOutput.Contains("successful"))
			throw new CSharpCompilationFailed(creationError, creationOutput, generatedCode);
	}

	private static string RunDotnetAndReturnOutput(string folder, string argument, out string error)
	{
		var process = new Process
		{
			StartInfo = new ProcessStartInfo
			{
				WorkingDirectory = folder,
				FileName = "dotnet",
				Arguments = argument,
				UseShellExecute = false,
				RedirectStandardOutput = true,
				RedirectStandardError = true
			}
		};
		process.Start();
		error = process.StandardError.ReadToEnd();
		return process.StandardOutput.ReadToEnd();
	}

	public sealed class
		CSharpCompilationFailed(string error, string actualText, string generatedCode) : Exception(
		error + Environment.NewLine + actualText + Environment.NewLine + nameof(generatedCode) + ":" +
		Environment.NewLine + generatedCode);

	[Test]
	[Category("Slow")]
	public void InvalidConsoleAppWillGiveUsCompilationError() =>
		Assert.That(
			() => GenerateNewConsoleAppAndReturnOutput(
				nameof(InvalidConsoleAppWillGiveUsCompilationError), "lafine=soeu"),
			Throws.InstanceOf<CSharpCompilationFailed>().And.Message.Contains("The build failed."));

	[Test]
	[Category("Manual")]
	public void GenerateDirectoryGetFilesProgram()
	{
		using var program = new Type(package, new TypeLines(nameof(GenerateDirectoryGetFilesProgram),
			"has App", "has logger", "has directory = \".\"", "Run", "\tfor directory.GetFiles", "\t\tlogger.Log(value)")).ParseMembersAndMethods(parser);
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo("Program.cs" + Environment.NewLine));
	}

	[Category("Manual")]
	[Test]
	public Task ArithmeticFunction() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ArithmeticFunction));

	public async Task GenerateCSharpByReadingStrictProgramAndCompareWithOutput(string programName,
		Package? overridePackage = null)
	{
		using var program = await ReadStrictFileAndCreateType(programName, overridePackage);
		program.Methods[0].GetBodyAndParseIfNeeded();
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(generatedCode,
			Is.EqualTo(string.Join(Environment.NewLine,
				await File.ReadAllLinesAsync(Path.Combine(await GetExampleFolder(),
					$"Output/{programName}.cs")))), generatedCode);
	}

	private async Task<Type>
		ReadStrictFileAndCreateType(string programName, Package? overridePackage = null) =>
		new Type(overridePackage ?? TestPackage.Instance,
			new TypeLines(programName,
				await File.ReadAllLinesAsync(Path.Combine(await GetExampleFolder(),
					$"{programName}.strict")))).ParseMembersAndMethods(parser);

	private static async Task<string> GetExampleFolder()
	{
		const string ExamplesSubFolder = "Examples";
		const string DevelopmentExamplesFolder = Repositories.StrictDevelopmentFolderPrefix + ExamplesSubFolder;
		if (Directory.Exists(DevelopmentExamplesFolder))
			return DevelopmentExamplesFolder;
		const string ExamplesPackageName = "Strict.Examples";
		return await Repositories.DownloadAndExtractRepository(
				new Uri(Repositories.StrictPrefixUri.AbsoluteUri + ExamplesSubFolder), ExamplesPackageName).
			ConfigureAwait(false);
	}

	[Test]
	public Task ReduceButGrow() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ReduceButGrow));

	[Test]
	public Task Fibonacci() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(Fibonacci));

	[Test]
	public Task ReverseList() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ReverseList));

	[Test]
	public Task RemoveExclamation() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(RemoveExclamation));

	[Test]
	public async Task ExecuteOperation()
	{
		using var register = await ReadStrictFileAndCreateType("Register", TestPackage.Instance);
		using var instruction = await ReadStrictFileAndCreateType("Instruction", TestPackage.Instance);
		using var statement = await ReadStrictFileAndCreateType("Statement", TestPackage.Instance);
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ExecuteOperation),
			TestPackage.Instance);
	}

	[Test]
	public async Task LinkedListAnalyzer()
	{
		using var _ = await ReadStrictFileAndCreateType("Node");
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(LinkedListAnalyzer));
	}

	[Test]
	public Task RemoveParentheses() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(RemoveParentheses));

	[Test]
	public Task RemoveDuplicateWords() =>
		GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(RemoveDuplicateWords));
}