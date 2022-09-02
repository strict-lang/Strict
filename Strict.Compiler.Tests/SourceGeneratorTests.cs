using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

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
	[Ignore("TODO: Not yet done")]
	public void GenerateCSharpClass()
	{
		var program = CreateHelloWorldProgramType();
		var file = generator.Generate(program);
		Assert.That(file.ToString(), Is.EqualTo(@"namespace SourceGeneratorTests;

public class Program
{
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}"));
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void CreateFileAndWriteIntoIt()
	{
		var program = new Type(package, new TypeLines(nameof(CreateFileAndWriteIntoIt),
			"implement App", "has file = \"" + TemporaryFile + "\"", "has log", "Run", "\tfile.Write(\"Hello\")")).ParseMembersAndMethods(parser);
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
		var program = new Type(package, new TypeLines(nameof(GenerateFileReadProgram), "implement App", "has file = \"" + TestTxt + "\"", "has log", "Run", "\tlog.Write(file.Read)")).ParseMembersAndMethods(parser);
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
		if (error.Length > 0)
			throw new CSharpCompilationFailed(error, actualText, generatedCode);
		return actualText;
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

	public sealed class CSharpCompilationFailed : Exception
	{
		public CSharpCompilationFailed(string error, string actualText, string generatedCode) : base(
			error + Environment.NewLine + actualText + Environment.NewLine + nameof(generatedCode) +
			":" + Environment.NewLine + generatedCode) { }
	}

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
		var program = new Type(package, new TypeLines(nameof(GenerateDirectoryGetFilesProgram),
			"implement App", "has log", "has directory = \".\"", "Run", "\tfor value in directory.GetFiles", "\t\tlog.Write(value)")).ParseMembersAndMethods(parser);
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo("Program.cs" + Environment.NewLine));
	}

	[Category("Manual")] // Unit Tests in strict are not compiling; after removing them this test works
	[Test]
	public async Task ArithmeticFunction() =>
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ArithmeticFunction));

	public async Task GenerateCSharpByReadingStrictProgramAndCompareWithOutput(string programName)
	{
		var lines = await File.ReadAllLinesAsync(Path.Combine(ExampleFolder, $"{programName}.strict"));
		var program = new Type(package, new TypeLines(programName, lines)).ParseMembersAndMethods(parser);
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(generatedCode,
			Is.EqualTo(string.Join(Environment.NewLine, await File.ReadAllLinesAsync(Path.Combine(ExampleFolder, $"Output/{programName}.cs")))), generatedCode);
	}

	private static string ExampleFolder => Path.Combine(Repositories.DevelopmentFolder, "Examples");

	[Ignore("this test will work once for loop is working")]
	[Test]
	public async Task ReduceButGrow() =>
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ReduceButGrow));

	[Ignore("for is not working yet - Tests in strict file are not compiling; Also variable are immutable in strict which causes this program to fail")]
	[Test]
	public async Task Fibonacci() =>
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(Fibonacci));

	[Ignore("this test will work once for loop is working")]
	[Test]
	public async Task ReverseList() =>
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(ReverseList));

	[Ignore("this test will work once for loop and not operator is working")]
	[Test]
	public async Task RemoveExclamation() =>
		await GenerateCSharpByReadingStrictProgramAndCompareWithOutput(nameof(RemoveExclamation));
}