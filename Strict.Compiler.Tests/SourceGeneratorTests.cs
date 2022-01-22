using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

public sealed class SourceGeneratorTests : TestCSharpGenerator
{
	[Test]
	public void GenerateCSharpInterface()
	{
		var app = new Type(package, "DummyApp", parser).Parse("Run");
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

	[Test]
	public void CreateFileWriteIntoItReadItAndThenDeleteIt()
	{
		var program = new Type(package, nameof(CreateFileWriteIntoItReadItAndThenDeleteIt), parser).Parse(@"implement App
has file = """ + TemporaryFile + @"""
has log
Run
	file.Write(""Hello"")
	log.Write(file.Read())
	file.Delete()");
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo("Hello"));
		Assert.That(File.Exists(Path.Combine(ProjectFolder, TemporaryFile)), Is.False);
	}

	private const string TemporaryFile = "temp.txt";

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void GenerateFileReadProgram()
	{
		var program = new Type(package, nameof(GenerateFileReadProgram), parser).Parse(@"implement App
has file = """ + TestTxt + @"""
has log
Run
	log.Write(file.Read())");
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo(ExpectedText));
		Assert.That(File.Exists(Path.Combine(ProjectFolder, TestTxt)), Is.True);
	}

	private const string ProjectFolder = nameof(GenerateFileReadProgram);
	private const string ExpectedText = "Hello World\r\n";
	private const string TestTxt = "test.txt";

	private static string GenerateNewConsoleAppAndReturnOutput(string folder, string generatedCode)
	{
		if (!Directory.Exists(folder))
			CreateFolderOnlyOnce(folder, generatedCode);
		File.WriteAllText(Path.Combine(folder, "Program.cs"), generatedCode);
		var actualText = RunDotnetAndReturnOutput(folder, "run", out var error);
		if (error.Length > 0)
			throw new CompilationFailed(error, actualText, generatedCode);
		return actualText;
	}

	private static void CreateFolderOnlyOnce(string folder, string generatedCode)
	{
		var creationOutput =
			RunDotnetAndReturnOutput("", "new console --force --name " + folder, out var creationError);
		if (!creationOutput.Contains("successful"))
			throw new CompilationFailed(creationError, creationOutput, generatedCode);
		File.WriteAllText(Path.Combine(folder, TestTxt), ExpectedText);
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

	private sealed class CompilationFailed : Exception
	{
		public CompilationFailed(string error, string actualText, string generatedCode) : base(error +
			Environment.NewLine + actualText + Environment.NewLine + nameof(generatedCode) + ":" +
			Environment.NewLine + generatedCode) { }
	}

	[Test]
	[Category("Slow")]
	public void InvalidConsoleAppWillGiveUsCompilationError() =>
		Assert.That(
			() => GenerateNewConsoleAppAndReturnOutput(
				nameof(InvalidConsoleAppWillGiveUsCompilationError), "lafine=soeu"),
			Throws.InstanceOf<CompilationFailed>().And.Message.Contains("The build failed."));
}