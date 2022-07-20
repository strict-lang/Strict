using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
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

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void CreateFileAndWriteIntoIt()
	{
		var program = new Type(package, nameof(CreateFileAndWriteIntoIt), parser).Parse(@"implement App
has file = """ + TemporaryFile + @"""
has log
Run
	file.Write(""Hello"")");
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
		var program = new Type(package, nameof(GenerateFileReadProgram), parser).Parse(@"implement App
has file = """ + TestTxt + @"""
has log
Run
	log.Write(file.Read)");
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo(ExpectedText + "\r\n"));
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
	[Category("Manual")] // TODO: List should be working before making this test work
	public void GenerateDirectoryGetFilesProgram()
	{
		var program = new Type(package, nameof(GenerateDirectoryGetFilesProgram), parser).Parse(@"implement App
has log
has directory = "".""
Run
	for value in directory.GetFiles
		log.Write(value)");
		var generatedCode = generator.Generate(program).ToString()!;
		Assert.That(GenerateNewConsoleAppAndReturnOutput(ProjectFolder, generatedCode),
			Is.EqualTo("Program.cs" + "\r\n"));
	}
}