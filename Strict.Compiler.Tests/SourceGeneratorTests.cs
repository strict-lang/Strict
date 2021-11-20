using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

public class SourceGeneratorTests
{
	[SetUp]
	public async Task CreateGenerator()
	{
		parser = new MethodExpressionParser();
		await new Repositories(parser).LoadFromUrl(Repositories.StrictUrl);
		package = new Package(nameof(SourceGeneratorTests));
		generator = new CSharpGenerator();
	}

	private MethodExpressionParser parser = null!;
	private Package package = null!;
	private SourceGenerator generator = null!;

	[Test]
	public void GenerateCSharpInterface()
	{
		var app = new Type(package, "DummyApp", parser).Parse("Run");
		var file = generator.Generate(app);
		Assert.That(file.ToString(), Is.EqualTo(@"public interface DummyApp
{
	void Run();
}"));
	}

	[Test]
	public void GenerateCSharpClass()
	{
		var program = new Type(package, "Program", parser).Parse(@"implement App
has log
Run
	log.Write(""Hello World"")");
		var file = generator.Generate(program);
		Assert.That(file.ToString(), Is.EqualTo(@"public class Program
{
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}"));
	}

	[Test]
	public void GenerateFileReadProgram()
	{
		const string ExpectedText = "Black friday is coming!\r\n";
		File.WriteAllText("test.txt", ExpectedText);
		var program = new Type(package, nameof(GenerateFileReadProgram), parser).Parse(@"implement App
has file = ""test.txt""
has log
Run
	log.Write(file.Read())");
		var generatedCode = generator.Generate(program).ToString();
		Assert.That(GenerateNewConsoleApp(generatedCode), Is.EqualTo(ExpectedText));
	}

	private static string GenerateNewConsoleApp(string? generatedCode)
	{
		const string FileReadProgramDirectory = "GenerateFileReadProgram";
		if (Directory.Exists(FileReadProgramDirectory))
			Directory.Delete(FileReadProgramDirectory, true);
		var projectCreationProcess =
			Process.Start("dotnet", "new console --name " + FileReadProgramDirectory);
		projectCreationProcess.WaitForExit();
		File.WriteAllText(FileReadProgramDirectory + "/Program.cs", generatedCode);
		var process = new Process
		{
			StartInfo = new ProcessStartInfo
			{
				FileName = "dotnet",
				Arguments = "run " + "Program.cs",
				UseShellExecute = false,
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				WorkingDirectory = FileReadProgramDirectory
			}
		};
		process.Start();
		var actualText = process.StandardOutput.ReadToEnd();
		var error = process.StandardError.ReadToEnd();
		if (error.Length > 0)
			throw new CompilationFailed(error, actualText);
		return actualText;
	}

	private sealed class CompilationFailed : Exception
	{
		public CompilationFailed(string error, string actualText) : base(error +
			Environment.NewLine + actualText) { }
	}
}