using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.LanguageServer.Tests;

public sealed class ScrunchAnalyzerTests
{
	private string folder = "";

	[SetUp]
	public void CreateTempFolder()
	{
		folder = Path.Combine(Path.GetTempPath(), "scrunch-mcp-" + Guid.NewGuid());
		Directory.CreateDirectory(folder);
	}

	[TearDown]
	public void DeleteTempFolder()
	{
		if (Directory.Exists(folder))
			Directory.Delete(folder, true);
	}

	[Test]
	public void AnalyzeReportsEmptyLineAsFailure()
	{
		var file = Write("Broken.strict", "has number\n\nAdd(num Number) Number\n\tnum + number");
		var report = ScrunchAnalyzer.AnalyzeFile(TestPackage.Instance, file);
		Assert.That(report.Ok, Is.False);
		Assert.That(report.Problems, Is.Not.Empty);
		Assert.That(report.Problems[0].Message, Does.Contain("Empty line").IgnoreCase.Or.
			Contain("EmptyLine"));
	}

	[Test]
	public void AnalyzePassingFileIsOk()
	{
		var file = Write("Adder.strict", "has number\nAdd(num Number) Number\n\t5 is 5\n\tnum + number");
		var report = ScrunchAnalyzer.AnalyzeFile(TestPackage.Instance, file);
		Assert.That(report.Ok, Is.True);
		Assert.That(report.Cached, Is.False);
		Assert.That(report.TestsFailed, Is.EqualTo(0));
	}

	[Test]
	public void FreshBinaryIsCachedPassWithoutRerun()
	{
		var file = Write("Cached.strict", "has number\nAdd(num Number) Number\n\t5 is 5\n\tnum + number");
		File.WriteAllBytes(Path.ChangeExtension(file, ".strictbinary"), [1, 2, 3]);
		File.SetLastWriteTimeUtc(Path.ChangeExtension(file, ".strictbinary"),
			File.GetLastWriteTimeUtc(file).AddSeconds(1));
		var report = ScrunchAnalyzer.AnalyzeFile(TestPackage.Instance, file);
		Assert.That(report.Ok, Is.True);
		Assert.That(report.Cached, Is.True);
	}

	[Test]
	public void ForceIgnoresStaleLookingCacheAndParses()
	{
		var file = Write("Forced.strict", "has number\n\nAdd(num Number) Number\n\tnum + number");
		File.WriteAllBytes(Path.ChangeExtension(file, ".strictbinary"), [1]);
		File.SetLastWriteTimeUtc(Path.ChangeExtension(file, ".strictbinary"),
			File.GetLastWriteTimeUtc(file).AddSeconds(1));
		var report = ScrunchAnalyzer.AnalyzeFile(TestPackage.Instance, file, true);
		Assert.That(report.Ok, Is.False);
		Assert.That(report.Cached, Is.False);
	}

	[Test]
	public void McpToolsListIncludesCheckAndStatus()
	{
		var response = McpServer.Handle("""{"jsonrpc":"2.0","id":1,"method":"tools/list"}""",
			TestPackage.Instance);
		Assert.That(response, Does.Contain("\"check\""));
		Assert.That(response, Does.Contain("\"status\""));
	}

	private string Write(string name, string text)
	{
		var path = Path.Combine(folder, name);
		File.WriteAllText(path, text.Replace("\n", Environment.NewLine));
		return path;
	}
}
