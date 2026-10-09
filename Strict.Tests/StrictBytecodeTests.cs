using System.IO.Compression;
using Strict.Bytecode;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Tests;

/// <summary>
/// Bytes written by the Strict Bytecode package are read by the C# side.
/// </summary>
[Category("Slow")]
public sealed class StrictBytecodeTests
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

	[Test]
	public async Task StrictZipWriterOutputOpensWithZipArchive()
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		await new Runner(Path.Combine(root, "Bytecode", "ZipWriter" + Type.Extension),
			"ZipWriter(ZipEntry(\"hello.txt\", (104, 105)), ZipEntry(\"data.bin\", (1, 2, 3))).Bytes").Run();
		using var zip = new ZipArchive(new MemoryStream(LastNumbersLine()));
		Assert.That(zip.Entries.Select(entry => entry.FullName + "=" + string.Join(",", Read(entry))),
			Is.EqualTo(new[] { "hello.txt=104,105", "data.bin=1,2,3" }));
	}

	[Test]
	public async Task StrictWrittenBinaryRunsOnTheVirtualMachine()
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		await new Runner(Path.Combine(root, "Bytecode", "BinaryFile" + Type.Extension),
			"BinaryFile(TypeEntry(\"Hello\", MethodEntry(\"Run\", \"None\", " +
			"InstructionEntry(45, \"Hello from a Strict written binary\", List(Number))))).Bytes").Run();
		var binaryPath = Path.Combine(Path.GetTempPath(), nameof(StrictBytecodeTests),
			"Hello" + BinaryExecutable.Extension);
		Directory.CreateDirectory(Path.GetDirectoryName(binaryPath)!);
		await File.WriteAllBytesAsync(binaryPath, LastNumbersLine());
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(binaryPath).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("Hello from a Strict written binary"));
	}

	private byte[] LastNumbersLine() =>
		consoleWriter.ToString().Split('\n').Last(line => line.StartsWith('(')).Trim().
			Trim('(', ')').Split(", ").Select(byte.Parse).ToArray();

	private static byte[] Read(ZipArchiveEntry entry)
	{
		using var stream = entry.Open();
		using var memory = new MemoryStream();
		stream.CopyTo(memory);
		return memory.ToArray();
	}
}