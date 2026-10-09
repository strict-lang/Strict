using System.IO.Compression;
using Strict.Bytecode;
using Strict.Language;
using Strict.Expressions;
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
		await new Runner(Path.Combine(Root, "Bytecode", "ZipWriter" + Type.Extension),
			"ZipWriter(ZipEntry(\"hello.txt\", (104, 105)), ZipEntry(\"data.bin\", (1, 2, 3))).Bytes").Run();
		using var zip = new ZipArchive(new MemoryStream(LastNumbersLine()));
		Assert.That(zip.Entries.Select(entry => entry.FullName + "=" + string.Join(",", Read(entry))),
			Is.EqualTo(new[] { "hello.txt=104,105", "data.bin=1,2,3" }));
	}

	[TestCase("HelloLogger")]
	[TestCase("NativeArithmetic")]
	[TestCase("NativeConditions")]
	[TestCase("NativeLoop")]
	[TestCase("Greeter")]
	[TestCase("Fibonacci")]
	[TestCase("AreaCalculator")]
	[TestCase("SimpleCalculator")]
	[TestCase("TemperatureConverter")]
	[TestCase("GcdCalculator")]
	[TestCase("FizzBuzz")]
	[TestCase("AutofilledMutable")]
	[TestCase("Pixel")]
	[TestCase("DirProbe")]
	[TestCase("ProcessProbe")]
	public async Task StrictCompiledExampleRunsLikeCSharp(string example)
	{
		var source = Root + "/Examples/" + example + Type.Extension;
		await new Runner(source).Run();
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var expected = Execute(Path.ChangeExtension(source, BinaryExecutable.Extension), basePackage);
		await new Runner(Root + "/Bytecode/FileCompiler" + Type.Extension, source + " " + Root).Run();
		var binaryPath = Path.Combine(Path.GetTempPath(), nameof(StrictBytecodeTests),
			example + BinaryExecutable.Extension);
		Directory.CreateDirectory(Path.GetDirectoryName(binaryPath)!);
		await File.WriteAllBytesAsync(binaryPath, LastNumbersLine());
		Assert.That(Execute(binaryPath, basePackage), Is.EqualTo(expected));
	}

	private static string Root =>
		Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)).Replace('\\', '/');

	private string Execute(string binaryPath, Package basePackage)
	{
		consoleWriter.GetStringBuilder().Clear();
		var machine = new VirtualMachine(new BinaryExecutable(binaryPath, basePackage));
		return consoleWriter + (machine.Execute().Returns is { HasValue: true } returns
			? "Returns " + returns
			: "");
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