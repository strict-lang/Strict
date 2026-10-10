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
			"ZipWriter(ZipEntry(\"hello.txt\", (104, 105)), " +
			"ZipEntry(\"data.bin\", (1, 2, 3))).Bytes").Run();
		using var zip = new ZipArchive(new MemoryStream(LastNumbersLine()));
		Assert.That(
			zip.Entries.Select(entry => entry.FullName + "=" + string.Join(",", Read(entry))),
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
	[TestCase("NumberSummer")]
	[TestCase("MemoryPressure")]
	[TestCase("NumberStats")]
	[TestCase("Grade")]
	[TestCase("Sum", 5, 10, 20)]
	public async Task StrictCompiledExampleRunsLikeCSharp(string example, params double[] numbers)
	{
		var source = Root + "/Examples/" + example + Type.Extension;
		await new Runner(source).Run();
		var expected = await Execute(Path.ChangeExtension(source, BinaryExecutable.Extension), numbers);
		await new Runner(Root + "/Bytecode/FileCompiler" + Type.Extension, source + " " + Root).
			Run();
		var binaryPath = Path.Combine(Path.GetTempPath(), nameof(StrictBytecodeTests),
			example + BinaryExecutable.Extension);
		Directory.CreateDirectory(Path.GetDirectoryName(binaryPath)!);
		await File.WriteAllBytesAsync(binaryPath, LastNumbersLine());
		Assert.That(await Execute(binaryPath, numbers), Is.EqualTo(expected));
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
	[TestCase("NumberSummer")]
	[TestCase("MemoryPressure")]
	[TestCase("NumberStats")]
	[TestCase("Grade")]
	public async Task StrictVirtualMachineRunsLikeCSharp(string example)
	{
		var source = Root + "/Examples/" + example + Type.Extension;
		await new Runner(source).Run();
		consoleWriter.GetStringBuilder().Clear();
		new VirtualMachine(new BinaryExecutable(Path.ChangeExtension(source, BinaryExecutable.Extension))).
			Execute();
		var expected = consoleWriter.ToString();
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(Root + "/Runtime/Execute" + Type.Extension, source + " " + Root).Run();
		var output = consoleWriter.ToString();
		Assert.That(output[..output.LastIndexOf("Executed ", StringComparison.Ordinal)],
			Is.EqualTo(expected));
	}

	private static string Root =>
		Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)).
			Replace('\\', '/');

	/// <summary>
	/// Binaries load self-contained like the CLI does, program arguments go through the Runner.
	/// </summary>
	private async Task<string> Execute(string binaryPath, double[] numbers)
	{
		consoleWriter.GetStringBuilder().Clear();
		if (numbers.Length > 0)
		{
			await new Runner(binaryPath, string.Join(" ", numbers)).Run();
			var output = consoleWriter.ToString();
			return output[..output.LastIndexOf("Executed ", StringComparison.Ordinal)];
		}
		var machine = new VirtualMachine(new BinaryExecutable(binaryPath));
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