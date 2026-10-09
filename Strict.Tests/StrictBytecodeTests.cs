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
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var variables = CreateNumbersVariable(basePackage, numbers);
		var expected = Execute(Path.ChangeExtension(source, BinaryExecutable.Extension),
			basePackage, variables);
		await new Runner(Root + "/Bytecode/FileCompiler" + Type.Extension, source + " " + Root).
			Run();
		var binaryPath = Path.Combine(Path.GetTempPath(), nameof(StrictBytecodeTests),
			example + BinaryExecutable.Extension);
		Directory.CreateDirectory(Path.GetDirectoryName(binaryPath)!);
		await File.WriteAllBytesAsync(binaryPath, LastNumbersLine());
		Assert.That(Execute(binaryPath, basePackage, variables), Is.EqualTo(expected));
	}

	private static string Root =>
		Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)).
			Replace('\\', '/');

	private static Dictionary<string, ValueInstance>? CreateNumbersVariable(Package basePackage,
		double[] numbers)
	{
		if (numbers.Length == 0)
			return null;
		var numberType = basePackage.GetType(Type.Number);
		return new Dictionary<string, ValueInstance>
		{
			["numbers"] = new(basePackage.GetType(Type.List).GetGenericImplementation(numberType),
				numbers.Select(number => new ValueInstance(numberType, number)).ToArray())
		};
	}

	private string Execute(string binaryPath, Package basePackage,
		IReadOnlyDictionary<string, ValueInstance>? variables)
	{
		consoleWriter.GetStringBuilder().Clear();
		var machine = new VirtualMachine(new BinaryExecutable(binaryPath, basePackage));
		return consoleWriter + (machine.Execute(initialVariables: variables).Returns is
			{ HasValue: true } returns
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