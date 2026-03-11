using Strict.Bytecode.Serialization;
using Strict.Language.Tests;

namespace Strict.Tests;

public sealed class RunnerTests
{
	[SetUp]
	public void CreateTextWriter()
	{
		writer = new StringWriter();
		rememberConsole = Console.Out;
		Console.SetOut(writer);
	}

	private StringWriter writer = null!;
	private TextWriter rememberConsole = null!;

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsole);

	[Test]
	public void RunSimpleCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, StrictFilePath).Run();
		Assert.That(writer.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
	}

	private const string StrictFilePath = "Examples/SimpleCalculator.strict";

	[Test]
	public void RunWithFullDiagnostics()
	{
		using var _ = new Runner(TestPackage.Instance, StrictFilePath, true).Run();
		Assert.That(writer.ToString().Length, Is.GreaterThan(1000));
	}

	[Test]
	public void RunFromBytecodeFileProducesSameOutput()
	{
		var binaryFilePath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		try
		{
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
			Assert.That(File.Exists(binaryFilePath), Is.True,
				BytecodeSerializer.Extension + " file should have been created");
			writer.GetStringBuilder().Clear();
			using var runner = new Runner(TestPackage.Instance, binaryFilePath).Run();
			Assert.That(writer.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		}
		finally
		{
			if (File.Exists(binaryFilePath))
				File.Delete(binaryFilePath);
		}
	}

	[Test]
	public void RunFromBytecodeFileWithoutStrictSourceFile()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		var copiedSourceFilePath = Path.Combine(tempDirectory, Path.GetFileName(StrictFilePath));
		var copiedBinaryFilePath = Path.ChangeExtension(copiedSourceFilePath, BytecodeSerializer.Extension);
		try
		{
			File.Copy(StrictFilePath, copiedSourceFilePath);
			new Runner(TestPackage.Instance, copiedSourceFilePath).Run().Dispose();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			writer.GetStringBuilder().Clear();
			File.Delete(copiedSourceFilePath);
			using var _ = new Runner(TestPackage.Instance, copiedBinaryFilePath).Run();
			Assert.That(writer.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		}
		finally
		{
			if (Directory.Exists(tempDirectory))
				Directory.Delete(tempDirectory, true);
		}
	}

	[Test]
	public void RunFizzBuzz()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/FizzBuzz.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("FizzBuzz(3) = Fizz"));
		Assert.That(output, Does.Contain("FizzBuzz(5) = Buzz"));
		Assert.That(output, Does.Contain("FizzBuzz(15) = FizzBuzz"));
		Assert.That(output, Does.Contain("FizzBuzz(7) = 7"));
	}

	[Test]
	public void RunAreaCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/AreaCalculator.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Area: 50"));
		Assert.That(output, Does.Contain("Perimeter: 30"));
	}

	[Test]
	public void RunGreeter()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/Greeter.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
	}

	[Test]
	public void SaveStrictBinaryWithTypeBytecodeEntriesOnly()
	{
		var binaryFilePath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		try
		{
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
			using var archive = System.IO.Compression.ZipFile.OpenRead(binaryFilePath);
			var entries = archive.Entries.Select(entry => entry.FullName).ToList();
			Assert.That(
				entries.Any(entry => entry.EndsWith(BytecodeSerializer.BytecodeEntryExtension,
					StringComparison.OrdinalIgnoreCase)), Is.False);
			Assert.That(entries.Any(entry => entry.Contains("#", StringComparison.Ordinal)), Is.False);
			Assert.That(entries, Does.Contain("Examples/SimpleCalculator.bytecode"));
			Assert.That(entries, Does.Contain("Strict/Number.bytecode"));
			Assert.That(entries, Does.Contain("Strict/Logger.bytecode"));
			Assert.That(entries.Count, Is.LessThanOrEqualTo(4));
			Assert.That(new FileInfo(binaryFilePath).Length, Is.LessThan(1200));
		}
		finally
		{
			if (File.Exists(binaryFilePath))
				File.Delete(binaryFilePath);
		}
	}

	[Test]
	public void ExportOnlyUsedMethodsForBaseTypes()
	{
		var binaryFilePath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		try
		{
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
			using var archive = System.IO.Compression.ZipFile.OpenRead(binaryFilePath);
			var numberMethodCount = ReadMethodHeaderCount(archive, "Strict/Number.bytecode");
			Assert.That(numberMethodCount, Is.LessThanOrEqualTo(3));
		}
		finally
		{
			if (File.Exists(binaryFilePath))
				File.Delete(binaryFilePath);
		}
	}

	private static int ReadMethodHeaderCount(System.IO.Compression.ZipArchive archive,
		string entryName)
	{
		var entry = archive.GetEntry(entryName) ?? throw new InvalidOperationException(entryName);
		using var reader = new BinaryReader(entry.Open());
		_ = reader.ReadBytes(6);
		var version = reader.ReadByte();
		if (version != BytecodeSerializer.Version)
			throw new InvalidOperationException("Expected version " + BytecodeSerializer.Version);
		var nameCount = reader.Read7BitEncodedInt();
		for (var nameIndex = 0; nameIndex < nameCount; nameIndex++)
			_ = reader.ReadString();
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
		{
			_ = reader.Read7BitEncodedInt();
			_ = reader.Read7BitEncodedInt();
			if (reader.ReadBoolean())
				throw new InvalidOperationException("Unexpected initial value in compact metadata");
		}
		return reader.Read7BitEncodedInt();
	}
}