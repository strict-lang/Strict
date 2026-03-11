using System.IO.Compression;
using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeDeserializerTests : TestBytecode
{
	[Test]
	public void ZipWithNoBytecodeEntriesThrows()
	{
		var filePath = CreateEmptyZipWithDummyEntry();
		Assert.That(() => new BytecodeDeserializer(filePath, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("no entries"));
	}

	private static string CreateEmptyZipWithDummyEntry()
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		zip.CreateEntry("dummy.txt");
		return filePath;
	}

	private static string GetTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "desertest" + fileCounter++ + BytecodeSerializer.Extension);

	private static int fileCounter;
	private static readonly byte[] MagicBytes = "Strict"u8.ToArray();

	[Test]
	public void EntryWithBadMagicBytesThrows()
	{
		var filePath = CreateZipWithBytecodeEntry(writer =>
			writer.Write(new byte[] { 0xBA, 0xAD, 0xBA, 0xAD, 0xBA, 0xAD, 0x01 }));
		Assert.That(() => new BytecodeDeserializer(filePath, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("magic bytes"));
	}

	private static string CreateZipWithBytecodeEntry(Action<BinaryWriter> writeContent)
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		var entry = zip.CreateEntry("test.bytecode");
		using var entryStream = entry.Open();
		using var writer = new BinaryWriter(entryStream);
		writeContent(writer);
		return filePath;
	}

	[Test]
	public void VersionZeroThrows()
	{
		var filePath = CreateZipWithBytecodeEntry(writer =>
		{
			writer.Write(MagicBytes);
			writer.Write((byte)0);
		});
		Assert.That(() => new BytecodeDeserializer(filePath, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidVersion>().With.Message.Contains("version"));
	}

	[Test]
	public void UnknownValueKindThrows()
	{
		var filePath = CreateZipWithBytecodeEntry(writer =>
		{
			writer.Write(MagicBytes);
			writer.Write(BytecodeSerializer.Version);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(1);
			writer.Write((byte)InstructionType.LoadConstantToRegister);
			writer.Write((byte)Register.R0);
			writer.Write((byte)0xFF);
		});
		Assert.That(() => new BytecodeDeserializer(filePath, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("Unknown ValueKind"));
	}

	[Test]
	public void UnknownExpressionKindThrows()
	{
		var filePath = CreateZipWithBytecodeEntry(writer =>
		{
			writer.Write(MagicBytes);
			writer.Write(BytecodeSerializer.Version);
			writer.Write7BitEncodedInt(3);
			writer.Write("Number");
			writer.Write("Run");
			writer.Write("None");
			writer.Write7BitEncodedInt(1);
			writer.Write((byte)InstructionType.Invoke);
			writer.Write((byte)Register.R0);
			writer.Write(true);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(0);
			writer.Write(false);
			writer.Write7BitEncodedInt(1);
			writer.Write((byte)0xFF);
		});
		Assert.That(() => new BytecodeDeserializer(filePath, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("Unknown ExpressionKind"));
	}
}