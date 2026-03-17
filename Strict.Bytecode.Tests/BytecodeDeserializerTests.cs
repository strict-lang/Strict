using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeDeserializerTests : TestBytecode
{
	[Test]
	public void ZipWithNoBytecodeEntriesCreatesEmptyStrictBinary()
	{
		var filePath = CreateEmptyZipWithDummyEntry();
		var binary = new BinaryExecutable(filePath, TestPackage.Instance);
		Assert.That(binary.MethodsPerType, Is.Empty);
	}

	[Test]
	public void EntryWithBadMagicBytesThrows()
	{
		var filePath = CreateZipWithSingleEntry([0xBA, 0xAD, 0xBA, 0xAD, 0xBA, 0xAD, 0x01]);
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryType.InvalidBytecodeEntry>().With.Message.
				Contains("magic bytes"));
	}

	[Test]
	public void VersionZeroThrows()
	{
		var filePath = CreateZipWithSingleEntry(BuildEntryBytes(writer =>
		{
			writer.Write(MagicBytes);
			writer.Write((byte)0);
		}));
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryType.InvalidVersion>().With.Message.Contains("version"));
	}

	[Test]
	public void UnknownValueKindThrows()
	{
		var filePath = CreateZipWithSingleEntry(BuildEntryBytes(writer =>
		{
			WriteHeader(writer, ["member", "Number"]);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(1);
			writer.Write(true);
			writer.Write((byte)InstructionType.LoadConstantToRegister);
			writer.Write((byte)Register.R0);
			writer.Write((byte)0xFF);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(0);
		}));
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryExecutable.InvalidFile>().With.Message.Contains("Unknown ValueKind"));
	}

	[Test]
	public void UnknownExpressionKindThrows()
	{
		var filePath = CreateZipWithSingleEntry(BuildEntryBytes(writer =>
		{
			WriteHeader(writer, ["member", "Number", "Run", "None"]);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(1);
			writer.Write(true);
			writer.Write((byte)InstructionType.Invoke);
			writer.Write((byte)Register.R0);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(2);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(3);
			writer.Write(false);
			writer.Write7BitEncodedInt(1);
			writer.Write((byte)0xFF);
			writer.Write((byte)0);
			writer.Write((byte)Register.R0);
			writer.Write7BitEncodedInt(0);
		}));
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryExecutable.InvalidFile>().With.Message.Contains("Unknown ExpressionKind"));
	}

	private static void WriteHeader(BinaryWriter writer, string[] names)
	{
		writer.Write(MagicBytes);
		writer.Write(BinaryType.Version);
		writer.Write7BitEncodedInt(names.Length);
		foreach (var name in names)
			writer.Write(name);
	}

	private static string CreateEmptyZipWithDummyEntry()
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		zip.CreateEntry("dummy.txt");
		return filePath;
	}

	private static string CreateZipWithSingleEntry(byte[] entryBytes)
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		var entry = zip.CreateEntry("Number" + BinaryType.BytecodeEntryExtension);
		using var stream = entry.Open();
		stream.Write(entryBytes);
		return filePath;
	}

	private static string GetTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "strictbinary" + fileCounter++ + BinaryExecutable.Extension);

	private static int fileCounter;
	private static readonly byte[] MagicBytes = "Strict"u8.ToArray();

	private static byte[] BuildEntryBytes(Action<BinaryWriter> writeContent)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		writeContent(writer);
		writer.Flush();
		return stream.ToArray();
	}
}