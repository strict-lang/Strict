using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = System.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeDeserializerTests : TestBytecode
{
	[Test]
	public void ZipWithNoBytecodeEntriesThrows()
	{
		var filePath = CreateEmptyZipWithDummyEntry();
		Assert.That(() => new BytecodeDeserializer(filePath).Deserialize(TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("no"));
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
	public void EntryWithBadMagicBytesThrows() =>
		Assert.That(() => new BytecodeDeserializer(new Dictionary<string, byte[]>
			{
				["test"] = [0xBA, 0xAD, 0xBA, 0xAD, 0xBA, 0xAD, 0x01]
			}, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("magic bytes"));

	[Test]
	public void VersionZeroThrows() =>
		Assert.That(() => new BytecodeDeserializer(new Dictionary<string, byte[]>
			{
				["test"] = BuildEntryBytes(writer =>
				{
					writer.Write(MagicBytes);
					writer.Write((byte)0);
				})
			}, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidVersion>().With.Message.Contains("version"));

	[Test]
	public void UnknownValueKindThrows() =>
		Assert.That(() => new BytecodeDeserializer(new Dictionary<string, byte[]>
			{
				["test"] = BuildEntryBytes(writer =>
				{
					writer.Write(MagicBytes);
					writer.Write(BytecodeSerializer.Version);
					writer.Write7BitEncodedInt(0);
					writer.Write7BitEncodedInt(0);
					writer.Write7BitEncodedInt(0);
					writer.Write7BitEncodedInt(1);
					writer.Write((byte)InstructionType.LoadConstantToRegister);
					writer.Write((byte)Register.R0);
					writer.Write((byte)0xFF);
					writer.Write7BitEncodedInt(0);
				})
			}, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("Unknown ValueKind"));

	[Test]
	public void UnknownExpressionKindThrows() =>
		Assert.That(() => new BytecodeDeserializer(new Dictionary<string, byte[]>
			{
				["test"] = BuildEntryBytes(writer =>
				{
					writer.Write(MagicBytes);
					writer.Write(BytecodeSerializer.Version);
					writer.Write7BitEncodedInt(3);
					writer.Write("Number");
					writer.Write("Run");
					writer.Write("None");
					writer.Write7BitEncodedInt(0);
					writer.Write7BitEncodedInt(0);
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
					writer.Write7BitEncodedInt(0);
				})
			}, TestPackage.Instance),
			Throws.TypeOf<BytecodeDeserializer.InvalidBytecodeFileException>().With.Message.
				Contains("Unknown ExpressionKind"));

	private static byte[] BuildEntryBytes(Action<BinaryWriter> writeContent)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		writeContent(writer);
		writer.Flush();
		return stream.ToArray();
	}
}