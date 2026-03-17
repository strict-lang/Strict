/*
using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

public sealed class BytecodeSerializer
{
	public BytecodeSerializer(Dictionary<string, IList<Instruction>> instructionsByType,
		string outputFolder, string packageName)
	{
		OutputFilePath = Path.Combine(outputFolder, packageName + Extension);
		using var fileStream = new FileStream(OutputFilePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		WriteBytecodeEntries(zip, instructionsByType);
	}

	public BytecodeSerializer(Binary usedTypes) => this.usedTypes = usedTypes;
	private readonly Binary? usedTypes;
	public string OutputFilePath { get; } = string.Empty;
	public const string Extension = ".strictbinary";
	public const string BytecodeEntryExtension = ".bytecode";
	internal static readonly byte[] StrictMagicBytes = "Strict"u8.ToArray();
	public const byte Version = 1;

	public void Serialize(string filePath)
	{
		if (usedTypes == null)
			throw new InvalidOperationException("Binary was not provided.");
		using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		foreach (var (fullTypeName, membersAndMethods) in usedTypes.MethodsPerType)
		{
			var entry = zip.CreateEntry(fullTypeName + BytecodeEntryExtension, CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			membersAndMethods.Write(writer);
		}
	}

	public static Dictionary<string, byte[]> SerializeToEntryBytes(
		Dictionary<string, IList<Instruction>> instructionsByType)
	{
		var result = new Dictionary<string, byte[]>(instructionsByType.Count, StringComparer.Ordinal);
		foreach (var (typeName, instructions) in instructionsByType)
		{
			using var stream = new MemoryStream();
			using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
			WriteEntry(writer, instructions);
			writer.Flush();
			result[typeName] = stream.ToArray();
		}
		return result;
	}

	private static void WriteBytecodeEntries(ZipArchive zip,
		Dictionary<string, IList<Instruction>> instructionsByType)
	{
		foreach (var (typeName, instructions) in instructionsByType)
		{
			var entry = zip.CreateEntry(typeName + BytecodeEntryExtension, CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			WriteEntry(writer, instructions);
		}
	}

	private static void WriteEntry(BinaryWriter writer, IList<Instruction> instructions)
	{
		writer.Write(StrictMagicBytes);
		writer.Write(Version);
		var table = new NameTable();
		foreach (var instruction in instructions)
			table.CollectStrings(instruction);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			instruction.Write(writer, table);
	}

	internal static void WriteExpression(BinaryWriter writer, Expression expr, NameTable table)
	{
		switch (expr)
		{
		case Value { Data.IsText: true } val:
			writer.Write((byte)ExpressionKind.TextValue);
			writer.Write7BitEncodedInt(table[val.Data.Text]);
			break;
		case Value val when val.Data.GetType().IsBoolean:
			writer.Write((byte)ExpressionKind.BooleanValue);
			writer.Write7BitEncodedInt(table[val.Data.GetType().Name]);
			writer.Write(val.Data.Boolean);
			break;
		case Value val when val.Data.GetType().IsNumber:
			if (IsSmallNumber(val.Data.Number))
			{
				writer.Write((byte)ExpressionKind.SmallNumberValue);
				writer.Write((byte)(int)val.Data.Number);
			}
			else if (IsIntegerNumber(val.Data.Number))
			{
				writer.Write((byte)ExpressionKind.IntegerNumberValue);
				writer.Write((int)val.Data.Number);
			}
			else
			{
				writer.Write((byte)ExpressionKind.NumberValue);
				writer.Write(val.Data.Number);
			}
			break;
		case Value val:
			throw new NotSupportedException("WriteExpression not supported value: " + val);
		case MemberCall memberCall:
			writer.Write((byte)ExpressionKind.MemberRef);
			writer.Write7BitEncodedInt(table[memberCall.Member.Name]);
			writer.Write7BitEncodedInt(table[memberCall.Member.Type.Name]);
			writer.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				WriteExpression(writer, memberCall.Instance, table);
			break;
		case Binary binary:
			writer.Write((byte)ExpressionKind.BinaryExpr);
			writer.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(writer, binary.Instance!, table);
			WriteExpression(writer, binary.Arguments[0], table);
			break;
		case MethodCall methodCall:
			writer.Write((byte)ExpressionKind.MethodCallExpr);
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.Name]);
			writer.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(writer, methodCall.Instance, table);
			writer.Write7BitEncodedInt(methodCall.Arguments.Count);
			foreach (var argument in methodCall.Arguments)
				WriteExpression(writer, argument, table);
			break;
		default:
			writer.Write((byte)ExpressionKind.VariableRef);
			writer.Write7BitEncodedInt(table[expr.ToString()]);
			writer.Write7BitEncodedInt(table[expr.ReturnType.Name]);
			break;
		}
	}

	internal static void WriteMethodCallData(BinaryWriter writer, MethodCall? methodCall,
		Registry? registry, NameTable table)
	{
		writer.Write(methodCall != null);
		if (methodCall != null)
		{
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.Name]);
			writer.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(writer, methodCall.Instance, table);
			writer.Write7BitEncodedInt(methodCall.Arguments.Count);
			foreach (var argument in methodCall.Arguments)
				WriteExpression(writer, argument, table);
		}
		writer.Write(registry != null);
		if (registry == null)
			return;
		writer.Write((byte)registry.NextRegister);
		writer.Write((byte)registry.PreviousRegister);
	}

	private static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);
}
*/