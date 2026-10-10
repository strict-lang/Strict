using System.IO.Compression;
using System.Runtime.CompilerServices;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;

[assembly: InternalsVisibleTo("Strict")]
[assembly: InternalsVisibleTo("Strict.Optimizers")]

namespace Strict.Bytecode;

public sealed partial class BinaryExecutable
{
	/// <summary>
	/// Writes to a temporary file first, so readers never see a partially written binary.
	/// </summary>
	public void Serialize(string filePath)
	{
		var temporaryPath = filePath + "." + Guid.NewGuid().ToString("N") + ".tmp";
		try
		{
			WriteZip(temporaryPath);
			File.Move(temporaryPath, filePath, true);
		}
		finally
		{
			if (File.Exists(temporaryPath))
				File.Delete(temporaryPath);
		}
	}

	/// <summary>
	/// Same input gives byte-identical binaries, caches and comparisons do not depend on time.
	/// </summary>
	private static readonly DateTimeOffset DeterministicEntryTime = new(2020, 1, 1, 0, 0, 0,
		TimeSpan.Zero);

	private void WriteZip(string filePath)
	{
		using var fileStream = new FileStream(filePath, FileMode.CreateNew, FileAccess.ReadWrite);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, false);
		foreach (var (fullTypeName, membersAndMethods) in MethodsPerType)
		{
			var entry = zip.CreateEntry(fullTypeName + BinaryType.BytecodeEntryExtension,
				CompressionLevel.Optimal);
			entry.LastWriteTime = DeterministicEntryTime;
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			membersAndMethods.Write(writer);
		}
	}

	internal static void WriteValueInstance(BinaryWriter writer, ValueInstance val, NameTable table)
	{
		if (val.IsText)
		{
			writer.Write((byte)ValueKind.Text);
			writer.Write7BitEncodedInt(table[val.Text]);
			return;
		}
		if (val.IsList)
		{
			writer.Write((byte)ValueKind.List);
			writer.Write7BitEncodedInt(table[val.List.ReturnType.FullName]);
			var items = val.List.Items;
			writer.Write7BitEncodedInt(items.Count);
			foreach (var item in items)
				WriteValueInstance(writer, item, table);
			return;
		}
		if (val.IsDictionary)
		{
			writer.Write((byte)ValueKind.Dictionary);
			writer.Write7BitEncodedInt(table[val.GetType().FullName]);
			var items = val.GetDictionaryItems();
			writer.Write7BitEncodedInt(items.Count);
			foreach (var kvp in items)
			{
				WriteValueInstance(writer, kvp.Key, table);
				WriteValueInstance(writer, kvp.Value, table);
			}
			return;
		}
		var type = val.GetType();
		if (type.IsBoolean)
		{
			writer.Write((byte)ValueKind.Boolean);
			writer.Write(val.Boolean);
		}
		else if (type.IsNone)
		{
			writer.Write((byte)ValueKind.None);
		}
		else if (type.IsNumber)
		{
			if (IsSmallNumber(val.Number))
			{
				writer.Write((byte)ValueKind.SmallNumber);
				writer.Write((byte)(int)val.Number);
			}
			else if (IsIntegerNumber(val.Number))
			{
				writer.Write((byte)ValueKind.IntegerNumber);
				writer.Write((int)val.Number);
			}
			else
			{
				writer.Write((byte)ValueKind.Number);
				writer.Write(val.Number);
			}
		}
		else
		{
			throw new ValueInstanceNotSupported(val); //ncrunch: no coverage
		}
	}

	public static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);

	public class ValueInstanceNotSupported(ValueInstance instance) : Exception(instance.ToString());

	internal static void WriteExpression(BinaryWriter writer, Expression expr, NameTable table)
	{
		switch (expr)
		{
		case List list:
			writer.Write((byte)ExpressionKind.ListExpr);
			writer.Write7BitEncodedInt(table[list.ReturnType.FullName]);
			writer.Write7BitEncodedInt(list.Values.Count);
			foreach (var value in list.Values)
				WriteExpression(writer, value, table);
			break;
		case Value { Data.IsText: true } val:
			writer.Write((byte)ExpressionKind.TextValue);
			writer.Write7BitEncodedInt(table[val.Data.Text]);
			break;
		case Value val when val.Data.GetType().IsBoolean:
			writer.Write((byte)ExpressionKind.BooleanValue);
			writer.Write7BitEncodedInt(table[val.Data.GetType().FullName]);
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
			throw new ValueInstanceNotSupported(val.Data);
		case MemberCall memberCall:
			writer.Write((byte)ExpressionKind.MemberRef);
			writer.Write7BitEncodedInt(table[memberCall.Member.Name]);
			writer.Write7BitEncodedInt(table[memberCall.Member.Type.FullName]);
			writer.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				// ReSharper disable TailRecursiveCall
				WriteExpression(writer, memberCall.Instance, table);
			break;
		case Binary binary:
			writer.Write((byte)ExpressionKind.BinaryExpr);
			writer.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(writer, binary.Instance!, table);
			WriteExpression(writer, binary.Arguments[0], table);
			break;
		case ListCall listCall:
			writer.Write((byte)ExpressionKind.ListCallExpr);
			writer.Write7BitEncodedInt(table[listCall.ReturnType.FullName]);
			WriteExpression(writer, listCall.List, table);
			WriteExpression(writer, listCall.Index, table);
			writer.Write(listCall.SecondIndex != null);
			if (listCall.SecondIndex != null)
				WriteExpression(writer, listCall.SecondIndex, table);
			break;
		case MethodCall methodCall:
			writer.Write((byte)ExpressionKind.MethodCallExpr);
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.FullName]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			foreach (var parameter in methodCall.Method.Parameters)
			{
				writer.Write7BitEncodedInt(table[parameter.Name]);
				writer.Write7BitEncodedInt(table[parameter.Type.FullName]);
			}
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.FullName]);
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
			writer.Write7BitEncodedInt(table[expr.ReturnType.FullName]);
			break;
		}
	}

	internal static void WriteMethodCallData(BinaryWriter writer, MethodCall? methodCall,
		Registry? registry, NameTable table)
	{
		writer.Write(methodCall != null);
		if (methodCall != null)
		{
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.FullName]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			foreach (var parameter in methodCall.Method.Parameters)
			{
				writer.Write7BitEncodedInt(table[parameter.Name]);
				writer.Write7BitEncodedInt(table[parameter.Type.FullName]);
			}
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.FullName]);
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
}
