using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Writes optimized <see cref="Instruction" /> lists per type into a compact .strictbinary ZIP.
/// The ZIP contains one entry per type named {typeName}.bytecode.
/// Entry layout: magic(6) + version(1) + string-table + instruction-count(7bit) + instructions.
/// </summary>
public sealed class BytecodeSerializer
{
	/// <summary>
	/// Serializes all types and their instructions into a .strictbinary ZIP file.
	/// The output file is named {packageName}.strictbinary in the given output folder.
	/// </summary>
	public BytecodeSerializer(Dictionary<string, IList<Instruction>> instructionsByType,
		string outputFolder, string packageName)
	{
		OutputFilePath = Path.Combine(outputFolder, packageName + Extension);
		using var fileStream = new FileStream(OutputFilePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		WriteBytecodeEntries(zip, instructionsByType);
	}

	/// <summary>
	/// Serializes package type data into compact .bytecode entries with members, method signatures,
	/// and method instruction bodies.
	/// </summary>
	public BytecodeSerializer(IReadOnlyList<TypeBytecodeData> types, string outputFolder,
		string packageName)
	{
		OutputFilePath = Path.Combine(outputFolder, packageName + Extension);
		using var fileStream = new FileStream(OutputFilePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		foreach (var typeData in types)
		{
			var entry = zip.CreateEntry(typeData.EntryPath + BytecodeEntryExtension,
				CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			WriteTypeEntry(writer, typeData);
		}
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

	private static void WriteTypeEntry(BinaryWriter writer, TypeBytecodeData typeData)
	{
		writer.Write(EntryMagicBytes);
		writer.Write(Version);
		var table = CreateTypeEntryNameTable(typeData);
		table.Write(writer);
		WriteMembers(writer, typeData.Members, table);
		WriteMethodHeaders(writer, typeData.Methods, table);
		writer.Write7BitEncodedInt(typeData.RunInstructions.Count);
		foreach (var instruction in typeData.RunInstructions)
			WriteInstruction(writer, instruction, table);
		writer.Write7BitEncodedInt(typeData.MethodInstructions.Count);
		foreach (var methodData in typeData.MethodInstructions)
		{
			writer.Write7BitEncodedInt(table[methodData.Key.Name]);
			writer.Write7BitEncodedInt(methodData.Key.Parameters.Count);
			writer.Write7BitEncodedInt(methodData.Value.Count);
			foreach (var instruction in methodData.Value)
				WriteInstruction(writer, instruction, table);
		}
	}

	private static NameTable CreateTypeEntryNameTable(TypeBytecodeData typeData)
	{
		var table = new NameTable(typeData.RunInstructions);
		foreach (var methodInstructions in typeData.MethodInstructions.Values)
		foreach (var instruction in methodInstructions)
			table.CollectInstruction(instruction);
		table.Add(typeData.TypeName);
		foreach (var member in typeData.Members)
		{
			table.Add(member.Name);
			table.Add(member.TypeName);
		}
		foreach (var method in typeData.Methods)
		{
			table.Add(method.Name);
			table.Add(method.ReturnTypeName);
			foreach (var parameter in method.Parameters)
			{
				table.Add(parameter.Name);
				table.Add(parameter.TypeName);
			}
		}
		return table;
	}

	private static void WriteMembers(BinaryWriter writer, IReadOnlyList<MemberBytecodeData> members,
		NameTable table)
	{
		writer.Write7BitEncodedInt(members.Count);
		foreach (var member in members)
		{
			writer.Write7BitEncodedInt(table[member.Name]);
			WriteTypeReference(writer, member.TypeName, table);
			writer.Write(false);
		}
	}

	private static void WriteMethodHeaders(BinaryWriter writer,
		IReadOnlyList<MethodBytecodeData> methods, NameTable table)
	{
		writer.Write7BitEncodedInt(methods.Count);
		foreach (var method in methods)
		{
			writer.Write7BitEncodedInt(table[method.Name]);
			writer.Write7BitEncodedInt(method.Parameters.Count);
			foreach (var parameter in method.Parameters)
			{
				writer.Write7BitEncodedInt(table[parameter.Name]);
				WriteTypeReference(writer, parameter.TypeName, table);
			}
			WriteTypeReference(writer, method.ReturnTypeName, table);
		}
	}

	private static void WriteTypeReference(BinaryWriter writer, string typeName, NameTable table)
	{
		switch (typeName)
		{
		case Type.None:
			writer.Write(TypeRefNone);
			break;
		case Type.Boolean:
			writer.Write(TypeRefBoolean); //ncrunch: no coverage
			break; //ncrunch: no coverage
		case Type.Number:
			writer.Write(TypeRefNumber);
			break;
		case Type.Text:
			writer.Write(TypeRefText);
			break;
		case Type.List:
			//ncrunch: no coverage start
			writer.Write(TypeRefList);
			break;
		case Type.Dictionary:
			writer.Write(TypeRefDictionary);
			break; //ncrunch: no coverage end
		default:
			writer.Write(TypeRefCustom);
			writer.Write7BitEncodedInt(table[typeName]);
			break;
		}
	}

	public string OutputFilePath { get; }
	public const string BytecodeEntryExtension = ".bytecode";
	internal static readonly byte[] EntryMagicBytes = "Strict"u8.ToArray();
	public const byte Version = 1;
	public const string Extension = ".strictbinary";

	/// <summary>
	/// Serializes all types and their instructions into in-memory .bytecode entry payloads.
	/// </summary>
	public static Dictionary<string, byte[]> SerializeToEntryBytes(
		Dictionary<string, IList<Instruction>> instructionsByType)
	{
		var result = new Dictionary<string, byte[]>(instructionsByType.Count,
			StringComparer.Ordinal);
		foreach (var (typeName, instructions) in instructionsByType)
		{
			var typeData = new TypeBytecodeData(typeName, typeName,
				Array.Empty<MemberBytecodeData>(),
				Array.Empty<MethodBytecodeData>(),
				instructions,
				new Dictionary<MethodBytecodeData, IList<Instruction>>());
			using var stream = new MemoryStream();
			using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
			WriteTypeEntry(writer, typeData);
			writer.Flush();
			result[typeName] = stream.ToArray();
		}
		return result;
	}

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);

	private static void WriteEntry(BinaryWriter writer, IList<Instruction> instructions)
	{
		writer.Write(EntryMagicBytes);
		writer.Write(Version);
		var table = new NameTable(instructions);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			WriteInstruction(writer, instruction, table);
	}

	private static void WriteInstruction(BinaryWriter writer, Instruction instruction,
		NameTable table)
	{
		switch (instruction)
		{
		case LoadConstantInstruction loadConst:
			writer.Write((byte)InstructionType.LoadConstantToRegister);
			writer.Write((byte)loadConst.Register);
			WriteValueInstance(writer, loadConst.ValueInstance, table);
			break;
		case LoadVariableToRegister loadVar:
			writer.Write((byte)InstructionType.LoadVariableToRegister);
			writer.Write((byte)loadVar.Register);
			writer.Write7BitEncodedInt(table[loadVar.Identifier]);
			break;
		case StoreVariableInstruction storeVar:
			writer.Write((byte)InstructionType.StoreConstantToVariable);
			WriteValueInstance(writer, storeVar.ValueInstance, table);
			writer.Write7BitEncodedInt(table[storeVar.Identifier]);
			writer.Write(storeVar.IsMember);
			break;
		case StoreFromRegisterInstruction storeReg:
			writer.Write((byte)InstructionType.StoreRegisterToVariable);
			writer.Write((byte)storeReg.Register);
			writer.Write7BitEncodedInt(table[storeReg.Identifier]);
			break;
		case SetInstruction set:
			writer.Write((byte)InstructionType.Set);
			WriteValueInstance(writer, set.ValueInstance, table);
			writer.Write((byte)set.Register);
			break;
		case BinaryInstruction binary:
			writer.Write((byte)binary.InstructionType);
			writer.Write((byte)binary.Registers.Length);
			foreach (var reg in binary.Registers)
				writer.Write((byte)reg);
			break;
		case Invoke invoke:
			writer.Write((byte)InstructionType.Invoke);
			writer.Write((byte)invoke.Register);
			WriteMethodCallData(writer, invoke.Method, invoke.PersistedRegistry, table);
			break;
		case ReturnInstruction ret:
			writer.Write((byte)InstructionType.Return);
			writer.Write((byte)ret.Register);
			break;
		case LoopBeginInstruction loopBegin:
			writer.Write((byte)InstructionType.LoopBegin);
			writer.Write((byte)loopBegin.Register);
			writer.Write(loopBegin.IsRange);
			if (loopBegin.IsRange)
				writer.Write7BitEncodedInt((int)loopBegin.EndIndex!.Value);
			break;
		case LoopEndInstruction loopEnd:
			writer.Write((byte)InstructionType.LoopEnd);
			writer.Write7BitEncodedInt(loopEnd.Steps);
			break;
		case JumpIfNotZero jumpIfNotZero:
			writer.Write((byte)InstructionType.JumpIfNotZero);
			writer.Write7BitEncodedInt(jumpIfNotZero.Steps);
			writer.Write((byte)jumpIfNotZero.Register);
			break;
		case JumpIf jumpIf:
			//ncrunch: no coverage start
			writer.Write((byte)jumpIf.InstructionType);
			writer.Write7BitEncodedInt(jumpIf.Steps);
			break; //ncrunch: no coverage end
		case Jump jump:
			writer.Write((byte)jump.InstructionType);
			writer.Write7BitEncodedInt(jump.InstructionsToSkip);
			break;
		case JumpToId jumpToId:
			writer.Write((byte)jumpToId.InstructionType);
			writer.Write7BitEncodedInt(jumpToId.Id);
			break;
		case WriteToListInstruction writeToList:
			writer.Write((byte)InstructionType.InvokeWriteToList);
			writer.Write((byte)writeToList.Register);
			writer.Write7BitEncodedInt(table[writeToList.Identifier]);
			break;
		case WriteToTableInstruction writeToTable:
			writer.Write((byte)InstructionType.InvokeWriteToTable);
			writer.Write((byte)writeToTable.Key);
			writer.Write((byte)writeToTable.Value);
			writer.Write7BitEncodedInt(table[writeToTable.Identifier]);
			break;
		case RemoveInstruction remove:
			writer.Write((byte)InstructionType.InvokeRemove);
			writer.Write7BitEncodedInt(table[remove.Identifier]);
			writer.Write((byte)remove.Register);
			break;
		case ListCallInstruction listCall:
			writer.Write((byte)InstructionType.ListCall);
			writer.Write((byte)listCall.Register);
			writer.Write((byte)listCall.IndexValueRegister);
			writer.Write7BitEncodedInt(table[listCall.Identifier]);
			break;
		}
	}

	private static void WriteValueInstance(BinaryWriter writer, ValueInstance val, NameTable table)
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
			writer.Write7BitEncodedInt(table[val.List.ReturnType.Name]);
			var items = val.List.Items;
			writer.Write7BitEncodedInt(items.Count);
			foreach (var item in items)
				WriteValueInstance(writer, item, table);
			return;
		}
		if (val.IsDictionary)
		{
			writer.Write((byte)ValueKind.Dictionary);
			writer.Write7BitEncodedInt(table[val.GetType().Name]);
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
			return;
		}
		if (type.IsNone)
		{
			writer.Write((byte)ValueKind.None);
			return;
		}
		if (type.IsNumber)
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
			throw new NotSupportedException("WriteValueInstance not supported value: " + val); //ncrunch: no coverage
	}

	private static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	private static void WriteExpression(BinaryWriter writer, Expression expr, NameTable table)
	{
		switch (expr)
		{
		case Value { Data.IsText: true } val:
			writer.Write((byte)ExpressionKind.TextValue);
			writer.Write7BitEncodedInt(table[val.Data.Text]);
			break;
		case Value val when val.Data.GetType().IsBoolean:
			//ncrunch: no coverage start
			writer.Write((byte)ExpressionKind.BooleanValue);
			writer.Write7BitEncodedInt(table[val.Data.GetType().Name]);
			writer.Write(val.Data.Boolean);
			break; //ncrunch: no coverage end
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
			{ //ncrunch: no coverage start
				writer.Write((byte)ExpressionKind.NumberValue);
				writer.Write(val.Data.Number);
			} //ncrunch: no coverage end
			break;
		case Value val:
			throw new NotSupportedException("WriteExpression not supported value: " + val); //ncrunch: no coverage
		case MemberCall memberCall:
			writer.Write((byte)ExpressionKind.MemberRef);
			writer.Write7BitEncodedInt(table[memberCall.Member.Name]);
			writer.Write7BitEncodedInt(table[memberCall.Member.Type.Name]);
			writer.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				// ReSharper disable TailRecursiveCall
				WriteExpression(writer, memberCall.Instance, table); //ncrunch: no coverage
			break;
		case Binary binary:
			writer.Write((byte)ExpressionKind.BinaryExpr);
			writer.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(writer, binary.Instance!, table);
			WriteExpression(writer, binary.Arguments[0], table);
			break;
		case MethodCall mc:
			writer.Write((byte)ExpressionKind.MethodCallExpr);
			writer.Write7BitEncodedInt(table[mc.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[mc.Method.Name]);
			writer.Write7BitEncodedInt(mc.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[mc.ReturnType.Name]);
			writer.Write(mc.Instance != null);
			if (mc.Instance != null)
				WriteExpression(writer, mc.Instance, table);
			writer.Write7BitEncodedInt(mc.Arguments.Count);
			foreach (var arg in mc.Arguments)
				WriteExpression(writer, arg, table);
			break;
		default:
			writer.Write((byte)ExpressionKind.VariableRef);
			writer.Write7BitEncodedInt(table[expr.ToString()]);
			writer.Write7BitEncodedInt(table[expr.ReturnType.Name]);
			break;
		}
	}

	private static void WriteMethodCallData(BinaryWriter writer, MethodCall? methodCall,
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
			foreach (var arg in methodCall.Arguments)
				WriteExpression(writer, arg, table);
		}
		writer.Write(registry != null);
		if (registry != null)
		{
			writer.Write((byte)registry.NextRegister);
			writer.Write((byte)registry.PreviousRegister);
		}
	}

	private const byte TypeRefNone = 0;
	private const byte TypeRefBoolean = 1;
	private const byte TypeRefNumber = 2;
	private const byte TypeRefText = 3;
	private const byte TypeRefList = 4;
	private const byte TypeRefDictionary = 5;
	private const byte TypeRefCustom = byte.MaxValue;
}