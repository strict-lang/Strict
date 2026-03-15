using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

public sealed class BytecodeSerializer(StrictBinary usedTypes)
{
	/*obs
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
			var typeData = new TypeBytecodeData(typeName, typeName,
				Array.Empty<MemberBytecodeData>(),
				Array.Empty<MethodBytecodeData>(),
				instructions,
				new Dictionary<MethodBytecodeData, IList<Instruction>>());
	*
	public void Serialize(string filePath)
	{
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

	public const string Extension = ".strictbinary";
	public const string BytecodeEntryExtension = ".bytecode";

/*already in NameTable
	private static NameTable CreateTypeEntryNameTable(StrictBinary.BytecodeMembersAndMethods membersAndMethods)
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
*
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
*
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

	private static void WriteEntry(BinaryWriter writer, IList<Instruction> instructions)
	{
		writer.Write(StrictMagicBytes);
		writer.Write(Version);
		var table = new NameTable(instructions);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			WriteInstruction(writer, instruction, table);
	}
*/
	private static void WriteInstruction(BinaryWriter writer, Instruction instruction,
		NameTable table)
	{
		switch (instruction)
		{
		case LoadConstantInstruction loadConst:
			break;
		case LoadVariableToRegister loadVar:
			break;
		case StoreVariableInstruction storeVar:
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
		case PrintInstruction print:
			writer.Write((byte)InstructionType.Print);
			writer.Write7BitEncodedInt(table[print.TextPrefix]);
			writer.Write(print.ValueRegister.HasValue);
			if (print.ValueRegister.HasValue)
			{
				writer.Write((byte)print.ValueRegister.Value);
				writer.Write(print.ValueIsText);
			}
			break;
		}
	}

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
			//ncrunch: no coverage start
			writer.Write((byte)ExpressionKind.BinaryExpr);
			writer.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(writer, binary.Instance!, table);
			WriteExpression(writer, binary.Arguments[0], table);
			break; //ncrunch: no coverage end
		case MethodCall mc:
			writer.Write((byte)ExpressionKind.MethodCallExpr);
			writer.Write7BitEncodedInt(table[mc.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[mc.Method.Name]);
			writer.Write7BitEncodedInt(mc.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[mc.ReturnType.Name]);
			writer.Write(mc.Instance != null);
			if (mc.Instance != null)
				WriteExpression(writer, mc.Instance, table); //ncrunch: no coverage
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
}