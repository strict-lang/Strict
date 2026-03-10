using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Saves optimized <see cref="Instruction" /> lists to a compact .strict_binary ZIP file
/// and restores them by resolving type and method references from a <see cref="Package" />.
/// The ZIP contains one entry per serialized unit named {typeName}.bytecode.
/// Entry layout: magic(6) + version(1) + string-table + instruction-count(7bit) + instructions.
/// All counts and lengths use Write7BitEncodedInt. String identifiers/names use string-table indices.
/// SmallNumber ValueKind stores 0–255 values in 1 byte (type implied). No source path is stored.
/// </summary>
public static class BytecodeSerializer
{
	public static void Serialize(IList<Instruction> instructions, string outputFilePath,
		string typeName = "main")
	{
		using var fileStream = new FileStream(outputFilePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		var entry = zip.CreateEntry(typeName + BytecodeEntryExtension, CompressionLevel.Optimal);
		using var entryStream = entry.Open();
		using var writer = new BinaryWriter(entryStream);
		WriteEntry(writer, instructions);
	}

	private const string BytecodeEntryExtension = ".bytecode";

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

	private static readonly byte[] EntryMagicBytes = "Strict"u8.ToArray();
	public const byte Version = 1;
	public const string Extension = ".strictbinary";

	public static List<Instruction> Deserialize(string zipFilePath, Package package)
	{
		try
		{
			using var zip = ZipFile.OpenRead(zipFilePath);
			if (zip.Entries.Count == 0)
				throw new InvalidBytecodeFileException(Extension + " ZIP contains no entries");
			using var entryStream = zip.Entries[0].Open();
			using var reader = new BinaryReader(entryStream);
			return ReadEntry(reader, package);
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid " + Extension + "ZIP file: " + ex.Message);
		}
	}

	internal static List<Instruction> DeserializeEntry(Stream entryStream, Package package)
	{
		using var reader = new BinaryReader(entryStream, System.Text.Encoding.UTF8, leaveOpen: true);
		return ReadEntry(reader, package);
	}

	private static List<Instruction> ReadEntry(BinaryReader reader, Package package)
	{
		ValidateMagicAndVersion(reader);
		var table = new NameTable(reader);
		var count = reader.Read7BitEncodedInt();
		var instructions = new List<Instruction>(count);
		for (var index = 0; index < count; index++)
			instructions.Add(ReadInstruction(reader, package, table.ToArray()));
		return instructions;
	}

	private static void ValidateMagicAndVersion(BinaryReader reader)
	{
		var magic = reader.ReadBytes(EntryMagicBytes.Length);
		if (!magic.SequenceEqual(EntryMagicBytes))
			throw new InvalidBytecodeFileException("Entry does not start with 'Strict' magic bytes");
		var fileVersion = reader.ReadByte();
		if (fileVersion is 0 or > Version)
			throw new InvalidVersion(fileVersion);
	}

	public sealed class InvalidBytecodeFileException(string message)
		: Exception("Not a valid Strict bytecode (" + Extension + ") file: " + message);

	public sealed class InvalidVersion(byte fileVersion) : Exception("File version: " +
		fileVersion + ", this runtime only supports up to version " + Version);

	private enum ValueKind : byte
	{
		None,
		/// <summary>
		/// 0–255 stored as 1 byte; Number type is implied
		/// </summary>
		SmallNumber,
		/// <summary>
		/// Any 32-bit signed integer value, no floating point.
		/// </summary>
		IntegerNumber,
		/// <summary>
		/// Any other number is stored as a 64-bit double floating point number (default in Strict)
		/// </summary>
		Number,
		Text,
		Boolean,
		List,
		//TODO: need tests
		Character,
		Name,
		Dictionary
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
			writer.Write((byte)jumpIf.InstructionType);
			writer.Write7BitEncodedInt(jumpIf.Steps);
			break;
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
			writer.Write7BitEncodedInt(items.Length);
			foreach (var item in items)
				WriteValueInstance(writer, item, table);
			return;
		}
		var type = val.GetType();
		if (type.IsBoolean)
		{
			writer.Write((byte)ValueKind.Boolean);
			writer.Write7BitEncodedInt(table[type.Name]);
			writer.Write(val.Boolean);
			return;
		}
		if (type.IsNone)
		{
			writer.Write((byte)ValueKind.None);
			writer.Write7BitEncodedInt(table[type.Name]);
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
			throw new NotSupportedException("WriteValueInstance not supported value: " + val);
	}

	private static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);

	private enum ExpressionKind : byte
	{
		/// <summary>
		/// Store any small number as just 1 extra byte (only values 0–255 would work)
		/// </summary>
		SmallNumberValue,
		/// <summary>
		/// 4-byte integer number, second most common like int in c-type languages.
		/// </summary>
		IntegerNumberValue,
		/// <summary>
		/// 8-byte double floating point number for everything else
		/// </summary>
		NumberValue,
		/// <summary>
		/// Stored as a NameTable index
		/// </summary>
		TextValue,
		/// <summary>
		/// NameTable index + 1-byte bool
		/// </summary>
		BooleanValue,
		/// <summary>
		/// NameTable index (name) + NameTable index (type)
		/// </summary>
		VariableRef,
		/// <summary>
		/// NameTable index (name) + NameTable index (type) + optional instance
		/// </summary>
		MemberRef,
		/// <summary>
		/// NameTable index (op) + left + right
		/// </summary>
		BinaryExpr,
		/// <summary>
		/// NameTable indices + optional instance + args
		/// </summary>
		MethodCallExpr
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
				// ReSharper disable TailRecursiveCall
				WriteExpression(writer, memberCall.Instance, table);
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

	private static Instruction ReadInstruction(BinaryReader reader, Package package, string[] table)
	{
		var type = (InstructionType)reader.ReadByte();
		return type switch
		{
			InstructionType.LoadConstantToRegister => ReadLoadConstant(reader, package, table),
			InstructionType.LoadVariableToRegister => ReadLoadVariable(reader, table),
			InstructionType.StoreConstantToVariable => ReadStoreVariable(reader, package, table),
			InstructionType.StoreRegisterToVariable => ReadStoreFromRegister(reader, table),
			InstructionType.Set => ReadSet(reader, package, table),
			InstructionType.Invoke => ReadInvoke(reader, package, table),
			InstructionType.Return => new ReturnInstruction((Register)reader.ReadByte()),
			InstructionType.LoopBegin => ReadLoopBegin(reader),
			InstructionType.LoopEnd => new LoopEndInstruction(reader.Read7BitEncodedInt()),
			InstructionType.JumpIfNotZero => ReadJumpIfNotZero(reader),
			InstructionType.Jump => new Jump(reader.Read7BitEncodedInt()),
			InstructionType.JumpIfTrue => new Jump(reader.Read7BitEncodedInt(), InstructionType.JumpIfTrue),
			InstructionType.JumpIfFalse => new Jump(reader.Read7BitEncodedInt(), InstructionType.JumpIfFalse),
			InstructionType.JumpEnd => new JumpToId(InstructionType.JumpEnd, reader.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfFalse => new JumpToId(InstructionType.JumpToIdIfFalse, reader.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfTrue => new JumpToId(InstructionType.JumpToIdIfTrue, reader.Read7BitEncodedInt()),
			InstructionType.InvokeWriteToList => ReadWriteToList(reader, table),
			InstructionType.InvokeWriteToTable => ReadWriteToTable(reader, table),
			InstructionType.InvokeRemove => ReadRemove(reader, table),
			InstructionType.ListCall => ReadListCall(reader, table),
			_ when IsBinaryOp(type) => ReadBinary(reader, type),
			_ => throw new InvalidBytecodeFileException("Unknown instruction type: " + type)
		};
	}

	private static bool IsBinaryOp(InstructionType t) =>
		t is > InstructionType.StoreSeparator and < InstructionType.BinaryOperatorsSeparator;

	private static LoadConstantInstruction ReadLoadConstant(BinaryReader r, Package package,
		string[] table) =>
		new((Register)r.ReadByte(), ReadValueInstance(r, package, table));

	private static LoadVariableToRegister ReadLoadVariable(BinaryReader r, string[] table) =>
		new((Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static StoreVariableInstruction ReadStoreVariable(BinaryReader r, Package package,
		string[] table) =>
		new(ReadValueInstance(r, package, table), table[r.Read7BitEncodedInt()], r.ReadBoolean());

	private static StoreFromRegisterInstruction ReadStoreFromRegister(BinaryReader r,
		string[] table) =>
		new((Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static SetInstruction ReadSet(BinaryReader r, Package package, string[] table) =>
		new(ReadValueInstance(r, package, table), (Register)r.ReadByte());

	private static BinaryInstruction ReadBinary(BinaryReader r, InstructionType type)
	{
		var count = r.ReadByte();
		var registers = new Register[count];
		for (var i = 0; i < count; i++)
			registers[i] = (Register)r.ReadByte();
		return new BinaryInstruction(type, registers);
	}

	private static Invoke ReadInvoke(BinaryReader r, Package package, string[] table)
	{
		var register = (Register)r.ReadByte();
		var (methodCall, registry) = ReadMethodCallData(r, package, table);
		return new Invoke(register, methodCall!, registry!);
	}

	private static LoopBeginInstruction ReadLoopBegin(BinaryReader r)
	{
		var reg = (Register)r.ReadByte();
		var isRange = r.ReadBoolean();
		return isRange
			? new LoopBeginInstruction(reg, (Register)r.Read7BitEncodedInt())
			: new LoopBeginInstruction(reg);
	}

	private static JumpIfNotZero ReadJumpIfNotZero(BinaryReader r) =>
		new(r.Read7BitEncodedInt(), (Register)r.ReadByte());

	private static WriteToListInstruction ReadWriteToList(BinaryReader r, string[] table) =>
		new((Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static WriteToTableInstruction ReadWriteToTable(BinaryReader r, string[] table) =>
		new((Register)r.ReadByte(), (Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static RemoveInstruction ReadRemove(BinaryReader r, string[] table) =>
		new(table[r.Read7BitEncodedInt()], (Register)r.ReadByte());

	private static ListCallInstruction ReadListCall(BinaryReader r, string[] table) =>
		new((Register)r.ReadByte(), (Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static ValueInstance ReadValueInstance(BinaryReader r, Package package, string[] table)
	{
		var kind = (ValueKind)r.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table[r.Read7BitEncodedInt()]),
			ValueKind.None => new ValueInstance(package.GetType(table[r.Read7BitEncodedInt()])),
			ValueKind.Boolean =>
				new ValueInstance(package.GetType(table[r.Read7BitEncodedInt()]), r.ReadBoolean()),
			ValueKind.SmallNumber =>
				new ValueInstance(package.GetType(Type.Number), r.ReadByte()),
			ValueKind.IntegerNumber =>
				new ValueInstance(package.GetType(Type.Number), r.ReadInt32()),
			ValueKind.Number =>
				new ValueInstance(package.GetType(Type.Number), r.ReadDouble()),
			ValueKind.List => ReadListValueInstance(r, package, table),
			_ => throw new InvalidBytecodeFileException("Unknown ValueKind: " + kind + " at byte " +
				r.BaseStream.Position)
		};
	}

	private static ValueInstance ReadListValueInstance(BinaryReader r, Package package,
		string[] table)
	{
		var typeName = table[r.Read7BitEncodedInt()];
		var count = r.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var i = 0; i < count; i++)
			items[i] = ReadValueInstance(r, package, table);
		return new ValueInstance(package.GetType(typeName), items);
	}

	private static Expression ReadExpression(BinaryReader r, Package package, string[] table)
	{
		var kind = (ExpressionKind)r.ReadByte();
		return kind switch
		{
			ExpressionKind.SmallNumberValue => new Number(package, r.ReadByte()),
			ExpressionKind.IntegerNumberValue => new Number(package, r.ReadInt32()),
			ExpressionKind.NumberValue => new Number(package, r.ReadDouble()),
			ExpressionKind.TextValue => new Text(package, table[r.Read7BitEncodedInt()]),
			ExpressionKind.BooleanValue => ReadBooleanValue(r, package, table),
			ExpressionKind.VariableRef => ReadVariableRef(r, package, table),
			ExpressionKind.MemberRef => ReadMemberRef(r, package, table),
			ExpressionKind.BinaryExpr => ReadBinaryExpr(r, package, table),
			ExpressionKind.MethodCallExpr => ReadMethodCallExpr(r, package, table),
			_ => throw new InvalidBytecodeFileException("Unknown ExpressionKind: " + kind + " at " +
				r.BaseStream.Position)
		};
	}

	private static Value ReadBooleanValue(BinaryReader r, Package package, string[] table)
	{
		var type = package.GetType(table[r.Read7BitEncodedInt()]);
		return new Value(type, new ValueInstance(type, r.ReadBoolean()));
	}

	private static Expression ReadVariableRef(BinaryReader r, Package package, string[] table)
	{
		var name = table[r.Read7BitEncodedInt()];
		var type = package.GetType(table[r.Read7BitEncodedInt()]);
		var param = new Parameter(type, name, new Value(type, new ValueInstance(type)));
		return new ParameterCall(param);
	}

	private static MemberCall ReadMemberRef(BinaryReader r, Package package, string[] table)
	{
		var memberName = table[r.Read7BitEncodedInt()];
		var memberTypeName = table[r.Read7BitEncodedInt()];
		var hasInstance = r.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(r, package, table)
			: null;
		var anyBaseType = package.GetType(Type.Number);
		var fakeMember = new Member(anyBaseType, memberName + " " + memberTypeName, null);
		return new MemberCall(instance, fakeMember);
	}

	private static Binary ReadBinaryExpr(BinaryReader r, Package package, string[] table)
	{
		var operatorName = table[r.Read7BitEncodedInt()];
		var left = ReadExpression(r, package, table);
		var right = ReadExpression(r, package, table);
		var operatorMethod = FindOperatorMethod(package, operatorName, left.ReturnType);
		return new Binary(left, operatorMethod, [right]);
	}

	private static Method FindOperatorMethod(Package package, string operatorName,
		Type preferredType)
	{
		var method = preferredType.Methods.FirstOrDefault(m => m.Name == operatorName);
		if (method != null)
			return method;
		foreach (var typeName in new[] { Type.Number, Type.Text, Type.Boolean })
		{
			method = package.GetType(typeName).Methods.FirstOrDefault(m => m.Name == operatorName);
			if (method != null)
				return method;
		}
		throw new MethodNotFoundException(operatorName);
	}

	private static MethodCall ReadMethodCallExpr(BinaryReader r, Package package, string[] table)
	{
		var declaringTypeName = table[r.Read7BitEncodedInt()];
		var methodName = table[r.Read7BitEncodedInt()];
		var paramCount = r.Read7BitEncodedInt();
		var returnTypeName = table[r.Read7BitEncodedInt()];
		var hasInstance = r.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(r, package, table)
			: null;
		var argCount = r.Read7BitEncodedInt();
		var args = new Expression[argCount];
		for (var i = 0; i < argCount; i++)
			args[i] = ReadExpression(r, package, table);
		var declaringType = package.GetType(declaringTypeName);
		var method = FindMethod(declaringType, methodName, paramCount);
		var returnType = returnTypeName != method.ReturnType.Name
			? package.GetType(returnTypeName)
			: null;
		return new MethodCall(method, instance, args, returnType);
	}

	private static (MethodCall? MethodCall, Registry? Registry) ReadMethodCallData(BinaryReader r,
		Package package, string[] table)
	{
		MethodCall? methodCall = null;
		if (r.ReadBoolean())
		{
			var declaringTypeName = table[r.Read7BitEncodedInt()];
			var methodName = table[r.Read7BitEncodedInt()];
			var paramCount = r.Read7BitEncodedInt();
			var returnTypeName = table[r.Read7BitEncodedInt()];
			var hasInstance = r.ReadBoolean();
			var instance = hasInstance
				? ReadExpression(r, package, table)
				: null;
			var argCount = r.Read7BitEncodedInt();
			var args = new Expression[argCount];
			for (var i = 0; i < argCount; i++)
				args[i] = ReadExpression(r, package, table);
			var declaringType = package.GetType(declaringTypeName);
			var method = FindMethod(declaringType, methodName, paramCount);
			var returnType = returnTypeName != method.ReturnType.Name
				? package.GetType(returnTypeName)
				: null;
			methodCall = new MethodCall(method, instance, args, returnType);
		}
		Registry? registry = null;
		if (r.ReadBoolean())
		{
			registry = new Registry();
			var nextRegisterCount = r.ReadByte();
			var prev = (Register)r.ReadByte();
			for (var i = 0; i < nextRegisterCount; i++)
				registry.AllocateRegister();
			registry.PreviousRegister = prev;
		}
		return (methodCall, registry);
	}

	private static Method FindMethod(Type type, string methodName, int paramCount)
	{
		var method =
			type.Methods.FirstOrDefault(m =>
				m.Name == methodName && m.Parameters.Count == paramCount) ??
			type.Methods.FirstOrDefault(m => m.Name == methodName);
		if (method != null)
			return method;
		if (type.AvailableMethods.TryGetValue(methodName, out var available))
		{
			var found = available.FirstOrDefault(m => m.Parameters.Count == paramCount) ??
				available.FirstOrDefault();
			if (found != null)
				return found;
		}
		throw new MethodNotFoundException(methodName);
	}

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found") { }
}