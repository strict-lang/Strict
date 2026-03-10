using System.Collections;
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
	private static readonly byte[] EntryMagicBytes = "Strict"u8.ToArray();
	private const byte Version = 2;
	public const string Extension = ".strict_binary";
	private const string BytecodeEntryExtension = ".bytecode";

	public static void Serialize(IList<Instruction> instructions, string outputPath,
		string typeName = "main")
	{
		using var fileStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		var entry = zip.CreateEntry(typeName + BytecodeEntryExtension, CompressionLevel.Optimal);
		using var entryStream = entry.Open();
		using var writer = new BinaryWriter(entryStream);
		var stringTable = BuildStringTable(instructions);
		WriteEntry(writer, instructions, stringTable);
	}

	private static void WriteEntry(BinaryWriter w, IList<Instruction> instructions,
		StringTable stringTable)
	{
		w.Write(EntryMagicBytes);
		w.Write(Version);
		WriteStringTable(w, stringTable);
		w.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			WriteInstruction(w, instruction, stringTable);
	}

	public static List<Instruction> Deserialize(string zipFilePath, Package package)
	{
		try
		{
			using var zip = ZipFile.OpenRead(zipFilePath);
			if (zip.Entries.Count == 0)
				throw new InvalidBytecodeFileException("ZIP contains no entries");
			using var entryStream = zip.Entries[0].Open();
			using var reader = new BinaryReader(entryStream);
			return ReadEntry(reader, package);
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid ZIP file: " + ex.Message);
		}
	}

	internal static List<Instruction> DeserializeEntry(Stream entryStream, Package package)
	{
		using var reader = new BinaryReader(entryStream, System.Text.Encoding.UTF8, leaveOpen: true);
		return ReadEntry(reader, package);
	}

	private static List<Instruction> ReadEntry(BinaryReader r, Package package)
	{
		ValidateMagicAndVersion(r);
		var stringTable = ReadStringTable(r);
		var count = r.Read7BitEncodedInt();
		var instructions = new List<Instruction>(count);
		for (var i = 0; i < count; i++)
			instructions.Add(ReadInstruction(r, package, stringTable));
		return instructions;
	}

	private static void ValidateMagicAndVersion(BinaryReader reader)
	{
		var magic = reader.ReadBytes(EntryMagicBytes.Length);
		if (!magic.SequenceEqual(EntryMagicBytes))
			throw new InvalidBytecodeFileException("Entry does not start with 'Strict' magic");
		var fileVersion = reader.ReadByte();
		if (fileVersion is 0 or > Version)
			throw new InvalidVersion(fileVersion);
	}

	public sealed class InvalidBytecodeFileException(string message)
		: Exception("Not a valid Strict bytecode (" + Extension + ") file: " + message);

	public sealed class InvalidVersion(byte fileVersion) : Exception("File version: " +
		fileVersion + ", this runtime only supports up to version " + Version);

	// ── String table ──────────────────────────────────────────────────────────

	private sealed class StringTable : IEnumerable<string>
	{
		private readonly List<string> strings = [];
		private readonly Dictionary<string, int> index = new(StringComparer.Ordinal);

		public int Add(string s)
		{
			if (index.TryGetValue(s, out var i))
				return i;
			i = strings.Count;
			strings.Add(s);
			index[s] = i;
			return i;
		}

		public int this[string s] => index[s];
		public int Count => strings.Count;
		public IEnumerator<string> GetEnumerator() => strings.GetEnumerator();
		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
	}

	private static StringTable BuildStringTable(IList<Instruction> instructions)
	{
		var table = new StringTable();
		foreach (var inst in instructions)
			CollectStrings(inst, table);
		return table;
	}

	private static void CollectStrings(Instruction inst, StringTable table)
	{
		switch (inst)
		{
		case LoadVariableToRegister loadVar:
			table.Add(loadVar.Identifier);
			break;
		case StoreVariableInstruction storeVar:
			table.Add(storeVar.Identifier);
			CollectValueInstanceStrings(storeVar.ValueInstance, table);
			break;
		case StoreFromRegisterInstruction storeReg:
			table.Add(storeReg.Identifier);
			break;
		case SetInstruction set:
			CollectValueInstanceStrings(set.ValueInstance, table);
			break;
		case LoadConstantInstruction loadConst:
			CollectValueInstanceStrings(loadConst.ValueInstance, table);
			break;
		case Invoke invoke when invoke.Method != null:
			CollectMethodCallStrings(invoke.Method, table);
			break;
		case WriteToListInstruction writeList:
			table.Add(writeList.Identifier);
			break;
		case WriteToTableInstruction writeTable:
			table.Add(writeTable.Identifier);
			break;
		case RemoveInstruction remove:
			table.Add(remove.Identifier);
			break;
		case ListCallInstruction listCall:
			table.Add(listCall.Identifier);
			break;
		}
	}

	private static void CollectValueInstanceStrings(ValueInstance val, StringTable table)
	{
		if (val.IsText)
		{
			table.Add(val.Text);
			return;
		}
		if (val.IsList)
		{
			table.Add(val.List.ReturnType.Name);
			foreach (var item in val.List.Items)
				CollectValueInstanceStrings(item, table);
			return;
		}
		var type = val.GetTypeExceptText();
		if (type.IsBoolean || type.IsNone)
			table.Add(type.Name);
		else if (!IsSmallNumber(val.Number))
			table.Add(type.Name);
		// SmallNumber: no type name needed (Number is implied)
	}

	private static void CollectMethodCallStrings(MethodCall mc, StringTable table)
	{
		table.Add(mc.Method.Type.Name);
		table.Add(mc.Method.Name);
		table.Add(mc.ReturnType.Name);
		if (mc.Instance != null)
			CollectExpressionStrings(mc.Instance, table);
		foreach (var arg in mc.Arguments)
			CollectExpressionStrings(arg, table);
	}

	private static void CollectExpressionStrings(Expression expr, StringTable table)
	{
		switch (expr)
		{
		case Value val when val.Data.IsText:
			table.Add(val.Data.Text);
			break;
		case Value val when val.Data.GetTypeExceptText().IsBoolean:
			table.Add(val.Data.GetTypeExceptText().Name);
			break;
		case Value val when !IsSmallNumber(val.Data.Number):
			table.Add(val.Data.GetTypeExceptText().Name);
			break;
		case MemberCall memberCall:
			table.Add(memberCall.Member.Name);
			table.Add(memberCall.Member.Type.Name);
			if (memberCall.Instance != null)
				CollectExpressionStrings(memberCall.Instance, table);
			break;
		case Binary binary:
			table.Add(binary.Method.Name);
			CollectExpressionStrings(binary.Instance!, table);
			CollectExpressionStrings(binary.Arguments[0], table);
			break;
		case MethodCall mc:
			CollectMethodCallStrings(mc, table);
			break;
		default:
			table.Add(expr.ToString());
			table.Add(expr.ReturnType.Name);
			break;
		}
	}

	private static void WriteStringTable(BinaryWriter w, StringTable table)
	{
		w.Write7BitEncodedInt(table.Count);
		foreach (var s in table)
			w.Write(s);
	}

	private static string[] ReadStringTable(BinaryReader r)
	{
		var count = r.Read7BitEncodedInt();
		var table = new string[count];
		for (var i = 0; i < count; i++)
			table[i] = r.ReadString();
		return table;
	}

	// ── ValueKind ─────────────────────────────────────────────────────────────

	private enum ValueKind : byte
	{
		None,
		Number,
		Text,
		Boolean,
		List,
		SmallNumber,  // 0–255 stored as 1 byte; Number type is implied
		Character,
		Name,
		Dictionary
	}

	private static bool IsSmallNumber(double v) =>
		v >= 0 && v <= 255 && v == Math.Floor(v);

	// ── Write helpers ─────────────────────────────────────────────────────────

	private static void WriteInstruction(BinaryWriter w, Instruction inst, StringTable table)
	{
		switch (inst)
		{
		case LoadConstantInstruction loadConst:
			w.Write((byte)InstructionType.LoadConstantToRegister);
			w.Write((byte)loadConst.Register);
			WriteValueInstance(w, loadConst.ValueInstance, table);
			break;
		case LoadVariableToRegister loadVar:
			w.Write((byte)InstructionType.LoadVariableToRegister);
			w.Write((byte)loadVar.Register);
			w.Write7BitEncodedInt(table[loadVar.Identifier]);
			break;
		case StoreVariableInstruction storeVar:
			w.Write((byte)InstructionType.StoreConstantToVariable);
			WriteValueInstance(w, storeVar.ValueInstance, table);
			w.Write7BitEncodedInt(table[storeVar.Identifier]);
			w.Write(storeVar.IsMember);
			break;
		case StoreFromRegisterInstruction storeReg:
			w.Write((byte)InstructionType.StoreRegisterToVariable);
			w.Write((byte)storeReg.Register);
			w.Write7BitEncodedInt(table[storeReg.Identifier]);
			break;
		case SetInstruction set:
			w.Write((byte)InstructionType.Set);
			WriteValueInstance(w, set.ValueInstance, table);
			w.Write((byte)set.Register);
			break;
		case BinaryInstruction binary:
			w.Write((byte)binary.InstructionType);
			w.Write((byte)binary.Registers.Length);
			foreach (var reg in binary.Registers)
				w.Write((byte)reg);
			break;
		case Invoke invoke:
			w.Write((byte)InstructionType.Invoke);
			w.Write((byte)invoke.Register);
			WriteMethodCallData(w, invoke.Method, invoke.PersistedRegistry, table);
			break;
		case ReturnInstruction ret:
			w.Write((byte)InstructionType.Return);
			w.Write((byte)ret.Register);
			break;
		case LoopBeginInstruction loopBegin:
			w.Write((byte)InstructionType.LoopBegin);
			w.Write((byte)loopBegin.Register);
			w.Write(loopBegin.IsRange);
			if (loopBegin.IsRange)
				w.Write7BitEncodedInt((int)loopBegin.EndIndex!.Value);
			break;
		case LoopEndInstruction loopEnd:
			w.Write((byte)InstructionType.LoopEnd);
			w.Write7BitEncodedInt(loopEnd.Steps);
			break;
		case JumpIfNotZero jumpIfNotZero:
			w.Write((byte)InstructionType.JumpIfNotZero);
			w.Write7BitEncodedInt(jumpIfNotZero.Steps);
			w.Write((byte)jumpIfNotZero.Register);
			break;
		case JumpIf jumpIf:
			w.Write((byte)jumpIf.InstructionType);
			w.Write7BitEncodedInt(jumpIf.Steps);
			break;
		case Jump jump:
			w.Write((byte)jump.InstructionType);
			w.Write7BitEncodedInt(jump.InstructionsToSkip);
			break;
		case JumpToId jumpToId:
			w.Write((byte)jumpToId.InstructionType);
			w.Write7BitEncodedInt(jumpToId.Id);
			break;
		case WriteToListInstruction writeToList:
			w.Write((byte)InstructionType.InvokeWriteToList);
			w.Write((byte)writeToList.Register);
			w.Write7BitEncodedInt(table[writeToList.Identifier]);
			break;
		case WriteToTableInstruction writeToTable:
			w.Write((byte)InstructionType.InvokeWriteToTable);
			w.Write((byte)writeToTable.Key);
			w.Write((byte)writeToTable.Value);
			w.Write7BitEncodedInt(table[writeToTable.Identifier]);
			break;
		case RemoveInstruction remove:
			w.Write((byte)InstructionType.InvokeRemove);
			w.Write7BitEncodedInt(table[remove.Identifier]);
			w.Write((byte)remove.Register);
			break;
		case ListCallInstruction listCall:
			w.Write((byte)InstructionType.ListCall);
			w.Write((byte)listCall.Register);
			w.Write((byte)listCall.IndexValueRegister);
			w.Write7BitEncodedInt(table[listCall.Identifier]);
			break;
		}
	}

	private static void WriteValueInstance(BinaryWriter w, ValueInstance val, StringTable table)
	{
		if (val.IsText)
		{
			w.Write((byte)ValueKind.Text);
			w.Write7BitEncodedInt(table[val.Text]);
			return;
		}
		if (val.IsList)
		{
			w.Write((byte)ValueKind.List);
			w.Write7BitEncodedInt(table[val.List.ReturnType.Name]);
			var items = val.List.Items;
			w.Write7BitEncodedInt(items.Length);
			foreach (var item in items)
				WriteValueInstance(w, item, table);
			return;
		}
		var type = val.GetTypeExceptText();
		if (type.IsBoolean)
		{
			w.Write((byte)ValueKind.Boolean);
			w.Write7BitEncodedInt(table[type.Name]);
			w.Write(val.Boolean);
			return;
		}
		if (type.IsNone)
		{
			w.Write((byte)ValueKind.None);
			w.Write7BitEncodedInt(table[type.Name]);
			return;
		}
		if (IsSmallNumber(val.Number))
		{
			w.Write((byte)ValueKind.SmallNumber);
			w.Write((byte)(int)val.Number);
			return;
		}
		w.Write((byte)ValueKind.Number);
		w.Write7BitEncodedInt(table[type.Name]);
		w.Write(val.Number);
	}

	private enum ExpressionKind : byte
	{
		SmallNumberValue,   // 1 extra byte (0–255)
		NumberValue,        // 8-byte double
		TextValue,          // string-table index
		BooleanValue,       // string-table index + 1-byte bool
		VariableRef,        // string-table index (name) + string-table index (type)
		MemberRef,          // string-table index (name) + string-table index (type) + optional instance
		BinaryExpr,         // string-table index (op) + left + right
		MethodCallExpr      // string-table indices + optional instance + args
	}

	private static void WriteExpression(BinaryWriter w, Expression expr, StringTable table)
	{
		switch (expr)
		{
		case Value val when val.Data.IsText:
			w.Write((byte)ExpressionKind.TextValue);
			w.Write7BitEncodedInt(table[val.Data.Text]);
			break;
		case Value val when val.Data.GetTypeExceptText().IsBoolean:
			w.Write((byte)ExpressionKind.BooleanValue);
			w.Write7BitEncodedInt(table[val.Data.GetTypeExceptText().Name]);
			w.Write(val.Data.Boolean);
			break;
		case Value val when IsSmallNumber(val.Data.Number):
			w.Write((byte)ExpressionKind.SmallNumberValue);
			w.Write((byte)(int)val.Data.Number);
			break;
		case Value val:
			w.Write((byte)ExpressionKind.NumberValue);
			w.Write(val.Data.Number);
			break;
		case MemberCall memberCall:
			w.Write((byte)ExpressionKind.MemberRef);
			w.Write7BitEncodedInt(table[memberCall.Member.Name]);
			w.Write7BitEncodedInt(table[memberCall.Member.Type.Name]);
			w.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				WriteExpression(w, memberCall.Instance, table);
			break;
		case Binary binary:
			w.Write((byte)ExpressionKind.BinaryExpr);
			w.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(w, binary.Instance!, table);
			WriteExpression(w, binary.Arguments[0], table);
			break;
		case MethodCall mc:
			w.Write((byte)ExpressionKind.MethodCallExpr);
			w.Write7BitEncodedInt(table[mc.Method.Type.Name]);
			w.Write7BitEncodedInt(table[mc.Method.Name]);
			w.Write7BitEncodedInt(mc.Method.Parameters.Count);
			w.Write7BitEncodedInt(table[mc.ReturnType.Name]);
			w.Write(mc.Instance != null);
			if (mc.Instance != null)
				WriteExpression(w, mc.Instance, table);
			w.Write7BitEncodedInt(mc.Arguments.Count);
			foreach (var arg in mc.Arguments)
				WriteExpression(w, arg, table);
			break;
		default:
			w.Write((byte)ExpressionKind.VariableRef);
			w.Write7BitEncodedInt(table[expr.ToString()]);
			w.Write7BitEncodedInt(table[expr.ReturnType.Name]);
			break;
		}
	}

	private static void WriteMethodCallData(BinaryWriter w, MethodCall? methodCall,
		Registry? registry, StringTable table)
	{
		w.Write(methodCall != null);
		if (methodCall != null)
		{
			w.Write7BitEncodedInt(table[methodCall.Method.Type.Name]);
			w.Write7BitEncodedInt(table[methodCall.Method.Name]);
			w.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			w.Write7BitEncodedInt(table[methodCall.ReturnType.Name]);
			w.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(w, methodCall.Instance, table);
			w.Write7BitEncodedInt(methodCall.Arguments.Count);
			foreach (var arg in methodCall.Arguments)
				WriteExpression(w, arg, table);
		}
		w.Write(registry != null);
		if (registry != null)
		{
			w.Write((byte)registry.NextRegister);
			w.Write((byte)registry.PreviousRegister);
		}
	}

	// ── Read helpers ──────────────────────────────────────────────────────────

	private static Instruction ReadInstruction(BinaryReader r, Package package, string[] table)
	{
		var type = (InstructionType)r.ReadByte();
		return type switch
		{
			InstructionType.LoadConstantToRegister => ReadLoadConstant(r, package, table),
			InstructionType.LoadVariableToRegister => ReadLoadVariable(r, table),
			InstructionType.StoreConstantToVariable => ReadStoreVariable(r, package, table),
			InstructionType.StoreRegisterToVariable => ReadStoreFromRegister(r, table),
			InstructionType.Set => ReadSet(r, package, table),
			InstructionType.Invoke => ReadInvoke(r, package, table),
			InstructionType.Return => new ReturnInstruction((Register)r.ReadByte()),
			InstructionType.LoopBegin => ReadLoopBegin(r),
			InstructionType.LoopEnd => new LoopEndInstruction(r.Read7BitEncodedInt()),
			InstructionType.JumpIfNotZero => ReadJumpIfNotZero(r),
			InstructionType.Jump => new Jump(r.Read7BitEncodedInt()),
			InstructionType.JumpIfTrue => new Jump(r.Read7BitEncodedInt(), InstructionType.JumpIfTrue),
			InstructionType.JumpIfFalse => new Jump(r.Read7BitEncodedInt(), InstructionType.JumpIfFalse),
			InstructionType.JumpEnd => new JumpToId(InstructionType.JumpEnd, r.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfFalse => new JumpToId(InstructionType.JumpToIdIfFalse, r.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfTrue => new JumpToId(InstructionType.JumpToIdIfTrue, r.Read7BitEncodedInt()),
			InstructionType.InvokeWriteToList => ReadWriteToList(r, table),
			InstructionType.InvokeWriteToTable => ReadWriteToTable(r, table),
			InstructionType.InvokeRemove => ReadRemove(r, table),
			InstructionType.ListCall => ReadListCall(r, table),
			_ when IsBinaryOp(type) => ReadBinary(r, type),
			_ => throw new InvalidBytecodeFileException("Unknown instruction type: " + type)
		};
	}

	private static bool IsBinaryOp(InstructionType t) =>
		t > InstructionType.StoreSeparator && t < InstructionType.BinaryOperatorsSeparator;

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
		if (isRange)
		{
			var endReg = (Register)r.Read7BitEncodedInt();
			return new LoopBeginInstruction(reg, endReg);
		}
		return new LoopBeginInstruction(reg);
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
			ValueKind.Number =>
				new ValueInstance(package.GetType(table[r.Read7BitEncodedInt()]), r.ReadDouble()),
			ValueKind.SmallNumber =>
				new ValueInstance(package.GetType(Type.Number), (double)r.ReadByte()),
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
