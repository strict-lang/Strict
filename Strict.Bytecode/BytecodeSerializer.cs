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
/// Results are cached per file path+modification-time so repeated loads skip ZIP I/O entirely.
/// </summary>
public static class BytecodeSerializer
{
	private sealed class CachedBinaryData(
		Package localPackage,
		Dictionary<string, List<Instruction>> instructions,
		DateTime modified)
	{
		public Package LocalPackage { get; } = localPackage;
		public Dictionary<string, List<Instruction>> Instructions { get; } = instructions;
		public DateTime Modified { get; } = modified;
	}

	private static readonly Dictionary<string, CachedBinaryData> binaryCache =
		new(StringComparer.OrdinalIgnoreCase);
	private static readonly object cacheLock = new();

	/// <summary>
	/// Opens the ZIP once, loads embedded source types and deserializes bytecode, then caches
	/// the result keyed by absolute file path + modification time. Subsequent calls for the same
	/// unchanged file return the cached Package and Instructions without any file I/O.
	/// The local package uses a "bin-" prefix (e.g. "bin-SimpleCalculator") so it never collides
	/// with source-runner packages (directory names) or Strict type names (must start uppercase).
	/// </summary>
	public static (Package LocalPackage, Dictionary<string, List<Instruction>> Instructions)
		LoadTypesAndDeserializeAll(string zipFilePath, Package basePackage)
	{
		var fullPath = Path.GetFullPath(zipFilePath);
		var mtime = File.GetLastWriteTimeUtc(fullPath);
		var packageName = "bin-" + Path.GetFileNameWithoutExtension(fullPath);
		lock (cacheLock)
		{
			if (binaryCache.TryGetValue(fullPath, out var cached) && cached.Modified == mtime)
				return (cached.LocalPackage, cached.Instructions);
			if (binaryCache.TryGetValue(fullPath, out var stale))
			{
				stale.LocalPackage.Dispose();
				binaryCache.Remove(fullPath);
			}
			// Evict any other entry whose cached package has the same name under the same parent.
			// Rare (only when different file paths produce the same binary name, e.g. in tests).
			foreach (var key in binaryCache.Keys.ToList())
			{
				var entry = binaryCache[key];
				if (entry.LocalPackage.Name == packageName && entry.LocalPackage.Parent == basePackage)
				{
					entry.LocalPackage.Dispose();
					binaryCache.Remove(key);
				}
			}
			var localPackage = new Package(basePackage, packageName);
			Dictionary<string, List<Instruction>> instructions;
			try
			{
				instructions = LoadFromZipOnce(fullPath, localPackage);
			}
			catch
			{
				localPackage.Dispose();
				throw;
			}
			binaryCache[fullPath] = new CachedBinaryData(localPackage, instructions, mtime);
			return (localPackage, instructions);
		}
	}

	private static Dictionary<string, List<Instruction>> LoadFromZipOnce(string fullPath,
		Package package)
	{
		try
		{
			using var zip = ZipFile.OpenRead(fullPath);
			LoadEmbeddedTypesFromZip(zip, package);
			return DeserializeAllFromZip(zip, package);
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid " + Extension + " ZIP file: " + ex.Message);
		}
	}

	public static void Serialize(IList<Instruction> instructions, string outputFilePath,
		string typeName = "main", string? sourceDirectoryPath = null)
	{
		using var fileStream = new FileStream(outputFilePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		var entry = zip.CreateEntry(typeName + BytecodeEntryExtension, CompressionLevel.Optimal);
		using (var entryStream = entry.Open())
		using (var writer = new BinaryWriter(entryStream))
			WriteEntry(writer, instructions);
		if (!string.IsNullOrWhiteSpace(sourceDirectoryPath) && Directory.Exists(sourceDirectoryPath))
			AddSourceEntries(zip, sourceDirectoryPath);
	}

	private static void AddSourceEntries(ZipArchive zip, string sourceDirectoryPath)
	{
		foreach (var sourceFilePath in Directory.GetFiles(sourceDirectoryPath, "*" + Type.Extension))
		{
			var sourceEntry = zip.CreateEntry(Path.GetFileName(sourceFilePath), CompressionLevel.Optimal);
			using var sourceStream = sourceEntry.Open();
			using var writer = new StreamWriter(sourceStream);
			writer.Write(File.ReadAllText(sourceFilePath));
		}
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

	public static Dictionary<string, Type> LoadEmbeddedTypes(string zipFilePath, Package package)
	{
		using var zip = ZipFile.OpenRead(zipFilePath);
		return LoadEmbeddedTypesFromZip(zip, package);
	}

	private static Dictionary<string, Type> LoadEmbeddedTypesFromZip(ZipArchive zip, Package package)
	{
		var sourceEntries = zip.Entries.Where(entry =>
			entry.Name.EndsWith(Type.Extension, StringComparison.OrdinalIgnoreCase)).ToList();
		if (sourceEntries.Count == 0)
			return new Dictionary<string, Type>(StringComparer.Ordinal);
		var typesByName = new Dictionary<string, Type>(StringComparer.Ordinal);
		foreach (var entry in sourceEntries)
		{
			var typeName = Path.GetFileNameWithoutExtension(entry.Name);
			if (package.FindDirectType(typeName) != null)
				continue;
			using var entryStream = entry.Open();
			using var reader = new StreamReader(entryStream);
			var lines = ReadLines(reader);
			typesByName[typeName] = new Type(package, new TypeLines(typeName, lines));
		}
		foreach (var type in typesByName.Values)
			type.ParseMembersAndMethods(new MethodExpressionParser());
		return typesByName;
	}

	private static string[] ReadLines(StreamReader reader)
	{
		var text = reader.ReadToEnd();
		return text.Replace("\r\n", "\n", StringComparison.Ordinal).
			Split('\n', StringSplitOptions.None);
	}

	public static List<Instruction> Deserialize(string zipFilePath, Package package)
	{
		var entries = DeserializeAll(zipFilePath, package);
		return entries.Values.First();
	}

	public static Dictionary<string, List<Instruction>> DeserializeAll(string zipFilePath,
		Package package)
	{
		try
		{
			using var zip = ZipFile.OpenRead(zipFilePath);
			return DeserializeAllFromZip(zip, package);
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid " + Extension + " ZIP file: " + ex.Message);
		}
	}

	private static Dictionary<string, List<Instruction>> DeserializeAllFromZip(ZipArchive zip,
		Package package)
	{
		var bytecodeEntries = zip.Entries.Where(entry =>
			entry.Name.EndsWith(BytecodeEntryExtension, StringComparison.OrdinalIgnoreCase)).ToList();
		if (bytecodeEntries.Count == 0)
			throw new InvalidBytecodeFileException(Extension + " ZIP contains no entries");
		foreach (var entry in bytecodeEntries)
			EnsureTypeExists(package, Path.GetFileNameWithoutExtension(entry.Name));
		var result = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
		foreach (var entry in bytecodeEntries)
		{
			var typeName = Path.GetFileNameWithoutExtension(entry.Name);
			using var entryStream = entry.Open();
			result[typeName] = DeserializeEntry(entryStream, package);
		}
		return result;
	}

	private static void EnsureTypeExists(Package package, string typeName)
	{
		if (package.FindDirectType(typeName) != null)
			return;
		new Type(package, new TypeLines(typeName, Method.Run)).ParseMembersAndMethods(new MethodExpressionParser());
	}

	internal static List<Instruction> DeserializeEntry(Stream entryStream, Package package)
	{
		using var reader = new BinaryReader(entryStream, System.Text.Encoding.UTF8, leaveOpen: true);
		return ReadEntry(reader, package);
	}

	private static List<Instruction> ReadEntry(BinaryReader reader, Package package)
	{
		ValidateMagicAndVersion(reader);
		var tableArray = new NameTable(reader).ToArray();
		var numberType = package.GetType(Type.Number);
		var count = reader.Read7BitEncodedInt();
		var instructions = new List<Instruction>(count);
		for (var index = 0; index < count; index++)
			instructions.Add(ReadInstruction(reader, package, tableArray, numberType));
		return instructions;
	}

	private static void ValidateMagicAndVersion(BinaryReader reader)
	{
		Span<byte> magic = stackalloc byte[EntryMagicBytes.Length];
		reader.Read(magic);
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

	private static Instruction ReadInstruction(BinaryReader reader, Package package, string[] table,
		Type numberType)
	{
		var type = (InstructionType)reader.ReadByte();
		return type switch
		{
			InstructionType.LoadConstantToRegister => ReadLoadConstant(reader, package, table, numberType),
			InstructionType.LoadVariableToRegister => ReadLoadVariable(reader, table),
			InstructionType.StoreConstantToVariable => ReadStoreVariable(reader, package, table, numberType),
			InstructionType.StoreRegisterToVariable => ReadStoreFromRegister(reader, table),
			InstructionType.Set => ReadSet(reader, package, table, numberType),
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
		string[] table, Type numberType) =>
		new((Register)r.ReadByte(), ReadValueInstance(r, package, table, numberType));

	private static LoadVariableToRegister ReadLoadVariable(BinaryReader r, string[] table) =>
		new((Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static StoreVariableInstruction ReadStoreVariable(BinaryReader r, Package package,
		string[] table, Type numberType) =>
		new(ReadValueInstance(r, package, table, numberType), table[r.Read7BitEncodedInt()], r.ReadBoolean());

	private static StoreFromRegisterInstruction ReadStoreFromRegister(BinaryReader r,
		string[] table) =>
		new((Register)r.ReadByte(), table[r.Read7BitEncodedInt()]);

	private static SetInstruction ReadSet(BinaryReader r, Package package, string[] table,
		Type numberType) =>
		new(ReadValueInstance(r, package, table, numberType), (Register)r.ReadByte());

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

	private static ValueInstance ReadValueInstance(BinaryReader r, Package package, string[] table,
		Type numberType)
	{
		var kind = (ValueKind)r.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table[r.Read7BitEncodedInt()]),
			ValueKind.None => new ValueInstance(package.GetType(table[r.Read7BitEncodedInt()])),
			ValueKind.Boolean =>
				new ValueInstance(package.GetType(table[r.Read7BitEncodedInt()]), r.ReadBoolean()),
			ValueKind.SmallNumber => new ValueInstance(numberType, r.ReadByte()),
			ValueKind.IntegerNumber => new ValueInstance(numberType, r.ReadInt32()),
			ValueKind.Number => new ValueInstance(numberType, r.ReadDouble()),
			ValueKind.List => ReadListValueInstance(r, package, table, numberType),
			_ => throw new InvalidBytecodeFileException("Unknown ValueKind: " + kind + " at byte " +
				r.BaseStream.Position)
		};
	}

	private static ValueInstance ReadListValueInstance(BinaryReader r, Package package,
		string[] table, Type numberType)
	{
		var typeName = table[r.Read7BitEncodedInt()];
		var count = r.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var i = 0; i < count; i++)
			items[i] = ReadValueInstance(r, package, table, numberType);
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
		var type = EnsureResolvedType(package, table[r.Read7BitEncodedInt()]);
		return new Value(type, new ValueInstance(type, r.ReadBoolean()));
	}

	private static Expression ReadVariableRef(BinaryReader r, Package package, string[] table)
	{
		var name = table[r.Read7BitEncodedInt()];
		var type = EnsureResolvedType(package, table[r.Read7BitEncodedInt()]);
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
		var anyBaseType = EnsureResolvedType(package, Type.Number);
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
			method = EnsureResolvedType(package, typeName).Methods.FirstOrDefault(m => m.Name == operatorName);
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
		var declaringType = EnsureResolvedType(package, declaringTypeName);
		var returnType = EnsureResolvedType(package, returnTypeName);
		var method = FindMethod(declaringType, methodName, paramCount, returnType);
		var declaredReturnType = returnType != method.ReturnType
			? returnType
			: null;
		return new MethodCall(method, instance, args, declaredReturnType);
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
			var declaringType = EnsureResolvedType(package, declaringTypeName);
			var returnType = EnsureResolvedType(package, returnTypeName);
			var method = FindMethod(declaringType, methodName, paramCount, returnType);
			var methodReturnType = returnType != method.ReturnType
				? returnType
				: null;
			methodCall = new MethodCall(method, instance, args, methodReturnType);
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

	private static Type EnsureResolvedType(Package package, string typeName)
	{
		var resolved = package.FindType(typeName) ?? package.FindFullType(typeName);
		if (resolved != null)
			return resolved;
		if (typeName.EndsWith(')') && typeName.Contains('('))
			return package.GetType(typeName);
		if (char.IsLower(typeName[0]))
			throw new TypeNotFoundForBytecode(typeName);
		EnsureTypeExists(package, typeName);
		return package.GetType(typeName);
	}

	private static Method FindMethod(Type type, string methodName, int paramCount, Type returnType)
	{
		var method = type.Methods.FirstOrDefault(existingMethod =>
			existingMethod.Name == methodName && existingMethod.Parameters.Count == paramCount) ??
			type.Methods.FirstOrDefault(existingMethod => existingMethod.Name == methodName);
		if (method != null)
			return method;
		if (type.AvailableMethods.TryGetValue(methodName, out var availableMethods))
		{
			var found = availableMethods.FirstOrDefault(existingMethod =>
				existingMethod.Parameters.Count == paramCount) ?? availableMethods.FirstOrDefault();
			if (found != null)
				return found;
		}
		var methodHeader = BuildMethodHeader(methodName, paramCount, returnType);
		var createdMethod = new Method(type, 0, new MethodExpressionParser(), [methodHeader]);
		type.Methods.Add(createdMethod);
		return createdMethod;
	}

	private static string BuildMethodHeader(string methodName, int paramCount, Type returnType)
	{
		if (paramCount == 0)
			return returnType.IsNone
				? methodName
				: methodName + " " + returnType.Name;
		var parameters = Enumerable.Range(1, paramCount).
			Select(parameterIndex => "argument" + parameterIndex + " " + Type.Number);
		return methodName + "(" + string.Join(", ", parameters) + ") " + returnType.Name;
	}

	public sealed class TypeNotFoundForBytecode(string typeName)
		: Exception("Type '" + typeName + "' not found while deserializing bytecode") { }

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found") { }
}