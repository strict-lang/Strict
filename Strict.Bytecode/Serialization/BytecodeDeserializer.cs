using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Restores <see cref="Instruction" /> lists from a compact .strictbinary ZIP file by resolving
/// type and method references from a <see cref="Language.Package" />.
/// Results are cached per file path+modification-time so repeated loads skip ZIP I/O entirely.
/// </summary>
public sealed class BytecodeDeserializer
{
	/// <summary>
	/// Deserializes all bytecode from a .strictbinary ZIP, creating a child package for types.
	/// </summary>
	public BytecodeDeserializer(string filePath, Package basePackage)
	{
		var fullPath = Path.GetFullPath(filePath);
		var packageName = Path.GetFileNameWithoutExtension(fullPath);
		Package = new Package(basePackage, packageName);
		try
		{
			using var zip = ZipFile.OpenRead(fullPath);
			Instructions = DeserializeAllFromZip(zip, Package, Path.GetDirectoryName(fullPath));
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid " + BytecodeSerializer.Extension +
				" ZIP file: " + ex.Message);
		}
	}

	public Package Package { get; }
	public Dictionary<string, List<Instruction>> Instructions { get; }

	private static Dictionary<string, List<Instruction>> DeserializeAllFromZip(ZipArchive zip,
		Package package, string? sourceDirectory)
	{
		var bytecodeEntries = zip.Entries.Where(entry =>
			entry.Name.EndsWith(BytecodeEntryExtension, StringComparison.OrdinalIgnoreCase)).ToList();
		if (bytecodeEntries.Count == 0)
			throw new InvalidBytecodeFileException(BytecodeSerializer.Extension +
				" ZIP contains no entries");
		var typeNames = new string[bytecodeEntries.Count];
		for (var entryIndex = 0; entryIndex < bytecodeEntries.Count; entryIndex++)
		{
			typeNames[entryIndex] = Path.GetFileNameWithoutExtension(bytecodeEntries[entryIndex].Name);
			EnsureTypeExists(package, typeNames[entryIndex], sourceDirectory);
		}
		var result = new Dictionary<string, List<Instruction>>(bytecodeEntries.Count, StringComparer.Ordinal);
		for (var entryIndex = 0; entryIndex < bytecodeEntries.Count; entryIndex++)
		{
			using var entryStream = bytecodeEntries[entryIndex].Open();
			result[typeNames[entryIndex]] = DeserializeEntry(entryStream, package);
		}
		return result;
	}

	private const string BytecodeEntryExtension = ".bytecode";

	public sealed class InvalidBytecodeFileException(string message) : Exception(
		"Not a valid Strict bytecode (" + BytecodeSerializer.Extension + ") file: " + message);

	/// <summary>
	/// Deserializes all bytecode entries from in-memory .bytecode payloads.
	/// </summary>
	public BytecodeDeserializer(Dictionary<string, byte[]> entryBytesByType, Package basePackage,
		string packageName = "memory")
	{
		Package = new Package(basePackage, packageName + "-" + ++memoryPackageCounter);
		Instructions = DeserializeAllFromEntries(entryBytesByType, Package);
	}

	private static Dictionary<string, List<Instruction>> DeserializeAllFromEntries(
		Dictionary<string, byte[]> entryBytesByType, Package package)
	{
		if (entryBytesByType.Count == 0)
			throw new InvalidBytecodeFileException(BytecodeSerializer.Extension + //ncrunch: no coverage
				" ZIP contains no entries");
		foreach (var typeName in entryBytesByType.Keys)
			EnsureTypeExists(package, typeName, null);
		var result = new Dictionary<string, List<Instruction>>(entryBytesByType.Count,
			StringComparer.Ordinal);
		foreach (var (typeName, entryBytes) in entryBytesByType)
		{
			using var stream = new MemoryStream(entryBytes);
			result[typeName] = DeserializeEntry(stream, package);
		}
		return result;
	}

	private static void EnsureTypeExists(Package package, string typeName, string? sourceDirectory)
	{
		if (package.FindDirectType(typeName) != null)
			return; //ncrunch: no coverage
		if (sourceDirectory != null)
		{
			var sourceFile = Path.Combine(sourceDirectory, typeName + Type.Extension);
			if (File.Exists(sourceFile))
			{
				new Type(package, new TypeLines(typeName, File.ReadAllLines(sourceFile))).
					ParseMembersAndMethods(new MethodExpressionParser());
				return;
			}
		}
		new Type(package, new TypeLines(typeName, Method.Run)).ParseMembersAndMethods(
			new MethodExpressionParser());
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
		Span<byte> magic = stackalloc byte[BytecodeSerializer.EntryMagicBytes.Length];
		_ = reader.Read(magic);
		if (!magic.SequenceEqual(BytecodeSerializer.EntryMagicBytes))
			throw new InvalidBytecodeFileException("Entry does not start with 'Strict' magic bytes");
		var fileVersion = reader.ReadByte();
		if (fileVersion is 0 or > BytecodeSerializer.Version)
			throw new InvalidVersion(fileVersion);
	}

	public sealed class InvalidVersion(byte fileVersion) : Exception("File version: " +
		fileVersion + ", this runtime only supports up to version " + BytecodeSerializer.Version);

	private static Instruction ReadInstruction(BinaryReader reader, Package package, string[] table,
		Type numberType)
	{
		var type = (InstructionType)reader.ReadByte();
		return type switch
		{
			InstructionType.LoadConstantToRegister => ReadLoadConstant(reader, package, table,
				numberType),
			InstructionType.LoadVariableToRegister => ReadLoadVariable(reader, table),
			InstructionType.StoreConstantToVariable => ReadStoreVariable(reader, package, table,
				numberType),
			InstructionType.StoreRegisterToVariable => ReadStoreFromRegister(reader, table),
			InstructionType.Set => ReadSet(reader, package, table, numberType),
			InstructionType.Invoke => ReadInvoke(reader, package, table),
			InstructionType.Return => new ReturnInstruction((Register)reader.ReadByte()),
			InstructionType.LoopBegin => ReadLoopBegin(reader),
			InstructionType.LoopEnd => new LoopEndInstruction(reader.Read7BitEncodedInt()),
			InstructionType.JumpIfNotZero => ReadJumpIfNotZero(reader),
			InstructionType.Jump => new Jump(reader.Read7BitEncodedInt()),
			InstructionType.JumpIfTrue => new Jump(reader.Read7BitEncodedInt(),
				InstructionType.JumpIfTrue),
			InstructionType.JumpIfFalse => new Jump(reader.Read7BitEncodedInt(),
				InstructionType.JumpIfFalse),
			InstructionType.JumpEnd => new JumpToId(InstructionType.JumpEnd,
				reader.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfFalse => new JumpToId(InstructionType.JumpToIdIfFalse,
				reader.Read7BitEncodedInt()),
			InstructionType.JumpToIdIfTrue => new JumpToId(InstructionType.JumpToIdIfTrue,
				reader.Read7BitEncodedInt()),
			InstructionType.InvokeWriteToList => ReadWriteToList(reader, table),
			InstructionType.InvokeWriteToTable => ReadWriteToTable(reader, table),
			InstructionType.InvokeRemove => ReadRemove(reader, table),
			InstructionType.ListCall => ReadListCall(reader, table),
			_ when IsBinaryOp(type) => ReadBinary(reader, type),
			_ => throw new InvalidBytecodeFileException("Unknown instruction type: " + type) //ncrunch: no coverage
		};
	}

	private static bool IsBinaryOp(InstructionType type) =>
		type is > InstructionType.StoreSeparator and < InstructionType.BinaryOperatorsSeparator;

	private static LoadConstantInstruction ReadLoadConstant(BinaryReader reader, Package package,
		string[] table, Type numberType) =>
		new((Register)reader.ReadByte(), ReadValueInstance(reader, package, table, numberType));

	private static LoadVariableToRegister ReadLoadVariable(BinaryReader reader, string[] table) =>
		new((Register)reader.ReadByte(), table[reader.Read7BitEncodedInt()]);

	private static StoreVariableInstruction ReadStoreVariable(BinaryReader reader, Package package,
		string[] table, Type numberType) =>
		new(ReadValueInstance(reader, package, table, numberType), table[reader.Read7BitEncodedInt()],
			reader.ReadBoolean());

	private static StoreFromRegisterInstruction ReadStoreFromRegister(BinaryReader reader,
		string[] table) =>
		new((Register)reader.ReadByte(), table[reader.Read7BitEncodedInt()]);

	private static SetInstruction ReadSet(BinaryReader reader, Package package, string[] table,
		Type numberType) =>
		new(ReadValueInstance(reader, package, table, numberType), (Register)reader.ReadByte());

	private static BinaryInstruction ReadBinary(BinaryReader reader, InstructionType type)
	{
		var count = reader.ReadByte();
		var registers = new Register[count];
		for (var index = 0; index < count; index++)
			registers[index] = (Register)reader.ReadByte();
		return new BinaryInstruction(type, registers);
	}

	private static Invoke ReadInvoke(BinaryReader reader, Package package, string[] table)
	{
		var register = (Register)reader.ReadByte();
		var (methodCall, registry) = ReadMethodCallData(reader, package, table);
		return new Invoke(register, methodCall!, registry!);
	}

	private static LoopBeginInstruction ReadLoopBegin(BinaryReader reader)
	{
		var register = (Register)reader.ReadByte();
		var isRange = reader.ReadBoolean();
		return isRange
			? new LoopBeginInstruction(register, (Register)reader.Read7BitEncodedInt())
			: new LoopBeginInstruction(register);
	}

	private static JumpIfNotZero ReadJumpIfNotZero(BinaryReader reader) =>
		new(reader.Read7BitEncodedInt(), (Register)reader.ReadByte());

	private static WriteToListInstruction ReadWriteToList(BinaryReader reader, string[] table) =>
		new((Register)reader.ReadByte(), table[reader.Read7BitEncodedInt()]);

	private static WriteToTableInstruction ReadWriteToTable(BinaryReader reader, string[] table) =>
		new((Register)reader.ReadByte(), (Register)reader.ReadByte(),
			table[reader.Read7BitEncodedInt()]);

	private static RemoveInstruction ReadRemove(BinaryReader reader, string[] table) =>
		new(table[reader.Read7BitEncodedInt()], (Register)reader.ReadByte());

	private static ListCallInstruction ReadListCall(BinaryReader reader, string[] table) =>
		new((Register)reader.ReadByte(), (Register)reader.ReadByte(),
			table[reader.Read7BitEncodedInt()]);

	private static ValueInstance ReadValueInstance(BinaryReader reader, Package package,
		string[] table, Type numberType)
	{
		var kind = (ValueKind)reader.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table[reader.Read7BitEncodedInt()]),
			ValueKind.None => new ValueInstance(package.GetType(Type.None)),
			ValueKind.Boolean => new ValueInstance(package.GetType(Type.Boolean), reader.ReadBoolean()),
			ValueKind.SmallNumber => new ValueInstance(numberType, reader.ReadByte()),
			ValueKind.IntegerNumber => new ValueInstance(numberType, reader.ReadInt32()),
			ValueKind.Number => new ValueInstance(numberType, reader.ReadDouble()),
			ValueKind.List => ReadListValueInstance(reader, package, table, numberType),
			ValueKind.Dictionary => ReadDictionaryValueInstance(reader, package, table, numberType),
			_ => throw new InvalidBytecodeFileException("Unknown ValueKind: " + kind)
		};
	}

	private static ValueInstance ReadListValueInstance(BinaryReader reader, Package package,
		string[] table, Type numberType)
	{
		var typeName = table[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var index = 0; index < count; index++)
			items[index] = ReadValueInstance(reader, package, table, numberType);
		return new ValueInstance(package.GetType(typeName), items);
	}

	private static ValueInstance ReadDictionaryValueInstance(BinaryReader reader, Package package,
		string[] table, Type numberType)
	{
		var typeName = table[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new Dictionary<ValueInstance, ValueInstance>(count);
		for (var index = 0; index < count; index++)
		{
			var key = ReadValueInstance(reader, package, table, numberType);
			var value = ReadValueInstance(reader, package, table, numberType);
			items[key] = value;
		}
		return new ValueInstance(package.GetType(typeName), items);
	}

	private static Expression ReadExpression(BinaryReader reader, Package package, string[] table)
	{
		var kind = (ExpressionKind)reader.ReadByte();
		return kind switch
		{
			ExpressionKind.SmallNumberValue => new Number(package, reader.ReadByte()),
			ExpressionKind.IntegerNumberValue => new Number(package, reader.ReadInt32()),
			ExpressionKind.NumberValue =>
				new Number(package, reader.ReadDouble()), //ncrunch: no coverage
			ExpressionKind.TextValue => new Text(package, table[reader.Read7BitEncodedInt()]),
			ExpressionKind.BooleanValue =>
				ReadBooleanValue(reader, package, table), //ncrunch: no coverage
			ExpressionKind.VariableRef => ReadVariableRef(reader, package, table),
			ExpressionKind.MemberRef => ReadMemberRef(reader, package, table),
			ExpressionKind.BinaryExpr => ReadBinaryExpr(reader, package, table),
			ExpressionKind.MethodCallExpr => ReadMethodCallExpr(reader, package, table),
			_ => throw new InvalidBytecodeFileException("Unknown ExpressionKind: " + kind)
		};
	}

	//ncrunch: no coverage start
	private static Value ReadBooleanValue(BinaryReader reader, Package package, string[] table)
	{
		var type = EnsureResolvedType(package, table[reader.Read7BitEncodedInt()]);
		return new Value(type, new ValueInstance(type, reader.ReadBoolean()));
	} //ncrunch: no coverage end

	private static Expression ReadVariableRef(BinaryReader reader, Package package, string[] table)
	{
		var name = table[reader.Read7BitEncodedInt()];
		var type = EnsureResolvedType(package, table[reader.Read7BitEncodedInt()]);
		var param = new Parameter(type, name, new Value(type, new ValueInstance(type)));
		return new ParameterCall(param);
	}

	private static MemberCall ReadMemberRef(BinaryReader reader, Package package, string[] table)
	{
		var memberName = table[reader.Read7BitEncodedInt()];
		var memberTypeName = table[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, package, table)
			: null;
		var anyBaseType = EnsureResolvedType(package, Type.Number);
		var fakeMember = new Member(anyBaseType, memberName + " " + memberTypeName, null);
		return new MemberCall(instance, fakeMember);
	}

	private static Binary ReadBinaryExpr(BinaryReader reader, Package package, string[] table)
	{
		var operatorName = table[reader.Read7BitEncodedInt()];
		var left = ReadExpression(reader, package, table);
		var right = ReadExpression(reader, package, table);
		var operatorMethod = FindOperatorMethod(operatorName, left.ReturnType);
		return new Binary(left, operatorMethod, [right]);
	}

	private static Method FindOperatorMethod(string operatorName, Type preferredType) =>
		preferredType.Methods.FirstOrDefault(m => m.Name == operatorName) ?? throw new
			MethodNotFoundException(operatorName);

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found");

	private static MethodCall ReadMethodCallExpr(BinaryReader reader, Package package,
		string[] table)
	{
		var declaringTypeName = table[reader.Read7BitEncodedInt()];
		var methodName = table[reader.Read7BitEncodedInt()];
		var paramCount = reader.Read7BitEncodedInt();
		var returnTypeName = table[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, package, table)
			: null;
		var argCount = reader.Read7BitEncodedInt();
		var args = new Expression[argCount];
		for (var index = 0; index < argCount; index++)
			args[index] = ReadExpression(reader, package, table);
		var declaringType = EnsureResolvedType(package, declaringTypeName);
		var returnType = EnsureResolvedType(package, returnTypeName);
		var method = FindMethod(declaringType, methodName, paramCount, returnType);
		var declaredReturnType = returnType != method.ReturnType
			? returnType
			: null;
		return new MethodCall(method, instance, args, declaredReturnType);
	}

	private static (MethodCall? MethodCall, Registry? Registry) ReadMethodCallData(
		BinaryReader reader, Package package, string[] table)
	{
		MethodCall? methodCall = null;
		if (reader.ReadBoolean())
		{
			var declaringTypeName = table[reader.Read7BitEncodedInt()];
			var methodName = table[reader.Read7BitEncodedInt()];
			var paramCount = reader.Read7BitEncodedInt();
			var returnTypeName = table[reader.Read7BitEncodedInt()];
			var hasInstance = reader.ReadBoolean();
			var instance = hasInstance
				? ReadExpression(reader, package, table)
				: null;
			var argCount = reader.Read7BitEncodedInt();
			var args = new Expression[argCount];
			for (var index = 0; index < argCount; index++)
				args[index] = ReadExpression(reader, package, table);
			var declaringType = EnsureResolvedType(package, declaringTypeName);
			var returnType = EnsureResolvedType(package, returnTypeName);
			var method = FindMethod(declaringType, methodName, paramCount, returnType);
			var methodReturnType = returnType != method.ReturnType
				? returnType
				: null;
			methodCall = new MethodCall(method, instance, args, methodReturnType);
		}
		Registry? registry = null;
		if (reader.ReadBoolean())
		{
			registry = new Registry();
			var nextRegisterCount = reader.ReadByte();
			var prev = (Register)reader.ReadByte();
			for (var index = 0; index < nextRegisterCount; index++)
				registry.AllocateRegister();
			registry.PreviousRegister = prev;
		}
		return (methodCall, registry);
	}

	private static Type EnsureResolvedType(Package package, string typeName)
	{
		var resolved = package.FindType(typeName) ?? (typeName.Contains('.')
			? package.FindFullType(typeName)
			: null);
		if (resolved != null)
			return resolved;
		if (typeName.EndsWith(')') && typeName.Contains('('))
			return package.GetType(typeName); //ncrunch: no coverage
		if (char.IsLower(typeName[0]))
			throw new TypeNotFoundForBytecode(typeName);
		EnsureTypeExists(package, typeName, null);
		return package.GetType(typeName);
	}

	public sealed class TypeNotFoundForBytecode(string typeName)
		: Exception("Type '" + typeName + "' not found while deserializing bytecode");

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
		} //ncrunch: no coverage
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
		var parameters = Enumerable.Range(0, paramCount).Select(parameterIndex =>
			ParameterNames[parameterIndex % ParameterNames.Length] + " " + Type.Number);
		return methodName + "(" + string.Join(", ", parameters) + ") " + returnType.Name;
	}

	private static readonly string[] ParameterNames =
		["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];

	private static int memoryPackageCounter;
}