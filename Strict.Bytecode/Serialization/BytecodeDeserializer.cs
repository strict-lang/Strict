using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Loads all <see cref="Instruction" /> generated from BytecodeGenerator back from the compact
/// .strictbinary ZIP file. The VM or executable generation only needs <see cref="BytecodeTypes"/>
/// </summary>
public sealed class BytecodeDeserializer(string FilePath)
{
	/// <summary>
	/// Reads a .strictbinary ZIP and returns <see cref="BytecodeTypes"/> containing all type
	/// metadata (members, method signatures) and instruction bodies for each type.
	/// </summary>
	public BytecodeTypes Deserialize(Package basePackage)
	{
		var package = new Package(basePackage,
			Path.GetFileNameWithoutExtension(FilePath) + "-" + ++packageCounter);
		try
		{
			using var zip = ZipFile.OpenRead(FilePath);
			var bytecodeEntries = zip.Entries.Where(entry =>
				entry.FullName.EndsWith(BytecodeSerializer.BytecodeEntryExtension,
					StringComparison.OrdinalIgnoreCase)).ToList();
			if (bytecodeEntries.Count == 0)
				throw new InvalidBytecodeFileException(BytecodeSerializer.Extension +
					" ZIP contains no " + BytecodeSerializer.BytecodeEntryExtension + " entries");
			var typeEntries = bytecodeEntries.Select(entry => new TypeEntryData(
				GetEntryNameWithoutExtension(entry.FullName),
				ReadAllBytes(entry.Open()))).ToList();
			var result = new BytecodeTypes();
			foreach (var typeEntry in typeEntries)
				result.MethodsPerType[typeEntry.EntryName] =
					ReadTypeMetadataIntoBytecodeTypes(typeEntry, package);
			var runInstructions = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
			var methodInstructions =
				new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
			foreach (var typeEntry in typeEntries)
				ReadTypeInstructions(typeEntry, package, runInstructions, methodInstructions);
			PopulateInstructions(result, typeEntries, runInstructions, methodInstructions);
			return result;
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidBytecodeFileException("Not a valid " + BytecodeSerializer.Extension +
				" ZIP file: " + ex.Message);
		}
	}

	private static void PopulateInstructions(BytecodeTypes result,
		List<TypeEntryData> typeEntries, Dictionary<string, List<Instruction>> runInstructions,
		Dictionary<string, List<Instruction>> methodInstructions)
	{
		foreach (var typeEntry in typeEntries)
		{
			if (!result.MethodsPerType.TryGetValue(typeEntry.EntryName, out var typeMethods))
				continue;
			var typeName = GetTypeNameFromEntryName(typeEntry.EntryName);
			if (runInstructions.TryGetValue(typeName, out var runInstr) && runInstr.Count > 0)
				typeMethods.InstructionsPerMethod[BuildMethodInstructionKey(typeName, Method.Run,
					0)] = runInstr;
			foreach (var (key, instructions) in methodInstructions)
			{
				var parts = key.Split('|');
				if (parts[0] != typeName)
					continue;
				typeMethods.InstructionsPerMethod[key] = instructions;
			}
		}
	}

	public Package? Package { get; private set; }
	public Dictionary<string, List<Instruction>>? Instructions { get; private set; }
	public Dictionary<string, List<Instruction>>? PrecompiledMethods { get; private set; }

	/// <summary>
	/// Deserializes all bytecode entries from in-memory .bytecode payloads.
	/// </summary>
	public BytecodeDeserializer(Dictionary<string, byte[]> entryBytesByType, Package basePackage,
		string packageName = "memory") : this("")
	{
		Package = new Package(basePackage, packageName + "-" + ++packageCounter);
		(Instructions, PrecompiledMethods) = DeserializeAllFromEntries(entryBytesByType, Package);
	}

	private sealed class TypeEntryData(string entryName, byte[] bytes)
	{
		public string EntryName { get; } = entryName;
		public byte[] Bytes { get; } = bytes;
	}

	/// <summary>
	/// Reads type metadata (members and method signatures) from a bytecode entry and returns a
	/// <see cref="BytecodeTypes.TypeMembersAndMethods"/> with the captured data. Also creates
	/// the corresponding Language types for instruction deserialization.
	/// </summary>
	private static BytecodeTypes.TypeMembersAndMethods ReadTypeMetadataIntoBytecodeTypes(
		TypeEntryData typeEntry, Package package)
	{
		var typeMembersAndMethods = new BytecodeTypes.TypeMembersAndMethods();
		using var stream = new MemoryStream(typeEntry.Bytes);
		using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
		_ = ValidateMagicAndVersion(reader);
		var table = new NameTable(reader).ToArray();
		var type = EnsureTypeForEntry(package, typeEntry.EntryName);
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
		{
			var memberName = table[reader.Read7BitEncodedInt()];
			var memberTypeName = ReadTypeReferenceName(reader, table);
			_ = EnsureMember(type, memberName, memberTypeName);
			if (reader.ReadBoolean())
				_ = ReadExpression(reader, package, table); //ncrunch: no coverage
			typeMembersAndMethods.Members.Add(
				new BytecodeTypes.TypeMember(memberName, memberTypeName, null));
		}
		var methodCount = reader.Read7BitEncodedInt();
		for (var methodIndex = 0; methodIndex < methodCount; methodIndex++)
		{
			var methodName = table[reader.Read7BitEncodedInt()];
			var parameterCount = reader.Read7BitEncodedInt();
			var parameters = new string[parameterCount];
			for (var parameterIndex = 0; parameterIndex < parameterCount; parameterIndex++)
			{ //ncrunch: no coverage start
				var parameterName = table[reader.Read7BitEncodedInt()];
				var parameterType = ReadTypeReferenceName(reader, table);
				parameters[parameterIndex] = parameterName + " " + parameterType;
			} //ncrunch: no coverage end
			var returnTypeName = ReadTypeReferenceName(reader, table);
			EnsureMethod(type, methodName, parameters, returnTypeName);
		}
		return typeMembersAndMethods;
	}

	private static void ReadTypeMetadata(TypeEntryData typeEntry, Package package)
	{
		using var stream = new MemoryStream(typeEntry.Bytes);
		using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
		_ = ValidateMagicAndVersion(reader);
		var table = new NameTable(reader).ToArray();
		//TODO: need to think about this, Type is nice, but this is a fake type and maybe we can do without?
		var type = EnsureTypeForEntry(package, typeEntry.EntryName);
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
		{
			var memberName = table[reader.Read7BitEncodedInt()];
			var memberTypeName = ReadTypeReferenceName(reader, table);
			_ = EnsureMember(type, memberName, memberTypeName);
			if (reader.ReadBoolean())
				_ = ReadExpression(reader, package, table); //ncrunch: no coverage
		}
		var methodCount = reader.Read7BitEncodedInt();
		for (var methodIndex = 0; methodIndex < methodCount; methodIndex++)
		{
			var methodName = table[reader.Read7BitEncodedInt()];
			var parameterCount = reader.Read7BitEncodedInt();
			var parameters = new string[parameterCount];
			for (var parameterIndex = 0; parameterIndex < parameterCount; parameterIndex++)
			{ //ncrunch: no coverage start
				var parameterName = table[reader.Read7BitEncodedInt()];
				var parameterType = ReadTypeReferenceName(reader, table);
				parameters[parameterIndex] = parameterName + " " + parameterType;
			} //ncrunch: no coverage end
			var returnTypeName = ReadTypeReferenceName(reader, table);
			EnsureMethod(type, methodName, parameters, returnTypeName);
		}
	}

	private static Member EnsureMember(Type type, string memberName, string memberTypeName)
	{
		var existing = type.Members.FirstOrDefault(member => member.Name == memberName);
		if (existing != null)
			return existing;
		var member = new Member(type, memberName + " " + memberTypeName, null);
		type.Members.Add(member);
		return member;
	}

	private static void EnsureMethod(Type type, string methodName, string[] parameters,
		string returnTypeName)
	{
		if (type.Methods.Any(existingMethod => existingMethod.Name == methodName &&
			existingMethod.Parameters.Count == parameters.Length))
			return; //ncrunch: no coverage
		var header = parameters.Length == 0
			? returnTypeName == Type.None
				? methodName
				: methodName + " " + returnTypeName
			: methodName + "(" + string.Join(", ", parameters) + ") " + returnTypeName;
		type.Methods.Add(new Method(type, 0, new MethodExpressionParser(), [header]));
	}

	private static void ReadTypeInstructions(TypeEntryData typeEntry, Package package,
		Dictionary<string, List<Instruction>> runInstructions,
		Dictionary<string, List<Instruction>> methodInstructions)
	{
		using var stream = new MemoryStream(typeEntry.Bytes);
		using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
		_ = ValidateMagicAndVersion(reader);
		var table = new NameTable(reader).ToArray();
		var typeNameForKey = GetTypeNameFromEntryName(typeEntry.EntryName);
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
		{
			_ = reader.Read7BitEncodedInt();
			_ = ReadTypeReferenceName(reader, table);
			if (reader.ReadBoolean())
				_ = ReadExpression(reader, package, table); //ncrunch: no coverage
		}
		var methodCount = reader.Read7BitEncodedInt();
		for (var methodIndex = 0; methodIndex < methodCount; methodIndex++)
		{
			_ = reader.Read7BitEncodedInt();
			var parameterCount = reader.Read7BitEncodedInt();
			for (var parameterIndex = 0; parameterIndex < parameterCount; parameterIndex++)
			{ //ncrunch: no coverage start
				_ = reader.Read7BitEncodedInt();
				_ = ReadTypeReferenceName(reader, table);
			} //ncrunch: no coverage end
			_ = ReadTypeReferenceName(reader, table);
		}
		var numberType = package.GetType(Type.Number);
		var runInstructionCount = reader.Read7BitEncodedInt();
		runInstructions[typeNameForKey] = ReadInstructions(reader, package, table, numberType,
			runInstructionCount);
		var compiledMethodCount = reader.Read7BitEncodedInt();
		for (var methodIndex = 0; methodIndex < compiledMethodCount; methodIndex++)
		{
			var methodName = table[reader.Read7BitEncodedInt()];
			var parameterCount = reader.Read7BitEncodedInt();
			var instructionCount = reader.Read7BitEncodedInt();
			methodInstructions[BuildMethodInstructionKey(typeNameForKey, methodName,
				parameterCount)] = ReadInstructions(reader, package, table, numberType,
				instructionCount);
		}
	}

	private static List<Instruction> ReadInstructions(BinaryReader reader, Package package,
		string[] table, Type numberType, int instructionCount)
	{
		var instructions = new List<Instruction>(instructionCount);
		for (var instructionIndex = 0; instructionIndex < instructionCount; instructionIndex++)
			instructions.Add(ReadInstruction(reader, package, table, numberType));
		return instructions;
	}

	private static Type EnsureTypeForEntry(Package package, string entryName)
	{
		var segments = entryName.Split(Context.ParentSeparator, StringSplitOptions.RemoveEmptyEntries);
		if (segments.Length == 0)
			throw new InvalidBytecodeFileException("Invalid entry name: " + entryName); //ncrunch: no coverage
		var typeName = segments[^1];
		var existingType = package.FindType(typeName);
		if (existingType != null)
			return existingType;
		var targetPackage = package;
		for (var segmentIndex = 0; segmentIndex < segments.Length - 1; segmentIndex++)
			targetPackage = targetPackage.FindSubPackage(segments[segmentIndex]) ??
				new Package(targetPackage, segments[segmentIndex]);
		return targetPackage.FindDirectType(typeName) != null
			? targetPackage.GetType(typeName)
			: new Type(targetPackage, new TypeLines(typeName, Method.Run));
	}

	internal static string GetTypeNameFromEntryName(string entryName) =>
		entryName.Contains(Context.ParentSeparator)
			? entryName[(entryName.LastIndexOf(Context.ParentSeparator) + 1)..]
			: entryName;

	private static byte[] ReadAllBytes(Stream stream)
	{
		using var memory = new MemoryStream();
		stream.CopyTo(memory);
		return memory.ToArray();
	}

	private static string GetEntryNameWithoutExtension(string fullName)
	{
		var normalized = fullName.Replace('\\', '/');
		var extensionStart = normalized.LastIndexOf('.');
		return extensionStart > 0
			? normalized[..extensionStart]
			: normalized;
	}

	public sealed class InvalidBytecodeFileException(string message) : Exception(
		"Not a valid Strict bytecode (" + BytecodeSerializer.Extension + ") file: " + message);

	private static (Dictionary<string, List<Instruction>> RunInstructions,
		Dictionary<string, List<Instruction>> MethodInstructions)
		DeserializeAllFromEntries(Dictionary<string, byte[]> entryBytesByType, Package package)
	{
		if (entryBytesByType.Count == 0)
			throw new InvalidBytecodeFileException(BytecodeSerializer.Extension + //ncrunch: no coverage
				" ZIP contains no entries");
		var runInstructions = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
		var methodInstructions = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
		foreach (var entry in entryBytesByType)
		{
			var typeEntry = new TypeEntryData(entry.Key, entry.Value);
			ReadTypeMetadata(typeEntry, package);
			ReadTypeInstructions(typeEntry, package, runInstructions, methodInstructions);
		}
		return (runInstructions, methodInstructions);
	}

	private static void EnsureTypeExists(Package package, string typeName)
	{
		if (package.FindDirectType(typeName) == null)
			new Type(package, new TypeLines(typeName, Method.Run)).ParseMembersAndMethods(
				new MethodExpressionParser());
	}

	public static string BuildMethodInstructionKey(string typeName, string methodName,
		int parameterCount) =>
		typeName + "|" + methodName + "|" + parameterCount;

	internal static List<Instruction> DeserializeEntry(Stream entryStream, Package package)
	{
		using var reader = new BinaryReader(entryStream, System.Text.Encoding.UTF8, leaveOpen: true);
		return ReadEntry(reader, package);
	}

	private static List<Instruction> ReadEntry(BinaryReader reader, Package package)
	{
		_ = ValidateMagicAndVersion(reader);
		var tableArray = new NameTable(reader).ToArray();
		var numberType = package.GetType(Type.Number);
		var count = reader.Read7BitEncodedInt();
		var instructions = new List<Instruction>(count);
		for (var index = 0; index < count; index++)
			instructions.Add(ReadInstruction(reader, package, tableArray, numberType));
		return instructions;
	}

	private static byte ValidateMagicAndVersion(BinaryReader reader)
	{
		Span<byte> magic = stackalloc byte[BytecodeSerializer.EntryMagicBytes.Length];
		_ = reader.Read(magic);
		if (!magic.SequenceEqual(BytecodeSerializer.EntryMagicBytes))
			throw new InvalidBytecodeFileException("Entry does not start with 'Strict' magic bytes");
		var fileVersion = reader.ReadByte();
		return fileVersion is 0 or > BytecodeSerializer.Version
			? throw new InvalidVersion(fileVersion)
			: fileVersion;
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
			InstructionType.Print => ReadPrint(reader, table),
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

	private static PrintInstruction ReadPrint(BinaryReader reader, string[] table)
	{
		var textPrefix = table[reader.Read7BitEncodedInt()];
		var hasValue = reader.ReadBoolean();
		if (!hasValue)
			return new PrintInstruction(textPrefix);
		var reg = (Register)reader.ReadByte();
		var valueIsText = reader.ReadBoolean();
		return new PrintInstruction(textPrefix, reg, valueIsText);
	}

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
			ExpressionKind.TextValue => new Text(package, table[reader.Read7BitEncodedInt()]), //ncrunch: no coverage
			ExpressionKind.BooleanValue => ReadBooleanValue(reader, package, table), //ncrunch: no coverage
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
		return new Binary(left, operatorMethod, [right]); //ncrunch: no coverage
	} //ncrunch: no coverage

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
		EnsureTypeExists(package, typeName);
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
			return returnType.IsNone //ncrunch: no coverage
				? methodName
				: methodName + " " + returnType.Name;
		var parameters = Enumerable.Range(0, paramCount).Select(parameterIndex =>
			ParameterNames[parameterIndex % ParameterNames.Length] + " " + Type.Number);
		return methodName + "(" + string.Join(", ", parameters) + ") " + returnType.Name;
	}

	private static readonly string[] ParameterNames =
		["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
	private static int packageCounter;

	private static string ReadTypeReferenceName(BinaryReader reader, string[] table) =>
		reader.ReadByte() switch
		{
			TypeRefNone => Type.None,
			TypeRefBoolean => Type.Boolean, //ncrunch: no coverage
			TypeRefNumber => Type.Number,
			TypeRefText => Type.Text, //ncrunch: no coverage
			TypeRefList => Type.List, //ncrunch: no coverage
			TypeRefDictionary => Type.Dictionary, //ncrunch: no coverage
			TypeRefCustom => table[reader.Read7BitEncodedInt()],
			var unknownType => throw new InvalidBytecodeFileException( //ncrunch: no coverage
				"Unknown type ref: " + unknownType)
		};

	private const byte TypeRefNone = 0;
	private const byte TypeRefBoolean = 1;
	private const byte TypeRefNumber = 2;
	private const byte TypeRefText = 3;
	private const byte TypeRefList = 4;
	private const byte TypeRefDictionary = 5;
	private const byte TypeRefCustom = byte.MaxValue;
}