/*
using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// </summary>
public sealed class BytecodeDeserializer(string FilePath)
{
	/*obs

	private static void PopulateInstructions(StrictBinary result,
		List<TypeEntryData> typeEntries, Dictionary<string, List<Instruction>> runInstructions,
		Dictionary<string, List<Instruction>> methodInstructions)
	{
		foreach (var typeEntry in typeEntries)
		{
			if (!result.MethodsPerType.TryGetValue(typeEntry.EntryName, out var typeMethods))
				continue;
			var typeName = GetTypeNameFromEntryName(typeEntry.EntryName);
			if (runInstructions.TryGetValue(typeName, out var runInstr) && runInstr.Count > 0)
				typeMethods.InstructionsPerMethod[StrictBinary.GetMethodKey(Method.Run, 0, Type.None)] = runInstr;
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
	/// <see cref="StrictBinary.BytecodeMembersAndMethods"/> with the captured data. Also creates
	/// the corresponding Language types for instruction deserialization.
	/// </summary>
	private static StrictBinary.TypeMembersAndMethods ReadTypeMetadataIntoBytecodeTypes(
		TypeEntryData typeEntry, Package package)
	{
		var typeMembersAndMethods = new StrictBinary.TypeMembersAndMethods();
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
				new StrictBinary.TypeMember(memberName, memberTypeName, null));
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
	*
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
			var returnTypeName = table[reader.Read7BitEncodedInt()];
			var instructionCount = reader.Read7BitEncodedInt();
			methodInstructions[StrictBinary.GetMethodKey(methodName, parameterCount, returnTypeName)] =
				ReadInstructions(reader, package, table, numberType, instructionCount);
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

	[Obsolete("Nah")]
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




	/*obs
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
	*
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

	//TODO: remove
	private static readonly string[] ParameterNames =
		["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
	private static int packageCounter;
/*stupid, just use Table!
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
	*/
//}