using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using System.IO.Compression;
using static Strict.Bytecode.Serialization.BytecodeDeserializer;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// After <see cref="BytecodeGenerator"/> generates all bytecode from the parsed expressions or
/// <see cref="BytecodeDeserializer"/> loads a .strictbinary ZIP file with the same bytecode,
/// this class contains the deserialized bytecode for each type used with each method used.
/// </summary>
public sealed class StrictBinary
{
	public StrictBinary(Package basePackage)
	{
		this.basePackage = basePackage;
		noneType = basePackage.GetType(Type.None);
		booleanType = basePackage.GetType(Type.Boolean);
		numberType = basePackage.GetType(Type.Number);
		characterType = basePackage.GetType(Type.Character);
		rangeType = basePackage.GetType(Type.Range);
		listType = basePackage.GetType(Type.List);
	}

	private readonly Package basePackage;
	internal Type noneType;
	internal Type booleanType;
	internal Type numberType;
	internal Type characterType;
	internal Type rangeType;
	internal Type listType;

	/// <summary>
	/// Reads a .strictbinary ZIP containing all type bytecode (used types, members, methods) and
	/// instruction bodies for each type.
	/// </summary>
	public StrictBinary(string filePath, Package basePackage) : this(basePackage)
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
			var result = new StrictBinary();
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

	/// <summary>
	/// Each key is a type.FullName (e.g. Strict/Number, Strict/ImageProcessing/Color), the Value
	/// contains all members of this type and all not stripped out methods that were actually used.
	/// </summary>
	public Dictionary<string, BytecodeMembersAndMethods> MethodsPerType = new();

	/// <summary>
	/// Writes optimized <see cref="Instruction" /> lists per type into a compact .strictbinary ZIP.
	/// The ZIP contains one entry per type named {typeName}.bytecode.
	/// Entry layout: magic(6) + version(1) + string-table + instruction-count(7bit) + instructions.
	/// </summary>
	public void Serialize(string filePath)
	{
		using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		foreach (var (fullTypeName, membersAndMethods) in MethodsPerType)
		{
			var entry = zip.CreateEntry(fullTypeName + BytecodeEntryExtension, CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			membersAndMethods.Write(writer);
		}
	}

	public const string Extension = ".strictbinary";
	public const string BytecodeEntryExtension = ".bytecode";

	public IReadOnlyList<Instruction>? FindInstructions(Type type, Method method) =>
		FindInstructions(type.FullName, method.Name, method.Parameters.Count, method.ReturnType.Name);

	public IReadOnlyList<Instruction>? FindInstructions(string fullTypeName, string methodName,
		int parametersCount, string returnType = "") =>
		MethodsPerType.TryGetValue(fullTypeName, out var methods)
			? methods.InstructionsPerMethodGroup.GetValueOrDefault(methodName)?.Find(m =>
				m.Parameters.Count == parametersCount && m.ReturnTypeName == returnType)?.Instructions
			: null;

	internal ValueInstance ReadValueInstance(BinaryReader reader, NameTable table)
	{
		var kind = (ValueKind)reader.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table.Names[reader.Read7BitEncodedInt()]),
			ValueKind.None => new ValueInstance(noneType),
			ValueKind.Boolean => new ValueInstance(booleanType, reader.ReadBoolean()),
			ValueKind.SmallNumber => new ValueInstance(numberType, reader.ReadByte()),
			ValueKind.IntegerNumber => new ValueInstance(numberType, reader.ReadInt32()),
			ValueKind.Number => new ValueInstance(numberType, reader.ReadDouble()),
			ValueKind.List => ReadListValueInstance(reader, table),
			ValueKind.Dictionary => ReadDictionaryValueInstance(reader, table),
			_ => throw new InvalidBytecodeFileException("Unknown ValueKind: " + kind)
		};
	}

	private ValueInstance ReadListValueInstance(BinaryReader reader, NameTable table)
	{
		var typeName = table.Names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var index = 0; index < count; index++)
			items[index] = ReadValueInstance(reader, table);
		return new ValueInstance(basePackage.GetType(typeName), items);
	}

	private ValueInstance ReadDictionaryValueInstance(BinaryReader reader, NameTable table)
	{
		var typeName = table.Names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new Dictionary<ValueInstance, ValueInstance>(count);
		for (var index = 0; index < count; index++)
		{
			var key = ReadValueInstance(reader, table);
			var value = ReadValueInstance(reader, table);
			items[key] = value;
		}
		return new ValueInstance(basePackage.GetType(typeName), items);
	}

	internal MethodCall ReadMethodCall(BinaryReader reader, NameTable table)
	{
		var declaringTypeName = table.Names[reader.Read7BitEncodedInt()];
		var methodName = table.Names[reader.Read7BitEncodedInt()];
		var paramCount = reader.Read7BitEncodedInt();
		var returnTypeName = table.Names[reader.Read7BitEncodedInt()];
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
		return new MethodCall(method, instance, args, methodReturnType);
	}
}