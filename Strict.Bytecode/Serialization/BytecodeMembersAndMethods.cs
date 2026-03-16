using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

public sealed class BytecodeMembersAndMethods
{
	/// <summary>
	/// Reads type metadata (members and method signatures) from a bytecode entry and returns a
	/// <see cref="StrictBinary.BytecodeMembersAndMethods"/> with the captured data. Also creates
	/// the corresponding Language types for instruction deserialization.
	/// </summary>
	public BytecodeMembersAndMethods(BinaryReader reader, StrictBinary binary, string typeFullName)
	{
		this.binary = binary;
		this.typeFullName = typeFullName;
		ValidateMagicAndVersion(reader);
		table = new NameTable(reader);
		var type = EnsureTypeForEntry();
		ReadMembers(reader, Members, type, binary);
		var methodCount = reader.Read7BitEncodedInt();
		for (var methodIndex = 0; methodIndex < methodCount; methodIndex++)
		{
			var methodName = table.Names[reader.Read7BitEncodedInt()];
			var parameterCount = reader.Read7BitEncodedInt();
			var parameters = new string[parameterCount];
			for (var parameterIndex = 0; parameterIndex < parameterCount; parameterIndex++)
			{
				var parameterName = table.Names[reader.Read7BitEncodedInt()];
				var parameterType = table.Names[reader.Read7BitEncodedInt()];
				parameters[parameterIndex] = parameterName + " " + parameterType;
			}
			var returnTypeName = table.Names[reader.Read7BitEncodedInt()];
			EnsureMethod(type, methodName, parameters, returnTypeName);
		}
	}

	private readonly StrictBinary binary;
	private readonly string typeFullName;

	private static void ValidateMagicAndVersion(BinaryReader reader)
	{
		Span<byte> magic = stackalloc byte[StrictMagicBytes.Length];
		_ = reader.Read(magic);
		if (!magic.SequenceEqual(StrictMagicBytes))
			throw new InvalidBytecodeEntry("Entry does not start with 'Strict' magic bytes");
		var fileVersion = reader.ReadByte();
		if (fileVersion is 0 or > Version)
			throw new InvalidVersion(fileVersion);
	}

	public const string BytecodeEntryExtension = ".bytecode";
	internal static readonly byte[] StrictMagicBytes = "Strict"u8.ToArray();
	public sealed class InvalidBytecodeEntry(string message) : Exception(message);
	public const byte Version = 1;

	public sealed class InvalidVersion(byte fileVersion) : Exception("File version: " +
		fileVersion + ", this runtime only supports up to version " + Version);

	private void ReadMembers(BinaryReader reader, List<BytecodeMember> members, Type? checkType,
		StrictBinary binary)
	{
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
			members.Add(new BytecodeMember(reader, table!, binary));
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

	//TODO: can we avoid this?
	private Type EnsureTypeForEntry()
	{
		var segments = typeFullName.Split(Context.ParentSeparator, StringSplitOptions.RemoveEmptyEntries);
		if (segments.Length == 0)
			throw new InvalidBytecodeEntry("Invalid entry name: " + typeFullName); //ncrunch: no coverage
		var typeName = segments[^1];
		var existingType = binary.basePackage.FindType(typeName);
		if (existingType != null)
			return existingType;
		var targetPackage = binary.basePackage;
		for (var segmentIndex = 0; segmentIndex < segments.Length - 1; segmentIndex++)
			targetPackage = targetPackage.FindSubPackage(segments[segmentIndex]) ??
				new Package(targetPackage, segments[segmentIndex]);
		return targetPackage.FindDirectType(typeName) != null
			? targetPackage.GetType(typeName)
			: new Type(targetPackage, new TypeLines(typeName, Method.Run));
	}

	public List<BytecodeMember> Members = new();
	public Dictionary<string, List<MethodInstructions>> InstructionsPerMethodGroup = new();

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

	public record MethodInstructions(IReadOnlyList<BytecodeMember> Parameters,
		string ReturnTypeName, IReadOnlyList<Instruction> Instructions);

	public NameTable Table => table ?? CreateNameTable();
	private NameTable? table;

	private NameTable CreateNameTable()
	{
		table = new NameTable();
		foreach (var member in Members)
			AddMemberNamesToTable(member);
		foreach (var (methodName, methods) in InstructionsPerMethodGroup)
		{
			table.Add(methodName);
			foreach (var method in methods)
			{
				table.Add(method.ReturnTypeName);
				foreach (var parameter in method.Parameters)
					AddMemberNamesToTable(parameter);
				foreach (var instruction in method.Instructions)
					table.CollectStrings(instruction);
			}
		}
		return table;
	}

	private void AddMemberNamesToTable(BytecodeMember member)
	{
		table!.Add(member.Name);
		table.Add(member.FullTypeName);
		if (member.InitialValueExpression != null)
			table.CollectStrings(member.InitialValueExpression);
	}

	public void Write(BinaryWriter writer)
	{
		writer.Write(StrictMagicBytes);
		writer.Write(Version);
		Table.Write(writer);
		WriteMembers(writer, Members);
		writer.Write7BitEncodedInt(InstructionsPerMethodGroup.Count);
		foreach (var methodGroup in InstructionsPerMethodGroup)
		{
			writer.Write7BitEncodedInt(Table[methodGroup.Key]);
			writer.Write7BitEncodedInt(methodGroup.Value.Count);
			foreach (var method in methodGroup.Value)
			{
				WriteMembers(writer, method.Parameters);
				writer.Write7BitEncodedInt(Table[method.ReturnTypeName]);
				foreach (var instruction in method.Instructions)
					instruction.Write(writer, table!);
			}
		}
	}

	private void WriteMembers(BinaryWriter writer, IReadOnlyList<BytecodeMember> members)
	{
		writer.Write7BitEncodedInt(members.Count);
		foreach (var member in members)
			member.Write(writer, table!);
	}

	public static string ReconstructMethodName(string methodName, MethodInstructions method) =>
		methodName + method.Parameters.ToBrackets() + (method.ReturnTypeName == Type.None
			? ""
			: " " + method.ReturnTypeName);
}