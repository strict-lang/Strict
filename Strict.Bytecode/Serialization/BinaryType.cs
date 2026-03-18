using Strict.Bytecode.Instructions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Read or write parsed type data (members, method and the used instructions)
/// </summary>
public sealed class BinaryType
{
	public BinaryType() => typeFullName = "";

	public BinaryType(BinaryReader reader, BinaryExecutable binary, string typeFullName)
	{
		this.binary = binary;
		this.typeFullName = typeFullName;
		ValidateMagicAndVersion(reader);
		table = new NameTable(reader);
		ReadMembers(reader, Members);
		var methodGroups = reader.Read7BitEncodedInt();
		for (var methodGroupIndex = 0; methodGroupIndex < methodGroups; methodGroupIndex++)
		{
			var methodName = table.Names[reader.Read7BitEncodedInt()];
			var overloadCount = reader.Read7BitEncodedInt();
			var overloads = new List<BinaryMethod>(overloadCount);
			for (var overloadIndex = 0; overloadIndex < overloadCount; overloadIndex++)
				overloads.Add(new BinaryMethod(reader, this, methodName));
			MethodGroups[methodName] = overloads;
		}
	}

	public BinaryType(BinaryExecutable binary, string typeFullName,
		Dictionary<string, List<BinaryMethod>> methodGroups,
		IReadOnlyList<BinaryMember>? members = null)
	{
		this.binary = binary;
		this.typeFullName = typeFullName;
		MethodGroups = methodGroups;
		if (members != null)
			Members = [.. members];
	}

	public sealed record BinaryMethod : global::Strict.Bytecode.Serialization.BinaryMethod
	{
		public BinaryMethod(List<BinaryMember> methodParameters, string returnTypeName,
			List<Instruction> methodInstructions)
			: base("", methodParameters, returnTypeName, methodInstructions) { }

		internal BinaryMethod(BinaryReader reader, BinaryType type, string methodName)
			: base(reader, type, methodName) { }
	}

	internal readonly BinaryExecutable? binary;
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

	/*TODO: can we avoid this? remove
	private static Member EnsureMember(Type type, string memberName, string memberTypeName)
	{
		var existing = type.Members.FirstOrDefault(member => member.Name == memberName);
		if (existing != null)
			return existing;
		var member = new Member(type, memberName + " " + memberTypeName, null);
		type.Members.Add(member);
		return member;
	}

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
*/
	private NameTable? table;

	internal void ReadMembers(BinaryReader reader, List<BinaryMember> members)
	{
		var numberOfMembers = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < numberOfMembers; memberIndex++)
			members.Add(new BinaryMember(reader, table!, binary!));
	}

	public List<BinaryMember> Members = new();
	public Dictionary<string, List<BinaryMethod>> MethodGroups = new();
	public NameTable Table => table ?? CreateNameTable();
	public bool UsesConsolePrint =>
		MethodGroups.Values.Any(methods => methods.Any(method => method.UsesConsolePrint));
	public int TotalInstructionCount =>
		MethodGroups.Values.Sum(methods => methods.Sum(method => method.instructions.Count));

	private NameTable CreateNameTable()
	{
		table = new NameTable();
		foreach (var member in Members)
			AddMemberNamesToTable(member);
		foreach (var (methodName, methods) in MethodGroups)
		{
			table.Add(methodName);
			foreach (var method in methods)
			{
				table.Add(method.ReturnTypeName);
				foreach (var parameter in method.parameters)
					AddMemberNamesToTable(parameter);
				foreach (var instruction in method.instructions)
					table.CollectStrings(instruction);
			}
		}
		return table;
	}

	private void AddMemberNamesToTable(BinaryMember member)
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
		writer.Write7BitEncodedInt(MethodGroups.Count);
		foreach (var methodGroup in MethodGroups)
		{
			writer.Write7BitEncodedInt(Table[methodGroup.Key]);
			writer.Write7BitEncodedInt(methodGroup.Value.Count);
			foreach (var method in methodGroup.Value)
				method.Write(writer, this);
		}
	}

	internal void WriteMembers(BinaryWriter writer, IReadOnlyList<BinaryMember> members)
	{
		writer.Write7BitEncodedInt(members.Count);
		foreach (var member in members)
			member.Write(writer, table!);
	}

	public static string ReconstructMethodName(string methodName, BinaryMethod method) =>
		methodName + method.parameters.ToBrackets() + (method.ReturnTypeName == Type.None
			? ""
			: " " + method.ReturnTypeName);
}