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
		table = new NameTable(reader, GetPredefinedTableEntries(typeFullName));
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
		var firstMagicByte = reader.ReadByte();
		if (firstMagicByte != StrictMagicByte)
			throw new InvalidBytecodeEntry("Entry does not start with compact 'S' magic byte");
		var secondByte = reader.ReadByte();
		byte fileVersion;
		if (secondByte == (byte)'t')
		{
			var legacyTail = reader.ReadBytes(4);
			if (legacyTail.Length != 4 || legacyTail[0] != (byte)'r' || legacyTail[1] != (byte)'i' ||
				legacyTail[2] != (byte)'c' || legacyTail[3] != (byte)'t')
				throw new InvalidBytecodeEntry("Entry does not start with supported magic bytes");
			fileVersion = reader.ReadByte();
		}
		else
			fileVersion = secondByte;
		if (fileVersion is 0 or > Version)
			throw new InvalidVersion(fileVersion);
	}

	public const string BytecodeEntryExtension = ".bytecode";
	internal const byte StrictMagicByte = (byte)'S';
	public sealed class InvalidBytecodeEntry(string message) : Exception(message);
	public const byte Version = 1;
	public sealed class InvalidVersion(byte fileVersion) : Exception("File version: " +
		fileVersion + ", this runtime only supports up to version " + Version);
	private NameTable? table;

	internal void ReadMembers(BinaryReader reader, List<BinaryMember> members)
	{
		var numberOfMembers = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < numberOfMembers; memberIndex++)
			members.Add(new BinaryMember(reader, table!, binary!));
	}

	public List<BinaryMember> Members = [];
	public Dictionary<string, List<BinaryMethod>> MethodGroups = [];
	public NameTable Table => table ?? CreateNameTable();
	public bool UsesConsolePrint =>
		MethodGroups.Values.Any(methods => methods.Any(method => method.UsesConsolePrint));
	public int TotalInstructionCount =>
		MethodGroups.Values.Sum(methods => methods.Sum(method => method.instructions.Count));

	private static readonly string[] BaseTypeNames =
	[
		Type.None, Type.Any, Type.Boolean, Type.Number, Type.Character, Type.Range, Type.Text,
		Type.Error, Type.ErrorWithValue, Type.Iterator, Type.List, Type.Logger, Type.App,
		Type.System, Type.File, Type.Directory, Type.TextWriter, Type.TextReader, Type.Stacktrace,
		Type.Mutable, Type.Dictionary
	];

	private static IEnumerable<string> GetPredefinedTableEntries(string typeFullName)
	{
		yield return "";
		yield return typeFullName;
		var separatorIndex = typeFullName.LastIndexOf(Context.ParentSeparator);
		yield return separatorIndex == -1
			? typeFullName
			: typeFullName[(separatorIndex + 1)..];
		foreach (var baseTypeName in BaseTypeNames)
		{
			yield return baseTypeName;
			yield return baseTypeName.ToLowerInvariant();
			yield return nameof(Strict) + Context.ParentSeparator + baseTypeName;
			yield return "TestPackage" + Context.ParentSeparator + baseTypeName;
		}
	}

	private NameTable CreateNameTable()
	{
		table = new NameTable(GetPredefinedTableEntries(typeFullName));
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
		writer.Write(StrictMagicByte);
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