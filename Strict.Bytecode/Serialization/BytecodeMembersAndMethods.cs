using Strict.Bytecode.Instructions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

public sealed class BytecodeMembersAndMethods
{
	public List<BytecodeMember> Members = new();
	public Dictionary<string, List<MethodInstructions>> InstructionsPerMethodGroup = new();

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
					instruction.Write(writer, table);
			}
		}
	}

	internal static readonly byte[] StrictMagicBytes = "Strict"u8.ToArray();
	public const byte Version = 1;

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