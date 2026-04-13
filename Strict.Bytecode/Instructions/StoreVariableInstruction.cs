using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public sealed class StoreVariableInstruction(ValueInstance constant, string identifier,
	bool isMember = false) : InstanceInstruction(InstructionType.StoreConstantToVariable, constant)
{
	public StoreVariableInstruction(BinaryReader reader, NameTable table, BinaryExecutable binary)
		: this(binary.ReadValueInstance(reader, table), table.names[reader.Read7BitEncodedInt()],
			reader.ReadBoolean()) { }

	public string Identifier { get; } = identifier;
	public bool IsMember { get; } = isMember;
	public override string ToString() => $"{base.ToString()} {Identifier}";

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
		writer.Write(IsMember);
	}
}