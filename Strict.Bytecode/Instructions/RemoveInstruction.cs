using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class RemoveInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.InvokeRemove, register)
{
	public RemoveInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}