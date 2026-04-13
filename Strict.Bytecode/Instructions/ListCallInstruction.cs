using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class ListCallInstruction(Register register, Register indexValueRegister,
	string identifier) : RegisterInstruction(InstructionType.ListCall, register)
{
	public ListCallInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), (Register)reader.ReadByte(),
			table.names[reader.Read7BitEncodedInt()]) { }

	public Register IndexValueRegister { get; } = indexValueRegister;
	public string Identifier { get; } = identifier;

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write((byte)IndexValueRegister);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}