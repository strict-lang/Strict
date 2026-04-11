using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class WriteToListInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.InvokeWriteToList, register)
{
	public WriteToListInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}