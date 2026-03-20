using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class WriteToTableInstruction(Register key, Register value, string identifier)
	: RegisterInstruction(InstructionType.InvokeWriteToTable, key)
{
	public WriteToTableInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), (Register)reader.ReadByte(),
			table.names[reader.Read7BitEncodedInt()]) { }

	public Register Value { get; } = value;
	public string Identifier { get; } = identifier;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write((byte)Value);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}