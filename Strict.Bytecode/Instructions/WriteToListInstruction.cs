using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class WriteToListInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.InvokeWriteToList, register)
{
	public WriteToListInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.Names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}