using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class LoadVariableToRegister(Register register, string identifier)
	: RegisterInstruction(InstructionType.LoadVariableToRegister, register)
{
	public LoadVariableToRegister(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.Names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;
	public override string ToString() => $"{InstructionType} {Identifier} {Register}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}