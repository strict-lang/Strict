using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class LoadVariableToRegister(Register register, string identifier)
	: RegisterInstruction(InstructionType.LoadVariableToRegister, register)
{
	public LoadVariableToRegister(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;
	public override string ToString() => $"{InstructionType} {Identifier} {Register}";

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}