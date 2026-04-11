using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class StoreFromRegisterInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.StoreRegisterToVariable, register)
{
	public StoreFromRegisterInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte(), table.names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;
	public override string ToString() => $"{base.ToString()} {Identifier}";

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}