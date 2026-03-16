using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class StoreFromRegisterInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.StoreRegisterToVariable, register)
{
	public StoreFromRegisterInstruction(BinaryReader reader, string[] table)
		: this((Register)reader.ReadByte(), table[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;
	public override string ToString() => $"{base.ToString()} {Identifier}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
	}
}