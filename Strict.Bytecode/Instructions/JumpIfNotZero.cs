using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class JumpIfNotZero(int instructionsToSkip, Register register)
	: Jump(instructionsToSkip, InstructionType.JumpIfNotZero)
{
	public JumpIfNotZero(BinaryReader reader)
		: this(reader.Read7BitEncodedInt(), (Register)reader.ReadByte()) { }

	public Register Register { get; } = register;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write((byte)Register);
	}
}