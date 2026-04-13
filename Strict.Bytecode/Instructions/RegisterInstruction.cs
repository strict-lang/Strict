using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public abstract class RegisterInstruction(InstructionType instructionType, Register register)
	: Instruction(instructionType)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{InstructionType} {Register}";

	protected override void WritePayload(BinaryWriter writer, NameTable table) =>
		writer.Write((byte)Register);
}