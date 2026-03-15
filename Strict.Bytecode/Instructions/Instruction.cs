using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public abstract class Instruction(InstructionType instructionType)
{
	public InstructionType InstructionType { get; } = instructionType;
	public override string ToString() => $"{InstructionType}";

	public virtual void Write(BinaryWriter writer, NameTable table) =>
		writer.Write((byte)InstructionType);
}