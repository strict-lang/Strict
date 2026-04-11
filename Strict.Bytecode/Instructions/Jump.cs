using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public class Jump(int instructionsToSkip, InstructionType instructionType = InstructionType.Jump)
	: Instruction(instructionType)
{
	public Jump(BinaryReader reader, InstructionType instructionType)
		: this(reader.Read7BitEncodedInt(), instructionType) { }

	public int InstructionsToSkip { get; } = instructionsToSkip;

	protected override void WritePayload(BinaryWriter writer, NameTable table) =>
		writer.Write7BitEncodedInt(InstructionsToSkip);
}	