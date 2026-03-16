using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public class Jump(int instructionsToSkip, InstructionType instructionType = InstructionType.Jump)
	: Instruction(instructionType)
{
	public Jump(BinaryReader reader, InstructionType instructionType)
		: this(reader.Read7BitEncodedInt(), instructionType) { }

	public int InstructionsToSkip { get; } = instructionsToSkip;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(InstructionsToSkip);
	}
}	