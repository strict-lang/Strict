namespace Strict.Runtime.Instructions;

public class Jump(int instructionsToSkip, InstructionType instructionType = InstructionType.Jump)
	: Instruction(instructionType)
{
	public int InstructionsToSkip { get; } = instructionsToSkip;
}