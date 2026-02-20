namespace Strict.Runtime.Statements;

public class Jump(int instructionsToSkip, Instruction instruction = Instruction.Jump)
	: Statement(instruction)
{
	public int InstructionsToSkip { get; } = instructionsToSkip;
}