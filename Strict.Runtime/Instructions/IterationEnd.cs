namespace Strict.Runtime.Instructions;

public sealed class LoopEndInstruction(int steps) : Instruction(InstructionType.LoopEnd)
{
	public int Steps { get; } = steps;
}