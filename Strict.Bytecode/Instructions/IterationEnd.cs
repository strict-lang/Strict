namespace Strict.Bytecode.Instructions;

public sealed class LoopEndInstruction(int steps) : Instruction(InstructionType.LoopEnd)
{
	public int Steps { get; } = steps;
	public LoopBeginInstruction? Begin { get; set; }
}