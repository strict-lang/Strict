namespace Strict.Bytecode.Instructions;

public class JumpIf(InstructionType instructionType, int steps) : Instruction(instructionType)
{
	public int Steps { get; } = steps;
}