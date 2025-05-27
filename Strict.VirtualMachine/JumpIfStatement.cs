namespace Strict.VirtualMachine;

public class JumpIfStatement(Instruction jumpInstruction, int steps)
	: JumpStatement(jumpInstruction)
{
	public int Steps { get; } = steps;
}