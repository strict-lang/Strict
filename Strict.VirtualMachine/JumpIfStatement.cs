namespace Strict.VirtualMachine;

public class JumpIfStatement : JumpStatement
{
	public JumpIfStatement(Instruction jumpInstruction, int steps) : base(jumpInstruction) =>
		Steps = steps;

	public int Steps { get; }
}