namespace Strict.VirtualMachine;

public sealed class IterationEndStatement : Statement
{
	public IterationEndStatement(int steps) => Steps = steps;
	public int Steps { get; }
	public override Instruction Instruction => Instruction.IterationEnd;
}