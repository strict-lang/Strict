namespace Strict.Runtime;

public sealed class IterationEndStatement(int steps) : Statement
{
	public int Steps { get; } = steps;
	public override Instruction Instruction => Instruction.IterationEnd;
}