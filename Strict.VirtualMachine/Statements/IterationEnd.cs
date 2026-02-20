namespace Strict.Runtime.Statements;

public sealed class IterationEnd(int steps) : Statement(Instruction.IterationEnd)
{
	public int Steps { get; } = steps;
}
