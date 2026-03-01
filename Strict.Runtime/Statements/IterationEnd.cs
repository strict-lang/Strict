namespace Strict.Runtime.Statements;

public sealed class LoopEndStatement(int steps) : Statement(Instruction.LoopEnd)
{
	public int Steps { get; } = steps;
}
