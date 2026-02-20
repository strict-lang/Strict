namespace Strict.Runtime.Statements;

public class JumpIf(Instruction instruction, int steps) : Statement(instruction)
{
	public int Steps { get; } = steps;
}
