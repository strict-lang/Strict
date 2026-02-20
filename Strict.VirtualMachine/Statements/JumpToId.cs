namespace Strict.Runtime.Statements;

public sealed class JumpToId(Instruction instruction, int id) : Statement(instruction)
{
	public int Id { get; } = id;
}
