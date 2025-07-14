namespace Strict.Runtime;

public sealed class JumpToIdStatement(Instruction jumpInstruction, int id)
	: JumpStatement(jumpInstruction)
{
	public int Id { get; } = id;
}