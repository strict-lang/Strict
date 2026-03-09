namespace Strict.Runtime.Instructions;

public sealed class JumpToId(InstructionType instructionType, int id)
	: Instruction(instructionType)
{
	public int Id { get; } = id;
}
