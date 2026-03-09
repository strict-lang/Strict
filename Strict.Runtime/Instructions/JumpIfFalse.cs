namespace Strict.Runtime.Instructions;

public sealed class JumpIfFalse(int instructionsToSkip, Register predicate)
	: Jump(instructionsToSkip, InstructionType.JumpIfFalse)
{
	public Register Predicate { get; } = predicate;
}