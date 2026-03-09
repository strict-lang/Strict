namespace Strict.Runtime.Instructions;

public class JumpIfTrue(int instructionsToSkip, Register predicate)
	: Jump(instructionsToSkip, InstructionType.JumpIfTrue)
{
	public Register Predicate { get; } = predicate;
}