namespace Strict.Runtime.Statements;

public sealed class JumpIfFalse(int instructionsToSkip, Register predicate)
	: Jump(instructionsToSkip, Instruction.JumpIfFalse)
{
	public Register Predicate { get; } = predicate;
}