namespace Strict.Runtime.Statements;

public class JumpIfTrue(int instructionsToSkip, Register predicate)
	: Jump(instructionsToSkip, Instruction.JumpIfTrue)
{
	public Register Predicate { get; } = predicate;
}