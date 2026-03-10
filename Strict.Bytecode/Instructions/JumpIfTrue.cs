namespace Strict.Bytecode.Instructions;

public class JumpIfTrue(int instructionsToSkip, Register predicate)
	: Jump(instructionsToSkip, InstructionType.JumpIfTrue)
{
	public Register Predicate { get; } = predicate;
}