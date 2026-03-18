namespace Strict.Bytecode.Instructions;

public sealed class JumpIfTrue(int instructionsToSkip, Register register)
	: Jump(instructionsToSkip, InstructionType.JumpIfTrue)
{
	public Register Register { get; } = register;
}
