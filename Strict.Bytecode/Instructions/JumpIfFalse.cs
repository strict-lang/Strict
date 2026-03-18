namespace Strict.Bytecode.Instructions;

public sealed class JumpIfFalse(int instructionsToSkip, Register register)
	: Jump(instructionsToSkip, InstructionType.JumpIfFalse)
{
	public Register Register { get; } = register;
}
