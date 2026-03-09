namespace Strict.Runtime.Instructions;

public sealed class JumpIfNotZero(int steps, Register register)
	: JumpIf(InstructionType.JumpIfNotZero, steps)
{
	public Register Register { get; } = register;
}
