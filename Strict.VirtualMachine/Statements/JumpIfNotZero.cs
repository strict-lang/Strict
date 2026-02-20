namespace Strict.Runtime.Statements;

public sealed class JumpIfNotZero(int steps, Register register)
	: JumpIf(Instruction.JumpIfNotZero, steps)
{
	public Register Register { get; } = register;
}
