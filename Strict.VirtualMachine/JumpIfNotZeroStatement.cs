namespace Strict.VirtualMachine;

public sealed class JumpIfNotZeroStatement(int steps, Register register)
	: JumpIfStatement(Instruction.JumpIfNotZero, steps)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{Instruction} {Steps} {Register}";
}