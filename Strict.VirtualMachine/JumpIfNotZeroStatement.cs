namespace Strict.VirtualMachine;

public sealed class JumpIfNotZeroStatement : JumpIfStatement
{
	public JumpIfNotZeroStatement(int steps, Register register) : base(Instruction.JumpIfNotZero,
		steps) =>
		Register = register;

	public Register Register { get; }
	public override string ToString() => $"{Instruction} {Steps} {Register}";
}