namespace Strict.VirtualMachine;

public sealed record Statement(Instruction Instruction, Register Register, double Value = 0)
{
	public Statement(Instruction instruction, double value = 0) : this(instruction, Register.None,
		value) { }
}