namespace Strict.VirtualMachine;

public sealed record Statement(Instruction Instruction, double Value = 0,
	params Register[] Registers)
{
	public Statement(Instruction instruction, params Register[] registers) : this(instruction, 0,
		registers) { }
}