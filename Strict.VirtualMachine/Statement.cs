namespace Strict.VirtualMachine;

public sealed record Statement(Instruction Instruction, Instance Instance = null!,
	params Register[] Registers)
{
	public Statement(Instruction instruction, params Register[] registers) : this(instruction, null!,
		registers) { }
}