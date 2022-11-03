namespace Strict.VirtualMachine;

public record Statement(Instruction Instruction, Instance? Instance, params Register[] Registers)
{
	public Statement(Instruction instruction, params Register[] registers) : this(instruction, null,
		registers) { }

	public Statement(Instruction instruction, Instance instance, string identifier) : this(
		instruction, instance) { }

	public override string ToString() =>
		$"{Instruction} {Instance?.Value}{(Registers.Length > 0 ? string.Join(", ", Registers) : "")}";
}