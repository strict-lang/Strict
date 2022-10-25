namespace Strict.VirtualMachine;

public sealed record LoadStatement(Register Register) : Statement(Instruction.Load,
	Register)
{
	public override string ToString() => $"{Instruction} {Register}";
}