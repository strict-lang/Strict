namespace Strict.VirtualMachine;

public sealed record LoadStatement(string Name, Register Register) : Statement(Instruction.Load,
	Register)
{
	public override string ToString() => $"{Instruction} {Name} {Register}";
}