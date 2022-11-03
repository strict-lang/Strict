namespace Strict.VirtualMachine;

public sealed record LoadVariableStatement(Register Register, string Identifier) : Statement(Instruction.Load,
	Register)
{
	public override string ToString() => $"{Instruction} {Register} {Identifier}";
}