namespace Strict.VirtualMachine;

public sealed record StoreStatement(Instance Instance, string Identifier) : Statement(
	Instruction.StoreVariable, Instance, Identifier)
{
	public override string ToString() => $"{Instruction.StoreVariable} {Instance?.Value} {Identifier}";
}