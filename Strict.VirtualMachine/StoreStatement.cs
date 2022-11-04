namespace Strict.VirtualMachine;

public sealed record StoreStatement(Instance Instance, string Identifier) : Statement(
	Instruction.StoreVariable, Instance)
{
	public override string ToString() => $"{Instruction.StoreVariable} {Instance?.Value} {Identifier}";
}