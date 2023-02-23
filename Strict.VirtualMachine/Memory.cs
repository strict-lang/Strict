namespace Strict.VirtualMachine;

public sealed class Memory
{
	public Dictionary<Register, Instance> Registers { get; init; } = new();
	public Dictionary<string, Instance> Variables = new();
}