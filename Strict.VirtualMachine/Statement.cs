namespace Strict.VirtualMachine;

public abstract class Statement
{
	public abstract Instruction Instruction { get; }
}