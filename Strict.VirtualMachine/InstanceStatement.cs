namespace Strict.VirtualMachine;

public abstract class InstanceStatement(Instance instance, Instruction instruction) : Statement
{
	public Instance Instance { get; } = instance;
	public override Instruction Instruction { get; } = instruction;
}