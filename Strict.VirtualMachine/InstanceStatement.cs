namespace Strict.VirtualMachine;

public abstract class InstanceStatement : Statement
{
	protected InstanceStatement(Instance instance, Instruction instruction)
	{
		Instance = instance;
		Instruction = instruction;
	}

	public Instance Instance { get; }
	public override Instruction Instruction { get; }
}