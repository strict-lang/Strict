namespace Strict.Runtime.Statements;

public abstract class InstanceStatement(Instruction instruction, Instance instance) : Statement(instruction)
{
	public Instance Instance { get; } = instance;
	public override string ToString() => $"{Instruction} {Instance.Value}";
}