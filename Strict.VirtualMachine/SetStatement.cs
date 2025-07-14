namespace Strict.Runtime;

public sealed class SetStatement(Instance instance, Register register)
	: InstanceStatement(instance, Instruction.Set)
{
	public Register Register { get; } = register;
}