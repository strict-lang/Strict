namespace Strict.VirtualMachine;

public sealed class SetStatement(Instance instance, Register register)
	: InstanceStatement(instance, Instruction.Set)
{
	public Register Register { get; } = register;
}