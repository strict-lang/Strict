namespace Strict.VirtualMachine;

public sealed class LoadConstantStatement(Register register, Instance instance)
	: RegisterStatement(register, Instruction.LoadConstant)
{
	public Instance Instance { get; } = instance;
}