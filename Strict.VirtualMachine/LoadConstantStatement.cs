namespace Strict.Runtime;

public sealed class LoadConstantStatement(Register register, Instance instance)
	: RegisterStatement(register, Instruction.LoadConstant)
{
	public Instance Instance { get; } = instance;
}