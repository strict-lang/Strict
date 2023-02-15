namespace Strict.VirtualMachine;

public sealed class LoadConstantStatement : RegisterStatement
{
	public Instance Instance { get; }

	public LoadConstantStatement(Register register, Instance instance) : base(register,
		Instruction.LoadConstant) =>
		Instance = instance;
}