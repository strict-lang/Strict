namespace Strict.VirtualMachine;

public sealed class SetStatement : InstanceStatement
{
	public SetStatement(Instance instance, Register register) : base(instance, Instruction.Set) =>
		Register = register;

	public Register Register { get; }
}