namespace Strict.Runtime.Statements;

public sealed class SetStatement(Instance instance, Register register)
	: InstanceStatement(Instruction.Set, instance)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{base.ToString()} {Register}";
}
