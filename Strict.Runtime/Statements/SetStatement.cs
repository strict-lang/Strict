using Strict.Expressions;

namespace Strict.Runtime.Statements;

public sealed class SetStatement(ValueInstance valueInstance, Register register)
	: InstanceStatement(Instruction.Set, valueInstance)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{base.ToString()} {Register}";
}
