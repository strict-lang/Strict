using Strict.Expressions;

namespace Strict.Runtime.Instructions;

public sealed class SetInstruction(ValueInstance valueInstance, Register register)
	: InstanceInstruction(InstructionType.Set, valueInstance)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{base.ToString()} {Register}";
}
