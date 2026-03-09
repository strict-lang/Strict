using Strict.Expressions;

namespace Strict.Runtime.Instructions;

public sealed class StoreVariableInstruction(ValueInstance constant, string identifier,
	bool isMember = false) : InstanceInstruction(InstructionType.StoreConstantToVariable, constant)
{
	public string Identifier { get; } = identifier;
	public bool IsMember { get; } = isMember;
	public override string ToString() => $"{base.ToString()} {Identifier}";
}