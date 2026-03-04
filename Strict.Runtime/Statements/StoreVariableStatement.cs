namespace Strict.Runtime.Statements;

public sealed class StoreVariableStatement(Instance constant, string identifier, bool isMember = false)
	: InstanceStatement(Instruction.StoreConstantToVariable, constant)
{
	public string Identifier { get; } = identifier;
	public bool IsMember { get; } = isMember;
	public override string ToString() => $"{base.ToString()} {Identifier}";
}