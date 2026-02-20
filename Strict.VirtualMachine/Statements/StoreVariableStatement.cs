namespace Strict.Runtime.Statements;

public sealed class StoreVariableStatement(Instance constant, string identifier)
	: InstanceStatement(Instruction.StoreConstantToVariable, constant)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{base.ToString()} {Identifier}";
}