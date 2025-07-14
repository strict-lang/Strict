namespace Strict.Runtime;

public sealed class StoreVariableStatement(Instance instance, string identifier)
	: InstanceStatement(instance, Instruction.StoreVariable)
{
	public string Identifier { get; } = identifier;

	public override string ToString() =>
		$"{Instruction.StoreVariable} {Instance.Value} {Identifier}";
}