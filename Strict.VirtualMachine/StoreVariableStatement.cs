namespace Strict.VirtualMachine;

public sealed class StoreVariableStatement : InstanceStatement
{
	public StoreVariableStatement(Instance instance, string identifier) :
		base(instance, Instruction.StoreVariable) =>
		Identifier = identifier;

	public string Identifier { get; }

	public override string ToString() =>
		$"{Instruction.StoreVariable} {Instance.Value} {Identifier}";
}