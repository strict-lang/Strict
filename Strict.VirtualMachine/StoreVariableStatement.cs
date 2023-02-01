namespace Strict.VirtualMachine;

public sealed class StoreVariableStatement : InstanceStatement
{
	public StoreVariableStatement(Instance instance, string identifier) : base(instance,
		Instruction.StoreVariable) =>
		Identifier = identifier;

	public string Identifier { get; }

	public override string ToString() =>
		$"{Instruction.StoreVariable} {Instance.Value} {Identifier}";
}

public sealed class SetStatement : InstanceStatement
{
	public SetStatement(Instance instance, Register register) : base(instance, Instruction.Set) =>
		Register = register;

	public Register Register { get; }
}