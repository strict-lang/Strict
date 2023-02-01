namespace Strict.VirtualMachine;

public sealed class LoadVariableStatement : RegisterStatement
{
	public string Identifier { get; }

	public LoadVariableStatement(Register register, string identifier) : base(register, Instruction.LoadConstant) =>
		Identifier = identifier;

	public override string ToString() => $"{Instruction} {Register} {Identifier}";
}