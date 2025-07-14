namespace Strict.Runtime;

public sealed class LoadVariableStatement(Register register, string identifier)
	: RegisterStatement(register, Instruction.Load)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{Instruction} {Register} {Identifier}";
}