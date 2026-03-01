namespace Strict.Runtime.Statements;

public sealed class LoadVariableToRegister(Register register, string identifier)
	: RegisterStatement(Instruction.LoadVariableToRegister, register)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{Instruction} {Identifier} {Register}";
}
