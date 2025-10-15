namespace Strict.Runtime.Statements;

public sealed class LoadVariableToRegister(string identifier, Register register)
	: RegisterStatement(Instruction.LoadVariableToRegister, register)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{Instruction} {Identifier} {Register}";
}
