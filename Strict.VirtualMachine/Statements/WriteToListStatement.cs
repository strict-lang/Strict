namespace Strict.Runtime.Statements;

public sealed class WriteToListStatement(Register register, string identifier)
	: RegisterStatement(Instruction.Invoke, register)
{
	public string Identifier { get; } = identifier;
}
