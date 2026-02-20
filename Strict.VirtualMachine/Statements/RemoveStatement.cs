namespace Strict.Runtime.Statements;

public sealed class RemoveStatement(string identifier, Register register)
	: RegisterStatement(Instruction.Remove, register)
{
	public string Identifier { get; } = identifier;
}
