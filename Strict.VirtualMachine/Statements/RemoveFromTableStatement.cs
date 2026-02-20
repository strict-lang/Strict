namespace Strict.Runtime.Statements;

public sealed class RemoveFromTableStatement(Register key, string identifier)
	: RegisterStatement(Instruction.RemoveFromTable, key)
{
	public string Identifier { get; } = identifier;
}
