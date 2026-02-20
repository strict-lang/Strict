namespace Strict.Runtime.Statements;

public sealed class RemoveFromTableStatement(Register key, string identifier)
	: RegisterStatement(Instruction.Invoke, key)
{
	public string Identifier { get; } = identifier;
}
