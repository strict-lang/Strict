namespace Strict.Runtime.Statements;

public sealed class WriteToTableStatement(Register key, Register value, string identifier)
	: Statement(Instruction.Invoke)
{
	public Register Key { get; } = key;
	public Register Value { get; } = value;
	public string Identifier { get; } = identifier;
}
