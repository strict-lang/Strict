namespace Strict.VirtualMachine;

public sealed class WriteToTableStatement(Register key, Register value, string identifier)
	: Statement
{
	public Register Key { get; } = key;
	public Register Value { get; } = value;
	public string Identifier { get; } = identifier;
	public override Instruction Instruction => Instruction.WriteToTable;
}