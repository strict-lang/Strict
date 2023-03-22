namespace Strict.VirtualMachine;

public sealed class WriteToTableStatement : Statement
{
	public WriteToTableStatement(Register key, Register value, string identifier)
	{
		Key = key;
		Value = value;
		Identifier = identifier;
	}

	public Register Key { get; }
	public Register Value { get; }
	public string Identifier { get; }
	public override Instruction Instruction => Instruction.WriteToTable;
}