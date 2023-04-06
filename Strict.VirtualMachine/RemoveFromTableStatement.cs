namespace Strict.VirtualMachine;

public class RemoveFromTableStatement : RegisterStatement
{
	public RemoveFromTableStatement(Register key, string identifier) : base(key,
		Instruction.RemoveFromTable) =>
		Identifier = identifier;

	public string Identifier { get; }
}