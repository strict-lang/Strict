namespace Strict.VirtualMachine;

public sealed class WriteToListStatement : RegisterStatement
{
	public string Identifier { get; }
	public WriteToListStatement(Register register, string identifier) : base(register, Instruction.WriteToList) => Identifier = identifier;
}