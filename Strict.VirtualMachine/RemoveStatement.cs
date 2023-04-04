namespace Strict.VirtualMachine;

public class RemoveStatement : RegisterStatement
{
	public RemoveStatement(string identifier, Register register) : base(register,
		Instruction.Remove) =>
		Identifier = identifier;

	public string Identifier { get; }
}