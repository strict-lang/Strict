namespace Strict.VirtualMachine;

public class ListCallStatement : RegisterStatement
{
	public ListCallStatement(Register register, Register indexValueRegister, string identifier) :
		base(register, Instruction.ListCall)
	{
		Identifier = identifier;
		IndexValueRegister = indexValueRegister;
	}

	public Register IndexValueRegister { get; }
	public string Identifier { get; }
}