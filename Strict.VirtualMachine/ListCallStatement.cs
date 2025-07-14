namespace Strict.Runtime;

public class ListCallStatement(Register register, Register indexValueRegister, string identifier)
	: RegisterStatement(register, Instruction.ListCall)
{
	public Register IndexValueRegister { get; } = indexValueRegister;
	public string Identifier { get; } = identifier;
}