namespace Strict.Runtime.Statements;

public sealed class ListCallStatement(Register register, Register indexValueRegister, string identifier)
	: RegisterStatement(Instruction.ListCall, register)
{
	public Register IndexValueRegister { get; } = indexValueRegister;
	public string Identifier { get; } = identifier;
}
