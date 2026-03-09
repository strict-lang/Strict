namespace Strict.Runtime.Instructions;

public sealed class ListCallInstruction(Register register, Register indexValueRegister,
	string identifier) : RegisterInstruction(InstructionType.ListCall, register)
{
	public Register IndexValueRegister { get; } = indexValueRegister;
	public string Identifier { get; } = identifier;
}