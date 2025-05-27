namespace Strict.VirtualMachine;

public sealed class WriteToListStatement(Register register, string identifier)
	: RegisterStatement(register, Instruction.WriteToList)
{
	public string Identifier { get; } = identifier;
}