namespace Strict.VirtualMachine;

public class RemoveStatement(string identifier, Register register)
	: RegisterStatement(register, Instruction.Remove)
{
	public string Identifier { get; } = identifier;
}