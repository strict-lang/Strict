namespace Strict.Bytecode.Instructions;

public sealed class WriteToListInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public string Identifier { get; } = identifier;
}