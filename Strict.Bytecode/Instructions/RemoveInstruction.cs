namespace Strict.Bytecode.Instructions;

public sealed class RemoveInstruction(string identifier, Register register)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public string Identifier { get; } = identifier;
}