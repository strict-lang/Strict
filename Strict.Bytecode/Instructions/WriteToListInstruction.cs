namespace Strict.Bytecode.Instructions;

public sealed class WriteToListInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.InvokeWriteToList, register)
{
	public string Identifier { get; } = identifier;
}