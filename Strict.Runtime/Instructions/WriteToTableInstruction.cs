namespace Strict.Runtime.Instructions;

public sealed class WriteToTableInstruction(Register key, Register value, string identifier)
	: Instruction(InstructionType.Invoke)
{
	public Register Key { get; } = key;
	public Register Value { get; } = value;
	public string Identifier { get; } = identifier;
}
