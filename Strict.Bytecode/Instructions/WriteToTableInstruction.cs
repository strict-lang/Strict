namespace Strict.Bytecode.Instructions;

public sealed class WriteToTableInstruction(Register key, Register value, string identifier)
	: Instruction(InstructionType.InvokeWriteToTable)
{
	public Register Key { get; } = key;
	public Register Value { get; } = value;
	public string Identifier { get; } = identifier;
}