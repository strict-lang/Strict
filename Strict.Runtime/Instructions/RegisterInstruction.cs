namespace Strict.Runtime.Instructions;

public abstract class RegisterInstruction(InstructionType instructionType, Register register)
	: Instruction(instructionType)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{InstructionType} {Register}";
}