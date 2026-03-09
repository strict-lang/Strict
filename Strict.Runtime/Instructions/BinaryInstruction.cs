namespace Strict.Runtime.Instructions;

public sealed class BinaryInstruction(InstructionType instructionType, params Register[] registers)
	: Instruction(instructionType)
{
	public Register[] Registers { get; } = registers;
	public override string ToString() => $"{InstructionType} {string.Join(" ", Registers)}";

	public bool IsConditional() =>
		InstructionType is > InstructionType.ArithmeticSeparator
			and < InstructionType.BinaryOperatorsSeparator;
}