namespace Strict.Runtime.Statements;

public sealed class Binary(Instruction instruction, params Register[] registers)
	: Statement(instruction)
{
	public Register[] Registers { get; } = registers;
	public override string ToString() => $"{Instruction} {string.Join(" ", Registers)}";

	public bool IsConditional() =>
		Instruction is > Instruction.ArithmeticSeparator and < Instruction.BinaryOperatorsSeparator;
}