namespace Strict.Runtime.Statements;

public sealed class Binary(Register first, Instruction instruction, Register second)
	: Statement(instruction)
{
	public Register First { get; } = first;
	public Register Second { get; } = second;
	public override string ToString() => $"{Instruction} {First} {Second}";

	public bool IsConditional() =>
		Instruction is > Instruction.ArithmeticSeparator and < Instruction.BinaryOperatorsSeparator;
}