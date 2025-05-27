namespace Strict.VirtualMachine;

public sealed class BinaryStatement(Instruction instruction, params Register[] registers)
	: Statement
{
	public Register[] Registers { get; } = registers;
	public override Instruction Instruction { get; } = instruction;
	public override string ToString() => $"{Instruction} {Registers}";

	public bool IsConditional() =>
		Instruction is >= Instruction.BinaryOperatorsSeparator and < Instruction.ConditionalSeparator;
}