namespace Strict.VirtualMachine;

public sealed class BinaryStatement : Statement
{
	public BinaryStatement(Instruction instruction, params Register[] registers)
	{
		Instruction = instruction;
		Registers = registers;
	}

	public Register[] Registers { get; }
	public override Instruction Instruction { get; }

	public bool IsConditional() =>
		Instruction is >= Instruction.BinaryOperatorsSeparator and < Instruction.ConditionalSeparator;
}