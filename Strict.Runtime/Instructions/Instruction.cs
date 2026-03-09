namespace Strict.Runtime.Instructions;

public abstract class Instruction(InstructionType instructionType)
{
	public InstructionType InstructionType { get; } = instructionType;
	public override string ToString() => $"{InstructionType}";
}
