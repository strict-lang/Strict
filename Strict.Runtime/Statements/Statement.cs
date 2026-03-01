namespace Strict.Runtime.Statements;

public abstract class Statement(Instruction instruction)
{
	public Instruction Instruction { get; } = instruction;
	public override string ToString() => $"{Instruction}";
}