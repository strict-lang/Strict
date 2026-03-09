namespace Strict.Runtime.Statements;

//TODO: Rename to Instruction, this is not really a statement like in programming language, we are doing low level instructions here (yes, they are statements too, but instruction is more clear that we are more on a assembly level)
public abstract class Statement(Instruction instruction)
{
	public Instruction Instruction { get; } = instruction;
	public override string ToString() => $"{Instruction}";
}