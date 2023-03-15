namespace Strict.VirtualMachine;

public sealed class LoopBeginStatement : Statement
{
	public LoopBeginStatement(string identifier) => Identifier = identifier;
	public string Identifier { get; }
	public override Instruction Instruction => Instruction.LoopBegin;
}