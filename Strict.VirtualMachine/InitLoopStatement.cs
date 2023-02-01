namespace Strict.VirtualMachine;

public sealed class InitLoopStatement : Statement
{
	public InitLoopStatement(string identifier) => Identifier = identifier;
	public string Identifier { get; }
	public override Instruction Instruction => Instruction.InitLoopStatement;
}