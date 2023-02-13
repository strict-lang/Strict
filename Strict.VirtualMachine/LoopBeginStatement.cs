namespace Strict.VirtualMachine;

public sealed class LoopBeginStatement : RegisterStatement
{
	public LoopBeginStatement(string identifier, Register iteratorRegister): base(iteratorRegister, Instruction.LoopBegin) => Identifier = identifier;
	public string Identifier { get; }
	public override Instruction Instruction => Instruction.LoopBegin;
}