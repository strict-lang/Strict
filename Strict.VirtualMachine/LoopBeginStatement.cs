namespace Strict.VirtualMachine;

public sealed class LoopBeginStatement : RegisterStatement
{
	public LoopBeginStatement(Register register) : base(register, Instruction.LoopBegin) { }
}