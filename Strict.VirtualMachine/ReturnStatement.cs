namespace Strict.VirtualMachine;

public sealed class ReturnStatement : RegisterStatement
{
	public ReturnStatement(Register register) : base(register, Instruction.Return) { }
}