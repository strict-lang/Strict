namespace Strict.VirtualMachine;

public sealed class LoopBeginStatement(Register register)
	: RegisterStatement(register, Instruction.LoopBegin);