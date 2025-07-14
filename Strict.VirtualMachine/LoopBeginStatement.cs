namespace Strict.Runtime;

public sealed class LoopBeginStatement(Register register)
	: RegisterStatement(register, Instruction.LoopBegin);