namespace Strict.Runtime.Statements;

public sealed class LoopBeginStatement(Register register)
	: RegisterStatement(Instruction.LoopBegin, register);
