namespace Strict.Runtime;

public sealed class ReturnStatement(Register register)
	: RegisterStatement(register, Instruction.Return);