namespace Strict.VirtualMachine;

public sealed class ReturnStatement(Register register)
	: RegisterStatement(register, Instruction.Return);