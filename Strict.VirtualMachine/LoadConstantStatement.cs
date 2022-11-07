namespace Strict.VirtualMachine;

public sealed record LoadConstantStatement(Register Register, Instance ConstantInstance) : Statement(
	Instruction.LoadConstant, ConstantInstance, Register);