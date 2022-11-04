namespace Strict.VirtualMachine;

public sealed record LoadConstantStatement(Register Register, Instance Instance) : Statement(
	Instruction.LoadConstant, Instance, Register);