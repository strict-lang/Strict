namespace Strict.VirtualMachine;

public sealed record LoadConstantStatement(Register Register) : Statement(Instruction.LoadConstant, Register);