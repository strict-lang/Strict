namespace Strict.VirtualMachine;

public record ReturnStatement(Register Register) : Statement(Instruction.Return, Register);