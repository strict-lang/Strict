namespace Strict.VirtualMachine;

public sealed record StoreFromRegisterStatement(Register Register, string Identifier) : Statement(Instruction.StoreFromRegister, Register);