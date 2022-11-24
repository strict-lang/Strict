namespace Strict.VirtualMachine;

public record JumpStatement(Instruction Instruction, int Steps = default, Register RegisterToCheckForZero = default) : Statement(Instruction);