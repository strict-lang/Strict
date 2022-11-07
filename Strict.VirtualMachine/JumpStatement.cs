namespace Strict.VirtualMachine;

public sealed record JumpStatement(Instruction Instruction, int Steps,
	Register RegisterToCheckForZero = new()) : Statement(Instruction);