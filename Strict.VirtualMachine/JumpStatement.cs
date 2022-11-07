namespace Strict.VirtualMachine;

public sealed record JumpStatement
	(Instruction Instruction, int Steps) : Statement(Instruction);