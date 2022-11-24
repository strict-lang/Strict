namespace Strict.VirtualMachine;

public sealed record JumpViaIdStatement(Instruction Instruction, int Id) : JumpStatement(Instruction);