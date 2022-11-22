namespace Strict.VirtualMachine;

public sealed record InitLoopStatement(string Identifier) : Statement(Instruction.InitLoopStatement);