namespace Strict.Runtime.Statements;

public sealed class Return(Register register) : RegisterStatement(Instruction.Return, register);