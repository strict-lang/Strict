namespace Strict.Runtime.Instructions;

public sealed class Return(Register register) : RegisterInstruction(InstructionType.Return, register);