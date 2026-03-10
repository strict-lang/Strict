namespace Strict.Bytecode.Instructions;

public sealed class ReturnInstruction(Register register)
	: RegisterInstruction(InstructionType.Return, register);