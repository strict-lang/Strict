namespace Strict.Bytecode.Instructions;

public sealed class JumpIf(InstructionType instructionType, int instructionsToSkip)
	: Jump(instructionsToSkip, instructionType);
