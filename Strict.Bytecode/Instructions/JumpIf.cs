namespace Strict.Bytecode.Instructions;

//TODO: remove again, this adds nothing!
public sealed class JumpIf(InstructionType instructionType, int instructionsToSkip)
	: Jump(instructionsToSkip, instructionType);
