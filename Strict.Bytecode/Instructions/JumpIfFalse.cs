namespace Strict.Bytecode.Instructions;

//TODO: remove again, this adds nothing! also register is not written/loaded, wtf is this? See JumpIfNotZero instead
public sealed class JumpIfFalse(int instructionsToSkip, Register register)
	: Jump(instructionsToSkip, InstructionType.JumpIfFalse)
{
	public Register Register { get; } = register;
}
