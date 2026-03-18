namespace Strict.Bytecode.Instructions;

//TODO: remove again, this adds nothing! also register is not written/loaded, wtf is this? See JumpIfNotZero instead
public sealed class JumpIfTrue(int instructionsToSkip, Register register)
	: Jump(instructionsToSkip, InstructionType.JumpIfTrue)
{
	public Register Register { get; } = register;
}
