namespace Strict.Runtime.Instructions;

public sealed class StoreFromRegisterInstruction(Register register, string identifier)
	: RegisterInstruction(InstructionType.StoreRegisterToVariable, register)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{base.ToString()} {Identifier}";
}
