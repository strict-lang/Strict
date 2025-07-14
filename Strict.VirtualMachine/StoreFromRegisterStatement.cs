namespace Strict.Runtime;

public sealed class StoreFromRegisterStatement(Register register, string identifier)
	: RegisterStatement(register, Instruction.StoreFromRegister)
{
	public string Identifier { get; } = identifier;
}