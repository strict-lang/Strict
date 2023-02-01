namespace Strict.VirtualMachine;

public sealed class StoreFromRegisterStatement : RegisterStatement
{
	public StoreFromRegisterStatement(Register register, string identifier) : base(register,
		Instruction.StoreFromRegister) =>
		Identifier = identifier;

	public string Identifier { get; }
}