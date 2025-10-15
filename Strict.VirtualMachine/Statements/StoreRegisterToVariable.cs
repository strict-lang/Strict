namespace Strict.Runtime.Statements;

public sealed class StoreRegisterToVariable(Register register, string identifier)
	: RegisterStatement(Instruction.StoreRegisterToVariable, register)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{base.ToString()} {Identifier}";
}
