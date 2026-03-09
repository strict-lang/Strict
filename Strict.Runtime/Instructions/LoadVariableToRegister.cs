namespace Strict.Runtime.Instructions;

public sealed class LoadVariableToRegister(Register register, string identifier)
	: RegisterInstruction(InstructionType.LoadVariableToRegister, register)
{
	public string Identifier { get; } = identifier;
	public override string ToString() => $"{InstructionType} {Identifier} {Register}";
}
