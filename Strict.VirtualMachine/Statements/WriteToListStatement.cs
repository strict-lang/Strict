namespace Strict.Runtime.Statements;

public sealed class WriteToListStatement(Register register, string identifier)
	: RegisterStatement(Instruction.WriteToList, register)
{
	public string Identifier { get; } = identifier;
}
