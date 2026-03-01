namespace Strict.Runtime.Statements;

public abstract class RegisterStatement(Instruction instruction, Register register)
	: Statement(instruction)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{Instruction} {Register}";
}